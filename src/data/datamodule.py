import logging
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from functools import partial
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from unimol_tools.data import Dictionary
from unimol_tools.config import MODEL_CONFIG
from unimol_tools.weights import get_weight_dir, weight_download

from src.data.components.conformer_dataset import Conformer3DDataset, unimol_tensor_collate_fn

log = logging.getLogger(__name__)


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Calcule le squelette de Bemis-Murcko pour une molécule SMILES.
    Pour les molécules acycliques (sans cycle), génère un squelette aliphatique générique."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        if scaffold:
            return scaffold
        # Pour les molécules sans cycle, squelette carboné générique
        generic_mol = MurckoScaffold.MakeScaffoldGeneric(mol)
        return Chem.MolToSmiles(generic_mol)
    except Exception:
        return ""


def scaffold_split(
    all_molecules: list[dict[str, Any]],
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sépare les molécules par familles de squelettes moléculaires (Random Scaffold Split).
    - Remplissage du Train en premier (80%) pour que les familles majeures aillent majoritairement en Train.
    - Mélange aléatoire (sans tri par taille) pour garantir une bonne représentativité des 138 odeurs.
    - Garantit une intersection strictement nulle de squelettes entre Train et Val (zéro data leakage).
    """
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, mol in enumerate(all_molecules):
        scaff = generate_scaffold(mol.get("smiles", ""))
        scaffold_to_indices[scaff].append(idx)

    rng = np.random.RandomState(seed)
    scaffold_groups = list(scaffold_to_indices.values())

    # Random Scaffold Split : mélange aléatoire sans tri destructif
    rng.shuffle(scaffold_groups)

    total_mols = len(all_molecules)
    train_cutoff = int(total_mols * (1.0 - val_split))

    train_indices: list[int] = []
    val_indices: list[int] = []

    for group in scaffold_groups:
        if len(train_indices) + len(group) <= train_cutoff:
            train_indices.extend(group)
        else:
            val_indices.extend(group)

    train_idx = np.array(train_indices, dtype=np.int64)
    val_idx = np.array(val_indices, dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class OdomolDataModule(LightningDataModule):
    def __init__(
        self,
        data_file: str | Path,
        val_split: float = 0.2,
        split_type: str = "stratified",
        batch_size: int = 32,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        seed: int = 42,
        jitter_min: float = 0.01,
        jitter_max: float = 0.04,
        permute_atoms: bool = True,
        max_pos_weight: float = 20.0,  # Plafond anti-overfitting
        dynamic_augmentation: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Conformer3DDataset | None = None
        self.data_val: Conformer3DDataset | None = None
        self.pos_weights: torch.Tensor | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.data_train is not None and self.data_val is not None:
            return

        # Charger le dictionnaire natif d'Uni-Mol
        weight_dir = get_weight_dir()
        dict_name = MODEL_CONFIG["dict"]["molecule_all_h"]
        dict_path = Path(weight_dir) / dict_name
        if not dict_path.is_file():
            weight_download(dict_name, weight_dir)
        with dict_path.open("r", encoding="utf-8") as dictionary_file:
            self.dictionary = Dictionary.load(dictionary_file)
        self.pad_idx = self.dictionary.pad()

        # 1. Chargement unique du fichier .pt en mémoire CPU
        payload = torch.load(
            self.hparams.data_file, map_location="cpu", weights_only=False
        )
        all_molecules: list[dict[str, Any]] = payload["molecules"]
        label_names: list[str] = payload.get("label_names", [])

        # 2. Extraction de la matrice des labels (N_molecules, 138)
        labels_matrix = np.stack(
            [mol["labels"].numpy() for mol in all_molecules], axis=0
        )
        indices = np.arange(len(all_molecules))

        # 3. Choix de la stratégie de split
        split_choice = str(self.hparams.split_type).lower()
        if split_choice == "scaffold":
            log.info("Application du Scaffold Split (Bemis-Murcko)...")
            train_idx, val_idx = scaffold_split(
                all_molecules=all_molecules,
                val_split=self.hparams.val_split,
                seed=self.hparams.seed,
            )
        elif split_choice == "stratified":
            log.info("Application du Split Stratifié Multi-Label...")
            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=self.hparams.val_split,
                random_state=self.hparams.seed,
            )
            train_idx, val_idx = next(msss.split(indices, labels_matrix))
        else:
            raise ValueError(
                f"split_type non reconnu : '{self.hparams.split_type}'. "
                "Valeurs acceptées : 'stratified', 'scaffold'."
            )

        log.info(
            f"Split effectué [{self.hparams.split_type}] -> "
            f"Train : {len(train_idx)} molécules ({len(train_idx)/len(all_molecules)*100:.1f}%), "
            f"Val : {len(val_idx)} molécules ({len(val_idx)/len(all_molecules)*100:.1f}%)"
        )

        # Vérification de la couverture des 138 odeurs dans Train et Val
        val_labels = labels_matrix[val_idx]
        train_pos = labels_matrix[train_idx].sum(axis=0)
        val_pos = val_labels.sum(axis=0)
        zero_in_train = int((train_pos == 0).sum())
        zero_in_val = int((val_pos == 0).sum())
        if zero_in_train > 0 or zero_in_val > 0:
            log.warning(
                f"Attention déséquilibre de split : {zero_in_train} classe(s) absente(s) du Train, "
                f"{zero_in_val} classe(s) absente(s) de Val."
            )
        else:
            log.info("Parfaite représentativité : les 138 classes d'odeurs sont toutes présentes dans Train et Val !")

        # --- CALCUL DYNAMIQUE DE POS_WEIGHT SUR LE TRAIN UNIQUEMENT ---
        train_labels = labels_matrix[train_idx]  # Shape: (N_train, 138)
        num_train = len(train_idx)

        pos_counts = train_labels.sum(axis=0)
        neg_counts = num_train - pos_counts

        # Protection contre la division par zéro et clipping pour éviter l'explosion de gradient
        pos_counts_safe = np.clip(pos_counts, a_min=1.0, a_max=None)
        raw_weights = neg_counts / pos_counts_safe
        clipped_weights = np.clip(raw_weights, a_min=1.0, a_max=self.hparams.max_pos_weight)

        self.pos_weights = torch.tensor(clipped_weights, dtype=torch.float32)

        train_molecules = [all_molecules[i] for i in train_idx]
        val_molecules = [all_molecules[i] for i in val_idx]

        # 4. Instanciation séparée :
        # - Train : is_train=True (20 conformères + permutation + jittering 3D)
        # - Val   : is_train=False (1 seul conformère fixe, zéro bruit, zéro permutation)
        self.data_train = Conformer3DDataset(
            molecules=train_molecules,
            label_names=label_names,
            dictionary=self.dictionary,
            is_train=True,
            jitter_min=self.hparams.jitter_min,
            jitter_max=self.hparams.jitter_max,
            permute_atoms=self.hparams.permute_atoms,
            dynamic_augmentation=self.hparams.dynamic_augmentation,
        )

        self.data_val = Conformer3DDataset(
            molecules=val_molecules,
            label_names=label_names,
            dictionary=self.dictionary,
            is_train=False,
        )

    def train_dataloader(self) -> DataLoader:
        collate_fn = partial(unimol_tensor_collate_fn, pad_idx=self.pad_idx)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers if self.hparams.num_workers > 0 else False
            ),
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        collate_fn = partial(unimol_tensor_collate_fn, pad_idx=self.pad_idx)
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers if self.hparams.num_workers > 0 else False
            ),
            shuffle=False,
            collate_fn=collate_fn,
        )
