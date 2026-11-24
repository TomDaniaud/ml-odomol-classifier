import json
import logging
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
from lightning import LightningDataModule
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import DataLoader
from unimol_tools.config import MODEL_CONFIG
from unimol_tools.data import Dictionary
from unimol_tools.weights import get_weight_dir, weight_download

from src.data.components.conformer_dataset import Conformer3DDataset, unimol_tensor_collate_fn
from src.data.components.conformer_unimol2_dataset import (
    Conformer3DUniMol2Dataset,
    unimol2_tensor_collate_fn,
)
from src.data.components.unimol2_transforms import extract_unimol2_graph_tensors

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


def scaffold_kfold_split(
    all_molecules: list[dict[str, Any]],
    num_folds: int = 5,
    fold_index: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Découpe les molécules en K plis disjoints par squelettes de Bemis-Murcko (Scaffold K-Fold).
    Chaque famille de squelettes est assignée intégralement à un seul pli afin de garantir
    zéro fuite structurelle entre Train et Val sur chacun des K plis, tout en équilibrant
    la taille des plis.
    """
    if not (0 <= fold_index < num_folds):
        raise ValueError(f"fold_index ({fold_index}) doit être compris entre 0 et {num_folds - 1}.")

    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, mol in enumerate(all_molecules):
        scaff = generate_scaffold(mol.get("smiles", ""))
        scaffold_to_indices[scaff].append(idx)

    rng = np.random.RandomState(seed)
    scaffold_groups = list(scaffold_to_indices.values())
    rng.shuffle(scaffold_groups)
    scaffold_groups.sort(key=len, reverse=True)

    folds: list[list[int]] = [[] for _ in range(num_folds)]
    fold_sizes = np.zeros(num_folds, dtype=np.int64)

    for group in scaffold_groups:
        target_fold = int(np.argmin(fold_sizes))
        folds[target_fold].extend(group)
        fold_sizes[target_fold] += len(group)

    val_idx = np.array(folds[fold_index], dtype=np.int64)
    train_indices: list[int] = []
    for k in range(num_folds):
        if k != fold_index:
            train_indices.extend(folds[k])
    train_idx = np.array(train_indices, dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class OdomolDataModule(LightningDataModule):
    _shared_payload_cache: dict[str, dict[str, Any]] = {}
    _shared_v2_graph_cache: dict[tuple[str, int], tuple[list[dict[str, torch.Tensor]], list[torch.Tensor]]] = {}

    def __init__(
        self,
        data_file: str | Path,
        val_split: float = 0.2,
        split_type: str = "stratified",
        num_folds: int = 1,
        fold_index: int = 0,
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
        backbone_version: str = "unimol",
        max_atoms: int = 256,
        split_file: str | None = None,  # requis pour split_type='benchmark'
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Conformer3DDataset | Conformer3DUniMol2Dataset | None = None
        self.data_val: Conformer3DDataset | Conformer3DUniMol2Dataset | None = None
        self.data_test: Conformer3DDataset | Conformer3DUniMol2Dataset | None = None
        self.pos_weights: torch.Tensor | None = None
        self.train_idx: np.ndarray | None = None
        self.val_idx: np.ndarray | None = None
        self.pad_idx: int = 0

    @property
    def _is_unimol2(self) -> bool:
        version = str(getattr(self.hparams, "backbone_version", "unimol")).lower()
        return version in {"unimol2", "unimolv2", "v2"}

    def setup(self, stage: str | None = None) -> None:
        if self.data_train is not None and self.data_val is not None:
            return

        if not self._is_unimol2:
            # Charger le dictionnaire natif d'Uni-Mol v1
            weight_dir = get_weight_dir()
            dict_name = MODEL_CONFIG["dict"]["molecule_all_h"]
            dict_path = Path(weight_dir) / dict_name
            if not dict_path.is_file():
                weight_download(dict_name, weight_dir)
            with dict_path.open("r", encoding="utf-8") as dictionary_file:
                self.dictionary = Dictionary.load(dictionary_file)
            self.dictionary.add_symbol("[MASK]", is_special=True)
            self.pad_idx = self.dictionary.pad()

        # 1. Chargement unique du fichier .pt en mémoire CPU (mis en cache entre les plis K-Fold)
        data_key = str(Path(self.hparams.data_file).resolve())
        if data_key not in self._shared_payload_cache:
            self._shared_payload_cache[data_key] = torch.load(
                self.hparams.data_file, map_location="cpu", weights_only=False
            )
        payload = self._shared_payload_cache[data_key]
        all_molecules: list[dict[str, Any]] = payload["molecules"]
        label_names: list[str] = payload.get("label_names", [])

        # 2. Extraction de la matrice des labels (N_molecules, 138)
        labels_matrix = np.stack([mol["labels"].numpy() for mol in all_molecules], axis=0)
        indices = np.arange(len(all_molecules))

        # 3. Choix de la stratégie de split (Hold-out classique ou Validation Croisée K-Fold)
        split_choice = str(self.hparams.split_type).lower()
        num_folds = int(getattr(self.hparams, "num_folds", 1))
        fold_index = int(getattr(self.hparams, "fold_index", 0))
        test_idx = None

        if split_choice == "benchmark":
            # Découpage figé du protocole POM (src/scripts/make_benchmark_split.py), en lignes du CSV.
            # Les 6 molécules sans conformère 3D sont absentes du .pt : elles sont ignorées.
            split = json.loads(Path(self.hparams.split_file).read_text())
            pos = {int(m["source_index"]): i for i, m in enumerate(all_molecules)}

            def rows_to_idx(rows: list[int]) -> np.ndarray:
                return np.array([pos[r] for r in rows if r in pos])

            val_idx = rows_to_idx(split["folds"][fold_index])
            train_idx = rows_to_idx([r for k, fold in enumerate(split["folds"]) if k != fold_index for r in fold])
            test_idx = rows_to_idx(split["test"])
        elif num_folds > 1:
            if not (0 <= fold_index < num_folds):
                raise ValueError(
                    f"fold_index ({fold_index}) invalide pour num_folds={num_folds} "
                    f"(doit être dans [0, {num_folds - 1}])."
                )
            if split_choice == "scaffold":
                log.info(f"Application du Scaffold K-Fold Split (pli {fold_index + 1}/{num_folds})...")
                train_idx, val_idx = scaffold_kfold_split(
                    all_molecules=all_molecules,
                    num_folds=num_folds,
                    fold_index=fold_index,
                    seed=self.hparams.seed,
                )
            elif split_choice == "stratified":
                log.info(f"Application du Split K-Fold Stratifié Multi-Label (pli {fold_index + 1}/{num_folds})...")
                mskf = MultilabelStratifiedKFold(
                    n_splits=num_folds,
                    shuffle=True,
                    random_state=self.hparams.seed,
                )
                all_splits = list(mskf.split(indices, labels_matrix))
                train_idx, val_idx = all_splits[fold_index]
            else:
                raise ValueError(
                    f"split_type non reconnu : '{self.hparams.split_type}'. "
                    "Valeurs acceptées : 'stratified', 'scaffold', 'benchmark'."
                )
            log.info(
                f"Split K-Fold [{self.hparams.split_type}, pli {fold_index + 1}/{num_folds}] -> "
                f"Train : {len(train_idx)} molécules ({len(train_idx) / len(all_molecules) * 100:.1f}%), "
                f"Val : {len(val_idx)} molécules ({len(val_idx) / len(all_molecules) * 100:.1f}%)"
            )
        else:
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
                f"Train : {len(train_idx)} molécules ({len(train_idx) / len(all_molecules) * 100:.1f}%), "
                f"Val : {len(val_idx)} molécules ({len(val_idx) / len(all_molecules) * 100:.1f}%)"
            )

        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.val_idx = np.asarray(val_idx, dtype=np.int64)

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

        # 4. Instanciation séparée selon la génération du backbone (Uni-Mol v1 vs Uni-Mol2) :
        if self._is_unimol2:
            max_atoms = int(getattr(self.hparams, "max_atoms", 128))
            v2_cache_key = (data_key, max_atoms)
            if v2_cache_key not in self._shared_v2_graph_cache:
                log.info(
                    f"Pré-calcul en mémoire des graphes topologiques 2D RDKit pour Uni-Mol2 "
                    f"({len(all_molecules)} molécules, max_atoms={max_atoms})..."
                )
                all_graphs = [
                    extract_unimol2_graph_tensors(mol["smiles"], max_atoms=max_atoms) for mol in all_molecules
                ]
                all_heavy_masks = [(mol["atomic_numbers"] != 1) for mol in all_molecules]
                self._shared_v2_graph_cache[v2_cache_key] = (all_graphs, all_heavy_masks)

            all_graphs, all_heavy_masks = self._shared_v2_graph_cache[v2_cache_key]
            train_graphs = [all_graphs[i] for i in train_idx]
            val_graphs = [all_graphs[i] for i in val_idx]
            train_heavy_masks = [all_heavy_masks[i] for i in train_idx]
            val_heavy_masks = [all_heavy_masks[i] for i in val_idx]

            self.data_train = Conformer3DUniMol2Dataset(
                molecules=train_molecules,
                graph_cache=train_graphs,
                heavy_masks=train_heavy_masks,
                label_names=label_names,
                is_train=True,
                jitter_min=self.hparams.jitter_min,
                jitter_max=self.hparams.jitter_max,
                max_atoms=max_atoms,
                dynamic_augmentation=self.hparams.dynamic_augmentation,
            )
            self.data_val = Conformer3DUniMol2Dataset(
                molecules=val_molecules,
                graph_cache=val_graphs,
                heavy_masks=val_heavy_masks,
                label_names=label_names,
                is_train=False,
                max_atoms=max_atoms,
            )
            if test_idx is not None:  # jeu de test tenu à l'écart (split 'benchmark')
                self.data_test = Conformer3DUniMol2Dataset(
                    molecules=[all_molecules[i] for i in test_idx],
                    graph_cache=[all_graphs[i] for i in test_idx],
                    heavy_masks=[all_heavy_masks[i] for i in test_idx],
                    label_names=label_names,
                    is_train=False,
                    max_atoms=max_atoms,
                )
        else:
            # - Train : is_train=True (20 conformères + permutation + jittering 3D)
            # - Val   : is_train=False (1 seul conformère fixe, zéro bruit, zéro permutation)
            max_atoms = int(getattr(self.hparams, "max_atoms", 256))
            self.data_train = Conformer3DDataset(
                molecules=train_molecules,
                label_names=label_names,
                dictionary=self.dictionary,
                is_train=True,
                jitter_min=self.hparams.jitter_min,
                jitter_max=self.hparams.jitter_max,
                permute_atoms=self.hparams.permute_atoms,
                max_atoms=max_atoms,
                dynamic_augmentation=self.hparams.dynamic_augmentation,
            )

            self.data_val = Conformer3DDataset(
                molecules=val_molecules,
                label_names=label_names,
                dictionary=self.dictionary,
                is_train=False,
                max_atoms=max_atoms,
            )
            if test_idx is not None:  # jeu de test tenu à l'écart (split 'benchmark')
                self.data_test = Conformer3DDataset(
                    molecules=[all_molecules[i] for i in test_idx],
                    label_names=label_names,
                    dictionary=self.dictionary,
                    is_train=False,
                    max_atoms=max_atoms,
                )

    def _get_collate_fn(self):
        if self._is_unimol2:
            return unimol2_tensor_collate_fn
        return partial(unimol_tensor_collate_fn, pad_idx=self.pad_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(self.hparams.persistent_workers if self.hparams.num_workers > 0 else False),
            shuffle=True,
            collate_fn=self._get_collate_fn(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(self.hparams.persistent_workers if self.hparams.num_workers > 0 else False),
            shuffle=False,
            collate_fn=self._get_collate_fn(),
        )
