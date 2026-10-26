from pathlib import Path
from typing import Any
import numpy as np
import torch
from functools import partial
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from unimol_tools.data import Dictionary
from unimol_tools.config import MODEL_CONFIG
from unimol_tools.weights import get_weight_dir, weight_download

from src.data.components.conformer_dataset import Conformer3DDataset, unimol_tensor_collate_fn


class OdomolDataModule(LightningDataModule):
    def __init__(
        self,
        data_file: str | Path,
        val_split: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        seed: int = 42,
        jitter_min: float = 0.01,
        jitter_max: float = 0.04,
        permute_atoms: bool = True,
        max_pos_weight: float = 20.0,  # Plafond anti-overfitting
        dynamic_augmentation: bool = True
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

        # 2. Extraction de la matrice des labels (N_molecules, 138) pour stratifier
        labels_matrix = np.stack(
            [mol["labels"].numpy() for mol in all_molecules], axis=0
        )
        indices = np.arange(len(all_molecules))

        # 3. Split stratifié multi-label au niveau des MOLÉCULES (aucun conformère partagé)
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=self.hparams.val_split,
            random_state=self.hparams.seed,
        )
        train_idx, val_idx = next(msss.split(indices, labels_matrix))

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