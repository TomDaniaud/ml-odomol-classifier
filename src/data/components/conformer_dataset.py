import logging
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.data.components.unimol_transforms import mol_to_unimol_tensors

log = logging.getLogger(__name__)


class Conformer3DDataset(Dataset):
    """
    Dataset PyTorch prenant les multi-conformères 3D générés par RDKit,
    appliquant les augmentations à la volée (choix du conformère, permutation,
    jittering 3D) et formatant les entrées pour Uni-Mol.
    """
    def __init__(
        self,
        molecules: list[dict[str, Any]],
        label_names: list[str],
        dictionary: Any,
        is_train: bool = True,
        jitter_min: float = 0.01,
        jitter_max: float = 0.04,
        permute_atoms: bool = True,
        max_atoms: int = 256,
        dynamic_augmentation: bool = True,
    ):
        """
        :param data_file: Chemin vers conformers_3d_dataset.pt.
        :param indices: Liste d'indices de molécules (pour séparer proprement Train et Val sans data leakage).
        :param is_train: Si True, active l'augmentation on-the-fly (tirage aléatoire, permutation, bruit 3D).
        :param jitter_min: Écart-type minimal du bruit gaussien 3D (en Angstroms).
        :param jitter_max: Écart-type maximal du bruit gaussien 3D (en Angstroms).
        :param permute_atoms: Si True, mélange l'ordre des atomes en entraînement.
        :param dynamic_augmentation: Active ou désactive les augmentations stochastiques d'entraînement.
        """
        super().__init__()
        self.molecules = molecules
        self.label_names = label_names
        self.dictionary = dictionary
        self.is_train = is_train
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max
        self.permute_atoms = permute_atoms
        self.max_atoms = max_atoms
        self.dynamic_augmentation = dynamic_augmentation

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], torch.Tensor]:
        """
        Retourne un dictionnaire représentant la géométrie 3D d'une molécule
        et le tenseur des labels multi-labels.
        """
        mol_data = self.molecules[index]
        conformers: list[torch.Tensor] = mol_data["conformers"]
        atom_symbols: list[str] = mol_data["atom_symbols"]
        atomic_numbers: torch.Tensor = mol_data["atomic_numbers"]
        labels: torch.Tensor = mol_data["labels"].to(dtype=torch.float32)

        if self.is_train:
            # 1. Tirage aléatoire d'un conformère parmi les ~20 disponibles
            conf_idx = torch.randint(0, len(conformers), (1,)).item()
            coords = conformers[conf_idx].clone()

            # 2. Permutation de l'ordre des atomes (équivalent SMILES randomization)
            if self.permute_atoms and self.dynamic_augmentation:
                perm = torch.randperm(coords.size(0))
                coords = coords[perm]
                atomic_numbers = atomic_numbers[perm]
                atom_symbols = [atom_symbols[i] for i in perm.tolist()]

            # 3. Jittering 3D : ajout de bruit gaussien continu sur (x, y, z)
            if self.jitter_max > 0.0 and self.dynamic_augmentation:
                std = torch.empty(1).uniform_(self.jitter_min, self.jitter_max).item()
                coords = coords + torch.randn_like(coords) * std
        else:
            # validation : deterministe
            coords = conformers[-1].clone()

        mol_tensors = mol_to_unimol_tensors(  # formatte les molécules augmentées au format unimol
            atom_symbols=atom_symbols,
            coords=coords,
            dictionary=self.dictionary,
            max_atoms=self.max_atoms,
        )
        return mol_tensors, labels


def unimol_tensor_collate_fn(
    batch: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
    pad_idx: int = 0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Assemble et padd les tenseurs de tailles variables en un batch 3D
    directement consommable par UniMolModel.forward().
    """
    mol_dicts, labels_list = zip(*batch)

    batch_size = len(mol_dicts)
    max_len = max(item["src_tokens"].size(0) for item in mol_dicts)

    # 1. Padding 1D pour src_tokens (Batch, Max_Len) et src_coord (Batch, Max_Len, 3)
    src_tokens = pad_sequence(
        [item["src_tokens"] for item in mol_dicts],
        batch_first=True,
        padding_value=pad_idx,
    )
    src_coord = pad_sequence(
        [item["src_coord"] for item in mol_dicts],
        batch_first=True,
        padding_value=0.0,
    )

    # 2. Padding 2D pour src_distance et src_edge_type (Batch, Max_Len, Max_Len)
    src_distance = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
    src_edge_type = torch.full(
        (batch_size, max_len, max_len), fill_value=pad_idx, dtype=torch.long
    )

    for i, item in enumerate(mol_dicts):
        length = item["src_tokens"].size(0)
        src_distance[i, :length, :length] = item["src_distance"]
        src_edge_type[i, :length, :length] = item["src_edge_type"]

    batched_tensors = {
        "src_tokens": src_tokens,
        "src_distance": src_distance,
        "src_coord": src_coord,
        "src_edge_type": src_edge_type,
    }

    batched_labels = torch.stack(labels_list, dim=0)

    return batched_tensors, batched_labels
