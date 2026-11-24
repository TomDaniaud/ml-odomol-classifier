import logging
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.components.unimol2_transforms import mol_to_unimol2_tensors

log = logging.getLogger(__name__)


class Conformer3DUniMol2Dataset(Dataset):
    """
    Dataset PyTorch adaptant les multi-conformères 3D existants au format Uni-Mol2 :
    - Filtrage des atomes lourds (sans hydrogènes explicites) via `heavy_masks`.
    - Augmentation à la volée en entraînement (tirage d'un conformère relaxé MMFF94 + jittering 3D).
    - Fusion avec les caractéristiques topologiques 2D RDKit pré-calculées (`graph_cache`).
    """

    def __init__(
        self,
        molecules: list[dict[str, Any]],
        graph_cache: list[dict[str, torch.Tensor]],
        heavy_masks: list[torch.Tensor],
        label_names: list[str],
        is_train: bool = True,
        jitter_min: float = 0.01,
        jitter_max: float = 0.04,
        max_atoms: int = 128,
        dynamic_augmentation: bool = True,
    ):
        """
        :param molecules: Liste des dictionnaires moléculaires issus de `conformers_3d_dataset_100_000.pt`
                          (filtrée sur les indices du split Train ou Val).
        :param graph_cache: Liste alignée des dictionnaires de tenseurs topologiques 2D RDKit pré-calculés
                            pour chaque molécule (`src_tokens`, `atom_feat`, `edge_feat`, `shortest_path`, etc.).
        :param heavy_masks: Liste alignée de masques booléens (`atomic_numbers != 1`) sélectionnant uniquement
                            les atomes lourds dans les conformères 3D.
        :param label_names: Liste ordonnée des noms des 138 descripteurs olfactifs.
        :param is_train: Si True, active l'échantillonnage aléatoire d'un conformère MMFF94 et le bruit 3D.
                         Si False, utilise de façon déterministe le dernier conformère sans bruit.
        :param jitter_min: Écart-type minimal du bruit gaussien 3D injecté sur (x, y, z) (en Angstroms).
        :param jitter_max: Écart-type maximal du bruit gaussien 3D injecté sur (x, y, z) (en Angstroms).
        :param max_atoms: Nombre maximal d'atomes lourds conservés par molécule (troncature de sécurité).
        :param dynamic_augmentation: Active ou désactive le bruit gaussien stochastique en entraînement.
        """
        super().__init__()
        self.molecules = molecules
        self.graph_cache = graph_cache
        self.heavy_masks = heavy_masks
        self.label_names = label_names
        self.is_train = is_train
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max
        self.max_atoms = max_atoms
        self.dynamic_augmentation = dynamic_augmentation

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Retourne le dictionnaire des tenseurs 2D/3D au format Uni-Mol2 pour une molécule
        ainsi que le vecteur cible multi-label associé.
        """
        mol_data = self.molecules[index]
        conformers: list[torch.Tensor] = mol_data["conformers"]
        heavy_mask: torch.Tensor = self.heavy_masks[index]
        labels: torch.Tensor = mol_data["labels"].to(dtype=torch.float32)

        if self.is_train:
            # 1. Tirage aléatoire parmi les conformères optimisés MMFF94
            start_idx = len(conformers) // 4 if len(conformers) >= 4 else 0
            conf_idx = int(torch.randint(start_idx, len(conformers), (1,)).item())
            heavy_coords = conformers[conf_idx][heavy_mask][: self.max_atoms].clone()

            # 2. Jittering 3D : bruit gaussien isotrope sur les atomes lourds
            if self.jitter_max > 0.0 and self.dynamic_augmentation:
                std = torch.empty(1).uniform_(self.jitter_min, self.jitter_max).item()
                heavy_coords = heavy_coords + torch.randn_like(heavy_coords) * std
        else:
            # Validation déterministe sur le dernier conformère optimisé
            heavy_coords = conformers[-1][heavy_mask][: self.max_atoms].clone()

        mol_tensors = mol_to_unimol2_tensors(
            graph_tensors=self.graph_cache[index],
            heavy_coords=heavy_coords,
        )
        return mol_tensors, labels


def unimol2_tensor_collate_fn(
    batch: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
    pad_idx: int = 0,
    attn_pad_value: float = -1e9,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Assemble et padde un batch de molécules de tailles variables pour `UniMolV2Model.forward()`.

    Note critique : `attn_bias` (taille `[B, Max_N + 1, Max_N + 1]` incluant `[CLS]` en indice 0)
    est utilisé comme masque additif avant le `softmax` d'attention dans Uni-Mol2. Les colonnes
    correspondant aux positions de padding (`>= N_i + 1`) sont donc masquées à `-1e9` (et non `0.0`)
    pour empêcher toute fuite d'attention vers les tokens de padding sans risque de `NaN`.
    """
    mol_dicts, labels_list = zip(*batch, strict=True)

    batch_size = len(mol_dicts)
    max_len = max(item["src_tokens"].size(0) for item in mol_dicts)

    # 1. Padding 1D : séquences par atome lourd
    src_tokens = pad_sequence(
        [item["src_tokens"] for item in mol_dicts],
        batch_first=True,
        padding_value=pad_idx,
    )
    degree = pad_sequence(
        [item["degree"] for item in mol_dicts],
        batch_first=True,
        padding_value=pad_idx,
    )
    atom_mask = pad_sequence(
        [item["atom_mask"] for item in mol_dicts],
        batch_first=True,
        padding_value=0,
    )
    atom_feat = pad_sequence(
        [item["atom_feat"] for item in mol_dicts],
        batch_first=True,
        padding_value=pad_idx,
    )
    src_coord = pad_sequence(
        [item["src_coord"] for item in mol_dicts],
        batch_first=True,
        padding_value=0.0,
    )

    # 2. Padding 2D : matrices par paires d'atomes lourds
    shortest_path = torch.full((batch_size, max_len, max_len), fill_value=pad_idx, dtype=torch.long)
    edge_feat = torch.full((batch_size, max_len, max_len, 3), fill_value=pad_idx, dtype=torch.long)
    pair_type = torch.full((batch_size, max_len, max_len, 2), fill_value=pad_idx, dtype=torch.long)

    # 3. Masque additif d'attention incluant le nœud virtuel [CLS] à l'indice 0
    attn_bias = torch.full(
        (batch_size, max_len + 1, max_len + 1),
        fill_value=attn_pad_value,
        dtype=torch.float32,
    )

    for i, item in enumerate(mol_dicts):
        length = item["src_tokens"].size(0)
        shortest_path[i, :length, :length] = item["shortest_path"]
        edge_feat[i, :length, :length, :] = item["edge_feat"]
        pair_type[i, :length, :length, :] = item["pair_type"]
        attn_bias[i, :, : length + 1] = 0.0

    batched_tensors = {
        "atom_feat": atom_feat,
        "atom_mask": atom_mask,
        "edge_feat": edge_feat,
        "shortest_path": shortest_path,
        "degree": degree,
        "pair_type": pair_type,
        "attn_bias": attn_bias,
        "src_tokens": src_tokens,
        "src_coord": src_coord,
    }
    batched_labels = torch.stack(labels_list, dim=0)

    return batched_tensors, batched_labels
