import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from unimol_tools.data.conformer import get_graph, get_graph_features


def extract_unimol2_graph_tensors(
    smiles: str,
    max_atoms: int = 128,
) -> dict[str, torch.Tensor]:
    """
    Extrait une seule fois les tenseurs topologiques 2D (RDKit sans hydrogènes)
    requis par Uni-Mol2 à partir d'une chaîne SMILES.

    Ces caractéristiques ne dépendent que de la connectivité chimique 2D et sont
    invariantes au choix du conformère 3D ainsi qu'au bruit gaussien (jittering).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES invalide : {smiles!r}")

    # Alignement strict avec le pré-traitement officiel de Uni-Mol2 (atomes lourds uniquement)
    mol = AllChem.RemoveAllHs(Chem.AddHs(mol))
    n_heavy = mol.GetNumAtoms()
    if n_heavy == 0:
        raise ValueError(f"Aucun atome lourd trouvé après suppression des hydrogènes pour : {smiles!r}")

    keep_n = min(n_heavy, max_atoms)
    mask = np.zeros(n_heavy, dtype=bool)
    mask[:keep_n] = True

    node_attr, edge_index, edge_attr = get_graph(mol)
    feat = get_graph_features(edge_attr, edge_index, node_attr, drop_feat=0, mask=mask)

    atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(keep_n)]
    src_tokens = torch.tensor(atomic_nums, dtype=torch.long).clamp(min=1, max=126)

    return {
        "src_tokens": src_tokens,
        "atom_feat": torch.from_numpy(feat["atom_feat"]).to(dtype=torch.long),
        "atom_mask": torch.from_numpy(feat["atom_mask"]).to(dtype=torch.long),
        "edge_feat": torch.from_numpy(feat["edge_feat"]).to(dtype=torch.long),
        "shortest_path": torch.from_numpy(feat["shortest_path"]).to(dtype=torch.long),
        "degree": torch.from_numpy(feat["degree"]).to(dtype=torch.long).clamp(min=0, max=127),
        "pair_type": torch.from_numpy(feat["pair_type"]).to(dtype=torch.long),
    }


def mol_to_unimol2_tensors(
    graph_tensors: dict[str, torch.Tensor],
    heavy_coords: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Assemble les tenseurs topologiques 2D pré-calculés avec les coordonnées 3D
    centrées des atomes lourds d'un conformère pour UniMolV2Model.
    """
    centered_coords = heavy_coords - heavy_coords.mean(dim=0, keepdim=True)
    mol_tensors = dict(graph_tensors)
    mol_tensors["src_coord"] = centered_coords.to(dtype=torch.float32)
    return mol_tensors
