import torch

def mol_to_unimol_tensors(
    atom_symbols: list[str],
    coords: torch.Tensor,
    dictionary: object,
    max_atoms: int = 256
) -> dict[str, torch.Tensor]:
    """
    Convertit une géométrie 3D (atomes + coordonnées) en tenseurs paddables 
    prêts pour UniMolModel.
    """
    bos_idx = dictionary.bos()
    eos_idx = dictionary.eos()
    unk_idx = dictionary.unk()
    vocab_size = len(dictionary)

    # 1. Troncature de sécurité
    if len(atom_symbols) > max_atoms:
        atom_symbols = atom_symbols[:max_atoms]
        coords = coords[:max_atoms]

    # 2. Centrage des coordonnées 3D (standard obligatoire d'Uni-Mol)
    coords = coords - coords.mean(dim=0, keepdim=True)

    # 3. Tokens d'atomes avec [CLS] et [SEP]
    atom_ids = [
        dictionary.index(sym) if sym in dictionary else unk_idx
        for sym in atom_symbols
    ]
    src_tokens = torch.tensor([bos_idx] + atom_ids + [eos_idx], dtype=torch.long)

    # 4. Coordonnées avec (0,0,0) pour les tokens spéciaux
    zero_coord = torch.zeros((1, 3), dtype=torch.float32)
    src_coord = torch.cat([zero_coord, coords, zero_coord], dim=0)

    # 5. Matrice de distances 3D
    src_distance = torch.zeros((len(src_tokens), len(src_tokens)), dtype=torch.float32)
    src_distance[1:-1, 1:-1] = torch.cdist(coords, coords, p=2.0)

    # 6. Types d'arêtes
    src_edge_type = src_tokens.view(-1, 1) * vocab_size + src_tokens.view(1, -1)

    return {
        "src_tokens": src_tokens,
        "src_coord": src_coord,
        "src_distance": src_distance,
        "src_edge_type": src_edge_type,
    }