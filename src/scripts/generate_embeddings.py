#!/usr/bin/env python
"""
Script to precompute and store Uni-Mol embeddings for odor molecules.

This script loads the molecule SMILES and multi-label odor descriptors from
`data/smiles_odor_dataset.csv`, computes the 3D molecular representation
(CLS embeddings, dimension 512) using Uni-Mol (`unimol-tools`), and saves
the resulting dataset in a single PyTorch file: `data/unimol_embeddings.pt`.

Saved dictionary keys:
- "embeddings": torch.FloatTensor of shape (N, 512)
- "labels": torch.FloatTensor of shape (N, 138)
- "smiles": list of str (length N)
- "label_names": list of str (138 descriptor names)
- "model_name": str ("unimolv1")
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.progress import track
from unimol_tools import UniMolRepr

console = Console()


def generate_and_save_embeddings(
    csv_path: Path | str = "data/smiles_odor_dataset.csv",
    output_path: Path | str = "data/unimol_embeddings.pt",
    model_name: str = "unimolv1",
    batch_size: int = 64,
):
    csv_file = Path(csv_path).resolve()
    output_file = Path(output_path).resolve()

    console.print("[bold cyan]=== Génération des Embeddings Uni-Mol ===[/bold cyan]")
    console.print(f"Fichier source : [green]{csv_file}[/green]")
    console.print(f"Fichier de sortie : [green]{output_file}[/green]")
    console.print(f"Modèle : [magenta]{model_name}[/magenta] | Batch size : [yellow]{batch_size}[/yellow]")

    if not csv_file.exists():
        raise FileNotFoundError(f"Le fichier {csv_file} est introuvable !")

    # 1. Chargement du CSV
    console.print("\n[bold]1. Lecture du fichier CSV...[/bold]")
    df = pd.read_csv(csv_file)
    smiles_col = df.columns[0]
    label_cols = list(df.columns[2:])

    smiles_list = df[smiles_col].astype(str).tolist()
    labels_matrix = df[label_cols].to_numpy(dtype=np.float32)

    total_mols = len(smiles_list)
    total_labels = len(label_cols)
    console.print(f"Nombre de molécules : [bold]{total_mols}[/bold]")
    console.print(f"Nombre de descripteurs d'odeurs : [bold]{total_labels}[/bold]")

    # 2. Initialisation de Uni-Mol
    use_cuda = torch.cuda.is_available()
    console.print(f"\n[bold]2. Initialisation de UniMolRepr (CUDA={use_cuda})...[/bold]")
    clf = UniMolRepr(
        data_type="molecule",
        remove_hs=False,
        model_name=model_name,
        use_cuda=use_cuda,
    )

    # 3. Calcul des embeddings par lots
    console.print(f"\n[bold]3. Calcul des embeddings (par lots de {batch_size})...[/bold]")
    all_embeddings = []
    valid_indices = []
    failed_molecules = []

    batches = [
        (i, smiles_list[i : i + batch_size], list(range(i, min(i + batch_size, total_mols))))
        for i in range(0, total_mols, batch_size)
    ]

    for start_idx, batch_smiles, batch_indices in track(batches, description="Calcul Uni-Mol"):
        try:
            reprs = clf.get_repr(batch_smiles, return_atomic_reprs=False)
            batch_emb = np.stack(reprs, axis=0)
            all_embeddings.append(batch_emb)
            valid_indices.extend(batch_indices)
        except Exception as e:
            console.print(f"[yellow]Avertissement sur le lot {start_idx} : tentative individuelle ({e})[/yellow]")
            for idx, s in zip(batch_indices, batch_smiles, strict=False):
                try:
                    single_repr = clf.get_repr([s], return_atomic_reprs=False)
                    all_embeddings.append(np.array(single_repr[0]).reshape(1, -1))
                    valid_indices.append(idx)
                except Exception as single_err:
                    failed_molecules.append((idx, s, str(single_err)))

    embeddings_np = np.concatenate(all_embeddings, axis=0)
    embeddings_tensor = torch.from_numpy(embeddings_np).float()
    labels_tensor = torch.from_numpy(labels_matrix[valid_indices]).float()
    filtered_smiles = [smiles_list[idx] for idx in valid_indices]

    console.print("\n[bold green]Embeddings calculés avec succès ![/bold green]")
    console.print(f"Dimensions embeddings : [cyan]{tuple(embeddings_tensor.shape)}[/cyan]")
    console.print(f"Dimensions labels : [cyan]{tuple(labels_tensor.shape)}[/cyan]")
    if failed_molecules:
        console.print(f"[red]Molécules échouées ({len(failed_molecules)}) :[/red] {failed_molecules[:5]}")

    # 4. Sauvegarde
    console.print(f"\n[bold]4. Sauvegarde dans {output_file}...[/bold]")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": embeddings_tensor,
        "labels": labels_tensor,
        "smiles": filtered_smiles,
        "label_names": label_cols,
        "model_name": model_name,
    }
    torch.save(payload, output_file)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    console.print(f"[bold green]Sauvegarde terminée ! Taille : {size_mb:.2f} Mo[/bold green]")

    # 5. Vérification
    console.print("\n[bold]5. Vérification de relecture...[/bold]")
    loaded = torch.load(output_file, map_location="cpu", weights_only=False)
    assert "embeddings" in loaded and "labels" in loaded
    assert loaded["embeddings"].shape[0] == loaded["labels"].shape[0]
    has_nan = torch.isnan(loaded["embeddings"]).any().item()
    console.print(f"Vérification NaN dans les embeddings : [cyan]{has_nan}[/cyan]")
    console.print("[bold green]Tout est prêt et validé ![/bold green]")


if __name__ == "__main__":
    generate_and_save_embeddings()
