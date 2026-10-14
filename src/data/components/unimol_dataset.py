import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class UnimolOdorDataset(Dataset):
    """
    Dataset PyTorch chargeant directement les embeddings précalculés Uni-Mol
    et les cibles multi-label associées depuis un fichier .pt.
    """

    def __init__(
        self,
        data_file: str | Path = "data/unimol_embeddings.pt",
    ):
        super().__init__()
        self.file_path = Path(data_file)
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Le fichier d'embeddings {self.file_path} est introuvable. "
                "Veuillez exécuter src/scripts/generate_embeddings.py d'abord."
            )

        log.info(f"Chargement des embeddings depuis {self.file_path}...")
        checkpoint = torch.load(self.file_path, map_location="cpu", weights_only=False)

        self.embeddings: torch.Tensor = checkpoint["embeddings"]  # Shape: (N, 512)
        self.labels: torch.Tensor = checkpoint["labels"]  # Shape: (N, 138)
        self.smiles_list: list[str] = checkpoint.get("smiles", [])
        self.label_names: list[str] = checkpoint.get("label_names", [])
        self.model_name: str = checkpoint.get("model_name", "unimolv1")

        log.info(
            f"Dataset Uni-Mol initialisé : {len(self.embeddings)} molécules, "
            f"dimension d'embedding={self.embeddings.shape[1]}, "
            f"{len(self.label_names)} classes d'odeurs."
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne l'embedding moléculaire et le vecteur multi-label cible.

        :param index: Index de l'échantillon.
        :return: (embedding_tensor, target_label_tensor)
        """
        return self.embeddings[index], self.labels[index]
