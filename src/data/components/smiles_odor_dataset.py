import logging
import math
from itertools import chain
from os import R_OK, access
from pathlib import Path
import csv

import numpy as np
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.progress import track
from rich.table import Table
from torchaudio.transforms import Resample

from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)

class SmilesOdorDataset(torch.utils.data.Dataset):
    """
    Dataset pour charger les molécuels SMILES, générer leur structure 3D
    et extraire les 138 descripteurs d'odeurs
    """

    def __init__(
        self,
        data_dir: str | Path,
    ):
        """
        Initialise the dataset.

        :param data_dir: Directory containing audio files.
        :type data_dir: Union[str, Path]
        """

        super().__init__()

        log.info("Setting up dataset.")

        # Input validation
        self.csv_path: Path = Path(data_dir)
        if not self.csv_path.exists(follow_symlinks=True):
            raise ValueError(f"`data_dir` not found. Got {self.csv_path=}")
        if not self.csv_path.is_file():
            raise ValueError(f"`data_dir` must be a file. Got {type(self.csv_path)=}")

        # Suppress fancy console status output if not rank zero to avoid glitchy output
        console = Console(quiet=rank_zero_only.rank != 0)  # ty:ignore[unresolved-attribute]

        self.smiles_list: list[str] = []
        self.labels: list[torch.Tensor] = []
        self.structure_3d: list[dict] = []
        self.label_names: list[str] = []


        # Check files in dataset are readable
        with console.status("Checking all files for read issues...", spinner="earth"):
            self._load_and_process()

        log.info(f"Finished initialising dataset. Dataset contains {len(self.smiles_list)} molecules.")


    def _smiles_to_3d(self, smiles: str):
        # 1. parser le code smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 2. Ajouter les atomes d'hydrogène
        mol = Chem.AddHs(mol)
        # 3. Generer les coordonnées 3D
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            # Fallback
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)

        # Si l'embedding a échoué (aucun conformère disponible)
        if res != 0 or mol.GetNumConformers() == 0:
            return None

        # 4. Optimiser les coordonnées
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

        # 5. Extraction des coordonnées et types d'atomes
        try:
            conf = mol.GetConformer()
            positions = conf.GetPositions()
            atomic_num = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        except Exception:
            return None

        return {
            "pos": torch.tensor(positions, dtype=torch.float32),
            "atomic_nums": torch.tensor(atomic_num, dtype=torch.long),
            "smiles": smiles,
        }


    def _load_and_process(self):
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.label_names = header[2:] 
            rows = list(reader)

            for row in track(rows, description="Precomputation of 3D Structures using RDKit"):
                smiles = row[0]
                label_vec = [float(val) for val in row[2:]]
                struct_3d = self._smiles_to_3d(smiles)
                if struct_3d is None:
                    log.warning(f"Could not process SMILES: {smiles}")
                    continue
                
                self.smiles_list.append(smiles)
                self.structure_3d.append(struct_3d)
                self.labels.append(torch.tensor(label_vec, dtype=torch.float32))


    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(
        self,
        index: int,
    ):
        data= {
            "smiles": self.smiles_list[index],
            "pos": self.structure_3d[index]["pos"],
            "atomic_nums": self.structure_3d[index]["atomic_nums"],
        }
        target = self.labels[index]
        return data, target
    
