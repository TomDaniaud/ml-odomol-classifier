import logging
import math
from itertools import chain
from os import R_OK, access
from pathlib import Path

import numpy as np
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.table import Table
from torchaudio.transforms import Resample

from utils.audio import audioread

log = logging.getLogger(__name__)


class AudioFileDataset(torch.utils.data.Dataset):
    """
    Basic file dataset.
    Each item in the dataset is load from a given directory.
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
        data_dir: Path = Path(data_dir)
        if not data_dir.exists(follow_symlinks=True):
            raise ValueError(f"`data_dir` not found. Got {data_dir=}")
        if not data_dir.is_dir():
            raise ValueError(f"`data_dir` must be a directory. Got {type(data_dir)=}")

        # Suppress fancy console status output if not rank zero to avoid glitchy output
        console = Console(quiet=rank_zero_only.rank != 0)  # ty:ignore[unresolved-attribute]

        # Check files in dataset are readable
        with console.status("Checking all files for read issues...", spinner="earth"):
            datas = ...

        self.datas: list[Path] = list(datas)
        log.info(f"Finished initialising dataset. Dataset contains {len(self.datas)} items.")

    def __len__(self):
        return len(self.datas)

    def __getitem__(
        self,
        index: int,
    ):
        data= self.datas[index]
        #parser le code smiles, generer sa structure 3D avec rdkit
        # retourner le tout plus les labels de vérité 
        # return (smiles, 3D), groundtruth
        return ..., ...
