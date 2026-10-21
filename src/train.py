import logging
import os
import sys
from pathlib import Path

# Configuration bas niveau de l'allocateur mémoire Intel oneAPI Level-Zero / XPU :
# - UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS : lève la limite heuristique de 4 Go par allocation contiguë.
# - PYTORCH_XPU_ALLOC_CONF : limite le découpage des segments pour prévenir la fragmentation mémoire.
os.environ.setdefault("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1")
os.environ.setdefault("PYTORCH_XPU_ALLOC_CONF", "max_split_size_mb:256")

# Fix Windows console encoding for Unicode/emojis (prevents UnicodeEncodeError cp1252)
if sys.platform == "win32":
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import hydra
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from utils.hardware import get_hardware_strategy
from utils.resolvers import register_resolvers

# Log full error tracebacks
os.environ["HYDRA_FULL_ERROR"] = "1"
# Set PYTHONPATH based on `pyproject.toml` file location
rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
# Set up logger, capture warnings and PyTorch Lightning logs
logging.captureWarnings(True)
log = logging.getLogger(__name__)
for name in ["lightning", "lightning.pytorch"]:
    lg = logging.getLogger(name)
    lg.handlers.clear()
    lg.propagate = True

register_resolvers()

@hydra.main(version_base="1.3", config_path="../configs", config_name="default.yaml")
def main(cfg: DictConfig):
    for handler in logging.root.handlers:
        if hasattr(handler, "stream") and hasattr(handler.stream, "reconfigure"):
            try:
                handler.stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    log.info("Starting a new training job.")
    log.info(f":card_index: {'[bold green]Job Name:[/]':<35}{cfg.run_name}")
    log.info(f":page_facing_up: {'[bold green]Description:[/]':<35}{cfg.description}")
    log.info(
        f":file_folder: {'[bold green]Output directory:[/]':<35}"
        f"./{Path(HydraConfig.get().run.dir).resolve().relative_to(rootutils.find_root())}"
    )

    if seed := cfg.get("seed"):
        log.info(f"Seed set to {seed}.")
        seed_everything(seed, workers=True, verbose=False)

    if float32_matmul_precision := cfg.get("float32_matmul_precision"):
        log.info(
            f"Using '{float32_matmul_precision}' precision float32 matmuls. "
            "'medium' or 'high' precision will trade off precision for speed. "
            "Configure this via `float32_matmul_precision` in the config."
        )
        torch.set_float32_matmul_precision(float32_matmul_precision)

    # Instantiate datamodule and trainer
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Instantiate callbacks
    callbacks: list[Callback] = []
    if cfg.callbacks is not None:
        for callback_cfg in cfg.callbacks.values():
            if isinstance(callback_cfg, DictConfig) and "_target_" in callback_cfg:
                callbacks.append(hydra.utils.instantiate(callback_cfg))
    else:
        log.warning("No callbacks found in config.")

    # Instantiate loggers
    loggers: list[Logger] = []
    if cfg.loggers is not None:
        for logger_cfg in cfg.loggers.values():
            if isinstance(logger_cfg, DictConfig) and "_target_" in logger_cfg:
                loggers.append(hydra.utils.instantiate(logger_cfg))
    else:
        log.warning("No loggers found in config.")

    # Detect hardware strategy (e.g. Intel Arc XPU, CUDA, MPS)
    strategy = get_hardware_strategy(cfg.trainer)
    trainer_kwargs = {"callbacks": callbacks, "logger": loggers}
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy

    # Instantiate trainer and start training
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, **trainer_kwargs)
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.get("ckpt_path"),
    )



if __name__ == "__main__":
    # Catch all errors to enable error logging via logger
    try:
        main()
    except Exception as e:
        log.exception(e)
