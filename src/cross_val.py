import contextlib
import copy
import functools
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Configuration bas niveau de l'allocateur mémoire Intel oneAPI Level-Zero / XPU :
os.environ.setdefault("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1")
os.environ.setdefault("PYTORCH_XPU_ALLOC_CONF", "max_split_size_mb:256")

# Fix Windows console encoding for Unicode/emojis (prevents UnicodeEncodeError cp1252)
if sys.platform == "win32":
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import hydra
import numpy as np
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from torchmetrics.classification import MultilabelAUROC

os.environ["HYDRA_FULL_ERROR"] = "1"
rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.models.benchmark_metrics import evaluate  # noqa: E402
from src.models.metrics import OptimalMultiLabelF1  # noqa: E402
from src.utils.hardware import get_hardware_strategy  # noqa: E402
from src.utils.resolvers import register_resolvers  # noqa: E402

logging.captureWarnings(True)
log = logging.getLogger(__name__)
for name in ["lightning", "lightning.pytorch"]:
    lg = logging.getLogger(name)
    lg.handlers.clear()
    lg.propagate = True

register_resolvers()
torch.serialization.add_safe_globals([functools.partial])


@torch.no_grad()
def collect_fold_predictions(
    model: LightningModule,
    datamodule: LightningDataModule,
    device: torch.device,
    split: str = "val",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Passe d'inférence sur le pli de validation (probabilités OOF) ou sur le jeu de test tenu à l'écart."""
    model.eval()
    model.to(device)
    probs_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []

    loader = datamodule.test_dataloader() if split == "test" else datamodule.val_dataloader()
    for batch in loader:
        inputs, targets = batch
        if isinstance(inputs, dict):
            inputs_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        else:
            inputs_dev = inputs.to(device)

        logits = model(inputs_dev)
        probs = torch.sigmoid(logits).detach().cpu()
        probs_chunks.append(probs)
        targets_chunks.append(targets.detach().cpu().to(dtype=torch.float32))

    return torch.cat(probs_chunks, dim=0), torch.cat(targets_chunks, dim=0)


def compute_nested_per_class_f1(
    fold_probs: dict[int, torch.Tensor],
    fold_targets: dict[int, torch.Tensor],
    thresholds: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """
    Calcule le F1-Macro à seuillage dynamique par classe sans fuite de données (Nested OOF) :
    - Pour chaque pli k, les 138 seuils optimaux (un par classe) sont calibrés sur les K-1 autres plis.
    - Ces seuils sont ensuite appliqués aux prédictions du pli k.
    Retourne le F1-Macro OOF imbriqué ainsi que les 138 seuils optimaux calibrés sur l'ensemble des plis.
    """
    fold_keys = sorted(fold_probs.keys())
    thresh_grid = thresholds.view(-1, 1, 1)  # (T, 1, 1)

    oof_binary_preds: list[torch.Tensor] = []
    oof_targets_all: list[torch.Tensor] = []

    for k in fold_keys:
        train_oof_probs = torch.cat([fold_probs[j] for j in fold_keys if j != k], dim=0)
        train_oof_targets = torch.cat([fold_targets[j] for j in fold_keys if j != k], dim=0)

        # Recherche vectorisée du meilleur seuil par classe sur les K-1 plis : (T, N_other, 138)
        preds_grid = (train_oof_probs.unsqueeze(0) >= thresh_grid).to(torch.float32)
        targets_exp = train_oof_targets.unsqueeze(0)

        tp = (preds_grid * targets_exp).sum(dim=1)  # (T, 138)
        fp = (preds_grid * (1.0 - targets_exp)).sum(dim=1)
        fn = ((1.0 - preds_grid) * targets_exp).sum(dim=1)
        denom = 2.0 * tp + fp + fn
        f1_grid = torch.where(denom > 0, (2.0 * tp) / denom, torch.zeros_like(denom))

        best_thresh_idx = torch.argmax(f1_grid, dim=0)  # (138,)
        best_class_thresholds = thresholds[best_thresh_idx]  # (138,)

        # Application sur le pli de test k
        val_preds_k = (fold_probs[k] >= best_class_thresholds.unsqueeze(0)).to(torch.float32)
        oof_binary_preds.append(val_preds_k)
        oof_targets_all.append(fold_targets[k])

    all_preds = torch.cat(oof_binary_preds, dim=0)
    all_targets = torch.cat(oof_targets_all, dim=0)

    tp_tot = (all_preds * all_targets).sum(dim=0)
    fp_tot = (all_preds * (1.0 - all_targets)).sum(dim=0)
    fn_tot = ((1.0 - all_preds) * all_targets).sum(dim=0)
    denom_tot = 2.0 * tp_tot + fp_tot + fn_tot
    f1_per_class = torch.where(
        denom_tot > 0, (2.0 * tp_tot) / denom_tot, torch.zeros_like(denom_tot)
    )
    nested_f1_macro = float(f1_per_class.mean().item())

    # Seuils optimaux par classe sur 100 % des prédictions OOF (pour l'inférence finale)
    full_probs = torch.cat([fold_probs[j] for j in fold_keys], dim=0)
    full_targets = torch.cat([fold_targets[j] for j in fold_keys], dim=0)
    full_grid = (full_probs.unsqueeze(0) >= thresh_grid).to(torch.float32)
    full_t_exp = full_targets.unsqueeze(0)
    tp_f = (full_grid * full_t_exp).sum(dim=1)
    fp_f = (full_grid * (1.0 - full_t_exp)).sum(dim=1)
    fn_f = ((1.0 - full_grid) * full_t_exp).sum(dim=1)
    denom_f = 2.0 * tp_f + fp_f + fn_f
    f1_f = torch.where(denom_f > 0, (2.0 * tp_f) / denom_f, torch.zeros_like(denom_f))
    final_thresholds = thresholds[torch.argmax(f1_f, dim=0)]

    return nested_f1_macro, final_thresholds


def release_accelerator_memory() -> None:
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(version_base="1.3", config_path="../configs", config_name="default.yaml")
def main(cfg: DictConfig) -> None:
    for handler in logging.root.handlers:
        if hasattr(handler, "stream") and hasattr(handler.stream, "reconfigure"):
            with contextlib.suppress(Exception):
                handler.stream.reconfigure(encoding="utf-8")

    num_folds = int(cfg.data.get("num_folds", 5))
    if num_folds <= 1:
        num_folds = 5
        log.info("num_folds <= 1 détecté dans la config ; passage automatique à num_folds=5.")

    base_output_dir = Path(HydraConfig.get().run.dir).resolve()
    log.info(f"Démarrage de la Validation Croisée {num_folds}-Fold.")
    log.info(f":card_index: {'[bold green]Job Name:[/]':<35}{cfg.run_name}")
    log.info(f":file_folder: {'[bold green]Output directory:[/]':<35}{base_output_dir}")

    if float32_matmul_precision := cfg.get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(float32_matmul_precision)

    tracked_keys = [
        "valid/loss",
        "valid/auroc_macro",
        "valid/f1_macro",
        "valid/f1_opt_macro",
        "valid/f1_opt_per_class_macro",
        "valid/f1_opt_micro",
    ]
    fold_metrics_list: list[dict[str, float]] = []
    fold_probs_dict: dict[int, torch.Tensor] = {}
    fold_targets_dict: dict[int, torch.Tensor] = {}
    fold_val_indices: dict[int, np.ndarray] = {}
    fold_test_probs: list[torch.Tensor] = []

    for fold_idx in range(num_folds):
        log.info("=" * 80)
        log.info(f"=== PLI {fold_idx + 1} / {num_folds} (fold_index={fold_idx}) ===")
        log.info("=" * 80)

        if seed := cfg.get("seed"):
            seed_everything(int(seed) + fold_idx, workers=True, verbose=False)

        fold_cfg = copy.deepcopy(cfg)
        OmegaConf.set_struct(fold_cfg, False)
        fold_cfg.data.num_folds = num_folds
        fold_cfg.data.fold_index = fold_idx

        fold_dir = base_output_dir / f"fold_{fold_idx}"
        fold_ckpt_dir = fold_dir / "checkpoints"

        if fold_cfg.get("callbacks"):
            if "model_checkpoint" in fold_cfg.callbacks:
                fold_cfg.callbacks.model_checkpoint.dirpath = str(fold_ckpt_dir)
            if "on_exception_checkpoint" in fold_cfg.callbacks:
                fold_cfg.callbacks.on_exception_checkpoint.dirpath = str(fold_ckpt_dir)

        if fold_cfg.get("loggers") and "tensorboard" in fold_cfg.loggers:
            base_version = str(cfg.run_name or "cross_val")
            fold_cfg.loggers.tensorboard.version = f"{base_version}/fold_{fold_idx}"

        datamodule: LightningDataModule = hydra.utils.instantiate(fold_cfg.data)
        model: LightningModule = hydra.utils.instantiate(fold_cfg.model)

        callbacks: list[Callback] = []
        ckpt_callback: ModelCheckpoint | None = None
        if fold_cfg.callbacks is not None:
            for callback_cfg in fold_cfg.callbacks.values():
                if isinstance(callback_cfg, DictConfig) and "_target_" in callback_cfg:
                    cb = hydra.utils.instantiate(callback_cfg)
                    if isinstance(cb, ModelCheckpoint):
                        ckpt_callback = cb
                    callbacks.append(cb)

        loggers: list[Logger] = []
        if fold_cfg.loggers is not None:
            for logger_cfg in fold_cfg.loggers.values():
                if isinstance(logger_cfg, DictConfig) and "_target_" in logger_cfg:
                    loggers.append(hydra.utils.instantiate(logger_cfg))

        strategy = get_hardware_strategy(fold_cfg.trainer)
        trainer_kwargs: dict[str, Any] = {
            "callbacks": callbacks,
            "logger": loggers,
            "default_root_dir": str(fold_dir),
        }
        if strategy is not None:
            trainer_kwargs["strategy"] = strategy

        trainer: Trainer = hydra.utils.instantiate(fold_cfg.trainer, **trainer_kwargs)
        trainer.fit(model=model, datamodule=datamodule)

        # Chargement du meilleur checkpoint du pli pour l'évaluation et les prédictions OOF
        best_ckpt_path = (
            ckpt_callback.best_model_path
            if (ckpt_callback is not None and ckpt_callback.best_model_path)
            else None
        )
        if best_ckpt_path and Path(best_ckpt_path).is_file():
            log.info(f"[Pli {fold_idx + 1}] Évaluation du meilleur checkpoint : {best_ckpt_path}")
            val_results = trainer.validate(
                model=model,
                datamodule=datamodule,
                ckpt_path=best_ckpt_path,
                weights_only=False,
                verbose=False,
            )
            state = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            val_results = trainer.validate(
                model=model, datamodule=datamodule, weights_only=False, verbose=False
            )

        val_dict = val_results[0] if val_results else {}
        fold_summary = {"fold": fold_idx}
        for key in tracked_keys:
            if key in val_dict:
                fold_summary[key] = float(val_dict[key])
            elif key in trainer.callback_metrics:
                fold_summary[key] = float(trainer.callback_metrics[key].item())
            else:
                fold_summary[key] = float("nan")
        fold_metrics_list.append(fold_summary)

        # Extraction des probabilités Out-Of-Fold (OOF)
        infer_device = (
            torch.device("xpu", 0)
            if (hasattr(torch, "xpu") and torch.xpu.is_available())
            else (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
        )
        probs_k, targets_k = collect_fold_predictions(model, datamodule, infer_device)
        fold_probs_dict[fold_idx] = probs_k
        fold_targets_dict[fold_idx] = targets_k
        if getattr(datamodule, "val_idx", None) is not None:
            fold_val_indices[fold_idx] = datamodule.val_idx

        # Split 'benchmark' : le modèle de ce pli prédit aussi le jeu de test tenu à l'écart
        if getattr(datamodule, "data_test", None) is not None:
            test_probs_k, test_targets = collect_fold_predictions(model, datamodule, infer_device, split="test")
            fold_test_probs.append(test_probs_k)

        del trainer, model, datamodule
        release_accelerator_memory()

    # --- AGRÉGATION DES MÉTRIQUES SUR LES K PLIS ---
    console = Console()
    table = Table(
        title=f"Bilan de la Validation Croisée ({num_folds}-Fold - {cfg.data.split_type})"
    )
    table.add_column("Pli", justify="center", style="cyan", no_wrap=True)
    for key in tracked_keys:
        table.add_column(key.replace("valid/", ""), justify="right")

    for fm in fold_metrics_list:
        row = [f"Fold {int(fm['fold']) + 1}"]
        for key in tracked_keys:
            row.append(f"{fm[key]:.4f}")
        table.add_row(*row)

    summary_stats: dict[str, dict[str, float]] = {}
    mean_row = ["Moyenne ± Écart-type"]
    for key in tracked_keys:
        vals = np.array([fm[key] for fm in fold_metrics_list], dtype=np.float64)
        mean_val = float(np.nanmean(vals))
        std_val = float(np.nanstd(vals))
        summary_stats[key] = {"mean": mean_val, "std": std_val}
        mean_row.append(f"[bold green]{mean_val:.4f} ± {std_val:.4f}[/bold green]")
    table.add_section()
    table.add_row(*mean_row)

    # --- CALCUL DES MÉTRIQUES OUT-OF-FOLD (OOF) GLOBALES ---
    all_oof_probs = torch.cat([fold_probs_dict[k] for k in range(num_folds)], dim=0)
    all_oof_targets = torch.cat([fold_targets_dict[k] for k in range(num_folds)], dim=0)

    oof_metric = OptimalMultiLabelF1(num_thresholds=19)
    oof_metric.update(all_oof_probs, all_oof_targets)
    oof_opt = oof_metric.compute()

    auroc_metric = MultilabelAUROC(num_labels=all_oof_targets.size(1), average="macro")
    oof_auroc = float(auroc_metric(all_oof_probs, all_oof_targets.to(torch.long)).item())

    nested_f1_per_class, optimal_class_thresholds = compute_nested_per_class_f1(
        fold_probs=fold_probs_dict,
        fold_targets=fold_targets_dict,
        thresholds=oof_metric.thresholds.cpu(),
    )

    oof_summary = {
        "oof/auroc_macro": oof_auroc,
        "oof/f1_opt_macro": float(oof_opt["f1_macro_opt"].item()),
        "oof/best_thresh_macro": float(oof_opt["thresh_macro"].item()),
        "oof/f1_opt_per_class_macro_oracle": float(oof_opt["f1_macro_per_class_opt"].item()),
        "oof/f1_opt_per_class_macro_nested": nested_f1_per_class,
        "oof/f1_opt_micro": float(oof_opt["f1_micro_opt"].item()),
        "oof/best_thresh_micro": float(oof_opt["thresh_micro"].item()),
    }

    oof_row = [
        "[bold yellow]Global OOF (N=100%)[/bold yellow]",
        "-",
        f"[bold yellow]{oof_summary['oof/auroc_macro']:.4f}[/bold yellow]",
        "-",
        f"[bold yellow]{oof_summary['oof/f1_opt_macro']:.4f}[/bold yellow]",
        f"[bold yellow]{oof_summary['oof/f1_opt_per_class_macro_nested']:.4f} (nested)[/bold yellow]",
        f"[bold yellow]{oof_summary['oof/f1_opt_micro']:.4f}[/bold yellow]",
    ]
    table.add_row(*oof_row)
    console.print(table)

    # Sauvegarde des résultats JSON et des prédictions OOF
    json_report = {
        "num_folds": num_folds,
        "split_type": str(cfg.data.split_type),
        "seed": int(cfg.get("seed", 64)),
        "per_fold_metrics": fold_metrics_list,
        "cross_val_mean_std": summary_stats,
        "global_oof_metrics": oof_summary,
    }
    report_path = base_output_dir / "cv_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    oof_payload = {
        "fold_probs": fold_probs_dict,
        "fold_targets": fold_targets_dict,
        "fold_val_indices": fold_val_indices,
        "optimal_class_thresholds": optimal_class_thresholds,
        "cv_summary": json_report,
    }
    oof_path = base_output_dir / "oof_predictions.pt"
    torch.save(oof_payload, oof_path)

    log.info(f"Rapport de validation croisée sauvegardé dans : {report_path}")
    log.info(f"Prédictions OOF et seuils optimaux par classe sauvegardés dans : {oof_path}")

    # --- TEST TENU À L'ÉCART, PROTOCOLE HAN ET AL. 2026 ---
    # Ensemble = moyenne des 5 modèles de pli ; seuils par label appris sur les prédictions OOF.
    if fold_test_probs:
        test_probs = torch.stack(fold_test_probs).mean(dim=0)
        result = evaluate(all_oof_probs.numpy(), all_oof_targets.numpy(), test_probs.numpy(), test_targets.numpy())
        for name, (value, low, high) in result.items():
            log.info(f"test/{name} = {value:.4f}  [IC 95 % : {low:.4f} – {high:.4f}]")
        (base_output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(e)
