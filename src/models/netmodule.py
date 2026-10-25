from pathlib import Path
from typing import cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerConfig, OptimizerLRSchedulerConfig
from torchmetrics import MeanMetric
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

from src.models.metrics import OptimalMultiLabelF1



class ClassifierLightningModule(LightningModule):
    """Module PyTorch Lightning pour la classification multi-label moléculaire (138 classes)."""

    def __init__(
        self,
        net: torch.nn.Module,
        nb_labels: int,
        optimizer: type[torch.optim.Optimizer],
        scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        compile: bool = False,
        use_pos_weights: bool = True,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "criterion"])

        self.net = cast(torch.nn.Module, torch.compile(net)) if compile else net
        self.optimizer_class = optimizer
        self.scheduler_class = scheduler

        self.nb_labels = nb_labels
        self.criterion = torch.nn.BCEWithLogitsLoss()  # Surchargé dans on_fit_start()
        self.use_pos_weights = use_pos_weights
        self.threshold = threshold

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # Métriques standard à seuil fixe (0.5)
        self.val_f1_macro = MultilabelF1Score(
            num_labels=self.nb_labels, average="macro", threshold=self.threshold
        )
        # self.val_f1_per_class = MultilabelF1Score(
        #     num_labels=self.nb_labels, average=None, threshold=self.threshold
        # )
        self.val_auroc = MultilabelAUROC(
            num_labels=self.nb_labels, average="macro"
        )

        # Métrique personnalisée conjointe (Macro + Micro optimaux)
        self.val_optimal_f1 = OptimalMultiLabelF1(num_thresholds=19)

    def on_fit_start(self) -> None:
        """Récupère et applique les poids de déséquilibre de classes depuis le DataModule."""
        dm = getattr(self.trainer, "datamodule", None)
        if self.use_pos_weights and dm is not None and getattr(dm, "pos_weights", None) is not None:
            weights = dm.pos_weights.to(device=self.device, dtype=torch.float32)
            self.register_buffer("pos_weight", weights)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, inputs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def model_step(self, batch: tuple[dict[str, torch.Tensor] | torch.Tensor, torch.Tensor]):
        inputs, targets = batch
        logits = self.forward(inputs)
        targets = targets.to(dtype=torch.float32)
        loss = self.criterion(logits, targets)

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).to(dtype=torch.float32)

        return loss, probs, preds, targets

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, _probs, _preds, _targets = self.model_step(batch)
        self.train_loss(loss.detach())
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, probs, preds, targets = self.model_step(batch)
        targets_int = targets.to(dtype=torch.long)

        self.val_loss(loss.detach())
        self.log("valid/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Métriques standard
        self.val_f1_macro.update(preds, targets_int)
        # self.val_f1_per_class.update(preds, targets_int)
        self.val_auroc.update(probs, targets_int)

        # Mise à jour de la métrique personnalisée (stockage sur CPU)
        self.val_optimal_f1.update(probs.detach().cpu(), targets.detach().cpu())

        self.log("valid/f1_macro", self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid/auroc_macro", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        # Calcul vectorisé des scores optimaux
        metrics = self.val_optimal_f1.compute()

        self.log("valid/f1_opt_macro", metrics["f1_macro_opt"], prog_bar=True)
        self.log("valid/best_thresh_macro", metrics["thresh_macro"], prog_bar=False)
        self.log("valid/f1_opt_micro", metrics["f1_micro_opt"], prog_bar=True)
        self.log("valid/best_thresh_micro", metrics["thresh_micro"], prog_bar=False)

        # Réinitialisation pour la prochaine époque
        self.val_optimal_f1.reset()

    def configure_optimizers(self) -> OptimizerConfig | OptimizerLRSchedulerConfig:
        optimizer = self.optimizer_class(params=self.net.parameters())
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "valid/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}