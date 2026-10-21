from typing import cast

import torch
from pathlib import Path
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerConfig, OptimizerLRSchedulerConfig
from torchmetrics import MeanMetric
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score


class ClassifierLightningModule(LightningModule):
    """
    Net Lightning module.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        nb_labels:int,
        optimizer: type[torch.optim.Optimizer],
        scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        compile: bool = False,
        use_pos_weights: bool = True,
    ):
        """
        Initialise the Lightning module.

        :param net: Model object to train and/or inference.
        :type net: torch.nn.Module
        :param optimizer: Optimizer class.
        :type optimizer: Type[torch.optim.Optimizer]
        :param scheduler: Optional learning rate scheduler class.
        :type scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]]
        """

        super().__init__()
        self.save_hyperparameters(ignore=["net", "criterion"])

        self.net = cast(torch.nn.Module, torch.compile(net)) if compile else net
        self.optimizer_class = optimizer
        self.scheduler_class = scheduler

        self.nb_labels = nb_labels
        self.criterion = torch.nn.BCEWithLogitsLoss() # surchargé dans on_fit_start()
        self.use_pos_weights = use_pos_weights

        self.threshold = 0.5
        self.val_probs_list: list[torch.Tensor] = []
        self.val_targets_list: list[torch.Tensor] = []

        # Perte
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # F1 Macro (moyenne non pondérée sur l'ensemble des 138 odeurs)
        self.val_f1_macro = MultilabelF1Score(
            num_labels=self.nb_labels, average="macro"
        )

        # F1 par classe (average=None renvoie un tenseur de 138 scores)
        self.val_f1_per_class = MultilabelF1Score(
            num_labels=self.nb_labels, average=None
        )

        # AUROC Macro
        self.val_auroc = MultilabelAUROC(
            num_labels=self.nb_labels, average="macro"
        )

    def on_fit_start(self) -> None:
        """
        Appelé automatiquement par Lightning juste après datamodule.setup()
        et après que le modèle a été envoyé sur le bon device (GPU/MPS/CPU).
        """
        dm = getattr(self.trainer, "datamodule", None)
        if self.use_pos_weights and dm is not None and getattr(dm, "pos_weights", None) is not None:
            # On transfère le tenseur de poids sur le même device que le modèle (ex: cuda:0 ou mps)
            weights = dm.pos_weights.to(device=self.device, dtype=torch.float32)
            self.register_buffer("pos_weight", weights)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def model_step(self, batch):
        inputs, targets = batch
        logits = self.forward(inputs)
        targets = targets.to(dtype=torch.float32)
        loss = self.criterion(logits, targets)

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).to(dtype=torch.float32)

        return loss, probs, preds, targets

    def training_step(
        self,
        batch,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data containing the inputs and targets.
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :return: A tensor of losses between model predictions and targets.
        :rtype: torch.Tensor
        """

        loss, _probs, _preds, _targets = self.model_step(batch)
        self.train_loss(loss.detach())  # Just for logging, so detach before accumulating
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ):
        """
        Perform a single validataion step on a batch of data from the validation set.

        :param batch: A batch of data containing the inputs and targets.
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :return: A tensor of losses between model predictions and targets.
        :rtype: torch.Tensor
        """
        loss, probs, preds, targets = self.model_step(batch)
        targets_int = targets.to(dtype=torch.long)
        self.val_loss(loss.detach())
        self.log(
            "valid/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # Mise à jour des métriques avec les bons formats :
        # - F1 prend preds binaire (ou probs) et targets (int)
        # - AUROC prend impérativement probs flottantes continues
        self.val_f1_macro.update(preds, targets_int)
        self.val_f1_per_class.update(preds, targets_int)
        self.val_auroc.update(probs, targets_int)

        self.val_probs_list.append(probs.detach().cpu())
        self.val_targets_list.append(targets.detach().cpu())

        self.log(
            "valid/f1_macro",
            self.val_f1_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "valid/auroc_macro",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        if not self.val_probs_list:
            return

        all_probs = torch.cat(self.val_probs_list, dim=0)
        all_targets = torch.cat(self.val_targets_list, dim=0)

        self.val_probs_list.clear()
        self.val_targets_list.clear()

        thresholds = torch.linspace(0.05, 0.95, steps=19).view(-1, 1, 1)

        preds_bin = (all_probs.unsqueeze(0) >= thresholds).to(torch.float32)
        targets_exp = all_targets.unsqueeze(0)

        tp = (preds_bin * targets_exp).sum(dim=1)
        fp = (preds_bin * (1.0 - targets_exp)).sum(dim=1)
        fn = ((1.0 - preds_bin) * targets_exp).sum(dim=1)

        # Calcul du F1 par classe puis moyenne Macro par seuil : Shape (19,)
        denom = 2.0 * tp + fp + fn
        f1_per_class = torch.where(denom > 0, (2.0 * tp) / denom, torch.zeros_like(denom))
        macro_f1_per_thresh = f1_per_class.mean(dim=1)

        best_idx = int(torch.argmax(macro_f1_per_thresh).item())
        best_f1 = macro_f1_per_thresh[best_idx].item()
        best_thresh = thresholds.view(-1)[best_idx].item()

        self.log("valid/f1_opt_global", best_f1, prog_bar=True)
        self.log("valid/best_threshold", best_thresh, prog_bar=False)

    def configure_optimizers(
        self,
    ) -> OptimizerConfig | OptimizerLRSchedulerConfig:
        trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        opt_kwargs = {}
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            # Sur GPU Intel Arc / Level-Zero sous Windows, l'implémentation multi-tenseurs
            # (_multi_tensor_adam, foreach=True) provoque un crash lors de `torch._foreach_sqrt`
            # (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, erreur 39). On force l'exécution unitaire (foreach=False).
            opt_kwargs["foreach"] = False
        try:
            optimizer = self.optimizer_class(params=trainable_params, **opt_kwargs)  # ty:ignore[missing-argument]
        except TypeError:
            optimizer = self.optimizer_class(params=trainable_params)  # ty:ignore[missing-argument]
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
