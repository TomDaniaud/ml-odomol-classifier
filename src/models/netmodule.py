from typing import cast

import torch
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
        criterion: torch.nn.Module,
        optimizer: type[torch.optim.Optimizer],
        scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
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

        self.net = cast(torch.nn.Module, torch.compile(net))
        self.optimizer_class = optimizer
        self.scheduler_class = scheduler
        self.criterion = criterion
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()

        self.nb_labels = nb_labels

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def model_step(self, batch):
        inputs, targets = batch
        logits = self.forward(inputs)
        targets = targets.to(dtype=torch.float32)
        loss = self.criterion(logits, targets)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).to(dtype=torch.float32)

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

        loss, _preds, _targets = self.model_step(batch)
        self.train_loss(loss.detach())  # Just for logging, so detach before accumulating
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
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
            "val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # Mise à jour des métriques avec les bons formats :
        # - F1 prend preds binaire (ou probs) et targets (int)
        # - AUROC prend impérativement probs flottantes continues
        self.val_f1_macro.update(preds, targets_int)
        self.val_f1_per_class.update(preds, targets_int)
        self.val_auroc.update(probs, targets_int)

        self.log(
            "val/f1_macro",
            self.val_f1_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/auroc_macro",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(
        self,
    ) -> OptimizerConfig | OptimizerLRSchedulerConfig:
        optimizer = self.optimizer_class(params=self.net.parameters())  # ty:ignore[missing-argument]
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
