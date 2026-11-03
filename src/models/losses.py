import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) pour la classification multi-label déséquilibrée.

    Référence : Ridnik et al., ICCV 2021.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True, #evite de paratiter le calcul du gradient
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.to(dtype=logits.dtype)

        probs_pos = probs
        probs_neg = 1.0 - probs

        # Marge asymétrique : coupe le gradient sous le seuil clip
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        # Pertes cross-entropy élémentaires
        los_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Facteur de pondération focal
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow(1.0 - pt, gamma)

            if self.disable_torch_grad_focal_loss:
                focal_weight = focal_weight.detach()

            loss = loss * focal_weight

        return -loss.sum() / logits.size(0)
