import torch
from torchmetrics import Metric


class OptimalMultiLabelF1(Metric):
    """Calcule conjointement les scores F1-Macro et F1-Micro optimaux

    ainsi que leurs seuils respectifs en une seule passe matricielle.
    """

    full_state_update: bool = False

    def __init__(
        self,
        num_thresholds: int = 19,
        thresh_min: float = 0.05,
        thresh_max: float = 0.95,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.register_buffer(
            "thresholds",
            torch.linspace(thresh_min, thresh_max, steps=num_thresholds),
        )
        self.add_state("probs_list", default=[], dist_reduce_fx="cat")
        self.add_state("targets_list", default=[], dist_reduce_fx="cat")

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        self.probs_list.append(probs.detach())
        self.targets_list.append(targets.detach())

    def compute(self) -> dict[str, torch.Tensor]:
        if not self.probs_list:
            zero = torch.tensor(0.0)
            mid = torch.tensor(0.5)
            return {
                "f1_macro_opt": zero,
                "thresh_macro": mid,
                "f1_micro_opt": zero,
                "thresh_micro": mid,
            }

        all_probs = torch.cat(self.probs_list, dim=0)
        all_targets = torch.cat(self.targets_list, dim=0)

        # Broadcasting sur les seuils : (K, N, 138)
        thresh_grid = self.thresholds.view(-1, 1, 1)
        preds_bin = (all_probs.unsqueeze(0) >= thresh_grid).to(torch.float32)
        targets_exp = all_targets.unsqueeze(0)

        # Matrice de confusion par seuil et par classe : (K, 138)
        tp_class = (preds_bin * targets_exp).sum(dim=1)
        fp_class = (preds_bin * (1.0 - targets_exp)).sum(dim=1)
        fn_class = ((1.0 - preds_bin) * targets_exp).sum(dim=1)

        # --- A. F1-Macro ---
        denom_macro = 2.0 * tp_class + fp_class + fn_class
        f1_per_class = torch.where(
            denom_macro > 0,
            (2.0 * tp_class) / denom_macro,
            torch.zeros_like(denom_macro),
        )
        macro_f1_scores = f1_per_class.mean(dim=1)  # (K,)
        best_macro_idx = torch.argmax(macro_f1_scores)

        # --- B. F1-Micro (somme des TP/FP/FN sur les 138 classes) ---
        tp_micro = tp_class.sum(dim=1)  # (K,)
        fp_micro = fp_class.sum(dim=1)
        fn_micro = fn_class.sum(dim=1)

        denom_micro = 2.0 * tp_micro + fp_micro + fn_micro
        micro_f1_scores = torch.where(
            denom_micro > 0,
            (2.0 * tp_micro) / denom_micro,
            torch.zeros_like(denom_micro),
        )
        best_micro_idx = torch.argmax(micro_f1_scores)

        return {
            "f1_macro_opt": macro_f1_scores[best_macro_idx],
            "thresh_macro": self.thresholds[best_macro_idx],
            "f1_micro_opt": micro_f1_scores[best_micro_idx],
            "thresh_micro": self.thresholds[best_micro_idx],
        }