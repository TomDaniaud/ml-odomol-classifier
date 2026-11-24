import logging
from collections.abc import Sequence

import torch
from torch import nn
from unimol_tools.models import UniMolV2Model

from src.models.components.head import ClassificationHead

log = logging.getLogger(__name__)


class _SafeUniMolV2Model(UniMolV2Model):
    """
    Sous-classe légère de `UniMolV2Model` corrigeant le chargement PyTorch 2.6+
    (`weights_only=False` requis pour les checkpoints Uni-Core/Fairseq contenant `argparse.Namespace`)
    et permettant de désactiver le chargement réseau lors des tests hors-ligne (`path == ""`).
    """

    def load_pretrained_weights(self, path: str | None, strict: bool = False) -> None:
        if not path:
            return
        log.info(f"Chargement des poids pré-entraînés Uni-Mol2 depuis {path}")
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        try:
            self.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            super().load_pretrained_weights(path=path, strict=strict)


class Odomol2FineTuneNet(nn.Module):
    """
    Architecture complète permettant l'extraction ou le fine-tuning du backbone
    Two-Track Transformer Uni-Mol2 (`84m`, `164m`, `310m`, `570m`, `1.1B`) couplé
    à la tête MLP `ClassificationHead`.
    """

    def __init__(
        self,
        model_size: str = "84m",
        in_channels: int | None = None,
        hidden_channels: int | Sequence[int] = 256,
        out_channels: int = 138,
        dropout: float = 0.4,
        unfreeze_last_n_layers: int = 0,
        num_layers: int = 2,
        pretrained_model_path: str | None = None,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.model_size = model_size
        self.unfreeze_last_n_layers = unfreeze_last_n_layers

        effective_ckpt_path = pretrained_model_path if load_pretrained else ""
        self.backbone = _SafeUniMolV2Model(
            output_dim=1,
            model_size=model_size,
            pretrained_model_path=effective_ckpt_path,
            remove_hs=True,
        )

        # Gel complet du backbone par défaut (y compris movement_pred_head et classification_head internes)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Dégel sélectif des N dernières couches de TransformerEncoderWithPairV2
        if unfreeze_last_n_layers > 0:
            for layer in self.backbone.encoder.layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            final_ln = getattr(self.backbone.encoder, "final_layer_norm", None)
            if final_ln is not None:
                for param in final_ln.parameters():
                    param.requires_grad = True

        embed_dim = int(self.backbone.args.encoder_embed_dim)
        resolved_in_channels = embed_dim if in_channels is None else int(in_channels)
        if resolved_in_channels != embed_dim:
            raise ValueError(
                f"in_channels={resolved_in_channels} incompatible avec Uni-Mol2 ({model_size}) "
                f"dont encoder_embed_dim={embed_dim}."
            )

        self.head = ClassificationHead(
            in_channels=resolved_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
        )

    def train(self, mode: bool = True) -> "Odomol2FineTuneNet":
        super().train(mode)
        # Maintient les couches gelées du backbone en mode eval (désactive leurs Dropouts internes)
        self.backbone.eval()
        if mode and self.unfreeze_last_n_layers > 0:
            for layer in self.backbone.encoder.layers[-self.unfreeze_last_n_layers :]:
                layer.train(True)
            final_ln = getattr(self.backbone.encoder, "final_layer_norm", None)
            if final_ln is not None:
                final_ln.train(True)
        return self

    def forward(self, batched_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Attend le dictionnaire de tenseurs Uni-Mol2 :
        `atom_feat`, `atom_mask`, `edge_feat`, `shortest_path`, `degree`,
        `pair_type`, `attn_bias`, `src_tokens`, `src_coord`.
        """
        grad_ctx = torch.enable_grad() if (self.training and self.unfreeze_last_n_layers > 0) else torch.no_grad()
        with grad_ctx:
            cls_repr = self.backbone(**batched_tensors, return_repr=True)

        return self.head(cls_repr)
