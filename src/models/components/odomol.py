from typing import Any
import torch
from torch import nn
from unimol_tools.models import UniMolModel

from src.models.components.head import ClassificationHead


class OdomolFineTuneNet(nn.Module):
    """
    Architecture complète permettant le fine-tuning du Graph Transformer Uni-Mol
    avec propagation des gradients de bout en bout.
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        out_channels: int = 138,
        dropout: float = 0.4,
        unfreeze_last_n_layers: int = 0,    #gel complet par défault
    ):
        super().__init__()
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.backbone = UniMolModel(output_dim=1, data_type="molecule", remove_hs=False)

        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n_layers > 0:
            for layer in self.backbone.encoder.layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.head = ClassificationHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def forward(self, batched_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Ici batched_tensors contient directement les tenseurs PyTorch :
        src_tokens, src_distance, src_coord, src_edge_type
        """
        cls_repr = self.backbone(
            src_tokens=batched_tensors["src_tokens"],
            src_distance=batched_tensors["src_distance"],
            src_coord=batched_tensors["src_coord"],
            src_edge_type=batched_tensors["src_edge_type"],
            return_repr=True,
        )

        return self.head(cls_repr)