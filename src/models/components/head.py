import torch
from torch import nn


class ClassificationHead(nn.Module):
    """
    MLP classification head for Uni-Mol representations.
    Outputs raw logits.
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 256,
        out_channels: int = 138,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: Molecule embedding tensor [Batch, in_channels]
        :return: Logits tensor [Batch, out_channels]
        """
        return self.net(inputs)