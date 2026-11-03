from collections.abc import Sequence

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
        hidden_channels: int | Sequence[int] = 256,
        out_channels: int = 138,
        dropout: float = 0.2,
        num_layers: int = 2,
    ):
        """
        :param in_channels: Dimension de l'embedding d'entrée Uni-Mol (ex: 512).
        :param hidden_channels: Dimension(s) des couches cachées (int ou liste/tuple d'int).
        :param out_channels: Nombre de labels de sortie (ex: 138).
        :param dropout: Taux de dropout.
        :param num_layers: Nombre total de couches linéaires du MLP (ex: 2 pour standard, 3 pour 3 couches).
        """
        super().__init__()

        if isinstance(hidden_channels, (list, tuple, Sequence)) and not isinstance(hidden_channels, (str, bytes)):
            hidden_dims = [int(h) for h in hidden_channels]
            num_layers = len(hidden_dims) + 1
        else:
            if num_layers < 1:
                raise ValueError(f"num_layers must be >= 1, got {num_layers}")
            hidden_dims = [int(hidden_channels)] * (num_layers - 1)

        layers: list[nn.Module] = []
        prev_dim = in_channels
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: Molecule embedding tensor [Batch, in_channels]
        :return: Logits tensor [Batch, out_channels]
        """
        return self.net(inputs)
