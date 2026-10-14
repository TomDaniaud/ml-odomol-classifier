import torch
from torch import nn
from unimol import UnimolBackbone

from head import ClassificationHead


class OdomolNet(nn.Module):
    """
    A network that run classification trough unimol embedding (backbone) then run the classifications layer (head).
    """

    def __init__(self, hidden_channels:int, nb_labels:int, dropout:int):
        """
        Initialize the model.
        """

        super().__init__()

        self.unimol = UnimolBackbone(
            ...
        )
        self.head = ClassificationHead(
            ...,        #in_channels = unimol out channels
            hidden_channels,
            nb_labels,
            dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform a single forward pass through the network.

        :param inputs: The input tensor.
        :type inputs: torch.Tensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """

        embeddings = self.unimol(inputs)
        return self.head(embeddings)
