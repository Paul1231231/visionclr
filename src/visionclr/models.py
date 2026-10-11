
import torch
from torch import nn
from torchvision.models import resnet18, resnet50


class SimCLRModel(nn.Module):
    """
    SimCLR network consisting of:
      1. ResNet-18 encoder
      2. Two-layer projection MLP

    Input:
        x: [batch_size, 3, image_size, image_size]

    Returns:
        h: [batch_size, 512]  encoder representations
        z: [batch_size, projection_dim] projection embeddings
    """

    def __init__(
        self,
        projection_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()

        backbone = resnet50(weights=None)

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1],
            nn.Flatten(),
        )

        self.projector = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z = self.projector(h)

        return h, z
