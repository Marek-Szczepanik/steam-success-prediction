"""Model A: vlastní CNN encoder obrázků."""
import torch
from torch import nn


class CNNEncoder(nn.Module):
    """Převede obrázek 3 x 128 x 128 na vektor délky out_dim."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # blok 1: dvě konvoluce, pak zmenšení na polovinu (128 -> 64)
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # blok 2: dvě konvoluce, pak zmenšení na polovinu (64 -> 32)
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # blok 3: dvě konvoluce, pak zmenšení na polovinu (32 -> 16)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # global average pooling: průměr každé mapy příznaků -> 64 kanálů x 1 x 1
            nn.AdaptiveAvgPool2d(1),
            # plně propojené vrstvy: vektor 64 hodnot -> vektor příznaků
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vrátí vektor příznaků [batch, out_dim] pro obrázky [batch, 3, 128, 128]."""
        return self.net(x)
