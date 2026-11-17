"""Architektury neuronových sítí pro predikci úspěchu her na Steamu."""
import torch
from torch import nn


class MetadataMLP(nn.Module):
    """MLP nad metadaty: vstupy -> skrytá vrstva -> ReLU -> jeden výstup (predikce y)."""

    def __init__(self, n_inputs: int):
        """
        Args:
            n_inputs: Počet vstupů (sloupců z make_features()).
            hidden: Počet neuronů skryté vrstvy.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vrátí predikci y pro každou hru v dávce (tvar [batch])."""
        return self.net(x).squeeze(1)
