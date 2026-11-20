"""MLP: základní stavební blok, samostatně nad metadaty i jako hlava hybridního modelu."""
import torch
from torch import nn


class MLP(nn.Module):
    """Vstupní vektor -> skrytá vrstva -> ReLU -> jeden výstup (predikce y)."""

    def __init__(self, n_inputs: int, hidden_size: int = 128):
        """
        Args:
            n_inputs: Délka vstupního vektoru.
            hidden_size: Počet neuronů skryté vrstvy.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vrátí predikci y pro každou hru v dávce (tvar [batch])."""
        return self.net(x).squeeze(1)
