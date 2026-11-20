"""Hybridní model: obrázky + metadata -> predikce y."""
import torch
from torch import nn

from models.mlp import MLP


class HybridModel(nn.Module):
    """Encoder převede obrázky na vektory, ty se zprůměrují,
    spojí s metadaty a MLP z nich spočítá predikci y."""

    def __init__(self, encoder: nn.Module, image_dim: int, n_meta_inputs: int, hidden_size: int = 128):
        """
        Args:
            encoder: Encoder obrázků.
            image_dim: Délka vektoru, který encoder vrací pro jeden obrázek.
            n_meta_inputs: Počet vstupů metadat.
            hidden_size: Počet neuronů skryté vrstvy MLP.
        """
        super().__init__()
        self.encoder = encoder
        self.mlp = MLP(image_dim + n_meta_inputs, hidden_size)

    def forward(self, meta: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Vrátí predikci y [batch] pro metadata [batch, n_meta_inputs] a obrázky [batch, n_images, 3, H, W]."""
        batch, n_images = images.shape[:2]
        # encoder zpracuje všechny obrázky v dávce najednou
        features = self.encoder(images.flatten(0, 1))
        # průměr vektorů obrázků pro každý vzorek
        image_vector = features.view(batch, n_images, -1).mean(dim=1)
        # spojení s metadaty a predikce
        return self.mlp(torch.cat([image_vector, meta], dim=1))
