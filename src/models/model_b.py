"""Model B: předtrénovaný encoder obrázků (ConvNeXt-Tiny natrénovaný na ImageNetu)."""
import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class PretrainedEncoder(nn.Module):
    """Převede obrázek 3 x H x W na vektor délky out_dim předtrénovanou sítí ConvNeXt-Tiny.

    Váhy encoderu jsou zmrazené, trénují se jen vrstvy za ním.
    """

    def __init__(self):
        super().__init__()
        self.net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        # odstranění klasifikační vrstvy pro třídy ImageNetu: výstupem je vektor příznaků
        self.net.classifier[2] = nn.Identity()
        # zmrazení: váhy encoderu se během trénování nemění
        for param in self.net.parameters():
            param.requires_grad = False
        # normalizace obrázků stejná jako při předtrénování na ImageNetu
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_dim = 768

    def train(self, mode: bool = True):
        """Přepne režim modulu, předtrénovaná síť ale zůstává vždy v režimu inference."""
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vrátí vektor příznaků [batch, out_dim] pro obrázky [batch, 3, H, W] s hodnotami 0-1."""
        return self.net((x - self.mean) / self.std)
