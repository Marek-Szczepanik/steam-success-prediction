"""Encoder pro experiment: obrazová část předtrénovaného modelu CLIP ViT-B/32 (OpenAI)."""
import open_clip
import torch
from torch import nn
from torch.nn import functional as F

# Velikost obrázků, na které byl CLIP předtrénován
CLIP_IMAGE_SIZE = 224


class CLIPEncoder(nn.Module):
    """Převede obrázek 3 x H x W na vektor délky out_dim obrazovým encoderem modelu CLIP.

    Váhy encoderu jsou zmrazené, trénují se jen vrstvy za ním.
    """

    # průměr a směrodatná odchylka pro normalizaci obrázků (buffery registrované v __init__)
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self):
        """Načte předtrénovaný model CLIP a ponechá z něj jen obrazový encoder."""
        super().__init__()
        # váhy OpenAI byly natrénovány s aktivací QuickGELU, model musí používat stejnou
        clip: nn.Module = open_clip.create_model("ViT-B-32", pretrained="openai", force_quick_gelu=True)
        self.net: nn.Module = clip.get_submodule("visual")
        # zmrazení: váhy encoderu se během trénování nemění
        for param in self.net.parameters():
            param.requires_grad = False
        # normalizace obrázků stejná jako při předtrénování CLIP
        self.register_buffer("mean", torch.tensor(open_clip.OPENAI_DATASET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(open_clip.OPENAI_DATASET_STD).view(1, 3, 1, 1))
        self.out_dim = 512

    def train(self, mode: bool = True):
        """Přepne režim modulu, předtrénovaná síť ale zůstává vždy v režimu inference."""
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vrátí vektor příznaků [batch, out_dim] pro obrázky [batch, 3, H, W] s hodnotami 0-1.

        Obrázky se nejdřív zvětší na velikost, na které byl CLIP předtrénován.
        """
        x = F.interpolate(x, size=CLIP_IMAGE_SIZE, mode="bicubic", align_corners=False)
        return self.net((x - self.mean) / self.std)
