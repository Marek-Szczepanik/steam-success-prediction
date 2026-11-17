"""Dataset pro PyTorch: metadata, obrázky a cílová proměnná."""
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from download_images import IMAGE_DIR

# Velikost obrázků na vstupu modelu (čtverec IMAGE_SIZE x IMAGE_SIZE)
IMAGE_SIZE = 128
# Soubory obrázků jednoho vzorku
IMAGE_FILES = ["header.jpg", "ss0.jpg", "ss1.jpg", "ss2.jpg"]


def load_image(path) -> torch.Tensor:
    """Načte obrázek, zmenší a ořízne ho na IMAGE_SIZE x IMAGE_SIZE a převede na tensor s hodnotami 0-1."""
    image = Image.open(path).convert("RGB")
    return to_tensor(ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE)))


class GameDataset(Dataset):
    """Vrací vzorky (metadata, obrázky, y)."""

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Args:
            X: Vstupy metadat z make_features(), index jsou app_id.
            y: Cílová proměnná se stejným indexem.
        """
        self.app_ids = list(X.index)
        self.meta = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        """Počet vzorků."""
        return len(self.app_ids)

    def __getitem__(self, i: int):
        """Vrátí i-tý vzorek: metadata [n_inputs], obrázky [n_images, 3, IMAGE_SIZE, IMAGE_SIZE] a y."""
        folder = IMAGE_DIR / str(self.app_ids[i])
        images = torch.stack([load_image(folder / name) for name in IMAGE_FILES])
        return self.meta[i], images, self.y[i]
