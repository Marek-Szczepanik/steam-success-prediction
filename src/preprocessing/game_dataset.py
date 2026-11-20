"""Dataset pro PyTorch: metadata, předzpracované obrázky a cílová proměnná."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .prepare_images import ARRAY_DIR


class GameDataset(Dataset):
    """Vrací vzorky (metadata, obrázky, y)."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, split: str):
        """
        Args:
            X: Vstupy metadat z make_features().
            y: Cílová proměnná se stejným indexem.
            split: Název množiny ("train", "val" nebo "test"), určuje soubor s obrázky.
        """
        self.meta = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32)
        self.images_path = ARRAY_DIR / f"{split}.npy"
        # pole s obrázky se otevře až v procesu, který data načítá
        self.images = None

    def __len__(self) -> int:
        """Počet vzorků."""
        return len(self.y)

    def __getitem__(self, i: int):
        """Vrátí i-tý vzorek: metadata [n_inputs], obrázky [n_images, 3, IMAGE_SIZE, IMAGE_SIZE] a y."""
        if self.images is None:
            self.images = np.load(self.images_path, mmap_mode="r")
        images = torch.from_numpy(np.array(self.images[i])).float() / 255   # hodnoty 0-1
        return self.meta[i], images, self.y[i]
