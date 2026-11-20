"""Dataset pro PyTorch: metadata, obrázky a cílová proměnná.

Obrázky se jednou předzpracují (zmenšení na IMAGE_SIZE x IMAGE_SIZE) a uloží do jednoho
pole uint8 pro každou množinu, takže se při trénování nemusí dekódovat ani zmenšovat.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from download_images import IMAGE_DIR
from prepare_data import PROCESSED_DIR

# Složka pro předzpracované obrázky (jeden soubor .npy pro každou množinu)
ARRAY_DIR = Path("data/images_128")
# Velikost obrázků na vstupu modelu (čtverec IMAGE_SIZE x IMAGE_SIZE)
IMAGE_SIZE = 128
# Soubory obrázků jednoho vzorku
IMAGE_FILES = ["header.jpg", "ss0.jpg", "ss1.jpg", "ss2.jpg"]


def load_images(app_id) -> np.ndarray:
    """Načte obrázky jednoho vzorku jako pole uint8 [n_images, 3, IMAGE_SIZE, IMAGE_SIZE]."""
    folder = IMAGE_DIR / str(app_id)
    images = []
    for name in IMAGE_FILES:
        image = ImageOps.fit(Image.open(folder / name).convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE))
        images.append(np.asarray(image).transpose(2, 0, 1))   # [H, W, 3] -> [3, H, W]
    return np.stack(images)


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


if __name__ == "__main__":
    ARRAY_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        app_ids = pd.read_pickle(PROCESSED_DIR / f"{name}.pkl").index
        # pole se zapisuje přímo na disk, celé by se do paměti nevešlo
        array = np.lib.format.open_memmap(
            ARRAY_DIR / f"{name}.npy", mode="w+", dtype=np.uint8,
            shape=(len(app_ids), len(IMAGE_FILES), 3, IMAGE_SIZE, IMAGE_SIZE))
        with ThreadPoolExecutor(8) as pool:
            for i, images in enumerate(pool.map(load_images, app_ids)):
                array[i] = images
        array.flush()
        print(f"{name}: {len(app_ids)} vzorků")
