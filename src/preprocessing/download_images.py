"""Stažení obrázků (header a screenshoty) pro hry, které prošly čištěním.

Spouští se po prepare_data.py. Stahování lze přerušit a spustit znovu,
již stažené obrázky se přeskočí. Nestažené obrázky se zapíšou do data/images/failed.txt.
"""
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from .prepare_data import MIN_SCREENSHOTS, PROCESSED_DIR

# Složka pro obrázky: data/images/<app_id>/header.jpg, ss0.jpg, ss1.jpg, ss2.jpg
IMAGE_DIR = Path("data/images")
# Soubor se seznamem obrázků, které se nepodařilo stáhnout
FAILED_LOG = IMAGE_DIR / "failed.txt"
# Kratší strana uloženého obrázku v pixelech
IMAGE_SIZE = 256
# Počet paralelních stahování
N_THREADS = 16


def image_jobs(df: pd.DataFrame) -> list[tuple[str, Path]]:
    """Vytvoří seznam dvojic (URL, cílový soubor) - header a prvních MIN_SCREENSHOTS screenshotů každé hry."""
    jobs = []
    for app_id, header, screenshots in zip(df.index, df["header_image"], df["screenshots"]):
        folder = IMAGE_DIR / str(app_id)
        jobs.append((header, folder / "header.jpg"))
        for i, url in enumerate(screenshots[:MIN_SCREENSHOTS]):
            jobs.append((url, folder / f"ss{i}.jpg"))
    return jobs


def fetch(url: str) -> bytes:
    """Stáhne obsah URL."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def download(job: tuple[str, Path]) -> str | None:
    """Stáhne jeden obrázek, zmenší ho a uloží jako JPEG.

    U screenshotů se nejdřív zkusí menší verze 600x338 (asi 10x méně dat),
    pokud neexistuje, stáhne se originál.

    Args:
        job: Dvojice (URL, cílový soubor).

    Returns:
        None při úspěchu (nebo pokud soubor už existuje), jinak popis chyby.
    """
    url, path = job
    if path.exists():
        return None

    small_url = url.replace(".1920x1080.jpg", ".600x338.jpg")
    urls = [small_url, url] if small_url != url else [url]
    error = ""
    for candidate in urls:
        try:
            image = Image.open(io.BytesIO(fetch(candidate))).convert("RGB")
            # zmenšení tak, aby kratší strana měla IMAGE_SIZE pixelů (menší obrázky se nezvětšují)
            scale = IMAGE_SIZE / min(image.size)
            if scale < 1:
                image = image.resize((round(image.width * scale), round(image.height * scale)))
            # uložení přes dočasný soubor, aby přerušené stahování nezanechalo poškozený obrázek
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            image.save(tmp, "JPEG", quality=90)
            tmp.replace(path)
            return None
        except Exception as e:
            error = str(e)
    return f"{url}\t{error}"


if __name__ == "__main__":
    df = pd.concat([pd.read_pickle(PROCESSED_DIR / f"{name}.pkl") for name in ["train", "val", "test"]])
    jobs = image_jobs(df)
    print(f"Her: {len(df)}, obrázků: {len(jobs)}")

    failed = []
    with ThreadPoolExecutor(N_THREADS) as pool:
        for i, error in enumerate(pool.map(download, jobs), start=1):
            if error:
                failed.append(error)
            if i % 5000 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} hotovo, chyb: {len(failed)}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_LOG.write_text("\n".join(failed), encoding="utf-8")
    print(f"Nestažené obrázky: {len(failed)} (seznam v {FAILED_LOG})")
