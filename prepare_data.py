"""Data preparation for the Steam game success prediction project.

Downloads the Steam games metadata from Hugging Face and loads it
into a pandas DataFrame for further processing.
"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

# Hugging Face dataset with Steam games metadata
REPO = "FronkonGames/steam-games-dataset"
# Dataset version
REVISION = "6c21457b9ebf0a6a7f5b91e11281d65d456ea30e"
# Folder for downloaded, unmodified data
RAW_DIR = Path("data/raw")


def download_metadata() -> Path:
    """Download games.json from Hugging Face unless it already exists.

    Returns:
        Path to the local games.json file.
    """
    path = RAW_DIR / "games.json"
    if not path.exists():
        hf_hub_download(REPO, "games.json", repo_type="dataset",
                        revision=REVISION, local_dir=RAW_DIR)
    return path


def load_metadata(path: Path) -> pd.DataFrame:
    """Load games.json into a DataFrame with one row per game.

    The JSON is a dict mapping a Steam appID to a dict of game fields.

    Args:
        path: Path to games.json.

    Returns:
        DataFrame indexed by app_id, one column per game field.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame.from_dict(raw, orient="index")
    df.index.name = "app_id"
    return df


if __name__ == "__main__":
    # First look at the data: size, column types and one example game
    df = load_metadata(download_metadata())
    print(df.shape)
    print(df.dtypes)
    print(df.iloc[0])
