"""
Příprava dat pro projekt predikce úspěchu her na Steamu.
Stáhne metadata her ze Steamu z Hugging Face, načte je do pandas DataFrame
a vyčistí je (rozhodnutí jsou zdůvodněna v eda.ipynb).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Dataset s metadaty her ze Steamu na Hugging Face
REPO = "FronkonGames/steam-games-dataset"
# Verze datasetu
REVISION = "6c21457b9ebf0a6a7f5b91e11281d65d456ea30e"
# Složka pro stažená, neupravená data
RAW_DIR = Path("data/raw")
# Složka pro vyčištěná a rozdělená data
PROCESSED_DIR = Path("data/processed")

# Poslední ponechaný rok vydání: do tohoto roku jsou počty recenzí spolehlivé (eda.ipynb, sekce 6)
MAX_RELEASE_YEAR = 2023
# Časové rozdělení podle roku vydání: train do TRAIN_END_YEAR včetně,
# validace VAL_YEAR, test MAX_RELEASE_YEAR
TRAIN_END_YEAR = 2021
VAL_YEAR = 2022
# Počet screenshotů, které používá model
MIN_SCREENSHOTS = 3
# Žánry softwaru (ne her); záznamy pouze s těmito žánry se vyřadí
SOFTWARE_GENRES = {
    "Utilities", "Design & Illustration", "Animation & Modeling", "Video Production",
    "Audio Production", "Game Development", "Photo Editing", "Web Publishing",
    "Software Training", "Accounting",
}
# Název jazyka: slova s velkým počátečním písmenem oddělená mezerou nebo " - ",
# volitelně s variantou v závorce, např. "Spanish - Spain", "Punjabi (Gurmukhi)".
# Zbytky HTML/BBCode a poznámky jako "(full audio)" jsou malými písmeny, proto se nezachytí.
LANGUAGE_PATTERN = re.compile(r"[A-Z][\w']*(?:(?: - | )[A-Z][\w']*)*(?: \([A-Z]\w*\))?")
# Kategorie popisující design hry; ostatní kategorie (Trading Cards, Remote Play,
# přístupnost, Family Sharing...) se přidávají až po vydání -> leakage (eda.ipynb, sekce 7)
DESIGN_CATEGORIES = [
    "Single-player", "Multi-player", "PvP", "Online PvP", "Co-op", "Online Co-op",
    "Shared/Split Screen", "Shared/Split Screen PvP", "Shared/Split Screen Co-op",
    "LAN PvP", "LAN Co-op", "MMO", "Cross-Platform Multiplayer",
    "VR Only", "VR Supported", "VR Support", "Tracked Controller Support",
    "Full controller support", "Partial Controller Support",
]
# Žánr Early Access odpovídá stavu v době stažení dat, ne v době vydání -> vyřazen
EXCLUDED_GENRES = {"Early Access"}
# Minimální počet her v trénovací množině, aby se žánr kódoval
MIN_GENRE_GAMES = 50
# Minimální podíl her v trénovací množině, aby se jazyk kódoval
MIN_LANGUAGE_SHARE = 0.01
# Sloupce ponechané po čištění: informace známé před vydáním + cílová proměnná
KEEP_COLS = [
    "name", "developers", "release", "price", "required_age",
    "genres", "categories", "supported_languages",
    "windows", "mac", "linux",
    "header_image", "screenshots",
    "n_reviews", "y",
]


def download_metadata() -> Path:
    """Stáhne games.json z Hugging Face, pokud ještě neexistuje.

    Returns:
        Cesta k lokálnímu souboru games.json.
    """
    path = RAW_DIR / "games.json"
    if not path.exists():
        hf_hub_download(REPO, "games.json", repo_type="dataset",
                        revision=REVISION, local_dir=RAW_DIR)
    return path


def load_metadata(path: Path) -> pd.DataFrame:
    """Načte games.json do DataFrame, jeden řádek odpovídá jedné hře.

    JSON je slovník, který mapuje Steam appID na slovník atributů hry.

    Args:
        path: Cesta ke games.json.

    Returns:
        DataFrame s indexem app_id, jeden sloupec pro každý atribut hry.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame.from_dict(raw, orient="index")
    df.index.name = "app_id"
    return df


def parse_languages(langs: list[str]) -> list[str]:
    """Vytáhne z chybně zapsaných položek názvy jazyků pomocí LANGUAGE_PATTERN.

    Args:
        langs: Původní seznam podporovaných jazyků jedné hry.

    Returns:
        Seřazený seznam unikátních názvů jazyků.
    """
    return sorted({lang for item in langs for lang in LANGUAGE_PATTERN.findall(item)})


def is_software(genres: list[str]) -> bool:
    """Vrátí True, pokud má záznam pouze softwarové žánry (nejde o hru)."""
    return len(genres) > 0 and all(g in SOFTWARE_GENRES for g in genres)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Provede čištění podle rozhodnutí z eda.ipynb a přidá cílovou proměnnou.

    Po každém kroku vypíše počet her.

    Args:
        df: Surová metadata z load_metadata().

    Returns:
        Vyčištěný DataFrame se sloupci KEEP_COLS a cílovou proměnnou y = log(1 + n_reviews).
    """
    df = df.copy()
    df["n_reviews"] = df["positive"] + df["negative"]
    df["release"] = pd.to_datetime(df["release_date"], format="mixed", errors="coerce")
    print("Původně:", len(df))

    # Spolehlivé počty recenzí jen do MAX_RELEASE_YEAR
    df = df[df["release"].dt.year <= MAX_RELEASE_YEAR]
    print(f"Vydáno do {MAX_RELEASE_YEAR}:", len(df))

    # Nulu recenzí nelze odlišit od chybějících dat
    df = df[df["n_reviews"] > 0]
    print("Aspoň 1 recenze:", len(df))

    # Obrázky jsou vstupem modelu
    df = df[(df["screenshots"].map(len) >= MIN_SCREENSHOTS) & (df["header_image"] != "")]
    print(f"Aspoň {MIN_SCREENSHOTS} screenshoty a header:", len(df))

    # Software, který není hrou
    df = df[~df["genres"].map(is_software)]
    print("Bez softwaru:", len(df))

    # Playtesty jsou testovací verze, ne vydané hry
    df = df[~df["name"].str.contains("playtest", case=False)]
    print("Bez playtestů:", len(df))

    # Stejná hra pod více app_id: ponechá se záznam s nejvíce recenzemi
    df = df.sort_values("n_reviews", ascending=False)
    key = df["name"].str.lower() + " | " + df["developers"].str.join(", ")
    df = df[~key.duplicated()].sort_index()
    print("Bez duplicit:", len(df))

    df["supported_languages"] = df["supported_languages"].map(parse_languages)
    df["y"] = np.log1p(df["n_reviews"])
    return df[KEEP_COLS]


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rozdělí data podle roku vydání na trénovací, validační a testovací množinu.

    Args:
        df: Vyčištěná data z clean().

    Returns:
        Trojice (train, val, test).
    """
    year = df["release"].dt.year
    train = df[year <= TRAIN_END_YEAR]
    val = df[year == VAL_YEAR]
    test = df[year == MAX_RELEASE_YEAR]
    return train, val, test



def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Spojité vstupy před škálováním: log ceny, rok vydání a log počtu jazyků."""
    return pd.DataFrame({
        "log_price": np.log1p(df["price"]),
        "year": df["release"].dt.year,
        "log_languages": np.log1p(df["supported_languages"].map(len)),
    })


def fit_features(train: pd.DataFrame) -> dict:
    """Naučí kódování vstupů pouze z trénovací množiny.

    Určí, které žánry a jazyky se budou kódovat, a průměr a směrodatnou
    odchylku spojitých vstupů pro standardizaci.

    Args:
        train: Trénovací množina ze split().

    Returns:
        Slovník s parametry kódování pro make_features().
    """
    genre_counts = train["genres"].explode().value_counts()
    language_counts = train["supported_languages"].explode().value_counts()
    min_language_games = int(MIN_LANGUAGE_SHARE * len(train))
    numeric = numeric_features(train)
    return {
        "genres": [g for g in genre_counts.index[genre_counts >= MIN_GENRE_GAMES] if g not in EXCLUDED_GENRES],
        "languages": list(language_counts.index[language_counts >= min_language_games]),
        "mean": numeric.mean(),
        "std": numeric.std(),
    }


def multi_hot(values: pd.Series, vocabulary: list[str], prefix: str) -> pd.DataFrame:
    """Pro každou hodnotu ze slovníku vytvoří sloupec 1/0 podle toho, zda ji hra má."""
    rows = [set(items) for items in values]
    return pd.DataFrame({prefix + v: [v in row for row in rows] for v in vocabulary}, index=values.index)


def make_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Převede metadata na číselné vstupy modelu.

    Args:
        df: Data ze split() (train, val nebo test).
        params: Parametry kódování z fit_features() (naučené na trainu).

    Returns:
        DataFrame vstupů typu float32 se stejnými sloupci pro všechny množiny.
    """
    numeric = (numeric_features(df) - params["mean"]) / params["std"]
    binary = pd.DataFrame({
        "free": df["price"] == 0,
        "windows": df["windows"],
        "mac": df["mac"],
        "linux": df["linux"],
    })
    genres = multi_hot(df["genres"], params["genres"], "genre_")
    categories = multi_hot(df["categories"], DESIGN_CATEGORIES, "category_")
    languages = multi_hot(df["supported_languages"], params["languages"], "language_")
    return pd.concat([numeric, binary, genres, categories, languages], axis=1).astype("float32")


if __name__ == "__main__":
    df = clean(load_metadata(download_metadata()))
    train, val, test = split(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_pickle(PROCESSED_DIR / f"{name}.pkl")
        print(f"{name}: {len(part)} her")
