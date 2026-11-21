"""Cílené experimenty s nejlepší konfigurací modelu B.

Experiment 1 (modality): jen metadata, jen obrázky. Varianta se všemi vstupy je finální model B.
Experiment 2 (encoder): obrazový encoder CLIP místo ConvNeXt-Tiny. Varianta s ConvNeXt
je finální model B.
Všechny varianty používají hyperparametry nejlepší konfigurace modelu B z config.json.
Souhrn (průměr a směrodatná odchylka přes seedy) se ukládá do results/experiments.csv,
spolu s referenčními výsledky finálního modelu B.
"""
import json
import time

import pandas as pd

from preprocessing.prepare_data import PROCESSED_DIR, fit_features, make_features
from train import CONFIG_FILE, DEVICE, MODEL_NAMES, RESULTS_DIR, RUNS_DIR, train_run
from utils.progress import Progress

# Varianty experimentů: (experiment, model, varianta)
EXPERIMENTS = [
    ("modality", "metadata", "metadata_only"),
    ("modality", "b", "images_only"),
    ("encoder", "clip", "clip"),
]
# Reference z finálního modelu B: (experiment, varianta)
REFERENCES = [("modality", "all_inputs"), ("encoder", "convnext")]


def variant_data(data: dict, variant: str) -> dict:
    """Pro variantu jen s obrázky vynechá všechny vstupy metadat, hlava pak dostane jen vektor obrázků."""
    if variant != "images_only":
        return data
    return {**data, "X_train": data["X_train"].iloc[:, :0], "X_val": data["X_val"].iloc[:, :0]}


def reference_records(seeds: list) -> list:
    """Výsledky finálních běhů modelu B jako referenční varianty obou experimentů."""
    records = []
    for seed in seeds:
        result = json.loads((RUNS_DIR / f"model_b_seed{seed}" / "result.json").read_text())
        for experiment, variant in REFERENCES:
            records.append({"experiment": experiment, "variant": variant, "seed": seed, **result})
    return records


def save_summary(records: list) -> None:
    """Uloží průměr a směrodatnou odchylku výsledků přes seedy do results/experiments.csv."""
    summary = (pd.DataFrame(records).groupby(["experiment", "variant"], sort=False)
               [["MAE", "RMSE", "best_epoch", "train_time_s"]].agg(["mean", "std"]).round(3))
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "experiments.csv")


if __name__ == "__main__":
    config = json.loads(CONFIG_FILE.read_text())
    train = pd.read_pickle(PROCESSED_DIR / "train.pkl")
    val = pd.read_pickle(PROCESSED_DIR / "val.pkl")

    # kódování vstupů se učí jen z trénovací množiny
    params = fit_features(train)
    data = {"X_train": make_features(train, params), "y_train": train["y"],
            "X_val": make_features(val, params), "y_val": val["y"]}

    seeds = config["seeds"]
    hp = config["models"]["b"]          # všechny varianty s nejlepší konfigurací modelu B
    progress = Progress(len(EXPERIMENTS) * len(seeds))
    print(f"Zařízení: {DEVICE} | {len(EXPERIMENTS)} variant x {len(seeds)} seedy | celkem {progress.total} běhů")

    records = reference_records(seeds)
    for experiment, model_name, variant in EXPERIMENTS:
        name = MODEL_NAMES[model_name][0]
        for i, seed in enumerate(seeds, start=1):
            result = train_run(model_name, hp, config, variant_data(data, variant), seed,
                               RUNS_DIR / f"exp_{variant}_seed{seed}", progress,
                               f"{name}, {variant}, běh {i}/{len(seeds)}",
                               {"experiment": experiment, "variant": variant})
            records.append({"experiment": experiment, "variant": variant, "seed": seed, **result})
        # průběžné uložení po každé variantě
        save_summary(records)

    print(pd.read_csv(RESULTS_DIR / "experiments.csv").to_string(index=False))
    print(f"\n[{time.strftime('%H:%M')}] hotovo, všech {progress.total} běhů dokončeno")
