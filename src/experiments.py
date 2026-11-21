"""Cílené experimenty s nejlepšími konfiguracemi modelů.

Experiment 1 (modality): model B jen s obrázky, bez metadat.
Experiment 2 (velikost trénovací množiny): modely A a B na podmnožinách trénovací množiny.
Varianty se všemi vstupy a se 100 % trénovacích dat jsou finální běhy z train.py.
Souhrn (průměr a směrodatná odchylka přes seedy) se ukládá do results/experiments.csv.
"""
import json
import time

import numpy as np
import pandas as pd

from preprocessing.prepare_data import PROCESSED_DIR, fit_features, make_features
from train import CONFIG_FILE, DEVICE, MODEL_NAMES, RESULTS_DIR, RUNS_DIR, train_run
from utils.progress import Progress


def experiment_plan(config: dict) -> list:
    """Seznam variant experimentů: (model, varianta, název varianty)."""
    plan = [("b", {"images_only": True}, "images_only")]
    for fraction in config["experiments"]["train_fractions"]:
        for model_name in ["a", "b"]:
            plan.append((model_name, {"train_fraction": fraction}, f"train{round(fraction * 100)}"))
    return plan


def variant_data(data: dict, variant: dict, subset_seed: int) -> tuple[dict, list | None]:
    """Připraví data pro variantu experimentu.

    Pro variantu jen s obrázky vynechá všechny vstupy metadat, hlava pak dostane jen vektor obrázků.
    Pro podíl trénovací množiny vybere náhodnou podmnožinu trénovacích her. Podmnožiny jsou stejné
    pro všechny modely a seedy a menší podmnožina je vždy součástí větší.

    Returns:
        Data pro trénování a pozice trénovacích her (None = celá trénovací množina).
    """
    if variant.get("images_only"):
        data = {**data, "X_train": data["X_train"].iloc[:, :0], "X_val": data["X_val"].iloc[:, :0]}
    if "train_fraction" not in variant:
        return data, None
    n_train = len(data["y_train"])
    order = np.random.default_rng(subset_seed).permutation(n_train)
    return data, order[:round(variant["train_fraction"] * n_train)].tolist()


def save_summary(records: list) -> None:
    """Uloží průměr a směrodatnou odchylku výsledků přes seedy do results/experiments.csv."""
    summary = (pd.DataFrame(records).groupby(["model", "variant"], sort=False)
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

    plan = experiment_plan(config)
    seeds = config["seeds"]
    progress = Progress(len(plan) * len(seeds))
    print(f"Zařízení: {DEVICE} | {len(plan)} variant x {len(seeds)} seedy | celkem {progress.total} běhů")

    records = []
    for model_name, variant, variant_name in plan:
        name, run_name = MODEL_NAMES[model_name]
        run_data, train_indices = variant_data(data, variant, config["experiments"]["subset_seed"])
        for i, seed in enumerate(seeds, start=1):
            result = train_run(model_name, config["models"][model_name], config, run_data, seed,
                               RUNS_DIR / f"exp_{run_name}_{variant_name}_seed{seed}", progress,
                               f"{name}, {variant_name}, běh {i}/{len(seeds)}", variant, train_indices)
            records.append({"model": name, "variant": variant_name, "seed": seed, **result})
        # průběžné uložení po každé variantě
        save_summary(records)

    print(pd.read_csv(RESULTS_DIR / "experiments.csv").to_string(index=False))
    print(f"\n[{time.strftime('%H:%M')}] hotovo, všech {progress.total} běhů dokončeno")
