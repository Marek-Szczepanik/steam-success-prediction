"""Vyhodnocení finálních modelů na validační a testovací množině.

Načte nejlepší checkpointy finálních běhů (runs/<běh>_seed<seed>/best.pt), spočítá MAE a RMSE,
změří rychlost inference a souhrn uloží do results/evaluation.csv. Predikce všech modelů
a seedů uloží do results/predictions.csv pro analýzu chyb.
Baseline se natrénují znovu na trénovací množině stejně jako při trénování.
"""
import json
import time

import pandas as pd
import torch

from preprocessing.prepare_data import PROCESSED_DIR, fit_features, make_features
from train import (CONFIG_FILE, DEVICE, MODEL_NAMES, RESULTS_DIR, RUNS_DIR, build_model, evaluate,
                   image_loader, predict, run_baselines, to_loader)

# Množiny, na kterých se modely vyhodnocují
SPLITS = ["val", "test"]
# Finální běhy modelů: model -> předpona složky běhu (CLIP z experimentu s encoderem)
FINAL_RUNS = {"metadata": "mlp_metadata", "a": "model_a", "b": "model_b", "clip": "exp_clip"}
# Velikost dávky pro vyhodnocení i měření rychlosti inference
BATCH_SIZE = 128
# Počet zahřívacích a měřených průchodů při měření rychlosti inference
WARMUP_STEPS = 5
TIMED_STEPS = 50


def make_loader(model_name: str, X: pd.DataFrame, y: pd.Series, split: str):
    """DataLoader pro vyhodnocení modelu (bez míchání)."""
    if model_name == "metadata":
        return to_loader(X, y, BATCH_SIZE, shuffle=False)
    return image_loader(X, y, split, BATCH_SIZE, shuffle=False)


def synchronize() -> None:
    """Počká, až GPU dokončí všechny výpočty (pro přesné měření času)."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def inference_speed(model: torch.nn.Module, loader) -> float:
    """Změří rychlost inference v hrách za sekundu.

    Měří se jen výpočet modelu na jedné dávce, která už je v paměti GPU (bez načítání dat):
    WARMUP_STEPS zahřívacích průchodů, potom TIMED_STEPS měřených průchodů.
    Výpočet probíhá stejně jako při vyhodnocení (režim inference, bfloat16).
    """
    *xb, _ = next(iter(loader))
    xb = [x.to(DEVICE) for x in xb]
    model.eval()
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        for _ in range(WARMUP_STEPS):
            model(*xb)
        synchronize()
        start = time.perf_counter()
        for _ in range(TIMED_STEPS):
            model(*xb)
        synchronize()
    return TIMED_STEPS * len(xb[0]) / (time.perf_counter() - start)


def baseline_records(data: dict, seeds: list) -> list:
    """Natrénuje baseline pro každý seed a vrátí jejich metriky na validační a testovací množině."""
    records = []
    for seed in seeds:
        results = {split: run_baselines(data["X_train"], data["y_train"], data[f"X_{split}"],
                                        data[f"y_{split}"], seed) for split in SPLITS}
        for name in results["val"]:
            records.append({"model": name, **{f"{split} {metric}": value for split in SPLITS
                                              for metric, value in results[split][name].items()}})
    return records


def model_records(model_name: str, data: dict, seeds: list) -> tuple[list, dict]:
    """Vyhodnotí finální běhy modelu pro všechny seedy.

    Returns:
        Pro každý seed metriky na validační a testovací množině, počet parametrů,
        čas trénování a rychlost inference. Dále predikce pro každou množinu
        ve tvaru {množina: {"<model> <seed>": predikce}}.
    """
    name = MODEL_NAMES[model_name][0]
    loaders = {split: make_loader(model_name, data[f"X_{split}"], data[f"y_{split}"], split) for split in SPLITS}
    records = []
    predictions: dict = {split: {} for split in SPLITS}
    for seed in seeds:
        run_dir = RUNS_DIR / f"{FINAL_RUNS[model_name]}_seed{seed}"
        hp = json.loads((run_dir / "config.json").read_text())
        model = build_model(model_name, data["X_train"].shape[1], hp["hidden_size"]).to(DEVICE)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=DEVICE))
        result = json.loads((run_dir / "result.json").read_text())

        record: dict = {"model": name}
        for split, loader in loaders.items():
            pred = predict(model, loader)
            predictions[split][f"{name} {seed}"] = pred
            metrics = evaluate(data[f"y_{split}"], pred)
            record.update({f"{split} {metric}": value for metric, value in metrics.items()})
        record["params total"] = sum(p.numel() for p in model.parameters())
        record["params trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        record["train time min"] = result["train_time_s"] / 60
        # medián času epochy je odolný vůči epochám protaženým např. uspáním počítače
        record["epoch time s"] = pd.read_csv(run_dir / "history.csv")["time_s"].median()
        record["inference games/s"] = inference_speed(model, loaders["test"])
        print(f"{name}, seed {seed}: test MAE {record['test MAE']:.3f}, "
              f"inference {record['inference games/s']:.0f} her/s")
        records.append(record)
    return records, predictions


if __name__ == "__main__":
    config = json.loads(CONFIG_FILE.read_text())
    splits = {name: pd.read_pickle(PROCESSED_DIR / f"{name}.pkl") for name in ["train", *SPLITS]}

    # kódování vstupů se učí jen z trénovací množiny
    params = fit_features(splits["train"])
    data = {}
    for name, df in splits.items():
        data[f"X_{name}"], data[f"y_{name}"] = make_features(df, params), df["y"]
    gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    print(f"Zařízení: {gpu} | dávka pro inferenci: {BATCH_SIZE} | seedy: {config['seeds']}")

    records = baseline_records(data, config["seeds"])
    # tabulky predikcí: skutečné y a predikce všech modelů a seedů pro každou množinu
    tables = {split: pd.DataFrame({"y": data[f"y_{split}"]}) for split in SPLITS}
    for model_name in FINAL_RUNS:
        model_rec, predictions = model_records(model_name, data, config["seeds"])
        records += model_rec
        for split in SPLITS:
            for column, pred in predictions[split].items():
                tables[split][column] = pred

    # metriky: průměr a směrodatná odchylka přes seedy, ostatní sloupce: průměr
    df = pd.DataFrame(records)
    metrics = [f"{split} {metric}" for split in SPLITS for metric in ["MAE", "RMSE"]]
    other = [column for column in df.columns if column not in ["model", *metrics]]
    summary = df.groupby("model", sort=False).agg({**{c: ["mean", "std"] for c in metrics},
                                                    **{c: "mean" for c in other}})
    summary.columns = [c if c in other else f"{c} {stat}" for c, stat in summary.columns]
    summary = summary.round(3)

    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "evaluation.csv")
    pd.concat(tables, names=["split", "app_id"]).round(4).to_csv(RESULTS_DIR / "predictions.csv")
    print(summary.to_string())
