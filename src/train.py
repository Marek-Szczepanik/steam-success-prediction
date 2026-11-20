"""Trénování a vyhodnocení modelů pro predikci úspěchu her na Steamu.

Hyperparametry se načítají ze souboru config.json.
S přepínačem --search se nejdřív spustí random search hyperparametrů s jedním seedem
a nejlepší konfigurace se potom natrénuje se všemi seedy.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.hybrid import HybridModel
from models.mlp import MLP
from models.model_a import CNNEncoder
from models.model_b import PretrainedEncoder
from prepare_data import PROCESSED_DIR, fit_features, make_features
from prepare_images import GameDataset

# Soubor s hyperparametry a prostorem pro random search
CONFIG_FILE = Path("config.json")
# Složka pro výsledky
RESULTS_DIR = Path("results")
# Složka pro jednotlivé běhy trénování
RUNS_DIR = Path("runs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Zobrazovaný název modelu a název jeho běhů ve složce RUNS_DIR
MODEL_NAMES = {
    "metadata": ("MLP (metadata)", "mlp_metadata"),
    "a": ("Model A", "model_a"),
    "b": ("Model B", "model_b"),
}
NUM_WORKERS = 6            # procesy pro paralelní načítání obrázků


def evaluate(y_true, y_pred) -> dict:
    """Spočítá MAE a RMSE na log škále."""
    error = np.asarray(y_pred) - np.asarray(y_true)
    return {"MAE": np.abs(error).mean(), "RMSE": np.sqrt((error ** 2).mean())}


def run_baselines(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame,
                  y_val: pd.Series, seed: int) -> dict:
    """Natrénuje baseline na trénovací množině a vyhodnotí je na validační.

    Průměr a lineární regrese jsou deterministické, vliv seedu má jen gradient boosting.

    Returns:
        Slovník {název modelu: metriky z evaluate()}.
    """
    predictions = {
        "Průměr": np.full(len(y_val), y_train.mean()),
        "Lineární regrese": LinearRegression().fit(X_train, y_train).predict(X_val),
        "Gradient boosting": HistGradientBoostingRegressor(random_state=seed).fit(X_train, y_train).predict(X_val),
    }
    return {name: evaluate(y_val, pred) for name, pred in predictions.items()}


def to_loader(X: pd.DataFrame, y: pd.Series, batch_size: int, shuffle: bool) -> DataLoader:
    """Převede vstupy a cílovou proměnnou na DataLoader."""
    dataset = TensorDataset(
        torch.tensor(X.to_numpy(), dtype=torch.float32),
        torch.tensor(y.to_numpy(), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def image_loader(X: pd.DataFrame, y: pd.Series, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    """DataLoader s metadaty a obrázky: dávky (metadata, obrázky, y)."""
    return DataLoader(GameDataset(X, y, split), batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, persistent_workers=True)


def make_loaders(model_name: str, data: dict, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Vytvoří trénovací a validační DataLoader pro daný model."""
    if model_name == "metadata":
        return (to_loader(data["X_train"], data["y_train"], batch_size, shuffle=True),
                to_loader(data["X_val"], data["y_val"], batch_size, shuffle=False))
    return (image_loader(data["X_train"], data["y_train"], "train", batch_size, shuffle=True),
            image_loader(data["X_val"], data["y_val"], "val", batch_size, shuffle=False))


def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """Vrátí predikce modelu pro všechny hry v loaderu."""
    model.eval()                    # režim inference
    # autograd nic nezaznamenává, výpočet v bfloat16 (mixed precision)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        pred = torch.cat([model(*[x.to(DEVICE) for x in xb]).float().cpu() for *xb, _ in loader])
    model.train()                   # zpět do trénovacího režimu
    return pred.numpy()


def train_model(model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                train_loader: DataLoader, val_loader: DataLoader, y_val: pd.Series,
                epochs: int, patience: int, run_dir: Path) -> dict:
    """Natrénuje model a vrátí metriky nejlepší epochy na validační množině.

    Po každé epoše spočítá validační metriky a doplní průběh trénování do run_dir/history.csv.
    Model s nejnižší validační MAE uloží do run_dir/best.pt.
    Pokud se validační MAE nezlepší patience epoch po sobě, trénink skončí dřív (early stopping).

    Returns:
        Metriky nejlepší epochy a číslo této epochy.
    """
    history = []
    best = {"MAE": np.inf, "best_epoch": 0}

    for epoch in range(1, epochs + 1):          # epocha: jeden průchod trénovací množinou
        start = time.time()
        total_loss, n_games = 0.0, 0
        for *xb, yb in train_loader:            # dávka: jeden krok; xb = vstupy modelu
            xb, yb = [x.to(DEVICE) for x in xb], yb.to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):   # mixed precision: výpočet v 16 bitech
                pred = model(*xb)
            loss = loss_fn(pred.float(), yb)    # 1 dopředný průchod a loss (loss ve float32)
            optimizer.zero_grad()               # 2 vynulování starých gradientů
            loss.backward()                     # 3 gradienty lossu
            optimizer.step()                    # 4 aktualizace parametrů
            total_loss += loss.item() * len(yb)
            n_games += len(yb)

        metrics = evaluate(y_val, predict(model, val_loader))
        epoch_time = time.time() - start        # čas epochy včetně validace
        history.append({"epoch": epoch, "train_loss": total_loss / n_games,
                        "val_MAE": metrics["MAE"], "val_RMSE": metrics["RMSE"],
                        "time_s": round(epoch_time, 1)})
        # průběžné uložení, aby průběh zůstal zachovaný i při přerušení
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"  epocha {epoch}: val MAE {metrics['MAE']:.3f}, {epoch_time:.0f} s")
        if metrics["MAE"] < best["MAE"]:
            best = {**metrics, "best_epoch": epoch}
            torch.save(model.state_dict(), run_dir / "best.pt")
        elif epoch - best["best_epoch"] >= patience:    # early stopping: model se přestal zlepšovat
            print(f"  early stopping, nejlepší epocha {best['best_epoch']}")
            break

    return best


def build_model(model_name: str, n_inputs: int, hidden_size: int) -> nn.Module:
    """Vytvoří model podle názvu z MODEL_NAMES."""
    if model_name == "metadata":
        return MLP(n_inputs, hidden_size)
    encoder = CNNEncoder() if model_name == "a" else PretrainedEncoder()
    return HybridModel(encoder, encoder.out_dim, n_inputs, hidden_size)


def train_run(model_name: str, hp: dict, config: dict, data: dict, seed: int, run_dir: Path) -> dict:
    """Natrénuje jeden model s danými hyperparametry a seedem.

    Do run_dir uloží použitou konfiguraci (config.json), průběh trénování (history.csv),
    nejlepší model (best.pt) a výsledek běhu (result.json).
    Pokud v run_dir už je dokončený běh se stejnou konfigurací, vrátí jeho výsledek
    bez trénování (navázání po přerušení).

    Returns:
        Metriky nejlepší epochy, číslo nejlepší epochy, celkový čas trénování a počet parametrů.
    """
    run_config = {"model": model_name, "seed": seed, "epochs": config["epochs"],
                  "patience": config["patience"], **hp}
    config_file, result_file = run_dir / "config.json", run_dir / "result.json"
    if result_file.exists() and json.loads(config_file.read_text()) == run_config:
        print(f"  přeskočeno, běh už je hotový: {run_dir}")
        return json.loads(result_file.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(run_config, indent=2))

    torch.manual_seed(seed)
    train_loader, val_loader = make_loaders(model_name, data, hp["batch_size"])
    model = build_model(model_name, data["X_train"].shape[1], hp["hidden_size"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])

    start = time.time()
    result = train_model(model, nn.HuberLoss(), optimizer, train_loader, val_loader, data["y_val"],
                         config["epochs"], config["patience"], run_dir)
    result["train_time_s"] = round(time.time() - start, 1)
    result["params_total"] = sum(p.numel() for p in model.parameters())
    result["params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result_file.write_text(json.dumps(result, indent=2, default=float))
    return result


def sample_hyperparams(space: dict, rng: np.random.Generator) -> dict:
    """Náhodně vybere hyperparametry z prostoru pro random search.

    Learning rate a weight decay se vybírají rovnoměrně na logaritmické škále,
    batch size a šířka skryté vrstvy ze seznamu možností.
    """
    return {
        "learning_rate": float(np.exp(rng.uniform(*np.log(space["learning_rate"])))),
        "weight_decay": float(np.exp(rng.uniform(*np.log(space["weight_decay"])))),
        "batch_size": int(rng.choice(space["batch_size"])),
        "hidden_size": int(rng.choice(space["hidden_size"])),
    }


def search(model_name: str, config: dict, data: dict) -> dict:
    """Random search hyperparametrů: n_trials náhodných konfigurací, každá s jedním seedem.

    Výsledky všech konfigurací seřazené podle validační MAE uloží do results/search_<model>.csv.

    Returns:
        Hyperparametry konfigurace s nejnižší validační MAE.
    """
    name, run_name = MODEL_NAMES[model_name]
    space = config["search"]
    rng = np.random.default_rng(space["seed"])
    results_file = RESULTS_DIR / f"search_{run_name}.csv"
    RESULTS_DIR.mkdir(exist_ok=True)
    trials = []
    for trial in range(1, space["n_trials"] + 1):
        hp = sample_hyperparams(space, rng)
        print(f"{name}, konfigurace {trial}/{space['n_trials']}: {hp}")
        result = train_run(model_name, hp, config, data, space["seed"],
                           RUNS_DIR / f"search_{run_name}_{trial:02d}")
        trials.append({"trial": trial, **hp, **result})
        # průběžné uložení po každé konfiguraci
        pd.DataFrame(trials).sort_values("MAE").round(4).to_csv(results_file, index=False)
    print(pd.read_csv(results_file))

    best = min(trials, key=lambda t: t["MAE"])
    return {key: best[key] for key in hp}


def run(model_name: str, config: dict, data: dict, hp: dict | None = None) -> None:
    """Natrénuje model se zvolenými hyperparametry pro všechny seedy
    a uloží průměr a směrodatnou odchylku metrik.

    Bez hp se použijí hyperparametry z config.json. U modelu nad metadaty vyhodnotí navíc i baseline.
    """
    name, run_name = MODEL_NAMES[model_name]
    hp = hp or config["models"][model_name]
    print(name, hp)
    runs = []
    for seed in config["seeds"]:
        print(f"Seed {seed}")
        result = run_baselines(data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed) \
            if model_name == "metadata" else {}
        metrics = train_run(model_name, hp, config, data, seed, RUNS_DIR / f"{run_name}_seed{seed}")
        result[name] = {"MAE": metrics["MAE"], "RMSE": metrics["RMSE"]}
        runs.append(pd.DataFrame(result).T)

    # průměr a směrodatná odchylka přes seedy pro každý model
    results = pd.concat(runs).groupby(level=0, sort=False).agg(["mean", "std"]).round(3)
    results.columns = [f"{metric} {stat}" for metric, stat in results.columns]
    print(f"Validační množina, {len(config['seeds'])} seedy:")
    print(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = "metadata_models.csv" if model_name == "metadata" else f"{run_name}.csv"
    results.to_csv(RESULTS_DIR / results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["metadata", "a", "b", "all"], default="metadata")
    parser.add_argument("--search", action="store_true",
                        help="random search a potom finální trénink nejlepší konfigurace")
    args = parser.parse_args()

    config = json.loads(CONFIG_FILE.read_text())
    train = pd.read_pickle(PROCESSED_DIR / "train.pkl")
    val = pd.read_pickle(PROCESSED_DIR / "val.pkl")

    # kódování vstupů se učí jen z trénovací množiny
    params = fit_features(train)
    data = {"X_train": make_features(train, params), "y_train": train["y"],
            "X_val": make_features(val, params), "y_val": val["y"]}
    print("Zařízení:", DEVICE, "| počet vstupů:", data["X_train"].shape[1], "| model:", args.model)

    # "all" spustí model A a potom model B
    model_names = ["a", "b"] if args.model == "all" else [args.model]
    for model_name in model_names:
        if args.search:
            # nejlepší konfigurace ze searche se rovnou natrénuje se všemi seedy
            run(model_name, config, data, search(model_name, config, data))
        else:
            run(model_name, config, data)
