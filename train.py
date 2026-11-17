"""Trénování a vyhodnocení modelů pro predikci úspěchu her na Steamu.

Spuštění:
    python train.py                 # baseline + MLP nad metadaty
    python train.py --model a       # model A: vlastní CNN + metadata
"""
import argparse
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
from prepare_data import PROCESSED_DIR, fit_features, make_features
from prepare_images import GameDataset

# Složka pro výsledky
RESULTS_DIR = Path("results")
# Složka pro jednotlivé běhy trénování
RUNS_DIR = Path("runs")
# Seedy pro opakované běhy
SEEDS = [123, 420, 666]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparametry trénování neuronových sítí
EPOCHS = 30
BATCH_SIZE = 256
IMAGE_BATCH_SIZE = 64      # modely s obrázky (4 obrázky na vzorek)
NUM_WORKERS = 6            # procesy pro paralelní načítání obrázků
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


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


def to_loader(X: pd.DataFrame, y: pd.Series, shuffle: bool) -> DataLoader:
    """Převede vstupy a cílovou proměnnou na DataLoader s dávkami velikosti BATCH_SIZE."""
    dataset = TensorDataset(
        torch.tensor(X.to_numpy(), dtype=torch.float32),
        torch.tensor(y.to_numpy(), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def image_loader(X: pd.DataFrame, y: pd.Series, split: str, shuffle: bool) -> DataLoader:
    """DataLoader s metadaty a obrázky: dávky (metadata, obrázky, y)."""
    return DataLoader(GameDataset(X, y, split), batch_size=IMAGE_BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, persistent_workers=True)


def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """Vrátí predikce modelu pro všechny hry v loaderu."""
    model.eval()                    # režim inference
    with torch.no_grad():           # autograd nic nezaznamenává
        pred = torch.cat([model(*[x.to(DEVICE) for x in xb]).cpu() for *xb, _ in loader])
    model.train()                   # zpět do trénovacího režimu
    return pred.numpy()


def train_model(model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                train_loader: DataLoader, val_loader: DataLoader, y_val: pd.Series,
                run_dir: Path) -> dict:
    """Natrénuje model a vrátí metriky nejlepší epochy na validační množině.

    Po každé epoše spočítá validační metriky.
    Model s nejnižší validační MAE uloží do run_dir/best.pt a průběh trénování do run_dir/history.csv.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = {"MAE": np.inf}

    for epoch in range(1, EPOCHS + 1):          # epocha: jeden průchod trénovací množinou
        start = time.time()
        total_loss, n_games = 0.0, 0
        for *xb, yb in train_loader:            # dávka: jeden krok; xb = vstupy modelu
            xb, yb = [x.to(DEVICE) for x in xb], yb.to(DEVICE)
            loss = loss_fn(model(*xb), yb)      # 1 dopředný průchod a loss
            optimizer.zero_grad()               # 2 vynulování starých gradientů
            loss.backward()                     # 3 gradienty lossu
            optimizer.step()                    # 4 aktualizace parametrů
            total_loss += loss.item() * len(yb)
            n_games += len(yb)

        metrics = evaluate(y_val, predict(model, val_loader))
        history.append({"epoch": epoch, "train_loss": total_loss / n_games,
                        "val_MAE": metrics["MAE"], "val_RMSE": metrics["RMSE"]})
        print(f"  epocha {epoch}: val MAE {metrics['MAE']:.3f}, {time.time() - start:.0f} s")
        if metrics["MAE"] < best["MAE"]:
            best = metrics
            torch.save(model.state_dict(), run_dir / "best.pt")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["metadata", "a"], default="metadata")
    args = parser.parse_args()

    train = pd.read_pickle(PROCESSED_DIR / "train.pkl")
    val = pd.read_pickle(PROCESSED_DIR / "val.pkl")

    # kódování vstupů se učí jen z trénovací množiny
    params = fit_features(train)
    X_train, y_train = make_features(train, params), train["y"]
    X_val, y_val = make_features(val, params), val["y"]
    if args.model == "metadata":
        train_loader = to_loader(X_train, y_train, shuffle=True)
        val_loader = to_loader(X_val, y_val, shuffle=False)
    else:
        train_loader = image_loader(X_train, y_train, "train", shuffle=True)
        val_loader = image_loader(X_val, y_val, "val", shuffle=False)
    print("Zařízení:", DEVICE, "| počet vstupů:", X_train.shape[1], "| model:", args.model)

    runs = []
    for seed in SEEDS:
        print(f"Seed {seed}")
        torch.manual_seed(seed)
        if args.model == "metadata":
            result = run_baselines(X_train, y_train, X_val, y_val, seed)
            model = MLP(X_train.shape[1])
            name, run_name = "MLP (metadata)", "mlp_metadata"
        else:
            result = {}
            encoder = CNNEncoder()
            model = HybridModel(encoder, encoder.out_dim, X_train.shape[1])
            name, run_name = "Model A", "model_a"
        model = model.to(DEVICE)

        loss_fn = nn.HuberLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        result[name] = train_model(model, loss_fn, optimizer, train_loader, val_loader,
                                   y_val, RUNS_DIR / f"{run_name}_seed{seed}")
        runs.append(pd.DataFrame(result).T)

    # průměr a směrodatná odchylka přes seedy pro každý model
    results = pd.concat(runs).groupby(level=0, sort=False).agg(["mean", "std"]).round(3)
    results.columns = [f"{metric} {stat}" for metric, stat in results.columns]
    print(f"Validační množina, {len(SEEDS)} seedy:")
    print(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = "metadata_models.csv" if args.model == "metadata" else "model_a.csv"
    results.to_csv(RESULTS_DIR / results_file)
