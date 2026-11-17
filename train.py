"""Trénování a vyhodnocení modelů pro predikci úspěchu her na Steamu."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from prepare_data import PROCESSED_DIR, fit_features, make_features

# Složka pro výsledky (tabulky)
RESULTS_DIR = Path("results")
# Náhodné semínko pro reprodukovatelnost
SEED = 42


def evaluate(y_true, y_pred) -> dict:
    """Spočítá MAE a RMSE na log škále."""
    error = np.asarray(y_pred) - np.asarray(y_true)
    return {"MAE": np.abs(error).mean(), "RMSE": np.sqrt((error ** 2).mean())}


def run_baselines(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Natrénuje baseline na trénovací množině a vyhodnotí je na validační."""
    params = fit_features(train)
    X_train = make_features(train, params)
    X_val = make_features(val, params)

    predictions = {
        "Průměr": np.full(len(val), train["y"].mean()),
        "Lineární regrese": LinearRegression().fit(X_train, train["y"]).predict(X_val),
        "Gradient boosting": HistGradientBoostingRegressor(random_state=SEED).fit(X_train, train["y"]).predict(X_val),
    }
    return pd.DataFrame({name: evaluate(val["y"], pred) for name, pred in predictions.items()}).T.round(3)


if __name__ == "__main__":
    train = pd.read_pickle(PROCESSED_DIR / "train.pkl")
    val = pd.read_pickle(PROCESSED_DIR / "val.pkl")

    results = run_baselines(train, val)
    print(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "baselines.csv")
