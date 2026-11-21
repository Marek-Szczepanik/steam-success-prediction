"""Grafy do zprávy: průběh trénování finálních modelů a analýza chyb na testovací množině.

Křivky se kreslí z runs/<běh>/history.csv, analýza chyb z results/predictions.csv
(vytváří evaluate.py). Grafy se ukládají jako PDF do docs/images/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import FINAL_RUNS
from train import CONFIG_FILE, MODEL_NAMES, RESULTS_DIR, RUNS_DIR

# Složka pro obrázky do zprávy
IMAGES_DIR = Path("docs/images")
# Barva a styl čáry každého modelu (pevné pořadí barev, styl čáry odliší modely i bez barev)
STYLES = {
    "metadata": ("#2a78d6", "-", "o"),
    "a": ("#eb6834", "--", "s"),
    "b": ("#1baf7a", "-.", "^"),
    "clip": ("#eda100", ":", "D"),
}
# Hranice intervalů skutečné hodnoty y pro analýzu chyb
Y_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 8, np.inf]


def style_axes(ax) -> None:
    """Jednotný vzhled os: bez horního a pravého rámečku, jemná vodorovná mřížka."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(seeds: list) -> None:
    """Trénovací ztráta a validační MAE po epochách pro všechny finální modely a seedy."""
    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(9, 3.4))
    for model_name in FINAL_RUNS:
        color, linestyle, _ = STYLES[model_name]
        for i, seed in enumerate(seeds):
            history = pd.read_csv(RUNS_DIR / f"{FINAL_RUNS[model_name]}_seed{seed}" / "history.csv")
            label = MODEL_NAMES[model_name][0] if i == 0 else None   # v legendě jen jednou
            ax_loss.plot(history["epoch"], history["train_loss"], color=color, linestyle=linestyle,
                         linewidth=1.5, label=label)
            ax_mae.plot(history["epoch"], history["val_MAE"], color=color, linestyle=linestyle, linewidth=1.5)
    ax_loss.set(xlabel="Epocha", ylabel="Trénovací Huberova ztráta")
    ax_mae.set(xlabel="Epocha", ylabel="Validační MAE")
    for ax in (ax_loss, ax_mae):
        style_axes(ax)
    ax_loss.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "krivky.pdf")


def plot_errors(seeds: list) -> None:
    """Průměrná chyba (predikce minus skutečnost) a MAE na testovací množině podle skutečné hodnoty y.

    Chyba každé hry se nejdřív zprůměruje přes seedy, potom přes hry v intervalu.
    """
    predictions = pd.read_csv(RESULTS_DIR / "predictions.csv")
    test = predictions.loc[predictions["split"] == "test"]
    y = test["y"].to_numpy()
    n_bins = len(Y_BINS) - 1
    bin_of_game = np.searchsorted(Y_BINS, y, side="right") - 1    # interval, do kterého hra patří
    counts = np.bincount(bin_of_game, minlength=n_bins)
    labels = [f"{low}–{high}\n({n})" if high != np.inf else f"≥{low}\n({n})"
              for low, high, n in zip(Y_BINS, Y_BINS[1:], counts)]
    x = np.arange(n_bins)

    fig, (ax_bias, ax_mae) = plt.subplots(1, 2, figsize=(9, 3.6))
    for model_name in FINAL_RUNS:
        name = MODEL_NAMES[model_name][0]
        color, linestyle, marker = STYLES[model_name]
        # chyby [hry, seedy]; každá hra má stejný počet seedů, průměr přes všechny prvky
        # je tedy průměr přes hry z průměrů přes seedy
        errors = test[[f"{name} {seed}" for seed in seeds]].to_numpy() - y[:, None]
        bias = [errors[bin_of_game == i].mean() for i in range(n_bins)]
        mae = [np.abs(errors[bin_of_game == i]).mean() for i in range(n_bins)]
        for ax, values in [(ax_bias, bias), (ax_mae, mae)]:
            ax.plot(x, values, color=color, linestyle=linestyle, marker=marker,
                    markersize=5, linewidth=1.5, label=name)
    ax_bias.axhline(0, color="#888888", linewidth=0.8)
    ax_bias.set(ylabel="Průměrná chyba (predikce − y)")
    ax_mae.set(ylabel="MAE")
    for ax in (ax_bias, ax_mae):
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlabel("Skutečné y (počet her v intervalu)")
        style_axes(ax)
    ax_bias.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "chyby.pdf")


if __name__ == "__main__":
    config = json.loads(CONFIG_FILE.read_text())
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_curves(config["seeds"])
    plot_errors(config["seeds"])
    print(f"Grafy uloženy do {IMAGES_DIR}")
