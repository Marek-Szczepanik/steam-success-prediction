"""Výpis průběhu trénování do konzole: aktuální běh, počet zbývajících běhů a odhad času."""
import time

import numpy as np


def format_time(seconds: float) -> str:
    """Převede počet sekund na čitelný text, např. "1 h 05 min"."""
    minutes = round(seconds / 60)
    if minutes < 1:
        return f"{seconds:.0f} s"
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


class Progress:
    """Sleduje postup všech běhů jednoho spuštění a vypisuje ho do konzole."""

    def __init__(self, total: int):
        """
        Args:
            total: Celkový počet běhů, které se v tomto spuštění natrénují.
        """
        self.total = total
        self.done = 0
        self.durations = []     # doby natrénovaných běhů, bez přeskočených

    def start(self, label: str, seed: int, hp: dict) -> None:
        """Vypíše záhlaví běhu: čas, pořadí, popis, seed a hyperparametry."""
        print(f"\n[{time.strftime('%H:%M')}] běh {self.done + 1}/{self.total}: {label}, seed {seed}")
        print(f"  lr {hp['learning_rate']:.2e}, weight decay {hp['weight_decay']:.2e}, "
              f"batch {hp['batch_size']}, šířka hlavy {hp['hidden_size']}")

    def finish(self, result: dict, duration: float | None) -> None:
        """Vypíše výsledek běhu, počet zbývajících běhů a odhad zbývajícího času.

        Args:
            result: Výsledek běhu z train_run().
            duration: Doba trénování v sekundách, u přeskočeného běhu None.
        """
        self.done += 1
        if duration is not None:
            self.durations.append(duration)
        remaining = self.total - self.done
        line = (f"  hotovo: val MAE {result['MAE']:.3f}, RMSE {result['RMSE']:.3f}, "
                f"nejlepší epocha {result['best_epoch']}")
        line += f", {format_time(duration)}" if duration is not None else " (přeskočeno, už hotové)"
        line += f" | zbývá {remaining} běhů"
        if remaining and self.durations:
            line += f", odhad {format_time(np.mean(self.durations) * remaining)}"
        print(line)
