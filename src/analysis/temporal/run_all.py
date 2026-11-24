"""Spustí celú časovú analýzu osoby A a grafy baseline modelov.

    py -3.11 -m src.analysis.temporal.run_all

Obrázky: results/figures/A/, tabuľky: results/tables/A/, kľúčové čísla: results/tables/A/key_numbers.json
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.analysis.temporal import checks, content, temporal, weather_impact
from src.analysis.temporal.common import CATS, CAT_COLORS, dump_keys, key, plt, save_fig
from src.data.config import path

MODEL_SK = {"persistence": "persistence", "seasonal": "sezónny priemer", "seasonal_ar1": "sezónny + AR(1) odchýlka",
            "linear_varx": "lineárny VARX (Ridge)"}
MODEL_COL = {"persistence": "#9aa5b1", "seasonal": "#e6550d", "seasonal_ar1": "#2e8b57", "linear_varx": "#1f5fa8"}


def baseline_figures() -> None:
    T = path("tables")
    h = pd.read_csv(T / "baseline_main_by_horizon.csv")
    c = pd.read_csv(T / "baseline_folds_by_weather.csv")
    key("baseline_main", {m: g.set_index("horizon")["MAE"].round(2).to_dict() for m, g in h.groupby("model")})
    key("baseline_main_rmse", {m: g.set_index("horizon")["RMSE"].round(2).to_dict() for m, g in h.groupby("model")})
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1, 2]})
    ax = axes[0]
    for m, g in h.groupby("model", sort=False):
        ax.errorbar(g["horizon"], g["MAE"], yerr=[g["MAE"] - g["MAE_lo"], g["MAE_hi"] - g["MAE"]], marker="o",
                    capsize=3, color=MODEL_COL[m], label=MODEL_SK[m])
    ax.set(title="Baseline – MAE na teste (nov–dec)", xlabel="horizont h (hodiny)", ylabel="MAE (min, 95 % CI)", xticks=[1, 3, 6])
    ax.legend()
    ax = axes[1]
    g = c[c["horizon"] == 1]
    models = list(MODEL_SK)
    w = 0.2
    for i, m in enumerate(models):
        s = g[g["model"] == m].set_index("weather_cat").reindex(CATS)
        x = np.arange(len(CATS)) + (i - 1.5) * w
        ax.bar(x, s["MAE"], w, yerr=[s["MAE"] - s["MAE_lo"], s["MAE_hi"] - s["MAE"]], color=MODEL_COL[m], capsize=2,
               label=MODEL_SK[m])
    n = g[g["model"] == models[0]].set_index("weather_cat").reindex(CATS)["n"]
    ax.set_xticks(range(len(CATS)), [f"{k}\n(n={int(v)})" for k, v in n.items()])
    ax.set(title="Baseline – MAE podľa počasia, h = 1 (4 štvrťročné foldy spolu)", ylabel="MAE (min, 95 % CI)")
    save_fig(fig, "A5_baseline")
    key("baseline_folds_h1_by_weather", {m: gg.set_index("weather_cat")["MAE"].round(2).to_dict()
                                         for m, gg in c[c["horizon"] == 1].groupby("model")})


def main() -> None:
    for name, mod in [("A1 obsah", content), ("A2 časové vlastnosti", temporal), ("A3 vplyv počasia", weather_impact),
                      ("A4 kontroly", checks)]:
        t = time.time()
        mod.run()
        print(f"{name}: {time.time() - t:.0f}s")
    baseline_figures()
    dump_keys()
    print("hotovo")


if __name__ == "__main__":
    main()
