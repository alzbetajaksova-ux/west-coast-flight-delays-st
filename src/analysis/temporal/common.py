"""Spoločné načítanie dát a štýl grafov pre časovú analýzu (osoba A)."""
from __future__ import annotations

import json
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.config import NODES, load_config, path  # noqa: E402

CFG = load_config()
CATS = CFG["weather_categories"]
AP_COLORS = dict(zip(NODES, ["#1f5fa8", "#5fa2dd", "#c0392b", "#e67e73", "#2e8b57", "#8fd19e"]))
CAT_COLORS = dict(zip(CATS, ["#9aa5b1", "#6baed6", "#08519c", "#b39ddb", "#e6550d"]))
MONTHS_SK = ["jan", "feb", "mar", "apr", "máj", "jún", "júl", "aug", "sep", "okt", "nov", "dec"]
DOW_SK = ["po", "ut", "st", "št", "pi", "so", "ne"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})

KEY: dict = {}   # kľúčové čísla pre kapitolu správy -> results/tables/A/key_numbers.json


def save_fig(fig, name: str) -> None:
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path("figures") / f"{name}.png")
    plt.close(fig)


def save_tab(df: pd.DataFrame, name: str, index: bool = False) -> pd.DataFrame:
    df.to_csv(path("tables") / f"{name}.csv", index=index, float_format="%.4g")
    return df


def key(name: str, value):
    """Zaznamená kľúčové číslo do key_numbers.json."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    KEY[name] = value
    return value


def dump_keys() -> None:
    f = path("tables") / "key_numbers.json"
    old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    old.update(KEY)
    f.write_text(json.dumps(old, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


@lru_cache(maxsize=1)
def flights() -> pd.DataFrame:
    f = pd.read_parquet(path("interim") / "flights_clean.parquet")
    return f


@lru_cache(maxsize=1)
def departures() -> pd.DataFrame:
    """Odlety zo 6 letísk v rámci osi 2018 (UTC), s miestnymi časovými kľúčmi."""
    f = flights()
    d = f[f["Origin"].isin(NODES)].copy()
    d["hour_utc"] = d["dep_utc"].dt.floor("h")
    d["hour_local"] = d["dep_local"].dt.hour
    d["dow"] = d["dep_local"].dt.day_of_week
    d["month"] = d["dep_local"].dt.month
    return d


@lru_cache(maxsize=1)
def panel() -> pd.DataFrame:
    """Hodinový panel (8 760 × 6) – lety + počasie + maska + miestne kľúče + časti rozdelenia."""
    P = path("processed")
    ts = pd.read_csv(P / "timestamps.csv", parse_dates=["time_utc", "time_local"])
    fh = pd.read_csv(path("interim") / "flights_hourly.csv", parse_dates=["time_utc"])
    wh = pd.read_csv(path("interim") / "weather_hourly.csv", parse_dates=["time_utc"])
    Y, M, C = np.load(P / "Y.npy"), np.load(P / "mask.npy"), np.load(P / "weather_cat.npy")
    rows = []
    for j, n in enumerate(NODES):
        d = ts.copy()
        d["airport"] = n
        d["y"] = np.where(M[:, j] > 0, Y[:, j], np.nan)
        d["valid"] = M[:, j] > 0
        d["cat"] = C[:, j]
        rows.append(d)
    p = pd.concat(rows, ignore_index=True)
    p = p.merge(fh, on=["time_utc", "airport"], how="left").merge(wh, on=["time_utc", "airport"], how="left",
                                                                    suffixes=("", "_w"))
    p["cat_name"] = p["cat"].map(dict(enumerate(CATS)))
    p["season"] = p["month"].map({12: "zima", 1: "zima", 2: "zima", 3: "jar", 4: "jar", 5: "jar",
                                  6: "leto", 7: "leto", 8: "leto", 9: "jeseň", 10: "jeseň", 11: "jeseň"})
    return p


def seasonal_anomaly(p: pd.DataFrame) -> pd.Series:
    """y mínus priemer letisko × miestna hodina × deň v týždni (opisná anomália, celý rok)."""
    base = p.groupby(["airport", "hour_local", "dow_local"])["y"].transform("mean")
    return p["y"] - base


def mean_ci(x: np.ndarray, reps: int = 500, seed: int = 0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(reps, len(x))).mean(1)
    return x.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975)
