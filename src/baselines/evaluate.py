"""Vyhodnotenie zo súborov preds_<model>_<variant>.npz (formát z dátového kontraktu).

Kľúče: y_true, y_pred, mask, weather_cat [S, H, 6]; t0 [S]; voliteľne horizons [H].
MAE/RMSE v minútach po horizontoch × kategóriách počasia × letiskách, s počtom testovacích
hodín a 95 % bootstrap intervalom. Bootstrap je blokový po dňoch (deň podľa t0), aby
rešpektoval autokoreláciu chýb v rámci dňa.

Modul je nezávislý od modelov; C ho môže rozšíriť v src/eval/ (napr. Diebold–Mariano).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import NODES, load_config


def save_predictions(file: Path, y_true, y_pred, mask, weather_cat, t0, horizons) -> None:
    np.savez_compressed(file, y_true=np.asarray(y_true, np.float32), y_pred=np.asarray(y_pred, np.float32),
                        mask=np.asarray(mask, np.uint8), weather_cat=np.asarray(weather_cat, np.int8),
                        t0=np.asarray(t0, np.int64), horizons=np.asarray(horizons, np.int64))


def load_predictions(files) -> dict:
    """Načíta jeden alebo viac npz (napr. 4 foldy) a spojí ich po osi S."""
    files = [files] if isinstance(files, (str, Path)) else list(files)
    parts = [dict(np.load(f)) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("y_true", "y_pred", "mask", "weather_cat", "t0")}
    out["horizons"] = parts[0].get("horizons", np.array(load_config()["window"]["H"]))
    return out


def _boot(abs_e, sq_e, day, reps, rng, ci):
    """Blokový bootstrap po dňoch pre MAE a RMSE."""
    days, inv = np.unique(day, return_inverse=True)
    s_abs = np.bincount(inv, abs_e, len(days))
    s_sq = np.bincount(inv, sq_e, len(days))
    cnt = np.bincount(inv, minlength=len(days)).astype(float)
    w = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=reps).astype(float)
    n = w @ cnt
    n[n == 0] = np.nan
    mae = (w @ s_abs) / n
    rmse = np.sqrt((w @ s_sq) / n)
    a = (1 - ci) / 2
    return np.nanquantile(mae, [a, 1 - a]), np.nanquantile(rmse, [a, 1 - a])


def metrics_table(p: dict, by=("horizon", "weather_cat"), reps: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Tabuľka metrík zoskupená podľa `by` ⊆ {horizon, weather_cat, airport}."""
    cfg = load_config()
    reps = reps or cfg["eval"]["bootstrap_reps"]
    ci = cfg["eval"]["ci"]
    names = cfg["weather_categories"]
    rng = np.random.default_rng(seed)
    S, Hn, N = p["y_true"].shape
    hz = np.broadcast_to(np.asarray(p["horizons"])[None, :, None], (S, Hn, N))
    ap = np.broadcast_to(np.arange(N)[None, None, :], (S, Hn, N))
    day = np.broadcast_to((np.asarray(p["t0"]) // 24)[:, None, None], (S, Hn, N))
    m = p["mask"] > 0
    df = pd.DataFrame({
        "horizon": hz[m], "weather_cat": p["weather_cat"][m], "airport": ap[m], "day": day[m],
        "err": (p["y_pred"] - p["y_true"])[m],
    })
    df["abs"], df["sq"] = df["err"].abs(), df["err"] ** 2
    rows = []
    groups = df.groupby(list(by)) if by else [((), df)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        (mae_lo, mae_hi), (rmse_lo, rmse_hi) = _boot(g["abs"].to_numpy(), g["sq"].to_numpy(), g["day"].to_numpy(),
                                                     reps, rng, ci)
        r = dict(zip(by, key))
        r.update(n=len(g), MAE=g["abs"].mean(), MAE_lo=mae_lo, MAE_hi=mae_hi,
                 RMSE=np.sqrt(g["sq"].mean()), RMSE_lo=rmse_lo, RMSE_hi=rmse_hi, bias=g["err"].mean())
        rows.append(r)
    out = pd.DataFrame(rows)
    if "weather_cat" in out:
        out["weather_cat"] = out["weather_cat"].map(dict(enumerate(names)))
    if "airport" in out:
        out["airport"] = out["airport"].map(dict(enumerate(NODES)))
    return out
