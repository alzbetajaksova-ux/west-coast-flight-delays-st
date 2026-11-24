"""Baseline modely – referencia pre TGCN, ConvLSTM a CNN-LSTM.

  persistence   – posledné platné y (do 24 h späť) na tom istom letisku
  seasonal      – priemer z tréningu podľa letisko × miestna hodina × deň v týždni
  seasonal_ar1  – sezónny priemer + posledná odchýlka utlmená φ_h (φ odhadnuté na tréningu po horizontoch)
  linear_varx   – lineárna viacrozmerná autoregresia so vstupmi počasia (VARX cez Ridge):
                  vstup = celé okno L × 6 × F (normalizované), výstup = H × 6; α vybrané na validácii

Výstupy: results/preds_<model>_<split>.npz pre split ∈ {main, Q1..Q4} a tabuľky metrík v results/tables/A/.
Spustenie:  py -3.11 -m src.baselines.baselines
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.baselines.evaluate import load_predictions, metrics_table, save_predictions
from src.data.config import load_config, path
from src.data.window_dataset import WindowDataset

SPLITS = ["main", "Q1", "Q2", "Q3", "Q4"]


def _local_keys():
    ts = pd.read_csv(path("processed") / "timestamps.csv")
    return ts["hour_local"].to_numpy(), ts["dow_local"].to_numpy()


def _last_valid(Y, M, t0, max_back=24):
    """Posledné platné y v čase <= t0 (pre každé letisko), inak NaN."""
    out = np.full((len(t0), Y.shape[1]), np.nan, dtype=np.float32)
    for back in range(max_back, -1, -1):        # bližšie hodnoty prepíšu vzdialenejšie
        t = t0 - back
        ok = (t >= 0)
        tt = np.clip(t, 0, None)
        v = np.where(M[tt] > 0, Y[tt], np.nan)
        v[~ok] = np.nan
        out = np.where(np.isnan(v), out, v)
    return out


def _seasonal_table(Y, M, train_idx, hod, dow):
    """Priemer y podľa (letisko, hodina, deň v týždni); fallback (letisko, hodina) a (letisko)."""
    N = Y.shape[1]
    tab = np.full((N, 24, 7), np.nan)
    for j in range(N):
        ok = train_idx[M[train_idx, j] > 0]
        d = pd.DataFrame({"h": hod[ok], "d": dow[ok], "y": Y[ok, j]})
        g3 = d.groupby(["h", "d"])["y"].agg(["mean", "size"])
        g2 = d.groupby("h")["y"].mean()
        for h in range(24):
            for w in range(7):
                if (h, w) in g3.index and g3.loc[(h, w), "size"] >= 5:
                    tab[j, h, w] = g3.loc[(h, w), "mean"]
                elif h in g2.index:
                    tab[j, h, w] = g2[h]
                else:
                    tab[j, h, w] = d["y"].mean()
    return tab


def _seasonal_at(tab, t, hod, dow):
    # t: [...], vracia [..., 6]
    return np.moveaxis(tab[:, hod[t], dow[t]], 0, -1)


def _train_hours(split_def, name):
    return np.concatenate([np.arange(a, b + 1) for a, b in split_def["splits"][name]["train"]])


def run_split(name: str, hod, dow) -> dict:
    cfg = load_config()
    H = np.array(cfg["window"]["H"])
    tr = WindowDataset(split=name, part="train")
    va = WindowDataset(split=name, part="val")
    te = WindowDataset(split=name, part="test")
    Y, M, C = te.Y, te.M, te.C
    train_idx = _train_hours(te.split_def, name)
    tab = _seasonal_table(Y, M, train_idx, hod, dow)

    def common(ds):
        tt = ds.t0[:, None] + H[None, :]
        return tt, Y[tt], M[tt], C[tt]

    preds = {}
    tt, yt, mt, ct = common(te)

    # 1) persistence (fallback: sezónny priemer, ak 24 h nebolo platné y)
    seas = _seasonal_at(tab, tt, hod, dow)                       # [S, H, 6]
    last = _last_valid(Y, M, te.t0)                              # [S, 6]
    pers = np.where(np.isnan(last)[:, None, :], seas, last[:, None, :])
    preds["persistence"] = pers

    # 2) sezónny priemer
    preds["seasonal"] = seas

    # 3) sezónny priemer + utlmená posledná odchýlka; φ_h = OLS cez počiatok na tréningu
    def anomalies(ds):
        tt_, y_, m_, _ = common(ds)
        last_ = _last_valid(Y, M, ds.t0)
        s0 = _seasonal_at(tab, ds.t0, hod, dow)
        a0 = last_ - s0
        a_h = y_ - _seasonal_at(tab, tt_, hod, dow)
        return a0, a_h, m_
    a0, ah, mh = anomalies(tr)
    phi = np.zeros(len(H))
    for k in range(len(H)):
        ok = (mh[:, k, :] > 0) & ~np.isnan(a0)
        x, y = a0[ok], ah[:, k, :][ok]
        phi[k] = float((x * y).sum() / (x * x).sum())
    a0t = np.nan_to_num(last - _seasonal_at(tab, te.t0, hod, dow))
    preds["seasonal_ar1"] = seas + phi[None, :, None] * a0t[:, None, :]

    # 4) VARX (Ridge) na celom normalizovanom okne
    def flat(ds):
        a = ds.as_arrays()
        return a["x"].reshape(len(ds), -1), a["y"], a["mask"]
    Xtr, ytr, mtr = flat(tr)
    Xva, yva, mva = flat(va)
    Xte, _, _ = flat(te)
    y_mu = tr.y_mu
    def fit(alpha, X, y):
        # neplatné ciele nahradíme sezónnym priemerom (nemajú vplyv na metriky, len stabilizujú fit)
        return Ridge(alpha=alpha).fit(X, y.reshape(len(y), -1) - y_mu)
    ytr_f = np.where(mtr > 0, ytr, _seasonal_at(tab, tr.t0[:, None] + H[None, :], hod, dow))
    best = None
    for alpha in [1, 10, 100, 1000, 10000]:
        mdl = fit(alpha, Xtr, ytr_f)
        p = mdl.predict(Xva).reshape(yva.shape) + y_mu
        mae = np.abs(p - yva)[mva > 0].mean()
        if best is None or mae < best[0]:
            best = (mae, alpha)
    ytrva = np.concatenate([ytr_f, np.where(mva > 0, yva, _seasonal_at(tab, va.t0[:, None] + H[None, :], hod, dow))])
    mdl = fit(best[1], np.concatenate([Xtr, Xva]), ytrva)
    preds["linear_varx"] = mdl.predict(Xte).reshape(yt.shape) + y_mu

    out_dir = path("results")
    for model, yp in preds.items():
        save_predictions(out_dir / f"preds_{model}_{name}.npz", yt, yp, mt, ct, te.t0, H)
    return {"phi": phi.round(3).tolist(), "ridge_alpha": best[1]}


def main() -> None:
    hod, dow = _local_keys()
    info = {}
    for name in SPLITS:
        t = time.time()
        info[name] = run_split(name, hod, dow)
        print(name, info[name], f"{time.time() - t:.0f}s")

    T = path("tables")
    models = ["persistence", "seasonal", "seasonal_ar1", "linear_varx"]
    res = path("results")
    rows_h, rows_cat, rows_ap, rows_fold_cat = [], [], [], []
    for m in models:
        p = load_predictions(res / f"preds_{m}_main.npz")
        rows_h.append(metrics_table(p, by=("horizon",)).assign(model=m))
        rows_cat.append(metrics_table(p, by=("horizon", "weather_cat")).assign(model=m))
        rows_ap.append(metrics_table(p, by=("horizon", "airport")).assign(model=m))
        pf = load_predictions([res / f"preds_{m}_{q}.npz" for q in ["Q1", "Q2", "Q3", "Q4"]])
        rows_fold_cat.append(metrics_table(pf, by=("horizon", "weather_cat")).assign(model=m))
    for rows, fn in [(rows_h, "baseline_main_by_horizon.csv"), (rows_cat, "baseline_main_by_weather.csv"),
                     (rows_ap, "baseline_main_by_airport.csv"), (rows_fold_cat, "baseline_folds_by_weather.csv")]:
        df = pd.concat(rows, ignore_index=True)
        cols = ["model"] + [c for c in df.columns if c != "model"]
        df[cols].round(3).to_csv(T / fn, index=False)
    pd.DataFrame(info).T.to_csv(T / "baseline_params.csv")
    print(pd.concat(rows_h)[["model", "horizon", "n", "MAE", "MAE_lo", "MAE_hi", "RMSE"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
