"""Zostavenie súborov dátového kontraktu (data/processed/).

  X_nodes.npy      [T, 6, F] float32 – surové (nenormalizované) príznaky uzlov
  Y.npy            [T, 6]    float32 – cieľ: priemerný DepDelay (min, orezaný), 0 tam, kde mask = 0
  mask.npy         [T, 6]    uint8   – 1 = platná hodina (>= 3 uskutočnené odlety)
  weather_cat.npy  [T, 6]    int8    – 0 Jasno, 1 Dážď slabý, 2 Dážď stredný/silný, 3 Hmla, 4 Sneh
  feature_names.json                  – názvy F príznakov a ktoré sa normalizujú
  timestamps.csv                      – index -> čas UTC/miestny, časti rozdelení
  split.json                          – hlavné rozdelenie + 4 štvrťročné foldy (rozsahy indexov)
  scaler.json                         – μ, σ z tréningovej časti každého rozdelenia

Spustenie:  py -3.11 -m src.data.build_processed
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.config import NODES, hours_index, load_config, path

FEATURES = [
    # (názov, normalizovať?)
    ("wc_jasno", False), ("wc_dazd_slaby", False), ("wc_dazd_silny", False), ("wc_hmla", False), ("wc_sneh", False),
    ("severity", True), ("precip_in", True), ("weather_active_frac", True), ("hours_since_event", True),
    ("n_sched", True), ("y_lag", True), ("y_valid", False), ("del15_share", True), ("cancel_share", True),
    ("n_arr", True), ("arr_delay_mean", True),
    ("hod_sin", False), ("hod_cos", False), ("dow_sin", False), ("dow_cos", False), ("doy_sin", False), ("doy_cos", False),
]
FEATURE_NAMES = [f for f, _ in FEATURES]
NORMALIZE = [f for f, n in FEATURES if n]
# príznaky, ktoré sú pri neplatnej hodine nastavené na 0 a po normalizácii sa majú znovu vynulovať
MASKED_BY = {"y_lag": "y_valid", "del15_share": "y_valid", "arr_delay_mean": "n_arr"}
HOURS_SINCE_CAP = 48


def _hours_since_event(wh_node: pd.DataFrame) -> pd.Series:
    """Počet hodín od poslednej hodiny s udalosťou počasia (kategória > 0), orezané na 48."""
    active = (wh_node["weather_cat"] > 0).to_numpy()
    out = np.empty(len(active))
    last = -10_000
    for i, a in enumerate(active):
        if a:
            last = i
        out[i] = min(i - last, HOURS_SINCE_CAP)
    return pd.Series(out, index=wh_node.index)


def build_arrays():
    cfg = load_config()
    idx = hours_index()
    T, N, F = len(idx), len(NODES), len(FEATURES)
    fh = pd.read_csv(path("interim") / "flights_hourly.csv", parse_dates=["time_utc"])
    wh = pd.read_csv(path("interim") / "weather_hourly.csv", parse_dates=["time_utc"])

    local = idx.tz_localize("UTC").tz_convert(cfg["local_tz"])
    hod = local.hour + local.minute / 60
    dow = local.day_of_week
    doy = local.day_of_year - 1
    cyc = {
        "hod_sin": np.sin(2 * np.pi * hod / 24), "hod_cos": np.cos(2 * np.pi * hod / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "doy_sin": np.sin(2 * np.pi * doy / 365), "doy_cos": np.cos(2 * np.pi * doy / 365),
    }

    X = np.zeros((T, N, F), dtype=np.float32)
    Y = np.zeros((T, N), dtype=np.float32)
    M = np.zeros((T, N), dtype=np.uint8)
    C = np.zeros((T, N), dtype=np.int8)
    k = {f: i for i, f in enumerate(FEATURE_NAMES)}
    min_dep = cfg["target"]["mask_min_departures"]

    for j, node in enumerate(NODES):
        f = fh[fh["airport"] == node].set_index("time_utc").reindex(idx)
        w_full = wh[wh["airport"] == node].set_index("time_utc").sort_index()
        hs = _hours_since_event(w_full).reindex(idx)
        w = w_full.reindex(idx)

        valid = (f["n_dep"] >= min_dep).to_numpy()
        M[:, j] = valid
        Y[:, j] = np.where(valid, f["dep_delay_mean"].fillna(0), 0)
        C[:, j] = w["weather_cat"].to_numpy()

        for c in range(5):
            X[:, j, c] = (w["weather_cat"] == c).to_numpy()
        X[:, j, k["severity"]] = w["severity"]
        X[:, j, k["precip_in"]] = w["precip_in"]
        X[:, j, k["weather_active_frac"]] = w["active_min"] / 60
        X[:, j, k["hours_since_event"]] = hs
        X[:, j, k["n_sched"]] = f["n_sched"]
        X[:, j, k["y_lag"]] = Y[:, j]
        X[:, j, k["y_valid"]] = valid
        X[:, j, k["del15_share"]] = np.where(valid, f["dep_del15_share"].fillna(0), 0)
        X[:, j, k["cancel_share"]] = f["cancel_share"]
        X[:, j, k["n_arr"]] = f["n_arr"]
        X[:, j, k["arr_delay_mean"]] = f["arr_delay_mean"].where(f["n_arr"] > 0, 0).fillna(0)
        for name, v in cyc.items():
            X[:, j, k[name]] = v
    assert np.isfinite(X).all()
    return idx, local, X, Y, M, C


def _ts(s: str, idx: pd.DatetimeIndex) -> int:
    return int(idx.get_loc(pd.Timestamp(s)))


def _runs(labels: np.ndarray, value: str) -> list[list[int]]:
    """Súvislé úseky indexov, kde labels == value (vrátane koncov)."""
    m = labels == value
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            out.append([i, j])
            i = j + 1
        else:
            i += 1
    return out


def _apply_gaps(labels: np.ndarray, gap: int) -> np.ndarray:
    """Na každej hranici dvoch rôznych častí vloží medzeru `gap` hodín.
    Medzera sa nikdy neberie z testu; pri hranici tréning→validácia sa berie z tréningu."""
    lab = labels.copy()
    bounds = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    for b in bounds:
        left, right = labels[b - 1], labels[b]
        if right == "test" or (left in ("train", "val") and right in ("train", "val")):
            lab[max(0, b - gap):b] = "gap"          # odrež koniec ľavého bloku
        else:                                        # left == test
            lab[b:min(len(lab), b + gap)] = "gap"   # odrež začiatok pravého bloku
    return lab


def build_splits(idx: pd.DatetimeIndex) -> tuple[dict, pd.DataFrame]:
    cfg = load_config()["split"]
    gap = cfg["gap_hours"]
    T = len(idx)
    labels_all = {}

    lab = np.full(T, "none", dtype=object)
    for part, (a, b) in cfg["main"].items():
        lab[_ts(a, idx):_ts(b, idx) + 1] = part
    labels_all["main"] = _apply_gaps(lab, gap)

    for fold in cfg["folds"]:
        lab = np.full(T, "train", dtype=object)
        a, b = _ts(fold["test"][0], idx), _ts(fold["test"][1], idx)
        lab[a:b + 1] = "test"
        # validácia = posledný kalendárny mesiac tréningu
        last_train = max(i for i in range(T) if lab[i] == "train")
        month = idx[last_train].to_period("M")
        mstart = _ts(str(month.start_time), idx)
        lab[mstart:last_train + 1] = "val"
        labels_all[fold["name"]] = _apply_gaps(lab, gap)

    split = {
        "description": ("Rozsahy sú inkluzívne indexy do časovej osi (timestamps.csv). Vzorka s posledným vstupným "
                        "indexom t0 patrí do časti P, ak všetky hodiny t0-L+1 … t0+max(H) ležia v jednom rozsahu P. "
                        f"Medzi časťami je medzera {gap} h (L + max H); z testu sa nikdy neodrezáva."),
        "gap_hours": gap,
        "splits": {name: {p: _runs(lab, p) for p in ("train", "val", "test")} for name, lab in labels_all.items()},
    }
    labels_df = pd.DataFrame({f"part_{k}": v for k, v in labels_all.items()})
    return split, labels_df


def build_scalers(X: np.ndarray, Y: np.ndarray, M: np.ndarray, split: dict) -> dict:
    k = {f: i for i, f in enumerate(FEATURE_NAMES)}
    out = {"feature_names": FEATURE_NAMES, "normalized": NORMALIZE, "masked_by": MASKED_BY, "splits": {}}
    for name, parts in split["splits"].items():
        tr = np.zeros(X.shape[0], dtype=bool)
        for a, b in parts["train"]:
            tr[a:b + 1] = True
        feats = {}
        for f in FEATURE_NAMES:
            if f not in NORMALIZE:
                feats[f] = {"mu": 0.0, "sigma": 1.0}
                continue
            v = X[tr, :, k[f]]
            if f in MASKED_BY:
                v = v[X[tr, :, k[MASKED_BY[f]]] > 0]
            v = v.ravel()
            sd = float(v.std())
            feats[f] = {"mu": float(v.mean()), "sigma": sd if sd > 1e-8 else 1.0}
        yv = Y[tr][M[tr] > 0]
        out["splits"][name] = {"features": feats, "y": {"mu": float(yv.mean()), "sigma": float(yv.std())}}
    return out


def main() -> None:
    cfg = load_config()
    P = path("processed")
    idx, local, X, Y, M, C = build_arrays()
    split, labels = build_splits(idx)
    scaler = build_scalers(X, Y, M, split)

    np.save(P / "X_nodes.npy", X)
    np.save(P / "Y.npy", Y)
    np.save(P / "mask.npy", M)
    np.save(P / "weather_cat.npy", C)
    ts = pd.DataFrame({
        "t": np.arange(len(idx)), "time_utc": idx,
        "time_local": local.tz_localize(None), "hour_local": local.hour,
        "dow_local": local.day_of_week, "month": idx.month,
    })
    pd.concat([ts, labels], axis=1).to_csv(P / "timestamps.csv", index=False)
    meta = {
        "nodes": NODES, "features": FEATURE_NAMES, "normalized": NORMALIZE, "masked_by": MASKED_BY,
        "weather_categories": cfg["weather_categories"],
        "shapes": {"X_nodes": list(X.shape), "Y": list(Y.shape), "mask": list(M.shape), "weather_cat": list(C.shape)},
        "notes": [
            "X_nodes je nenormalizovaný; normalizuje WindowDataset podľa scaler.json.",
            "y_lag/del15_share sú 0 tam, kde y_valid = 0; arr_delay_mean je 0, kde n_arr = 0.",
            "Čas = koniec plánovanej hodiny odletu (CRSDepTime) v UTC; hodina t pokrýva [t, t+1).",
        ],
    }
    with open(P / "feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    with open(P / "split.json", "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2, ensure_ascii=False)
    with open(P / "scaler.json", "w", encoding="utf-8") as fh:
        json.dump(scaler, fh, indent=2, ensure_ascii=False)

    names = cfg["weather_categories"]
    print("shapes", meta["shapes"])
    print("valid hours per node", dict(zip(NODES, M.sum(0).tolist())))
    print("valid airport-hours per category:", {names[c]: int(((C == c) & (M > 0)).sum()) for c in range(5)})
    print("mean Y valid per category:", {names[c]: round(float(Y[(C == c) & (M > 0)].mean()), 2) for c in range(5)})
    print(json.dumps(split["splits"], indent=None))


if __name__ == "__main__":
    main()
