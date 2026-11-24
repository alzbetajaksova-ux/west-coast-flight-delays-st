"""WindowDataset – spoločné okná (L hodín vstupu -> H horizontov) pre všetky modely.

Nezávisí od frameworku: v PyTorch funguje ako `torch.utils.data.Dataset` (má __len__/__getitem__,
vracia numpy, ktoré default collate prevedie na tenzory); pre Keras je `as_arrays()`.

    ds = WindowDataset(mode="graph", L=24, H=(1, 3, 6), split="main", part="train")
    x, y, m = ds[0]["x"], ds[0]["y"], ds[0]["mask"]      # [L,6,F], [3,6], [3,6]
    arr = ds.as_arrays()                                  # dict s osou S navyše

    # foldy pre hodnotenie po počasí: split in {"Q1","Q2","Q3","Q4"}
    # grid: mode="grid" načíta data/processed/X_grid.npy [T,C,17,9] (stavia D)

Normalizácia: μ, σ z scaler.json pre zvolené `split` (iba tréningová časť). Príznaky s maskou
(y_lag, del15_share, arr_delay_mean) sa po normalizácii na neplatných hodinách nastavia na 0.
Cieľ y sa vracia v minútach (normalize_y=False) alebo normalizovaný (normalize_y=True);
`ds.inverse_y(y)` vráti minúty.
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from src.data.config import load_config, path

WEATHER_FEATS = ["wc_jasno", "wc_dazd_slaby", "wc_dazd_silny", "wc_hmla", "wc_sneh",
                 "severity", "precip_in", "weather_active_frac", "hours_since_event"]


class WindowDataset:
    def __init__(self, mode: str = "graph", L: int | None = None, H: Iterable[int] | None = None,
                 split: str = "main", part: str = "train", normalize: bool = True,
                 normalize_y: bool = False, oracle_weather: bool = False, data_dir=None):
        cfg = load_config()
        assert mode in ("graph", "grid"), mode
        assert part in ("train", "val", "test"), part
        self.mode, self.split, self.part = mode, split, part
        self.L = int(L or cfg["window"]["L"])
        self.H = np.array(list(H or cfg["window"]["H"]), dtype=int)
        self.normalize_y, self.oracle_weather = normalize_y, oracle_weather
        P = path("processed") if data_dir is None else data_dir

        with open(P / "split.json", encoding="utf-8") as f:
            self.split_def = json.load(f)
        with open(P / "scaler.json", encoding="utf-8") as f:
            sc = json.load(f)
        self.feature_names = sc["feature_names"]
        s = sc["splits"][split]
        self.y_mu, self.y_sigma = s["y"]["mu"], s["y"]["sigma"]

        self.Y = np.load(P / "Y.npy")
        self.M = np.load(P / "mask.npy")
        self.C = np.load(P / "weather_cat.npy")
        X = np.load(P / "X_nodes.npy").astype(np.float32)
        self.X_nodes_raw = X
        if normalize:
            mu = np.array([s["features"][f]["mu"] for f in self.feature_names], dtype=np.float32)
            sd = np.array([s["features"][f]["sigma"] for f in self.feature_names], dtype=np.float32)
            X = (X - mu) / sd
            k = {f: i for i, f in enumerate(self.feature_names)}
            for feat, flag in sc["masked_by"].items():
                X[..., k[feat]] = np.where(self.X_nodes_raw[..., k[flag]] > 0, X[..., k[feat]], 0.0)
        self.X_nodes = X

        if mode == "grid":
            gpath = P / "X_grid.npy"
            if not gpath.exists():
                raise FileNotFoundError(f"{gpath} neexistuje – mriežku stavia D (src/grid/build_grid.py).")
            G = np.load(gpath, mmap_mode="r")
            gs = P / "scaler_grid.json"
            if normalize and gs.exists():
                with open(gs, encoding="utf-8") as f:
                    g = json.load(f)["splits"][split]
                G = (np.asarray(G, dtype=np.float32) - np.array(g["mu"], np.float32)[None, :, None, None]) \
                    / np.array(g["sigma"], np.float32)[None, :, None, None]
            self.X_grid = G

        hmax = int(self.H.max())
        t0 = [np.arange(a + self.L - 1, b - hmax + 1) for a, b in self.split_def["splits"][split][part]]
        self.t0 = np.concatenate(t0) if t0 else np.array([], dtype=int)
        self._wk = [self.feature_names.index(f) for f in WEATHER_FEATS]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.t0)

    def __getitem__(self, i: int) -> dict:
        t0 = int(self.t0[i])
        src = self.X_nodes if self.mode == "graph" else self.X_grid
        x = np.asarray(src[t0 - self.L + 1:t0 + 1], dtype=np.float32)
        tt = t0 + self.H
        y = self.Y[tt].astype(np.float32)
        if self.normalize_y:
            y = (y - self.y_mu) / self.y_sigma
        out = {"x": x, "y": y, "mask": self.M[tt].astype(np.float32),
               "weather_cat": self.C[tt].astype(np.int64), "t0": t0}
        if self.mode == "grid":
            out["x_nodes"] = self.X_nodes[t0 - self.L + 1:t0 + 1]
        if self.oracle_weather:   # iba označený experiment „oracle“: počasie v čase T+h
            out["x_future_weather"] = self.X_nodes[tt][..., self._wk]
        return out

    def as_arrays(self) -> dict:
        """Všetky vzorky naraz (pre Keras). Pozor: grid × celý tréning môže byť veľký."""
        items = [self[i] for i in range(len(self))]
        return {k: np.stack([it[k] for it in items]) for k in items[0]}

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_sigma + self.y_mu if self.normalize_y else y

    def __repr__(self) -> str:
        return (f"WindowDataset(mode={self.mode}, split={self.split}, part={self.part}, L={self.L}, "
                f"H={self.H.tolist()}, S={len(self)})")


if __name__ == "__main__":
    for sp in ["main", "Q1", "Q2", "Q3", "Q4"]:
        for pt in ["train", "val", "test"]:
            ds = WindowDataset(split=sp, part=pt)
            it = ds[0]
            print(ds, {k: getattr(v, "shape", v) for k, v in it.items()})
