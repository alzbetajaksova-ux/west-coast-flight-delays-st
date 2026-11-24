"""Načítanie spoločnej konfigurácie (config.yaml) a pomocné cesty."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolútna cesta podľa config['paths'][key]; priečinky sa vytvoria."""
    p = ROOT / load_config()["paths"][key]
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
    return p


def hours_index() -> pd.DatetimeIndex:
    """Časová os 2018 v UTC (8 760 hodín), bez časovej zóny (naive UTC)."""
    c = load_config()["period"]
    return pd.date_range(c["start"], c["end"], freq="h")


NODES: list[str] = load_config()["nodes"]
STATIONS: dict[str, str] = load_config()["stations"]
