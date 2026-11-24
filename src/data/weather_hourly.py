"""Počasie po hodinách (UTC).

Udalosť je aktívna v hodine [h, h+1), ak sa s ňou prekrýva interval [StartTime, EndTime).
Zrážky udalosti sa rozdelia do hodín úmerne prekryvu (v minútach).

Výstupy (data/interim/):
  weather_events_6.csv          – udalosti 2016–2022 na 6 letiskách (kontext, typickosť 2018)
  weather_region_events.csv     – udalosti v mriežke regiónu (počasie pre D, variant E2)
  weather_hourly.csv            – [8 760 h × 6 letísk]: one-hot typy, závažnosť, zrážky, kategória
  weather_region_hourly.parquet – riedka tabuľka (stanica × hodina) pre všetky stanice regiónu
  weather_stations_region.csv   – stanice regiónu so súradnicami a bunkou mriežky

Spustenie:  py -3.11 -m src.data.weather_hourly
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.config import NODES, STATIONS, hours_index, load_config, path

CAT = {"clear": 0, "rain_light": 1, "rain_heavy": 2, "fog": 3, "snow": 4}


def load_raw() -> pd.DataFrame:
    w = pd.read_csv(path("raw_weather"), engine="pyarrow")
    w = w.rename(columns={"StartTime(UTC)": "start", "EndTime(UTC)": "end", "Precipitation(in)": "precip"})
    w["start"] = pd.to_datetime(w["start"])
    w["end"] = pd.to_datetime(w["end"])
    return w


def region_mask(w: pd.DataFrame) -> pd.Series:
    g = load_config()["grid"]
    return (w["LocationLat"].ge(g["lat_min"]) & w["LocationLat"].lt(g["lat_max"])
            & w["LocationLng"].ge(g["lon_min"]) & w["LocationLng"].lt(g["lon_max"]))


def event_category(ev: pd.DataFrame) -> pd.Series:
    """Kategória jednej udalosti (pred prioritou v rámci hodiny)."""
    c = pd.Series(CAT["clear"], index=ev.index)
    t, s = ev["Type"], ev["Severity"]
    c[(t == "Rain") & (s == "Light")] = CAT["rain_light"]
    c[t == "Precipitation"] = CAT["rain_light"]
    c[(t == "Rain") & s.isin(["Moderate", "Heavy"])] = CAT["rain_heavy"]
    c[t.isin(["Storm", "Hail"])] = CAT["rain_heavy"]
    c[t == "Fog"] = CAT["fog"]
    c[t == "Snow"] = CAT["snow"]
    # Cold -> Jasno (0): nejde o zrážky ani zníženú viditeľnosť
    return c


# priorita pri prekryve: Sneh > Hmla > Dážď silný > Dážď slabý > Jasno
PRIORITY = np.array([0, 1, 2, 3, 4])  # rank podľa kódu kategórie (index = kód)


def explode_hours(ev: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """Udalosti -> riadky (udalosť, hodina) s minútami prekryvu, orezané na [t0, t1]."""
    ev = ev[(ev["end"] > t0) & (ev["start"] < t1 + pd.Timedelta(hours=1))].copy()
    ev["dur_min"] = ((ev["end"] - ev["start"]).dt.total_seconds() / 60).clip(lower=1)
    h_start = ev["start"].dt.floor("h")
    h_end = (ev["end"] - pd.Timedelta(seconds=1)).dt.floor("h")
    n = ((h_end - h_start) / pd.Timedelta(hours=1)).astype(int).clip(lower=0) + 1
    rep = ev.loc[ev.index.repeat(n)].copy()
    offs = np.concatenate([np.arange(k) for k in n.to_numpy()])
    rep["hour"] = h_start.loc[rep.index].to_numpy() + pd.to_timedelta(offs, unit="h")
    a = np.maximum(rep["start"].to_numpy(), rep["hour"].to_numpy())
    b = np.minimum(rep["end"].to_numpy(), (rep["hour"] + pd.Timedelta(hours=1)).to_numpy())
    rep["overlap_min"] = (b - a) / np.timedelta64(1, "m")
    rep = rep[rep["overlap_min"] > 0]
    rep["precip_h"] = rep["precip"].fillna(0) * rep["overlap_min"] / rep["dur_min"]
    rep = rep[(rep["hour"] >= t0) & (rep["hour"] <= t1)]
    return rep


def aggregate(rep: pd.DataFrame, key: str) -> pd.DataFrame:
    """(key, hour) -> one-hot typov, max závažnosť, zrážky, aktívne minúty, kategória."""
    cfg = load_config()
    sev = rep["Severity"].map(cfg["severity_map"]).fillna(1)
    rep = rep.assign(sev=sev, cat=event_category(rep))
    g = rep.groupby([key, "hour"])
    out = pd.DataFrame({
        "severity": g["sev"].max(),
        "precip_in": g["precip_h"].sum(),
        "active_min": g["overlap_min"].sum().clip(upper=60),
        "weather_cat": g["cat"].max(),   # kódy sú zoradené podľa priority
        "n_events": g.size(),
    })
    for t in cfg["weather_types"]:
        out[f"type_{t}"] = rep.assign(x=(rep["Type"] == t).astype(int)).groupby([key, "hour"])["x"].max()
    return out.reset_index()


def main() -> None:
    cfg = load_config()
    report: dict = {}
    w = load_raw()
    stations6 = [STATIONS[n] for n in NODES]
    st2node = {v: k for k, v in STATIONS.items()}

    w6 = w[w["AirportCode"].isin(stations6)].copy()
    w6.to_csv(path("interim") / "weather_events_6.csv", index=False)
    report["events_6_total"] = int(len(w6))
    report["events_6_2018"] = int((w6["start"].dt.year == 2018).sum())
    report["events_6_negative_duration"] = int((w6["end"] < w6["start"]).sum())

    reg = w[region_mask(w)].copy()
    g = cfg["grid"]
    reg["grid_row"] = ((g["lat_max"] - reg["LocationLat"]) // g["cell_deg"]).astype(int)   # riadok 0 = sever
    reg["grid_col"] = ((reg["LocationLng"] - g["lon_min"]) // g["cell_deg"]).astype(int)   # stĺpec 0 = západ
    stations = (reg.groupby("AirportCode")[["LocationLat", "LocationLng", "grid_row", "grid_col", "City", "State", "TimeZone"]]
                .first().reset_index())
    stations["is_node"] = stations["AirportCode"].isin(stations6)
    stations.to_csv(path("interim") / "weather_stations_region.csv", index=False)
    report["region_stations"] = int(len(stations))
    report["region_events_total"] = int(len(reg))
    report["region_rule"] = "lat in [32, 49), lon in [-125, -116) – polootvorené intervaly mriežky 17x9"

    t0 = pd.Timestamp(cfg["period"]["weather_load_from"])
    t1 = pd.Timestamp(cfg["period"]["end"])
    reg_win = reg[(reg["end"] > t0) & (reg["start"] <= t1 + pd.Timedelta(hours=1))]
    reg_win.to_csv(path("interim") / "weather_region_events.csv", index=False)

    rep = explode_hours(reg_win, t0, t1)
    agg_region = aggregate(rep, "AirportCode")
    agg_region.to_parquet(path("interim") / "weather_region_hourly.parquet", index=False)

    # 6 letísk na plnej osi (od 30. 12. 2017 kvôli vstupnému oknu)
    full_idx = pd.date_range(t0, t1, freq="h")
    a6 = agg_region[agg_region["AirportCode"].isin(stations6)]
    frames = []
    for node in NODES:
        sub = a6[a6["AirportCode"] == STATIONS[node]].set_index("hour").reindex(full_idx)
        sub = sub.drop(columns="AirportCode")
        sub = sub.fillna({c: 0 for c in sub.columns})
        sub.insert(0, "airport", node)
        frames.append(sub.rename_axis("time_utc").reset_index())
    wh = pd.concat(frames, ignore_index=True)
    int_cols = ["severity", "weather_cat", "n_events"] + [c for c in wh.columns if c.startswith("type_")]
    wh[int_cols] = wh[int_cols].astype(int)
    wh.to_csv(path("interim") / "weather_hourly.csv", index=False)

    in2018 = wh[wh["time_utc"].isin(hours_index())]
    names = cfg["weather_categories"]
    report["hours_per_category_2018_all"] = {names[k]: int(v) for k, v in in2018["weather_cat"].value_counts().sort_index().items()}
    report["hours_with_event_share_2018"] = in2018.groupby("airport").apply(
        lambda d: round(float((d["n_events"] > 0).mean()), 4), include_groups=False).to_dict()
    with open(path("interim") / "weather_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
