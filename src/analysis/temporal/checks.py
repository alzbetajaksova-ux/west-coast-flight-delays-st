"""A4 – Kontroly spojenia: Origin ↔ stanica, CRSDepTime „2400“, letný čas 11. 3. a 4. 11. 2018,
typickosť roka 2018 a tabuľka kategórií počasia v rozdeleniach."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analysis.temporal.common import (AP_COLORS, CATS, MONTHS_SK, NODES, departures, key, panel, plt, save_fig,
                                          save_tab)
from src.data.config import STATIONS, load_config, path


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def join_checks() -> None:
    ap = pd.read_csv(path("airports")).set_index("iata_code")
    st = pd.read_csv(path("interim") / "weather_stations_region.csv").set_index("AirportCode")
    grid = load_config()["grid"]
    rows = []
    for n in NODES:
        s = STATIONS[n]
        rows.append({"letisko": n, "stanica": s, "lat letisko": ap.loc[n, "latitude"], "lon letisko": ap.loc[n, "longitude"],
                     "lat stanica": st.loc[s, "LocationLat"], "lon stanica": st.loc[s, "LocationLng"],
                     "vzdialenosť (km)": _haversine(ap.loc[n, "latitude"], ap.loc[n, "longitude"],
                                                    st.loc[s, "LocationLat"], st.loc[s, "LocationLng"]),
                     "časové pásmo stanice": st.loc[s, "TimeZone"],
                     "bunka mriežky (riadok, stĺpec)": f"({st.loc[s, 'grid_row']}, {st.loc[s, 'grid_col']})"})
    df = save_tab(pd.DataFrame(rows), "A4_letisko_stanica")
    key("max_station_distance_km", float(df["vzdialenosť (km)"].max()))
    cells = {n: [int(st.loc[STATIONS[n], "grid_row"]), int(st.loc[STATIONS[n], "grid_col"])] for n in NODES}
    key("grid_cells_unique", len({tuple(v) for v in cells.values()}) == len(NODES))
    (path("processed") / "grid_index.json").write_text(json.dumps(
        {"nodes": NODES, "cells": cells, "grid": grid,
         "note": "riadok 0 = sever (49° N), stĺpec 0 = západ (125° W); bunka 1° × 1°, polootvorené intervaly"},
        indent=2, ensure_ascii=False), encoding="utf-8")

    rep = json.loads((path("interim") / "flights_dst_report.json").read_text(encoding="utf-8"))
    d = departures()
    # počet odletov po UTC hodinách okolo zmien času – overenie, že nevznikla diera ani zdvojenie
    out = []
    for day, label in [("2018-03-11", "jar (PST→PDT)"), ("2018-11-04", "jeseň (PDT→PST)")]:
        a = pd.Timestamp(day) + pd.Timedelta(hours=6)
        win = pd.date_range(a, a + pd.Timedelta(hours=8), freq="h")
        cnt = d[d["hour_utc"].isin(win)].groupby("hour_utc").size().reindex(win, fill_value=0)
        for t, c in cnt.items():
            out.append({"zmena": label, "hodina UTC": t, "miestne": t.tz_localize("UTC").tz_convert("America/Los_Angeles").strftime("%H:%M %Z"), "odletov": c})
    save_tab(pd.DataFrame(out), "A4_letny_cas_okolie")
    checks = pd.DataFrame([
        ("CRSDepTime = '2400'", rep["crs_dep_2400"]),
        ("CRSArrTime = '2400'", rep["crs_arr_2400"]),
        ("odlety v neexistujúcej hodine 11. 3. 02:00–02:59", rep["departures_dst"]["nonexistent_spring_forward"]),
        ("odlety v dvojznačnej hodine 4. 11. 01:00–01:59", rep["departures_dst"]["ambiguous_fall_back"]),
        ("prílety v dvojznačnej hodine 4. 11.", rep["arrivals_dst"]["ambiguous_fall_back"]),
        ("prílety s dátumom +1 deň (cez polnoc)", rep["arrival_date_shift_counts"].get("1", 0)),
        ("odlety mimo osi 2018 UTC", rep["departures_outside_utc_year"]),
        ("zrušené lety s vyplneným DepDelay", rep["cancelled_with_depdelay"]),
        ("DepDelay orezané na 360 min", rep["depdelay_clipped_high"]),
        ("max. vzdialenosť letisko–stanica (km)", round(float(df["vzdialenosť (km)"].max()), 2)),
    ], columns=["kontrola", "hodnota"])
    save_tab(checks, "A4_kontroly_spojenia")
    key("dst_ambiguous_departures", rep["departures_dst"]["ambiguous_fall_back"])
    key("clipped_360", rep["depdelay_clipped_high"])


def typical_year() -> None:
    w6 = pd.read_csv(path("interim") / "weather_events_6.csv", parse_dates=["start", "end"])
    w6["airport"] = w6["AirportCode"].str[1:]
    w6["year"], w6["month"] = w6["start"].dt.year, w6["start"].dt.month
    w6["hours"] = (w6["end"] - w6["start"]).dt.total_seconds() / 3600
    ev = w6.groupby(["airport", "year", "month"]).size().unstack("month", fill_value=0).stack().rename("n").reset_index()
    full = pd.MultiIndex.from_product([NODES, range(2016, 2023), range(1, 13)], names=["airport", "year", "month"])
    ev = ev.set_index(["airport", "year", "month"])["n"].reindex(full, fill_value=0).reset_index()
    ctx = ev[ev["year"] != 2018].groupby(["airport", "month"])["n"].agg(["mean", "std"])
    y18 = ev[ev["year"] == 2018].set_index(["airport", "month"])["n"]
    tab = ctx.join(y18.rename("2018"))
    tab["z"] = (tab["2018"] - tab["mean"]) / tab["std"].replace(0, np.nan)
    save_tab(tab.round(2).reset_index(), "A4_typickost_2018_mesiace")

    yr = w6.groupby(["airport", "year"]).agg(udalosti=("EventId", "size"), hodiny=("hours", "sum")).reset_index()
    piv = yr.pivot(index="year", columns="airport", values="udalosti")[NODES]
    save_tab(piv.reset_index(), "A4_udalosti_po_rokoch")
    rank = piv.rank(ascending=False).loc[2018]
    others = piv.drop(index=2018)
    ratio = piv.loc[2018] / others.mean()
    key("events_2018_vs_mean_ratio", ratio.round(2).to_dict())
    key("events_2018_rank_of_7", rank.astype(int).to_dict())
    key("events_total_2018_vs_mean", [int(piv.loc[2018].sum()), float(others.sum(1).mean())])
    by_type = w6.groupby(["year", "Type"]).size().unstack(fill_value=0)
    save_tab(by_type.reset_index(), "A4_udalosti_typ_po_rokoch")
    key("months_abs_z_gt2", int((tab["z"].abs() > 2).sum()))
    key("jan_vs_jul_events_7y", [int(w6[w6.month == 1].shape[0]), int(w6[w6.month == 7].shape[0])])

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
    for ax, n in zip(axes.ravel(), NODES):
        t = tab.loc[n]
        ax.fill_between(t.index, (t["mean"] - t["std"]).clip(lower=0), t["mean"] + t["std"], color="grey", alpha=0.25,
                        label="2016–22 bez 2018: ±1σ")
        ax.plot(t.index, t["mean"], color="grey", lw=1.5, label="priemer")
        ax.plot(t.index, t["2018"], color=AP_COLORS[n], marker="o", lw=2, label="2018")
        ax.set(title=f"{n}: udalosti počasia za mesiac", xticks=range(1, 13))
        ax.set_xticklabels(MONTHS_SK, rotation=45, fontsize=8)
    axes[0, 0].legend()
    save_fig(fig, "A4_typickost_2018")


def split_table() -> None:
    p = panel()
    v = p[p["valid"]]
    rows = []
    split = json.loads((path("processed") / "split.json").read_text(encoding="utf-8"))
    for name in ["main", "Q1", "Q2", "Q3", "Q4"]:
        col = f"part_{name}"
        for part in ["train", "val", "test", "gap"]:
            g = v[v[col] == part]
            r = {"rozdelenie": name, "časť": part, "hodín (letisko×h)": len(g)}
            for c in CATS:
                r[c] = int((g["cat_name"] == c).sum())
            r["priemer y"] = g["y"].mean()
            rows.append(r)
    df = save_tab(pd.DataFrame(rows), "A4_kategorie_v_rozdeleniach")
    folds_test = df[(df["rozdelenie"] != "main") & (df["časť"] == "test")][CATS].sum()
    key("folds_test_category_hours", folds_test.astype(int).to_dict())
    key("main_test_category_hours", df[(df["rozdelenie"] == "main") & (df["časť"] == "test")][CATS].iloc[0].astype(int).to_dict())
    rng = {k: {pt: [[str(pd.Timestamp(load_config()["period"]["start"]) + pd.Timedelta(hours=a)),
                     str(pd.Timestamp(load_config()["period"]["start"]) + pd.Timedelta(hours=b))] for a, b in r]
               for pt, r in v_.items()} for k, v_ in split["splits"].items()}
    rows = [{"rozdelenie": k, "časť": pt, "od (UTC)": a, "do (UTC)": b} for k, d in rng.items() for pt, r in d.items() for a, b in r]
    save_tab(pd.DataFrame(rows), "A4_rozdelenia_datumy")


def run() -> None:
    join_checks()
    typical_year()
    split_table()
