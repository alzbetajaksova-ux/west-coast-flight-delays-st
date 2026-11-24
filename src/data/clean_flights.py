"""Čistenie letov 2018 a agregácia na hodiny (UTC).

Výstupy (data/interim/):
  flights_clean.parquet  – všetky lety, ktoré štartujú alebo pristávajú na jednom zo 6 letísk
  flights_hourly.csv     – [8760 h × 6 letísk] agregáty odletov a príletov
  flights_dst_report.json – kontrola spojenia / letného času

Spustenie:  py -3.11 -m src.data.clean_flights
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.config import NODES, hours_index, load_config, path

COLS = [
    "FlightDate", "Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline",
    "Origin", "Dest", "CRSDepTime", "DepTime", "DepDelay", "DepDel15",
    "CRSArrTime", "ArrDelay", "ArrDel15", "Cancelled", "CancellationCode", "Diverted",
    "CRSElapsedTime", "Distance", "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
]
CAUSES = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]


def _hhmm_to_timedelta(s: pd.Series) -> pd.Series:
    """'HHMM' -> Timedelta; '2400' = 24:00 (nasledujúci deň 00:00)."""
    v = pd.to_numeric(s, errors="coerce")
    return pd.to_timedelta(v // 100, unit="h") + pd.to_timedelta(v % 100, unit="m")


def _localize(naive: pd.Series, tz: str, report: dict, key: str) -> pd.Series:
    """Miestny čas -> naive UTC. Zaznamená neexistujúce (11. 3.) a dvojznačné (4. 11.) časy."""
    probe = naive.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    nat_before = naive.isna()
    bad = probe.isna() & ~nat_before
    # rozlíšenie: neexistujúci čas (jar) vs. dvojznačný (jeseň)
    is_nonexist = bad & (naive.dt.month == 3)
    is_ambig = bad & (naive.dt.month == 11)
    report[key] = {
        "nonexistent_spring_forward": int(is_nonexist.sum()),
        "ambiguous_fall_back": int(is_ambig.sum()),
        "rule": "nonexistent -> posun dopredu (shift_forward); ambiguous -> letný čas (PDT, prvý výskyt)",
    }
    loc = naive.dt.tz_localize(tz, ambiguous=np.ones(len(naive), dtype=bool), nonexistent="shift_forward")
    return loc.dt.tz_convert("UTC").dt.tz_localize(None)


def load_raw() -> pd.DataFrame:
    cfg = load_config()
    f = pd.read_csv(
        path("raw_flights"), engine="pyarrow", usecols=COLS,
        dtype={"CRSDepTime": str, "DepTime": str, "CRSArrTime": str, "Tail_Number": str,
               "Flight_Number_Reporting_Airline": str, "CancellationCode": str},
    )
    six = set(cfg["nodes"])
    f = f[f["Origin"].isin(six) | f["Dest"].isin(six)].copy()
    f["FlightDate"] = pd.to_datetime(f["FlightDate"])
    return f


def clean(f: pd.DataFrame, report: dict) -> pd.DataFrame:
    cfg = load_config()
    tz = cfg["local_tz"]
    six = set(cfg["nodes"])
    report["n_rows_touching_6"] = int(len(f))
    report["crs_dep_2400"] = int((f["CRSDepTime"] == "2400").sum())
    report["crs_arr_2400"] = int((f["CRSArrTime"] == "2400").sum())

    # --- odlet (miestny čas pôvodného letiska) ---
    f["dep_local"] = f["FlightDate"] + _hhmm_to_timedelta(f["CRSDepTime"])
    dep_mask = f["Origin"].isin(six)
    f["dep_utc"] = pd.NaT
    f.loc[dep_mask, "dep_utc"] = _localize(f.loc[dep_mask, "dep_local"], tz, report, "departures_dst")
    f["dep_utc"] = pd.to_datetime(f["dep_utc"])

    # --- prílet do našich letísk (miestny čas cieľa = US/Pacific) ---
    # Dátum príletu: FlightDate alebo +1 deň – vyberie sa kandidát najbližší k dep_local + CRSElapsedTime
    # (rozdiel časových pásiem < 12 h, takže výber je jednoznačný).
    arr_mask = f["Dest"].isin(six)
    arr_tod = _hhmm_to_timedelta(f.loc[arr_mask, "CRSArrTime"])
    base = f.loc[arr_mask, "FlightDate"] + arr_tod
    expected = f.loc[arr_mask, "dep_local"] + pd.to_timedelta(f.loc[arr_mask, "CRSElapsedTime"], unit="m")
    cands = [base - pd.Timedelta(days=1), base, base + pd.Timedelta(days=1)]
    diffs = np.stack([(c - expected).abs().dt.total_seconds().fillna(np.inf).to_numpy() for c in cands])
    pick = diffs.argmin(axis=0)
    arr_local = cands[0].where(pick == 0, cands[1].where(pick == 1, cands[2]))
    f["arr_local"] = pd.NaT
    f.loc[arr_mask, "arr_local"] = arr_local
    f["arr_local"] = pd.to_datetime(f["arr_local"])
    f["arr_utc"] = pd.NaT
    f.loc[arr_mask, "arr_utc"] = _localize(f.loc[arr_mask, "arr_local"], tz, report, "arrivals_dst")
    f["arr_utc"] = pd.to_datetime(f["arr_utc"])
    report["arrival_date_shift_counts"] = {str(k - 1): int(v) for k, v in zip(*np.unique(pick, return_counts=True))}

    # --- extrémy ---
    lo, hi = cfg["target"]["clip_low"], cfg["target"]["clip_high"]
    f["DepDelay_clip"] = f["DepDelay"].clip(lo, hi)
    f["ArrDelay_clip"] = f["ArrDelay"].clip(lo, hi)
    d = f[dep_mask & f["DepDelay"].notna()]
    report["depdelay_clipped_high"] = int((d["DepDelay"] > hi).sum())
    report["depdelay_clipped_low"] = int((d["DepDelay"] < lo).sum())
    report["depdelay_clipped_share"] = float(((d["DepDelay"] > hi) | (d["DepDelay"] < lo)).mean())
    report["depdelay_max"] = float(d["DepDelay"].max())
    report["depdelay_min"] = float(d["DepDelay"].min())

    # zrušené lety nemajú DepDelay; odklonené odleteli, DepDelay majú
    report["cancelled_with_depdelay"] = int((f["Cancelled"].eq(1) & f["DepDelay"].notna()).sum())
    report["notcancelled_without_depdelay"] = int((f["Cancelled"].eq(0) & f["DepDelay"].isna() & dep_mask).sum())
    return f


def hourly(f: pd.DataFrame) -> pd.DataFrame:
    """Hodinové agregáty pre každé letisko na plnej osi 8 760 h (UTC)."""
    idx = hours_index()
    out = []
    for node in NODES:
        dep = f[f["Origin"] == node].copy()
        dep["hour"] = dep["dep_utc"].dt.floor("h")
        flown = dep[(dep["Cancelled"] == 0) & dep["DepDelay"].notna()]
        g_all = dep.groupby("hour")
        g = flown.groupby("hour")
        h = pd.DataFrame(index=idx)
        h["n_sched"] = g_all.size()
        h["n_cancelled"] = g_all["Cancelled"].sum()
        h["n_diverted"] = g_all["Diverted"].sum()
        h["n_dep"] = g.size()
        h["dep_delay_mean"] = g["DepDelay_clip"].mean()
        h["dep_delay_mean_raw"] = g["DepDelay"].mean()
        h["dep_delay_median"] = g["DepDelay"].median()
        h["dep_del15_share"] = g["DepDel15"].mean()
        for c in CAUSES:
            h[f"dep_{c}_sum"] = g[c].sum(min_count=1)

        arr = f[(f["Dest"] == node)].copy()
        arr["hour"] = arr["arr_utc"].dt.floor("h")
        landed = arr[(arr["Cancelled"] == 0) & (arr["Diverted"] == 0) & arr["ArrDelay"].notna()]
        ga = landed.groupby("hour")
        h["n_arr_sched"] = arr.groupby("hour").size()
        h["n_arr"] = ga.size()
        h["arr_delay_mean"] = ga["ArrDelay_clip"].mean()

        cnt = ["n_sched", "n_cancelled", "n_diverted", "n_dep", "n_arr_sched", "n_arr"]
        h[cnt] = h[cnt].fillna(0).astype(int)
        h["cancel_share"] = (h["n_cancelled"] / h["n_sched"]).where(h["n_sched"] > 0, 0.0)
        h.insert(0, "airport", node)
        out.append(h.rename_axis("time_utc").reset_index())
    return pd.concat(out, ignore_index=True)


def main() -> None:
    report: dict = {}
    f = load_raw()
    f = clean(f, report)
    # lety mimo osi 2018 v UTC (1. 1. ráno UTC = 31. 12. 2017 miestne; 31. 12. večer miestne = 2019 UTC)
    idx = hours_index()
    dep6 = f["dep_utc"].notna()
    report["departures_outside_utc_year"] = int((dep6 & ~f["dep_utc"].dt.floor("h").isin(idx)).sum())

    keep = [c for c in f.columns if c not in ("CRSDepTime", "DepTime", "CRSArrTime")] + ["CRSDepTime", "CRSArrTime"]
    f[keep].to_parquet(path("interim") / "flights_clean.parquet", index=False)
    hf = hourly(f)
    hf.to_csv(path("interim") / "flights_hourly.csv", index=False)
    report["hourly_rows"] = int(len(hf))
    with open(path("interim") / "flights_dst_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
