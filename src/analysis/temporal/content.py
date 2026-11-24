"""A1 – Obsah a charakteristiky dát: rozsah, počty, chýbajúce hodnoty, rozdelenie DepDelay,
zrušené lety, príčiny meškania, rozdelenie hodinového cieľa, trvanie udalostí počasia."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.temporal.common import (AP_COLORS, CATS, NODES, departures, flights, key, panel, plt, save_fig,
                                          save_tab)
from src.data.config import path

CAUSES = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
CAUSE_SK = {"CarrierDelay": "dopravca", "WeatherDelay": "počasie", "NASDelay": "NAS (letová prevádzka)",
            "SecurityDelay": "bezpečnosť", "LateAircraftDelay": "meškajúce lietadlo"}


def overview() -> None:
    wrep = json.loads((path("interim") / "weather_report.json").read_text(encoding="utf-8"))
    frep = json.loads((path("interim") / "flights_dst_report.json").read_text(encoding="utf-8"))
    f, d = flights(), departures()
    w6 = pd.read_csv(path("interim") / "weather_events_6.csv", parse_dates=["start", "end"])
    rows = [
        ("Počasie – udalosti v USA 2016–2022", 8_627_181),
        ("Počasie – stanice v regióne mriežky", wrep["region_stations"]),
        ("Počasie – udalosti v regióne mriežky", wrep["region_events_total"]),
        ("Počasie – udalosti na 6 letiskách 2016–2022", len(w6)),
        ("Počasie – udalosti na 6 letiskách 2018", int((w6["start"].dt.year == 2018).sum())),
        ("Lety 2018 – všetky (USA)", 7_206_195),
        ("Lety – štart alebo cieľ na 6 letiskách", frep["n_rows_touching_6"]),
        ("Lety – odlety zo 6 letísk", len(d)),
        ("Lety – priamo medzi 6 letiskami", int((f["Origin"].isin(NODES) & f["Dest"].isin(NODES)).sum())),
        ("Lety – prílety na 6 letísk", int(f["Dest"].isin(NODES).sum())),
        ("Odlety – zrušené", int(d["Cancelled"].sum())),
        ("Odlety – odklonené", int(d["Diverted"].sum())),
        ("Odlety mimo osi 2018 v UTC (31. 12. večer miestne)", frep["departures_outside_utc_year"]),
        ("Hodiny na osi (UTC)", 8760),
    ]
    save_tab(pd.DataFrame(rows, columns=["položka", "počet"]), "A1_prehlad_dat")
    key("n_departures", len(d))
    key("n_flights_touching", frep["n_rows_touching_6"])


def missing() -> None:
    d = departures()
    cols = ["Tail_Number", "DepDelay", "DepDel15", "ArrDelay", "CancellationCode"] + CAUSES
    rows = []
    for c in cols:
        na = d[c].isna() if c != "CancellationCode" else d[c].isna() | (d[c] == "")
        rows.append({"stĺpec": c, "chýba": int(na.sum()), "podiel": na.mean(),
                     "z toho zrušené": int((na & (d["Cancelled"] == 1)).sum())})
    df = save_tab(pd.DataFrame(rows), "A1_chybajuce_hodnoty")
    # príčiny meškania sú vyplnené práve vtedy, keď ArrDelay >= 15
    flown = d[(d["Cancelled"] == 0) & (d["Diverted"] == 0)]
    has_cause = flown["CarrierDelay"].notna()
    key("cause_filled_iff_arrdel15", float((has_cause == (flown["ArrDelay"] >= 15)).mean()))
    key("missing_depdelay", int(d["DepDelay"].isna().sum()))
    return df


def depdelay_distribution() -> None:
    d = departures()
    x = d["DepDelay"].dropna()
    q = x.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    summ = {
        "n": len(x), "priemer": x.mean(), "smer. odchýlka": x.std(), "medián": x.median(),
        "šikmosť": stats.skew(x), "špicatosť (excess)": stats.kurtosis(x),
        "min": x.min(), "max": x.max(),
        "podiel < 0 (skôr)": (x < 0).mean(), "podiel > 15 min": (x > 15).mean(),
        "podiel > 60 min": (x > 60).mean(), "podiel > 180 min": (x > 180).mean(),
        "podiel > 360 min (orezané)": (x > 360).mean(),
        "podiel minút z letov > 60 min": x[x > 60].sum() / x[x > 0].sum(),
    }
    summ.update({f"kvantil {k:g}": v for k, v in q.items()})
    save_tab(pd.Series(summ).rename("hodnota").rename_axis("štatistika").reset_index(), "A1_depdelay_statistiky")
    for k_, v in [("dep_mean", x.mean()), ("dep_median", x.median()), ("dep_p99", q[0.99]), ("dep_skew", stats.skew(x)),
                  ("dep_kurt", stats.kurtosis(x)), ("dep_share_neg", (x < 0).mean()), ("dep_share_gt15", (x > 15).mean()),
                  ("dep_share_gt60", (x > 60).mean()), ("dep_minutes_share_from_gt60", summ["podiel minút z letov > 60 min"]),
                  ("dep_max", x.max())]:
        key(k_, float(v))

    by_ap = d.groupby("Origin")["DepDelay"].agg(
        n="count", priemer="mean", medián="median", p90=lambda s: s.quantile(.9), p99=lambda s: s.quantile(.99),
        šikmosť=lambda s: stats.skew(s.dropna())).reindex(NODES)
    by_ap["podiel > 15 min"] = d.groupby("Origin")["DepDel15"].mean().reindex(NODES)
    by_ap["zrušené"] = d.groupby("Origin")["Cancelled"].mean().reindex(NODES)
    save_tab(by_ap.reset_index().rename(columns={"Origin": "letisko"}), "A1_depdelay_po_letiskach")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.hist(x.clip(-30, 180), bins=np.arange(-30, 182, 2), color="#1f5fa8")
    ax.axvline(x.median(), color="k", ls="--", lw=1, label=f"medián {x.median():.0f}")
    ax.axvline(x.mean(), color="#c0392b", ls="--", lw=1, label=f"priemer {x.mean():.1f}")
    ax.set(title="Rozdelenie DepDelay (orezané na −30…180)", xlabel="min", ylabel="počet letov")
    ax.legend()
    ax = axes[1]
    ax.hist(x, bins=np.arange(-60, 1540, 10), color="#1f5fa8", log=True)
    ax.axvline(360, color="#c0392b", ls=":", lw=1, label="orezanie cieľa 360 min")
    ax.set(title="Celý chvost (log os)", xlabel="min", ylabel="počet letov (log)")
    ax.legend()
    ax = axes[2]
    for n in NODES:
        s = np.sort(d.loc[d["Origin"] == n, "DepDelay"].dropna())
        ax.plot(s, 1 - np.arange(len(s)) / len(s), color=AP_COLORS[n], label=n)
    ax.set(xscale="symlog", yscale="log", xlim=(-30, 1500), title="Chvost P(DepDelay > x)", xlabel="min (symlog)")
    ax.legend(ncol=2)
    save_fig(fig, "A1_depdelay_rozdelenie")


def cancellations() -> None:
    d = departures()
    code = d.loc[d["Cancelled"] == 1, "CancellationCode"].map(
        {"A": "dopravca", "B": "počasie", "C": "NAS", "D": "bezpečnosť"})
    t = pd.crosstab(d.loc[d["Cancelled"] == 1, "Origin"], code).reindex(NODES)
    t["spolu"] = t.sum(1)
    t["podiel zrušených"] = d.groupby("Origin")["Cancelled"].mean().reindex(NODES)
    save_tab(t.reset_index().rename(columns={"Origin": "letisko"}), "A1_zrusene_po_letiskach")
    key("cancel_share", float(d["Cancelled"].mean()))
    key("cancel_weather_share_of_cancelled", float((code == "počasie").mean()))
    key("diverted_share", float(d["Diverted"].mean()))

    m = d.groupby(["month", "Origin"])["Cancelled"].mean().unstack()[NODES] * 100
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    for n in NODES:
        ax[0].plot(m.index, m[n], marker="o", color=AP_COLORS[n], label=n)
    ax[0].set(title="Podiel zrušených odletov po mesiacoch", ylabel="%", xticks=range(1, 13))
    ax[0].legend(ncol=3)
    t[[c for c in ["dopravca", "počasie", "NAS", "bezpečnosť"] if c in t]].plot.bar(stacked=True, ax=ax[1],
                                                                                   color=["#9aa5b1", "#08519c", "#e6550d", "#555"])
    ax[1].set(title="Príčina zrušenia (počet)", xlabel="")
    save_fig(fig, "A1_zrusene_lety")


def delay_causes() -> None:
    d = departures()
    s = d.groupby("Origin")[CAUSES].sum().reindex(NODES)
    share = s.div(s.sum(1), axis=0)
    out = share.rename(columns=CAUSE_SK)
    out["minúty spolu"] = s.sum(1)
    save_tab(out.reset_index().rename(columns={"Origin": "letisko"}), "A1_priciny_meskania")
    tot = s.sum() / s.sum().sum()
    for c in CAUSES:
        key(f"cause_share_{c}", float(tot[c]))
    key("weather_delay_share_by_airport", share["WeatherDelay"].round(4).to_dict())

    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = np.zeros(len(NODES))
    colors = ["#9aa5b1", "#08519c", "#e6550d", "#555555", "#f1c40f"]
    for c, col in zip(CAUSES, colors):
        ax.barh(NODES, share[c] * 100, left=left, color=col, label=CAUSE_SK[c])
        left += share[c].to_numpy() * 100
    ax.invert_yaxis()
    ax.set(title="Rozklad minút meškania podľa príčiny (lety s ArrDelay ≥ 15)", xlabel="% minút")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_fig(fig, "A1_priciny_meskania")


def hourly_target() -> None:
    p = panel()
    v = p[p["valid"]]
    key("valid_share", float(p["valid"].mean()))
    key("valid_hours_by_airport", p.groupby("airport")["valid"].sum().reindex(NODES).astype(int).to_dict())
    key("y_hourly_mean", float(v["y"].mean()))
    key("y_hourly_median", float(v["y"].median()))
    key("y_hourly_std", float(v["y"].std()))
    key("y_hourly_skew", float(stats.skew(v["y"])))
    key("n_dep_per_valid_hour_median", float(v["n_dep"].median()))
    night = p[p["hour_local"].between(1, 4)]
    key("night_departures_1_459", int(night["n_dep"].sum()))
    tab = p.groupby("airport").agg(platné_hodiny=("valid", "sum"), podiel_platných=("valid", "mean"),
                                   odletov_za_hodinu_medián=("n_dep", lambda s: s[s >= 3].median()),
                                   y_priemer=("y", "mean"), y_medián=("y", "median"), y_std=("y", "std")).reindex(NODES)
    save_tab(tab.reset_index(), "A1_hodinovy_ciel")

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].hist(v["y"], bins=np.arange(-20, 120, 2), color="#1f5fa8")
    ax[0].set(title="Hodinový cieľ y (platné hodiny)", xlabel="priemerný DepDelay v hodine (min)", ylabel="počet hodín")
    vm = p.groupby(["hour_local", "airport"])["valid"].mean().unstack()[NODES]
    for n in NODES:
        ax[1].plot(vm.index, vm[n] * 100, color=AP_COLORS[n], label=n)
    ax[1].set(title="Podiel platných hodín (≥ 3 odlety)", xlabel="miestna hodina", ylabel="%", xticks=range(0, 24, 3))
    ax[1].legend(ncol=2)
    for n in NODES:
        s = p.loc[(p["airport"] == n) & p["valid"]]
        ax[2].scatter(s["n_dep"], s["y"], s=2, alpha=0.15, color=AP_COLORS[n])
    ax[2].set(title="Šum cieľa vs. počet odletov v hodine", xlabel="odletov v hodine", ylabel="y (min)", ylim=(-20, 150))
    save_fig(fig, "A1_hodinovy_ciel")


def weather_events() -> None:
    w6 = pd.read_csv(path("interim") / "weather_events_6.csv", parse_dates=["start", "end"])
    w6["dur_h"] = (w6["end"] - w6["start"]).dt.total_seconds() / 3600
    w6["airport"] = w6["AirportCode"].str[1:]
    w6["year"] = w6["start"].dt.year
    t = pd.crosstab(w6["airport"], w6["Type"]).reindex(NODES)
    t["spolu"] = t.sum(1)
    t18 = pd.crosstab(w6.loc[w6["year"] == 2018, "airport"], w6.loc[w6["year"] == 2018, "Type"]).reindex(NODES).fillna(0)
    t18["spolu 2018"] = t18.sum(1)
    save_tab(t.join(t18.add_suffix(" (2018)")).reset_index(), "A1_udalosti_pocasia_typy")
    dur = w6.groupby("Type")["dur_h"].describe(percentiles=[.5, .9])[["count", "mean", "50%", "90%", "max"]]
    save_tab(dur.reset_index(), "A1_trvanie_udalosti")
    key("event_duration_median_h", float(w6["dur_h"].median()))
    key("event_duration_p90_h", float(w6["dur_h"].quantile(.9)))
    sev = pd.crosstab(w6["Type"], w6["Severity"])
    save_tab(sev.reset_index(), "A1_zavaznost_podla_typu")
    p = panel()
    hc = pd.crosstab(p["airport"], p["cat_name"]).reindex(NODES)[CATS]
    hcv = pd.crosstab(p.loc[p["valid"], "airport"], p.loc[p["valid"], "cat_name"]).reindex(NODES)[CATS]
    save_tab(hc.join(hcv.add_suffix(" (platné)")).reset_index(), "A1_hodiny_kategorii_2018")
    pr = w6[w6["year"] == 2018].groupby("Type")["precip"].describe()
    save_tab(pr.reset_index(), "A1_zrazky_2018")

    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    for typ, col in [("Rain", "#08519c"), ("Fog", "#b39ddb"), ("Snow", "#e6550d")]:
        ax[0].hist(w6.loc[w6["Type"] == typ, "dur_h"].clip(upper=12), bins=np.arange(0, 12.25, 0.25), alpha=0.6,
                   color=col, label=typ)
    ax[0].set(title="Trvanie udalostí počasia (6 letísk, 2016–2022)", xlabel="hodiny (orezané na 12)", ylabel="počet")
    ax[0].legend()
    hcv.div(hcv.sum(1), axis=0).mul(100)[CATS[1:]].plot.bar(ax=ax[1], color=["#6baed6", "#08519c", "#b39ddb", "#e6550d"])
    ax[1].set(title="Podiel platných hodín s počasím (2018)", ylabel="%", xlabel="")
    save_fig(fig, "A1_udalosti_pocasia")


def run() -> None:
    overview()
    missing()
    depdelay_distribution()
    cancellations()
    delay_causes()
    hourly_target()
    weather_events()
