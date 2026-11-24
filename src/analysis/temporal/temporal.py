"""A2 – Časové vlastnosti: denný, týždenný a ročný profil meškania a počasia, ACF/PACF, stacionarita."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src.analysis.temporal.common import (AP_COLORS, CATS, CAT_COLORS, DOW_SK, MONTHS_SK, NODES, departures, key,
                                          panel, plt, save_fig, save_tab)


def profiles() -> None:
    d = departures()
    p = panel()
    # --- denný profil (miestna hodina) ---
    dh = d.groupby(["hour_local", "Origin"])["DepDelay"].mean().unstack()[NODES]
    cnt = d.groupby(["hour_local", "Origin"]).size().unstack()[NODES]
    dh = dh.where(cnt >= 300)            # nočné hodiny s pár letmi skresľujú
    wx = p.groupby(["hour_local", "cat_name"]).size().unstack().fillna(0)
    wx = wx.div(wx.sum(1), axis=0)[CATS[1:]] * 100
    save_tab(dh.round(2).reset_index(), "A2_denny_profil_meskanie")
    save_tab(wx.round(2).reset_index(), "A2_denny_profil_pocasie")
    allh = d.groupby("hour_local")["DepDelay"].mean()
    allh = allh[d.groupby("hour_local").size() >= 1000]
    key("daily_profile_min_hour", int(allh.idxmin()))
    key("daily_profile_min", float(allh.min()))
    key("daily_profile_max_hour", int(allh.idxmax()))
    key("daily_profile_max", float(allh.max()))
    fog_h = p[p["cat"] == 3].groupby("hour_local").size()
    key("fog_peak_hour_local", int(fog_h.idxmax()))
    key("fog_share_5_10_local", float(fog_h.loc[5:10].sum() / fog_h.sum()))

    # --- týždenný profil ---
    dw = d.groupby(["dow", "Origin"])["DepDelay"].mean().unstack()[NODES]
    save_tab(dw.round(2).reset_index(), "A2_tyzdenny_profil")
    w_all = d.groupby("dow")["DepDelay"].mean()
    key("weekly_profile", w_all.round(2).rename(index=dict(enumerate(DOW_SK))).to_dict())

    # --- ročný profil (mesiace) ---
    dm = d.groupby(["month", "Origin"])["DepDelay"].mean().unstack()[NODES]
    dm["všetky"] = d.groupby("month")["DepDelay"].mean()
    save_tab(dm.round(2).reset_index(), "A2_mesacny_profil")
    key("monthly_mean_all", dm["všetky"].round(2).rename(index=dict(enumerate(MONTHS_SK, 1))).to_dict())
    key("month_min", MONTHS_SK[int(dm["všetky"].idxmin()) - 1])
    key("month_max", MONTHS_SK[int(dm["všetky"].idxmax()) - 1])
    wm = p.groupby(["month", "cat_name"]).size().unstack().fillna(0)[CATS[1:]]
    save_tab(wm.reset_index(), "A2_mesacny_profil_pocasie")

    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
    ax = axes[0, 0]
    for n in NODES:
        ax.plot(dh.index, dh[n], color=AP_COLORS[n], label=n, marker=".")
    ax.set(title="Denný profil DepDelay", xlabel="miestna hodina odletu", ylabel="min", xticks=range(0, 24, 3))
    ax.legend(ncol=2)
    ax = axes[0, 1]
    for n in NODES:
        ax.plot(dw.index, dw[n], color=AP_COLORS[n], label=n, marker="o")
    ax.set(title="Týždenný profil DepDelay", ylabel="min", xticks=range(7))
    ax.set_xticklabels(DOW_SK)
    ax = axes[0, 2]
    for n in NODES:
        ax.plot(dm.index, dm[n], color=AP_COLORS[n], label=n, marker="o")
    ax.plot(dm.index, dm["všetky"], color="k", lw=2.5, label="všetky")
    ax.set(title="Ročný profil DepDelay (2018)", ylabel="min", xticks=range(1, 13))
    ax.set_xticklabels(MONTHS_SK, rotation=45)
    ax.legend(ncol=2)
    ax = axes[1, 0]
    ax.stackplot(wx.index, *[wx[c] for c in wx.columns], colors=[CAT_COLORS[c] for c in wx.columns], labels=wx.columns)
    ax.set(title="Denný profil počasia (podiel hodín)", xlabel="miestna hodina", ylabel="% hodín", xticks=range(0, 24, 3))
    ax.legend(loc="upper right")
    ax = axes[1, 1]
    cn = d.groupby(["hour_local", "Origin"]).size().unstack()[NODES] / 365
    for n in NODES:
        ax.plot(cn.index, cn[n], color=AP_COLORS[n], label=n)
    ax.set(title="Počet odletov za hodinu (priemerný deň)", xlabel="miestna hodina", ylabel="odletov / h", xticks=range(0, 24, 3))
    ax = axes[1, 2]
    bottom = np.zeros(12)
    for c in CATS[1:]:
        ax.bar(wm.index, wm[c], bottom=bottom, color=CAT_COLORS[c], label=c)
        bottom += wm[c].to_numpy()
    ax.set(title="Hodiny s počasím po mesiacoch (6 letísk, 2018)", ylabel="letisko-hodiny", xticks=range(1, 13))
    ax.set_xticklabels(MONTHS_SK, rotation=45)
    ax.legend()
    save_fig(fig, "A2_profily")


def daily_series() -> None:
    p = panel()
    d = departures()
    dd = d.groupby([d["dep_local"].dt.date, "Origin"])["DepDelay"].mean().unstack()[NODES]
    dd.index = pd.to_datetime(dd.index)
    wd = p.assign(date=p["time_local"].dt.date).groupby(["date", "airport"])["cat"].apply(lambda s: (s > 0).sum()).unstack()[NODES]
    wd.index = pd.to_datetime(wd.index)
    save_tab(dd.round(2).reset_index(names="dátum"), "A2_denne_meskanie")
    fig, axes = plt.subplots(len(NODES), 1, figsize=(15, 12), sharex=True)
    for ax, n in zip(axes, NODES):
        ax.plot(dd.index, dd[n], color=AP_COLORS[n], lw=1)
        ax.plot(dd.index, dd[n].rolling(7, center=True).mean(), color="k", lw=1.2)
        ax2 = ax.twinx()
        ax2.bar(wd.index, wd[n], color="#6baed6", alpha=0.35, width=1)
        ax2.set_ylim(0, 72)
        ax2.set_ylabel("h počasia", color="#3182bd")
        ax2.grid(False)
        ax.set_ylabel(f"{n}\nmin")
    axes[0].set_title("Denný priemer DepDelay (farebne), 7-dňový kĺzavý priemer (čierne) a hodiny s počasím (modré stĺpce)")
    save_fig(fig, "A2_denne_rady")
    top = dd.stack().sort_values(ascending=False).head(10).rename("DepDelay").reset_index()
    top.columns = ["dátum", "letisko", "priemer DepDelay"]
    top["hodín počasia"] = [wd.loc[r["dátum"], r["letisko"]] for _, r in top.iterrows()]
    save_tab(top, "A2_najhorsie_dni")


def acf_analysis() -> None:
    p = panel()
    lags = 72
    acf_tab, pacf_tab, adf_rows = {}, {}, []
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    for n in NODES:
        s = p.loc[p["airport"] == n, "y"].to_numpy()
        a = acf(s, nlags=lags, missing="conservative")
        acf_tab[n] = a
        # PACF potrebuje súvislú radu: neplatné hodiny nahradíme sezónnym priemerom (hodina × deň)
        sub = p[p["airport"] == n]
        fill = sub.groupby(["hour_local", "dow_local"])["y"].transform("mean").fillna(sub["y"].mean())
        s_f = sub["y"].fillna(fill).to_numpy()
        pa = pacf(s_f, nlags=48, method="ywm")
        pacf_tab[n] = np.r_[pa, [np.nan] * (lags - 48)]
        axes[0, 0].plot(range(lags + 1), a, color=AP_COLORS[n], label=n)
        axes[0, 1].plot(range(49), pa, color=AP_COLORS[n], label=n, marker=".", ms=3)
        # denná rada: ADF + KPSS
        dser = sub.groupby(sub["time_local"].dt.date)["y"].mean().dropna()
        adf = adfuller(dser, autolag="AIC")
        kp = kpss(dser, regression="c", nlags="auto")
        adf_rows.append({"letisko": n, "ADF stat": adf[0], "ADF p": adf[1], "KPSS stat": kp[0], "KPSS p": kp[1],
                         "ACF lag 1 h": a[1], "ACF lag 24 h": a[24], "ACF lag 168 h": np.nan})
    conf = 1.96 / np.sqrt(len(s))
    for ax in axes[0]:
        ax.axhspan(-conf, conf, color="grey", alpha=0.2)
        ax.axhline(0, color="k", lw=0.8)
    axes[0, 0].set(title="ACF hodinového meškania y (neplatné hodiny vynechané)", xlabel="posun (h)", xticks=range(0, 73, 12))
    axes[0, 0].legend(ncol=3)
    axes[0, 1].set(title="PACF hodinového meškania y", xlabel="posun (h)", xticks=range(0, 49, 6))

    # anomália (bez denného/týždenného profilu) – koľko pamäte zostane po odstránení sezónnosti
    anom_acf = {}
    for n in NODES:
        sub = p[p["airport"] == n]
        an = sub["y"] - sub.groupby(["hour_local", "dow_local"])["y"].transform("mean")
        anom_acf[n] = acf(an.to_numpy(), nlags=lags, missing="conservative")
        axes[1, 0].plot(range(lags + 1), anom_acf[n], color=AP_COLORS[n], label=n)
    axes[1, 0].axhline(0, color="k", lw=0.8)
    axes[1, 0].set(title="ACF anomálie y (po odčítaní profilu hodina × deň)", xlabel="posun (h)", xticks=range(0, 73, 12))

    # ACF dennej rady (týždenná periodicita)
    d_acf = {}
    for n in NODES:
        sub = p[p["airport"] == n]
        dser = sub.groupby(sub["time_local"].dt.date)["y"].mean()
        d_acf[n] = acf(dser.to_numpy(), nlags=28, missing="conservative")
        axes[1, 1].plot(range(29), d_acf[n], color=AP_COLORS[n], marker="o", ms=3, label=n)
    axes[1, 1].axhspan(-1.96 / np.sqrt(365), 1.96 / np.sqrt(365), color="grey", alpha=0.2)
    axes[1, 1].set(title="ACF denného priemeru y", xlabel="posun (dni)", xticks=range(0, 29, 7))
    save_fig(fig, "A2_acf_pacf")

    for r in adf_rows:
        r["ACF lag 168 h"] = acf_tab[r["letisko"]][min(168, lags)] if lags >= 168 else np.nan
    adf_df = pd.DataFrame(adf_rows).drop(columns="ACF lag 168 h")
    adf_df["ACF anomálie lag 1 h"] = [anom_acf[n][1] for n in NODES]
    adf_df["ACF anomálie lag 6 h"] = [anom_acf[n][6] for n in NODES]
    adf_df["ACF denná lag 1 d"] = [d_acf[n][1] for n in NODES]
    adf_df["ACF denná lag 7 d"] = [d_acf[n][7] for n in NODES]
    save_tab(adf_df, "A2_acf_stacionarita")
    save_tab(pd.DataFrame(acf_tab).rename_axis("lag").reset_index(), "A2_acf_hodinova")
    save_tab(pd.DataFrame(pacf_tab).rename_axis("lag").reset_index(), "A2_pacf_hodinova")
    key("acf_lag1_mean", float(np.mean([acf_tab[n][1] for n in NODES])))
    key("acf_lag24_mean", float(np.mean([acf_tab[n][24] for n in NODES])))
    key("anom_acf_lag1_mean", float(np.mean([anom_acf[n][1] for n in NODES])))
    key("anom_acf_lag6_mean", float(np.mean([anom_acf[n][6] for n in NODES])))
    key("pacf_lag1_mean", float(np.mean([pacf_tab[n][1] for n in NODES])))
    key("pacf_lag2_mean", float(np.mean([pacf_tab[n][2] for n in NODES])))
    key("adf_p_max", float(adf_df["ADF p"].max()))
    key("kpss_p_min", float(adf_df["KPSS p"].min()))


def run() -> None:
    profiles()
    daily_series()
    acf_analysis()
