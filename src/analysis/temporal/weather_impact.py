"""A3 – Vplyv počasia v čase: meškanie podľa kategórie × letiska × dennej doby × sezóny,
event study (nábeh a doznievanie meškania okolo udalosti) a korelácia s posunom."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.temporal.common import (AP_COLORS, CATS, CAT_COLORS, NODES, key, mean_ci, panel, plt, save_fig,
                                          save_tab, seasonal_anomaly)

TOD_BINS = [(5, 9, "ráno 5–9"), (10, 13, "dopoludnie 10–13"), (14, 17, "popoludnie 14–17"), (18, 23, "večer 18–23")]
SEASONS = ["zima", "jar", "leto", "jeseň"]


def _tod(h):
    for a, b, name in TOD_BINS:
        if a <= h <= b:
            return name
    return "noc 0–4"


def by_category() -> None:
    p = panel()
    v = p[p["valid"]].copy()
    v["anom"] = seasonal_anomaly(p)[p["valid"]]
    rows = []
    for c in CATS:
        g = v[v["cat_name"] == c]
        m, lo, hi = mean_ci(g["y"].to_numpy())
        am, alo, ahi = mean_ci(g["anom"].to_numpy())
        rows.append({"kategória": c, "hodín": len(g), "priemer y": m, "95% CI dolná": lo, "95% CI horná": hi,
                     "medián y": g["y"].median(), "anomália (min)": am, "anom CI dolná": alo, "anom CI horná": ahi,
                     "podiel > 15 min (hod.)": g["dep_del15_share"].mean(), "zrušené": g["cancel_share"].mean()})
    cat = save_tab(pd.DataFrame(rows), "A3_meskanie_podla_kategorie")
    key("delay_by_category", cat.set_index("kategória")[["hodín", "priemer y", "medián y", "anomália (min)"]].round(2).to_dict("index"))

    t = v.pivot_table(index="airport", columns="cat_name", values="y", aggfunc="mean").reindex(NODES)[CATS]
    n = v.pivot_table(index="airport", columns="cat_name", values="y", aggfunc="size").reindex(NODES)[CATS]
    a = v.pivot_table(index="airport", columns="cat_name", values="anom", aggfunc="mean").reindex(NODES)[CATS]
    save_tab(t.round(2).join(n.add_suffix(" n")).reset_index(), "A3_kategoria_x_letisko")
    save_tab(a.round(2).join(n.add_suffix(" n")).reset_index(), "A3_kategoria_x_letisko_anomalia")
    key("rain_heavy_sfo_vs_clear", [float(t.loc["SFO", "Dážď stredný/silný"]), float(t.loc["SFO", "Jasno"])])
    key("fog_sea_vs_clear", [float(t.loc["SEA", "Hmla"]), float(t.loc["SEA", "Jasno"])])
    key("fog_lax_vs_clear", [float(t.loc["LAX", "Hmla"]), float(t.loc["LAX", "Jasno"])])
    key("rain_any_sfo_vs_clear", [float(v.loc[(v.airport == "SFO") & v.cat.isin([1, 2]), "y"].mean()),
                                  float(t.loc["SFO", "Jasno"])])

    # kategória × denná doba × sezóna
    v["tod"] = v["hour_local"].map(_tod)
    ts = v.pivot_table(index=["season", "tod"], columns="cat_name", values="anom", aggfunc="mean")[CATS]
    tn = v.pivot_table(index=["season", "tod"], columns="cat_name", values="anom", aggfunc="size")[CATS]
    save_tab(ts.round(2).join(tn.add_suffix(" n")).reset_index(), "A3_kategoria_x_doba_x_sezona")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1, 1.2, 1.5]})
    ax = axes[0]
    ax.bar(cat["kategória"], cat["priemer y"], yerr=[cat["priemer y"] - cat["95% CI dolná"], cat["95% CI horná"] - cat["priemer y"]],
           color=[CAT_COLORS[c] for c in cat["kategória"]], capsize=4)
    for i, r in cat.iterrows():
        ax.text(i, 0.5, f"n={r['hodín']}", ha="center", color="white", fontsize=8, rotation=90, va="bottom")
    ax.set(title="Priemerné meškanie podľa počasia", ylabel="min (95 % CI)")
    ax.tick_params(axis="x", rotation=30)
    ax = axes[1]
    im = ax.imshow(a.to_numpy(dtype=float), cmap="RdBu_r", vmin=-15, vmax=15, aspect="auto")
    for i in range(len(NODES)):
        for j in range(len(CATS)):
            val, cnt = a.iloc[i, j], n.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}\n({int(cnt)})", ha="center", va="center", fontsize=7.5)
    ax.set_xticks(range(len(CATS)), CATS, rotation=30)
    ax.set_yticks(range(len(NODES)), NODES)
    ax.grid(False)
    ax.set_title("Anomália (min): letisko × počasie (n h)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[2]
    tod_order = [b[2] for b in TOD_BINS]
    sub = ts.reindex([(s, t_) for s in SEASONS for t_ in tod_order])
    im = ax.imshow(sub[CATS[1:]].to_numpy(dtype=float).T, cmap="RdBu_r", vmin=-15, vmax=15, aspect="auto")
    ax.set_yticks(range(4), [c.replace(" stredný/silný", " str./silný") for c in CATS[1:]])
    ax.set_xticks(range(len(sub)), [f"{s[:3]}·{t_.split()[0]}" for s, t_ in sub.index], rotation=60, fontsize=8)
    ax.grid(False)
    ax.set_title("Anomália (min): počasie × sezóna × doba")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_fig(fig, "A3_pocasie_vs_meskanie")


def _episodes(active: np.ndarray, cat: np.ndarray, merge_gap: int = 1):
    """Súvislé epizódy aktívneho počasia (medzery <= merge_gap h sa spoja): (start, end, max_cat)."""
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return []
    eps, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap + 1:
            eps.append((s, prev))
            s = i
        prev = i
    eps.append((s, prev))
    return [(a, b, int(cat[a:b + 1].max())) for a, b in eps]


def event_study(pre: int = 6, post: int = 12) -> None:
    p = panel().copy()
    p["anom"] = seasonal_anomaly(p)
    rows = []
    for n in NODES:
        sub = p[p["airport"] == n].reset_index(drop=True)
        cat = sub["cat"].to_numpy()
        anom = sub["anom"].to_numpy()
        eps = _episodes(cat > 0, cat)
        for k, (a, b, mc) in enumerate(eps):
            nxt = eps[k + 1][0] if k + 1 < len(eps) else len(sub) + 999
            prv = eps[k - 1][1] if k > 0 else -999
            for rel in range(-pre, post + 1):
                # zarovnanie na koniec epizódy; po konci iba čisté hodiny (bez ďalšej udalosti)
                t = b + rel
                if 0 <= t < len(sub) and (rel <= 0 or t < nxt) and t > prv:
                    rows.append({"airport": n, "cat": mc, "dur": b - a + 1, "align": "koniec", "rel": rel, "anom": anom[t]})
                t = a + rel
                if 0 <= t < len(sub) and (rel >= 0 or t > prv) and t <= b + post and t < nxt:
                    rows.append({"airport": n, "cat": mc, "dur": b - a + 1, "align": "začiatok", "rel": rel, "anom": anom[t]})
    es = pd.DataFrame(rows).dropna(subset=["anom"])
    es["cat_name"] = es["cat"].map(dict(enumerate(CATS)))
    out = []
    for (al, c, r), g in es.groupby(["align", "cat_name", "rel"]):
        m, lo, hi = mean_ci(g["anom"].to_numpy(), reps=300)
        out.append({"zarovnanie": al, "kategória": c, "posun_h": r, "n": len(g), "anomália": m, "CI dolná": lo, "CI horná": hi})
    tab = save_tab(pd.DataFrame(out), "A3_event_study")

    # doznievanie: prvá hodina po konci, keď 95 % CI obsahuje 0 (a ostane tak 2 h)
    decay = {}
    for c in CATS[1:]:
        g = tab[(tab["zarovnanie"] == "koniec") & (tab["kategória"] == c) & (tab["posun_h"] >= 0)].sort_values("posun_h")
        ok = ((g["CI dolná"] <= 0) & (g["CI horná"] >= 0)).to_numpy()
        h = next((int(g["posun_h"].iloc[i]) for i in range(len(ok) - 1) if ok[i] and ok[i + 1]), None)
        decay[c] = {"anomália pri konci": float(g["anomália"].iloc[0]) if len(g) else None,
                    "hodín do odznenia": h, "epizód": int(es[(es["cat_name"] == c) & (es["align"] == "koniec") & (es["rel"] == 0)].shape[0])}
    key("event_decay", decay)
    n_eps = es[(es["align"] == "koniec") & (es["rel"] == 0)].groupby("cat_name").size()
    key("n_episodes_valid_end", n_eps.to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
    for ax, al in zip(axes, ["začiatok", "koniec"]):
        for c in CATS[1:]:
            g = tab[(tab["zarovnanie"] == al) & (tab["kategória"] == c)].sort_values("posun_h")
            if g["n"].median() < 15:
                continue
            ax.plot(g["posun_h"], g["anomália"], color=CAT_COLORS[c], marker="o", ms=3, label=f"{c}")
            ax.fill_between(g["posun_h"], g["CI dolná"], g["CI horná"], color=CAT_COLORS[c], alpha=0.15)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.set(title=f"Anomália meškania okolo {al.replace('začiatok', 'začiatku').replace('koniec', 'konca')} epizódy počasia",
               xlabel=f"hodiny od {al.replace('začiatok', 'začiatku').replace('koniec', 'konca')}", ylabel="min (95 % CI)")
    axes[1].legend()
    save_fig(fig, "A3_event_study")


def lagged_corr(max_lag: int = 12) -> None:
    p = panel().copy()
    p["anom"] = seasonal_anomaly(p)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4))
    for n in NODES:
        sub = p[p["airport"] == n].reset_index(drop=True)
        w = (sub["cat"] > 0).astype(float)
        r = [w.corr(sub["anom"].shift(-k)) for k in range(-6, max_lag + 1)]
        rows.append(pd.Series(r, index=range(-6, max_lag + 1), name=n))
        ax.plot(range(-6, max_lag + 1), r, color=AP_COLORS[n], marker=".", label=n)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set(title="Korelácia počasie(t) × anomália meškania(t+k) na tom istom letisku", xlabel="k (h)", ylabel="Pearson r")
    ax.legend(ncol=3)
    save_fig(fig, "A3_korelacia_s_posunom")
    df = pd.DataFrame(rows).T.rename_axis("posun_h").reset_index()
    save_tab(df, "A3_korelacia_s_posunom")
    key("lagcorr_k0", {n: round(float(df.loc[df.posun_h == 0, n].iloc[0]), 3) for n in NODES})
    key("lagcorr_k3", {n: round(float(df.loc[df.posun_h == 3, n].iloc[0]), 3) for n in NODES})


def run() -> None:
    by_category()
    event_study()
    lagged_corr()
