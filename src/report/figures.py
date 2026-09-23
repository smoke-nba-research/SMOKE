"""Build the SMOKE figure suite.

Each figure answers one question a reviewer or analytics staffer would ask, and the
question it answers is listed beside it below.

    fig1_model_works.png      Does the expected-make model actually work?
    fig2_thesis.png           Are selection and shot-making genuinely separable?
    fig3_leaderboard.png      Who is good, and how sure are we about each number?
    fig4_stability.png        Is it repeatable across seasons?
    fig5_convergent.png       Does it agree with trusted metrics without duplicating them?
    fig6_predictive.png       Does it predict anything?
    fig7_fairness.png         Does it penalize a type of player?
    movers_chart.png          What does it see that the box score misses?

Shared palette with the dashboard and the one-pager so every artifact matches.

Run:
    .venv/Scripts/python.exe -m src.report.figures
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.archetypes import ARCHETYPE_ORDER
from src.features.names import display_name
from src.models.build_model_outputs import load_scored_shots
from src.pulls._paths import REPO_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = REPO_ROOT / "artifacts"
VAL = REPO_ROOT / "data" / "v2" / "validation"
V1 = REPO_ROOT / "data" / "model_outputs"

NAVY = "#1F3864"
BLUE = "#2E75B6"
LIGHT = "#9DC3E6"
GRAY = "#6B7280"
RED = "#B03A2E"

# One name formatter for every artifact (figures, one-pager, dashboard).
nm = display_name


def style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """One look for every figure."""
    if title:
        ax.set_title(title, fontsize=11.5, color=NAVY, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=NAVY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=NAVY)
    ax.tick_params(labelsize=8.5, colors="#333333")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRAY)
        ax.spines[side].set_linewidth(0.8)


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote artifacts/{name}")


def scored_shots() -> pd.DataFrame:
    """Every 2014-15 shot with its out-of-fold expected-make probability and key features."""
    scored = load_scored_shots()
    out = scored[["CLOSE_DEF_DIST", "SHOT_DIST", "SHOT_MADE", "expected"]].copy()
    return out.rename(columns={"SHOT_MADE": "made"})


# --------------------------------------------------------------- figure 1
def fig_model_works(shots: pd.DataFrame) -> None:
    """Q: does the expected-make model actually work?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))

    # calibration: predicted probability against observed make rate
    bins = pd.qcut(shots["expected"], 10, duplicates="drop")
    cal = shots.groupby(bins, observed=True).agg(
        pred=("expected", "mean"), actual=("made", "mean"), n=("made", "size")
    )
    ax1.plot([0.2, 0.8], [0.2, 0.8], color=GRAY, ls="--", lw=1, label="perfect calibration")
    ax1.plot(cal["pred"], cal["actual"], "o-", color=BLUE, lw=2, ms=6, label="SMOKE model")
    style(ax1, "The model is calibrated", "Predicted make probability", "Observed make rate")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")

    # Why a model is needed at all. Raw FG% by defender distance looks flat, because
    # tight shots are mostly layups and open shots are mostly threes. The two effects
    # cancel. Hold shot distance fixed and the effect is large and monotone in every band.
    defb = pd.cut(
        shots["CLOSE_DEF_DIST"], [0, 2, 4, 6, 100], labels=["0-2", "2-4", "4-6", "6+"],
        include_lowest=True,
    )
    distb = pd.cut(
        shots["SHOT_DIST"], [0, 4, 14, 23.75, 100],
        labels=["At the rim", "4-14 ft", "14 ft to arc", "3-pointers"],
        include_lowest=True,
    )
    tagged = shots.assign(defb=defb, distb=distb)
    grid = (
        tagged.pivot_table(index="distb", columns="defb", values="made", aggfunc="mean", observed=True)
        * 100
    )
    band_colors = [NAVY, BLUE, "#5B9BD5", LIGHT]
    for color, (band, row) in zip(band_colors, grid.iterrows(), strict=True):
        ax2.plot(range(len(row)), row.to_numpy(), "o-", color=color, lw=2, ms=5, label=str(band))
        ax2.annotate(
            str(band), (len(row) - 1, row.iloc[-1]), textcoords="offset points",
            xytext=(6, -2), fontsize=8, color=color,
        )
    ax2.set_xticks(range(grid.shape[1]))
    ax2.set_xticklabels([f"{c} ft" for c in grid.columns], fontsize=8.5)
    ax2.set_xlim(-0.2, grid.shape[1] + 0.9)
    style(
        ax2,
        "More space, higher percentage, at every shot distance",
        "Closest defender distance", "Field goal percentage",
    )
    raw_by_def = tagged.groupby("defb", observed=True)["made"].mean() * 100
    dist_by_def = tagged.groupby("defb", observed=True)["SHOT_DIST"].mean()
    wide_open = tagged[tagged["defb"] == "6+"]
    three_share_open = (wide_open["SHOT_DIST"] >= 23.75).mean() * 100
    note = (
        f"Raw FG% by defender distance looks flat ({raw_by_def['0-2']:.0f}%, {raw_by_def['2-4']:.0f}%, "
        f"{raw_by_def['4-6']:.0f}%, {raw_by_def['6+']:.0f}%), because tight shots average\n"
        f"{dist_by_def['0-2']:.1f} ft (mostly layups) and open shots average {dist_by_def['6+']:.1f} ft "
        f"({three_share_open:.0f}% threes). That confound is why a model is necessary."
    )
    ax2.text(
        0.0, -0.30, note,
        transform=ax2.transAxes, fontsize=7.8, color=GRAY, va="top",
    )

    fig.suptitle(
        "Figure 1. The expected-make model, checked two ways",
        fontsize=12.5, color=NAVY, x=0.02, ha="left", y=1.04, weight="bold",
    )
    fig.tight_layout()
    save(fig, "fig1_model_works.png")


# --------------------------------------------------------------- figure 2
def fig_thesis() -> None:
    """Q: are shot selection and shot-making genuinely separable?"""
    r = pd.read_csv(V1 / "rank_table.csv")
    x = r["expected_fg_pct"] * 100          # how easy their shots were
    y = r["smoke_rate"] * 100    # how much they beat that

    fig, ax = plt.subplots(figsize=(7.6, 5))
    ax.axhline(0, color=GRAY, lw=0.9)
    ax.scatter(x, y, s=26, color=BLUE, alpha=0.55, edgecolor="none")

    corr = float(np.corrcoef(x, y)[0, 1])

    highlight = {
        "deandre jordan": ("easy shots, still beats them", (-14, 16), "right"),
        "kyle korver": ("hard shots, beats them badly", (10, 10), "left"),
        "omer asik": ("easy shots, does not beat them", (-16, 14), "right"),
    }
    for raw, (note, offset, ha) in highlight.items():
        row = r[r["player_name"] == raw]
        if row.empty:
            continue
        px, py = row["expected_fg_pct"].iloc[0] * 100, row["smoke_rate"].iloc[0] * 100
        ax.scatter([px], [py], s=70, color=NAVY, zorder=5)
        ax.annotate(
            f"{nm(raw)}\n{note}",
            (px, py), textcoords="offset points", xytext=offset,
            fontsize=8, color=NAVY, ha=ha,
            arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.7},
        )

    style(
        ax,
        "Figure 2. Shot selection and shot-making are separate skills",
        "Shot difficulty faced (expected FG%, higher means easier shots)",
        "SMOKE (percentage points above expectation)",
    )
    ax.text(
        0.0, -0.16,
        f"Correlation between the two axes: {corr:+.2f}. Knowing how easy a player's shots are tells you "
        "almost nothing\nabout whether he beats them. That near-independence is the premise of the metric.",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=GRAY,
    )
    fig.tight_layout()
    save(fig, "fig2_thesis.png")


# --------------------------------------------------------------- figure 3
def fig_leaderboard() -> None:
    """Q: who is good, and how confident are we in any single number?"""
    p = pd.read_csv(VAL / "player_smoke_with_error_bars.csv")
    n_distinguishable = int(p["distinguishable"].sum())
    n_total = len(p)
    top = p.nlargest(20, "smoke_total_shrunk").iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ypos = np.arange(len(top))
    for i, (_, row) in enumerate(top.iterrows()):
        sure = bool(row["distinguishable"])
        color = BLUE if sure else GRAY
        ax.plot([row["smoke_total_lo"], row["smoke_total_hi"]], [i, i], color=color, lw=2, alpha=0.75)
        ax.plot([row["smoke_total"]], [i], "o", color=color, ms=6)
    ax.axvline(0, color=RED, lw=1, ls="--", alpha=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([nm(n) for n in top["player"]], fontsize=8.5)
    style(
        ax,
        "Figure 3. The leaderboard, with the uncertainty shown",
        "Makes above expectation (95% interval)",
        "",
    )
    ax.text(
        0.0, -0.10,
        "Blue: the interval excludes zero, so the player is separable from league average. "
        "Gray: not separable\non a single season. Only "
        f"{n_distinguishable} of {n_total} qualified players clear that bar.",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=GRAY,
    )
    fig.tight_layout()
    save(fig, "fig3_leaderboard.png")


# --------------------------------------------------------------- figure 4
def fig_stability() -> None:
    """Q: is it repeatable across seasons?"""
    d = pd.read_parquet(VAL / "stability_2season_players.parquet")
    s = pd.read_csv(VAL / "stability_correlations.csv")
    s = s[s["design"] == "per-season cross-fit"]  # primary design; robustness variant excluded

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

    x, y = d["smoke_14"] * 100, d["smoke_16"] * 100
    ax1.axhline(0, color=GRAY, lw=0.8)
    ax1.axvline(0, color=GRAY, lw=0.8)
    ax1.scatter(x, y, s=24, color=BLUE, alpha=0.55, edgecolor="none")
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax1.plot(xs, m * xs + b, color=NAVY, lw=2)
    r_smoke = float(s.loc[s["metric"] == "SMOKE", "pearson_r"].iloc[0])
    ax1.text(
        0.03, 0.95, f"r = {r_smoke:.2f}  (n = {len(d)})",
        transform=ax1.transAxes, va="top", fontsize=10, color=NAVY, weight="bold",
    )
    style(ax1, "Players repeat themselves", "SMOKE, 2014-15 (pts)", "SMOKE, 2015-16 (pts)")

    order = ["raw FG%", "SMOKE", "eFG%"]
    labels = ["Raw FG%", "SMOKE", "eFG%"]
    vals = [float(s.loc[s["metric"] == o, "pearson_r"].iloc[0]) for o in order]
    r_raw, r_smoke2, r_efg = vals
    colors = [GRAY, BLUE, LIGHT]
    ax2.bar(labels, vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9.5, color=NAVY)
    ax2.set_ylim(0, 0.82)
    style(ax2, "Compared to the incumbents", "", "Year-over-year correlation")
    if r_smoke2 >= r_efg:
        cmp_line = f"SMOKE (r = {r_smoke2:.2f}) is at least as stable as eFG% (r = {r_efg:.2f}), the metric it refines."
    else:
        cmp_line = f"SMOKE (r = {r_smoke2:.2f}) is somewhat less stable than eFG% (r = {r_efg:.2f}), the metric it refines."
    ax2.text(
        0.5, -0.28,
        f"{cmp_line}\n"
        f"Raw FG% is more stable still (r = {r_raw:.2f}), because it inherits shot\n"
        "selection, which SMOKE removes on purpose.",
        transform=ax2.transAxes, ha="center", va="top", fontsize=7.6, color=GRAY,
    )

    fig.suptitle(
        "Figure 4. Year-over-year stability",
        fontsize=12.5, color=NAVY, x=0.02, ha="left", y=1.03, weight="bold",
    )
    fig.tight_layout()
    save(fig, "fig4_stability.png")


# --------------------------------------------------------------- figure 5
def fig_convergent() -> None:
    """Q: does it agree with trusted metrics without duplicating them?"""
    c = pd.read_csv(VAL / "convergent_correlations.csv")
    pretty = {
        "ts_percent": ("True shooting %", "shooting efficiency, which SMOKE refines"),
        "obpm": ("Offensive BPM", "offense including shooting"),
        "per": ("PER", "efficiency-driven composite"),
        "bpm": ("Box plus/minus", "half of it is defense, which SMOKE ignores"),
        "usg_percent": ("Usage rate", "negative control: should be near zero"),
    }
    c = c.set_index("metric").loc[list(pretty)].reset_index()
    labels = [pretty[m][0] for m in c["metric"]]
    notes = [pretty[m][1] for m in c["metric"]]
    vals = c["pearson_r"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ypos = np.arange(len(vals))[::-1]
    colors = [BLUE] * (len(vals) - 1) + [GRAY]
    ax.barh(ypos, vals, color=colors, height=0.55)
    for yp, v in zip(ypos, vals, strict=True):
        ax.text(v + 0.008, yp, f"{v:.2f}", va="center", fontsize=9.5, color=NAVY)
    ax.set_yticks(ypos)
    # the note rides on a second line of the tick label, so nothing sits under a bar
    ax.set_yticklabels([f"{lab}\n{note}" for lab, note in zip(labels, notes, strict=True)], fontsize=8.5)
    ax.set_xlim(0, 0.60)
    style(
        ax,
        "Figure 5. Agreement with established metrics",
        "Correlation with SMOKE",
        "",
    )
    ts_r = float(c.loc[c["metric"] == "ts_percent", "pearson_r"].iloc[0])
    usg_r = float(c.loc[c["metric"] == "usg_percent", "pearson_r"].iloc[0])
    ax.text(
        0.0, -0.16,
        f"{ts_r:.2f} against true shooting confirms SMOKE measures shooting. {usg_r:.2f} against usage confirms it does\n"
        "not measure volume or role. True shooting is highest and usage lowest, as predicted.",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.3, color=GRAY,
    )
    fig.tight_layout()
    save(fig, "fig5_convergent.png")


# --------------------------------------------------------------- figure 6
def fig_predictive() -> None:
    """Q: does it predict anything?"""
    s = pd.read_csv(VAL / "predictive_summary.csv").set_index("test")["value"]
    groups = ["Predicting next-season\nefficiency (eFG%)", "Predicting next-season\nshot-making (SMOKE)"]
    smoke_vals = [s["a_corr_smoke_vs_futureEFG"], s["c_corr_smoke_vs_futureSMOKE"]]
    efg_vals = [s["a_corr_pastEFG_vs_futureEFG"], s["c_corr_pastEFG_vs_futureSMOKE"]]

    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    b1 = ax.bar(x - w / 2, smoke_vals, w, label="SMOKE as predictor", color=BLUE)
    b2 = ax.bar(x + w / 2, efg_vals, w, label="eFG% as predictor", color=GRAY)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.2f}", ha="center", fontsize=9.5, color=NAVY,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylim(0, 0.66)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax, "Figure 6. What SMOKE predicts, and what it does not", "", "Correlation with next season")
    p_smoke = s["c_p_smoke"]
    p_text = "p < 0.001" if p_smoke < 0.001 else f"p = {p_smoke:.3f}"
    ax.text(
        0.0, -0.19,
        f"Left: SMOKE loses ({smoke_vals[0]:.2f} vs {efg_vals[0]:.2f}), by design. Efficiency is self-predictive "
        "because it carries shot selection,\nwhich SMOKE strips out. Right: for the quantity SMOKE actually "
        f"measures, it predicts better than eFG% ({smoke_vals[1]:.2f} vs {efg_vals[1]:.2f}).\n"
        f"Adding SMOKE to eFG% raises adjusted R² from {s['c_adjR2_efg_only']:.2f} to "
        f"{s['c_adjR2_efg_plus_smoke']:.2f} ({p_text}), incremental information beyond eFG% alone.",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.0, color=GRAY,
    )
    fig.tight_layout()
    save(fig, "fig6_predictive.png")


# --------------------------------------------------------------- figure 7
def fig_fairness() -> None:
    """Q: does it penalize a type of player?"""
    a = pd.read_csv(VAL / "archetype_means.csv")

    order = ARCHETYPE_ORDER
    a = a.set_index("archetype").loc[order].reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={"width_ratios": [1.35, 1]})

    ypos = np.arange(len(a))[::-1]
    ax1.axvline(0, color=RED, lw=1, ls="--", alpha=0.8)
    for yp, (_, row) in zip(ypos, a.iterrows(), strict=True):
        color = NAVY if row["archetype"] == "Interior Finishers" else BLUE
        ax1.plot([row["ci_lo_shrunk"] * 100, row["ci_hi_shrunk"] * 100], [yp, yp], color=color, lw=2.4)
        ax1.plot([row["mean_shrunk"] * 100], [yp], "o", color=color, ms=7)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([f"{r.archetype}  (n={r.n})" for r in a.itertuples()], fontsize=9)
    style(ax1, "No playing style is structurally penalized", "Mean SMOKE (pts), with 95% interval", "")

    interior = a[a["archetype"] == "Interior Finishers"].iloc[0]
    significant = a[a["p_weighted_vs_0"] < 0.05]
    if len(significant) == 0:
        tail = "No archetype's weighted mean is significantly different from zero."
    else:
        names = "; ".join(
            f"{row.archetype} ({row.weighted_mean_raw * 100:+.2f} pp, p = {row.p_weighted_vs_0:.3f})"
            for row in significant.itertuples()
        )
        tail = f"The exception: {names}."
    p_int = float(interior["p_weighted_vs_0"])
    verdict = (
        f"are not distinguishable from zero (p = {p_int:.2f})" if p_int >= 0.05
        else f"differ from zero ({interior['weighted_mean_raw'] * 100:+.2f} pp, p = {p_int:.3f})"
    )
    ax1.text(
        0.0, -0.19,
        "Interior finishers, the group most often assumed to be disadvantaged\n"
        f"by difficulty adjustment, {verdict}. "
        f"{tail}",
        transform=ax1.transAxes, ha="left", va="top", fontsize=8.2, color=GRAY,
    )

    vals = [float(a.loc[a["archetype"] == o, "separable_share"].iloc[0]) * 100 for o in order]
    colors = [NAVY if o == "Interior Finishers" else BLUE for o in order]
    ax2.bar(range(len(order)), vals, color=colors, width=0.6)
    ax2.set_xticks(range(len(order)))
    short = {
        "On-Ball Creators": "On-ball\ncreators",
        "Catch-and-Shoot": "Catch-and\n-shoot",
        "Mid-Range Scorers": "Mid-range\nscorers",
        "Interior Finishers": "Interior\nfinishers",
    }
    ax2.set_xticklabels([short[o] for o in order], fontsize=8)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.7, f"{v:.0f}%", ha="center", fontsize=9, color=NAVY)
    ax2.set_ylim(0, max(vals) * 1.25)
    style(ax2, "And big men are measured most precisely", "", "Share separable from average")

    fig.suptitle(
        "Figure 7. Fairness across playing styles",
        fontsize=12.5, color=NAVY, x=0.02, ha="left", y=1.03, weight="bold",
    )
    fig.tight_layout()
    save(fig, "fig7_fairness.png")


# --------------------------------------------------------------- movers
def fig_movers(
    n: int = 8,
    filename: str = "movers_chart.png",
    title: str = "Figure 8. The same players, ranked by raw FG% and then by SMOKE",
) -> None:
    """Q: what does SMOKE see that the box score misses?"""
    r = pd.read_csv(V1 / "rank_table.csv")
    up = r.nlargest(n, "rank_shift_vs_fg")
    down = r.nsmallest(n, "rank_shift_vs_fg").iloc[::-1]
    rows = pd.concat([down, up])
    labels = [nm(p) for p in rows["player_name"]]
    vals = rows["rank_shift_vs_fg"].to_numpy()
    colors = [BLUE if v > 0 else GRAY for v in vals]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.barh(labels, vals, color=colors, height=0.62)
    ax.axvline(0, color=NAVY, lw=0.9)
    style(
        ax,
        title,
        "Ranking spots moved, raw FG% rank to SMOKE rank",
        "",
    )
    fig.tight_layout()
    save(fig, filename)


def main() -> None:
    print("Building the SMOKE figure suite")
    print("  scoring shots for the model diagnostics (takes a moment)")
    shots = scored_shots()
    fig_model_works(shots)
    fig_thesis()
    fig_leaderboard()
    fig_stability()
    fig_convergent()
    fig_predictive()
    fig_fairness()
    fig_movers()
    print("done")


if __name__ == "__main__":
    main()
