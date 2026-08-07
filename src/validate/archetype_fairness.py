"""Phase 2.8 — archetype fairness: does SMOKE systematically punish any player type?

The ethics section's core worry (from the capstone): a shot-quality metric might
structurally bury interior big men — their shots are easy layups/dunks the model *expects*
them to make, so "beating expectation" is nearly impossible, and they'd look bad through no
fault of their own. If true, that's a fairness problem a team must know about before using
SMOKE to judge a center.

Test: cluster ALL qualified players (not just the overperformers — that would be circular)
into the capstone's four shot-selection archetypes, then ask whether any archetype's mean
SMOKE is systematically below zero. Because SMOKE is a residual, the league mean is ~0 by
construction; the question is whether it's ~0 *within each role* or whether one role carries
a structural penalty.

Run:
    .venv/Scripts/python.exe -m src.validate.archetype_fairness
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.features.kaggle_shot_quality import (
    PLAYER_NAME_COLUMN,
    TARGET_COLUMN,
    add_derived_features,
    load_kaggle_shot_logs,
)
from src.pulls._paths import REPO_ROOT
from src.validate.convergent import norm_name

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAL_DIR = REPO_ROOT / "data" / "v2" / "validation"
SMOKE_CSV = VAL_DIR / "player_smoke_with_error_bars.csv"
MIN_SHOTS = 150
FEATURES = ["avg_shot_dist", "avg_def_dist", "avg_touch_time", "avg_dribbles", "three_rate"]


def player_profiles() -> pd.DataFrame:
    """Per-player shot-selection profile for all players ≥ MIN_SHOTS (2014-15)."""
    s = add_derived_features(load_kaggle_shot_logs())
    s = s.dropna(subset=["SHOT_RESULT", PLAYER_NAME_COLUMN]).copy()
    s["is_three"] = (s["PTS_TYPE"] == 3).astype(int)
    prof = (
        s.groupby(PLAYER_NAME_COLUMN)
        .agg(
            shots=(TARGET_COLUMN, "size"),
            avg_shot_dist=("SHOT_DIST", "mean"),
            avg_def_dist=("CLOSE_DEF_DIST", "mean"),
            avg_touch_time=("TOUCH_TIME", "mean"),
            avg_dribbles=("DRIBBLES", "mean"),
            three_rate=("is_three", "mean"),
        )
        .reset_index()
    )
    return prof[prof["shots"] >= MIN_SHOTS].reset_index(drop=True)


def label_clusters(prof: pd.DataFrame) -> pd.DataFrame:
    """K-means (k=4) on shot-selection features, then name clusters by their defining trait."""
    X = StandardScaler().fit_transform(prof[FEATURES])
    prof = prof.copy()
    prof["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    cent = prof.groupby("cluster")[FEATURES].mean()

    labels: dict[int, str] = {}
    remaining = set(cent.index)

    def assign(cluster_id: int, name: str) -> None:
        labels[cluster_id] = name
        remaining.discard(cluster_id)

    # interior finishers = shortest average distance
    assign(cent.loc[list(remaining), "avg_shot_dist"].idxmin(), "Interior Finishers")
    # on-ball creators = most dribbles among the rest
    assign(cent.loc[list(remaining), "avg_dribbles"].idxmax(), "On-Ball Creators")
    # catch-and-shoot = highest three-rate among the rest
    assign(cent.loc[list(remaining), "three_rate"].idxmax(), "Catch-and-Shoot")
    # the last one = mid-range scorers
    labels[remaining.pop()] = "Mid-Range Scorers"

    prof["archetype"] = prof["cluster"].map(labels)
    return prof


def main() -> None:
    prof = label_clusters(player_profiles())
    prof["key"] = prof[PLAYER_NAME_COLUMN].map(norm_name)

    smoke = pd.read_csv(SMOKE_CSV)
    smoke["key"] = smoke["player"].map(norm_name)
    col = "smoke_rate_shrunk" if "smoke_rate_shrunk" in smoke.columns else "smoke_rate"
    df = prof.merge(smoke[["key", col, "distinguishable"]], on="key", how="inner").rename(
        columns={col: "smoke"}
    )

    print("=" * 80)
    print("PHASE 2.8 — archetype fairness: is any role systematically penalized? (2014-15)")
    print("=" * 80)
    print(f"  players clustered: {len(df)}  |  league mean SMOKE: {df['smoke'].mean():+.4f} (≈0 by construction)")

    print("\n  MEAN SMOKE BY ARCHETYPE  (the fairness table — want all near 0, none buried)")
    print("  " + "-" * 74)
    print(f"  {'archetype':22s} {'n':>4} {'mean SMOKE':>12} {'95% CI of mean':>20} {'p vs 0':>8}")
    print("  " + "-" * 74)
    order = ["On-Ball Creators", "Catch-and-Shoot", "Mid-Range Scorers", "Interior Finishers"]
    rows = []
    for arch in order:
        g = df[df["archetype"] == arch]["smoke"]
        if len(g) < 2:
            continue
        m = g.mean()
        se = g.std(ddof=1) / np.sqrt(len(g))
        ci = stats.t.interval(0.95, len(g) - 1, loc=m, scale=se)
        p = stats.ttest_1samp(g, 0.0).pvalue
        flag = "  ← ≠0" if p < 0.05 else ""
        print(f"  {arch:22s} {len(g):>4} {m:>+12.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]  {p:>7.3f}{flag}")
        rows.append({"archetype": arch, "n": len(g), "mean_smoke": round(m, 4),
                     "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4), "p_vs_0": round(p, 4)})
    print("  " + "-" * 74)

    # do archetypes differ from each other at all?
    groups = [df[df["archetype"] == a]["smoke"].to_numpy() for a in order if (df["archetype"] == a).any()]
    kw = stats.kruskal(*groups)
    print(f"\n  Kruskal-Wallis across archetypes: H = {kw.statistic:.2f}, p = {kw.pvalue:.3f}")
    print(f"    → archetypes {'DO' if kw.pvalue < 0.05 else 'do NOT'} differ in mean SMOKE.")

    # the specific worry: are interior finishers buried?
    interior = df[df["archetype"] == "Interior Finishers"]["smoke"]
    print(f"\n  THE KEY CHECK — Interior Finishers (n={len(interior)}): mean SMOKE {interior.mean():+.4f}")
    if stats.ttest_1samp(interior, 0.0).pvalue >= 0.05:
        print("    → NOT significantly different from 0. The metric does not structurally bury big men.")
    else:
        sign = "below" if interior.mean() < 0 else "above"
        print(f"    → significantly {sign} 0 — investigate whether this reflects real performance or bias.")

    # how many players of each archetype are even distinguishable from average?
    print("\n  distinguishable-from-average rate by archetype (context from 2.1):")
    for arch in order:
        g = df[df["archetype"] == arch]
        if len(g):
            print(f"    {arch:22s} {g['distinguishable'].mean():.0%} distinguishable  (mean shot dist "
                  f"{prof[prof['archetype'] == arch]['avg_shot_dist'].mean():.1f} ft)")

    df.sort_values("smoke", ascending=False).to_csv(VAL_DIR / "archetype_fairness.csv", index=False)
    pd.DataFrame(rows).to_csv(VAL_DIR / "archetype_means.csv", index=False)
    print(f"\n  wrote {VAL_DIR / 'archetype_fairness.csv'}  ({len(df)} players)")
    print(f"  wrote {VAL_DIR / 'archetype_means.csv'}")


if __name__ == "__main__":
    main()
