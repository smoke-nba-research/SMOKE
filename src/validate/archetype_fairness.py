"""Phase 2.8 — archetype fairness: does SMOKE systematically punish any player type?

The ethics section's core worry (from the capstone): a shot-quality metric might
structurally bury interior big men — their shots are easy layups/dunks the model *expects*
them to make, so "beating expectation" is nearly impossible, and they'd look bad through no
fault of their own. If true, that's a fairness problem a team must know about before using
SMOKE to judge a center.

What this test can and cannot show
----------------------------------
The archetypes are K-means clusters on player averages of the model's own inputs (shot
distance, defender distance, touch time, dribbles, three-point rate). A well-calibrated
model has near-zero mean residual within strata of its own features, so a near-zero mean
SMOKE per archetype is partly a *calibration* property, not an independent discovery; a
mean that does differ from zero is evidence of miscalibration in that region. The
fairness question proper is about *precision*: whether one role's players can be measured
at all. That is the separability rate by archetype, reported last, and it is the number
that answers the capstone's worry.

Two versions of the per-archetype mean are reported: the inverse-variance-weighted mean
of the raw rates (each player weighted by the precision of his own bootstrap estimate,
the appropriate test) and the plain mean of the shrunk rates (what the leaderboard shows).

Run:
    .venv/Scripts/python.exe -m src.validate.archetype_fairness
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.features.archetypes import ARCHETYPE_ORDER, build_player_style, cluster_archetypes
from src.features.kaggle_shot_quality import PLAYER_ID_COLUMN
from src.models.build_model_outputs import PLAYER_MIN_SHOTS, load_scored_shots
from src.pulls._paths import REPO_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAL_DIR = REPO_ROOT / "data" / "v2" / "validation"
SMOKE_CSV = VAL_DIR / "player_smoke_with_error_bars.csv"
MIN_SHOTS = PLAYER_MIN_SHOTS


def weighted_mean_test(rate: np.ndarray, se: np.ndarray) -> tuple[float, float, float]:
    """Inverse-variance-weighted mean of raw rates, its SE, and a two-sided p versus zero."""
    w = 1.0 / se**2
    mean = float(np.sum(w * rate) / np.sum(w))
    se_mean = float(np.sqrt(1.0 / np.sum(w)))
    p = float(2 * stats.norm.sf(abs(mean / se_mean)))
    return mean, se_mean, p


def main() -> None:
    shots = load_scored_shots()
    prof = cluster_archetypes(build_player_style(shots, min_shots=MIN_SHOTS))

    smoke = pd.read_csv(SMOKE_CSV)
    df = prof.merge(
        smoke[[PLAYER_ID_COLUMN, "smoke_rate", "smoke_rate_se", "smoke_rate_shrunk", "distinguishable"]],
        on=PLAYER_ID_COLUMN, how="inner",
    )
    assert len(df) == len(prof) == len(smoke), "archetype and SMOKE tables must cover the same players"

    print("=" * 80)
    print("PHASE 2.8 — archetype fairness: is any role systematically penalized? (2014-15)")
    print("=" * 80)
    print(f"  players clustered: {len(df)}  |  league mean shrunk SMOKE: {df['smoke_rate_shrunk'].mean():+.4f} (≈0 by construction)")

    print("\n  MEAN SMOKE BY ARCHETYPE")
    print("  " + "-" * 96)
    print(f"  {'archetype':20s} {'n':>4} {'weighted raw':>13} {'SE':>7} {'p vs 0':>8}   {'mean shrunk':>12} {'95% CI (shrunk)':>20} {'p':>7}")
    print("  " + "-" * 96)
    rows = []
    for arch in ARCHETYPE_ORDER:
        g = df[df["archetype"] == arch]
        if len(g) < 2:
            continue
        wmean, wse, wp = weighted_mean_test(g["smoke_rate"].to_numpy(), g["smoke_rate_se"].to_numpy())
        sh = g["smoke_rate_shrunk"]
        m, se = sh.mean(), sh.std(ddof=1) / np.sqrt(len(sh))
        ci = stats.t.interval(0.95, len(sh) - 1, loc=m, scale=se)
        p = stats.ttest_1samp(sh, 0.0).pvalue
        flag = "  ← ≠0" if wp < 0.05 else ""
        print(f"  {arch:20s} {len(g):>4} {wmean:>+13.4f} {wse:>7.4f} {wp:>8.3f}   {m:>+12.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}] {p:>7.3f}{flag}")
        rows.append(
            {
                "archetype": arch, "n": len(g),
                "weighted_mean_raw": round(wmean, 4), "weighted_se": round(wse, 4), "p_weighted_vs_0": round(wp, 4),
                "mean_shrunk": round(m, 4), "ci_lo_shrunk": round(ci[0], 4), "ci_hi_shrunk": round(ci[1], 4), "p_shrunk_vs_0": round(p, 4),
                "separable_share": round(float(g["distinguishable"].mean()), 4),
                "mean_shot_dist_ft": round(float(g["avg_shot_dist"].mean()), 1),
            }
        )
    print("  " + "-" * 96)

    groups = [df[df["archetype"] == a]["smoke_rate_shrunk"].to_numpy() for a in ARCHETYPE_ORDER if (df["archetype"] == a).any()]
    kw = stats.kruskal(*groups)
    print(f"\n  Kruskal-Wallis across archetypes (shrunk): H = {kw.statistic:.2f}, p = {kw.pvalue:.3f}")
    print(f"    → archetypes {'DO' if kw.pvalue < 0.05 else 'do NOT'} differ in mean SMOKE.")

    interior = df[df["archetype"] == "Interior Finishers"]
    wmean, wse, wp = weighted_mean_test(interior["smoke_rate"].to_numpy(), interior["smoke_rate_se"].to_numpy())
    print(f"\n  THE KEY CHECK — Interior Finishers (n={len(interior)}): weighted mean SMOKE {wmean:+.4f} (p = {wp:.3f})")
    if wp >= 0.05:
        print("    → NOT significantly different from 0. The metric does not structurally bury big men.")
    else:
        sign = "below" if wmean < 0 else "above"
        print(f"    → significantly {sign} 0 — investigate whether this reflects real performance or bias.")

    print("\n  SEPARABILITY BY ARCHETYPE (the fairness statistic: can this role's players be measured at all?)")
    for r in rows:
        print(f"    {r['archetype']:20s} {r['separable_share']:.0%} distinguishable from average  (mean shot dist {r['mean_shot_dist_ft']} ft)")

    df.sort_values("smoke_rate_shrunk", ascending=False).to_csv(VAL_DIR / "archetype_fairness.csv", index=False)
    means = pd.DataFrame(rows)
    means.attrs = {}
    means.to_csv(VAL_DIR / "archetype_means.csv", index=False)
    pd.DataFrame([{"stat": "kruskal_H", "value": round(float(kw.statistic), 3)}, {"stat": "kruskal_p", "value": round(float(kw.pvalue), 4)}]).to_csv(
        VAL_DIR / "archetype_omnibus.csv", index=False
    )
    print(f"\n  wrote {VAL_DIR / 'archetype_fairness.csv'}  ({len(df)} players)")
    print(f"  wrote {VAL_DIR / 'archetype_means.csv'}")
    print(f"  wrote {VAL_DIR / 'archetype_omnibus.csv'}")


if __name__ == "__main__":
    main()
