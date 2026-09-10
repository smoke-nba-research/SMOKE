"""Phase 2.1-2.3 — error bars, shrinkage, and within-season reliability for SMOKE.

All three answer one question a team's statistician asks first: *how much do we trust a
single player's number?* Built on the scored-shots artifact from
``src.models.build_model_outputs`` (2014-15, cross-fitted expected probabilities), so
every value here is consistent with the leaderboard, the figures, and the dashboard.

2.1 Error bars.  Bootstrap each player's shots 2,000x → 95% interval on makes-above-
    expected. Any player whose interval crosses zero is flagged "not distinguishable from
    expectation".

2.2 Shrinkage.  Empirical-Bayes (normal-normal) shrink each player's per-shot SMOKE rate
    toward the league mean, weighted by how precisely it's measured (the bootstrap SE
    from 2.1). The between-player variance is the DerSimonian-Laird estimator (inverse-
    variance weighted, from the Q statistic). Publish both raw and shrunk; lead with
    shrunk.

2.3 Reliability.  Two split-half designs, both Spearman-Brown corrected: (a) alternate
    shots in chronological order, and (b) first half of the season versus second half
    by date. The chronological interleave shares every game between halves, so it is the
    more optimistic design; the date split is the like-for-like comparison with the
    across-season test in 2.4. Both are reported.

Run:
    .venv/Scripts/python.exe -m src.validate.reliability
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.features.kaggle_shot_quality import (
    DATE_COLUMN,
    PLAYER_ID_COLUMN,
    PLAYER_NAME_COLUMN,
    TARGET_COLUMN,
)
from src.models.build_model_outputs import PLAYER_MIN_SHOTS, load_scored_shots
from src.pulls._paths import REPO_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = REPO_ROOT / "data" / "v2" / "validation"
MIN_SHOTS = PLAYER_MIN_SHOTS
N_BOOT = 2000
RNG = np.random.default_rng(42)
PLAYER_KEYS = [PLAYER_NAME_COLUMN, PLAYER_ID_COLUMN]


def score_all_shots() -> pd.DataFrame:
    """Every 2014-15 shot with its out-of-fold expected-make probability."""
    scored = load_scored_shots()
    out = scored[PLAYER_KEYS + [DATE_COLUMN, "SHOT_NUMBER", TARGET_COLUMN, "expected"]].copy()
    out["residual"] = out[TARGET_COLUMN] - out["expected"]  # per-shot SMOKE
    return out


def bootstrap_players(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-player point estimate + bootstrap 95% CI and SE.

    Resamples each player's shots with replacement N_BOOT times. Reports both the total
    (makes above expected) and the rate (FG% above expected), plus the SE of the rate
    (feeds shrinkage in 2.2).
    """
    rows = []
    for (player, player_id), grp in scored.groupby(PLAYER_KEYS):
        resid = grp["residual"].to_numpy()
        n = len(resid)
        if n < MIN_SHOTS:
            continue
        idx = RNG.integers(0, n, size=(N_BOOT, n))
        boot_rate = resid[idx].mean(axis=1)     # FG% above expected per replicate
        boot_total = boot_rate * n              # makes above expected per replicate
        lo, hi = np.percentile(boot_total, [2.5, 97.5])
        rows.append(
            {
                "player": player,
                PLAYER_ID_COLUMN: int(player_id),
                "shots": n,
                "smoke_total": float(resid.sum()),
                "smoke_total_lo": float(lo),
                "smoke_total_hi": float(hi),
                "smoke_rate": float(resid.mean()),
                "smoke_rate_se": float(boot_rate.std(ddof=1)),
                "distinguishable": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(rows)


def dersimonian_laird(x: np.ndarray, se: np.ndarray) -> tuple[float, float]:
    """Random-effects mean and between-player variance (DerSimonian and Laird, 1986).

    tau² = max(0, (Q − (k − 1)) / (Σw − Σw²/Σw)) with w = 1/se², Q the weighted sum of
    squares about the fixed-effect mean; mu is then the inverse-variance-weighted mean
    under the random-effects weights 1/(se² + tau²).
    """
    w = 1.0 / se**2
    fixed_mean = float(np.sum(w * x) / np.sum(w))
    q = float(np.sum(w * (x - fixed_mean) ** 2))
    k = len(x)
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / c)
    w_star = 1.0 / (se**2 + tau2)
    mu = float(np.sum(w_star * x) / np.sum(w_star))
    return mu, tau2


def empirical_bayes_shrink(players: pd.DataFrame) -> pd.DataFrame:
    """Normal-normal empirical-Bayes shrinkage of the SMOKE rate toward the league mean.

    Model: observed rate x_i = θ_i + ε_i, ε_i ~ N(0, se_i²); true skill θ_i ~ N(μ, τ²).
    Shrunk estimate: θ̂_i = μ + B_i·(x_i − μ),  B_i = τ²/(τ²+se_i²)  (reliability weight).
    High-volume players (small se) barely move; low-volume ones shrink hard toward μ.
    """
    x = players["smoke_rate"].to_numpy()
    se = players["smoke_rate_se"].to_numpy()
    mu, tau2 = dersimonian_laird(x, se)
    B = tau2 / (tau2 + se**2)
    out = players.copy()
    out["reliability"] = B                        # 0 = all sampling variation, 1 = all repeatable skill
    out["smoke_rate_shrunk"] = mu + B * (x - mu)
    out["smoke_total_shrunk"] = out["smoke_rate_shrunk"] * out["shots"]
    out.attrs["mu"] = mu
    out.attrs["tau"] = float(np.sqrt(tau2))
    return out


def split_half_reliability(scored: pd.DataFrame) -> pd.DataFrame:
    """Two split-half designs per player, correlated across players, Spearman-Brown corrected."""
    ordered = scored.sort_values(PLAYER_KEYS + [DATE_COLUMN, "SHOT_NUMBER"])
    rows = []
    for _, grp in ordered.groupby(PLAYER_KEYS):
        g = grp.reset_index(drop=True)
        if len(g) < MIN_SHOTS:
            continue
        mid_date = g[DATE_COLUMN].median()
        rows.append(
            {
                "alt_a": g.iloc[0::2]["residual"].mean(),
                "alt_b": g.iloc[1::2]["residual"].mean(),
                "date_a": g[g[DATE_COLUMN] <= mid_date]["residual"].mean(),
                "date_b": g[g[DATE_COLUMN] > mid_date]["residual"].mean(),
            }
        )
    h = pd.DataFrame(rows)
    out = []
    for design, a, b in [("alternating shots", "alt_a", "alt_b"), ("first vs second half by date", "date_a", "date_b")]:
        r = float(stats.pearsonr(h[a], h[b]).statistic)
        out.append({"design": design, "n": len(h), "half_r": round(r, 3), "spearman_brown": round(2 * r / (1 + r), 3)})
    return pd.DataFrame(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("PHASE 2.1-2.3 — error bars, shrinkage, within-season reliability (2014-15)")
    print("=" * 76)

    scored = score_all_shots()
    print(f"  scored {len(scored):,} shots (out-of-fold expected probabilities)")

    players = bootstrap_players(scored)
    print(f"  players ≥ {MIN_SHOTS} shots: {len(players)}  |  bootstrap replicates: {N_BOOT}")

    # 2.1 — how many survive the error bars?
    disting = players["distinguishable"].sum()
    half_width = (players["smoke_total_hi"] - players["smoke_total_lo"]) / 2
    print(f"\n  [2.1] distinguishable from expectation (95% CI excludes 0): "
          f"{disting}/{len(players)} ({disting / len(players):.0%})")
    print(f"        → {len(players) - disting} players are NOT statistically separable from average.")
    print(f"        shots per qualified player: mean {players['shots'].mean():.0f}, median {players['shots'].median():.0f}; "
          f"interval half-width: mean {half_width.mean():.1f} makes")

    # 2.2 — shrinkage
    players = empirical_bayes_shrink(players)
    print(f"\n  [2.2] empirical-Bayes shrinkage (DerSimonian-Laird): league mean μ = {players.attrs['mu']:+.4f} "
          f"FG% above exp, between-player SD τ = {players.attrs['tau']:.4f}")
    print(f"        reliability weight ranges {players['reliability'].min():.2f}–{players['reliability'].max():.2f} "
          f"(median {players['reliability'].median():.2f})")

    # 2.3 — within-season reliability
    rel = split_half_reliability(scored)
    print("\n  [2.3] split-half reliability:")
    for r in rel.itertuples():
        print(f"        {r.design:30s} r = {r.half_r:.3f}, Spearman-Brown = {r.spearman_brown:.3f}  (n={r.n})")

    # marquee leaderboard, shrunk, with error bars
    board = players.sort_values("smoke_total_shrunk", ascending=False).head(15)
    print("\n  TOP 15 by shrunk SMOKE (makes above expected)")
    print("  " + "-" * 72)
    print(f"  {'player':22s} {'shots':>5} {'raw':>7} {'95% CI':>16} {'shrunk':>7} {'rel':>5}")
    print("  " + "-" * 72)
    for _, r in board.iterrows():
        ci = f"[{r['smoke_total_lo']:+.0f}, {r['smoke_total_hi']:+.0f}]"
        flag = "" if r["distinguishable"] else "  (~avg)"
        print(f"  {r['player'][:22]:22s} {r['shots']:>5.0f} {r['smoke_total']:>+7.1f} "
              f"{ci:>16} {r['smoke_total_shrunk']:>+7.1f} {r['reliability']:>5.2f}{flag}")
    print("  " + "-" * 72)
    print("  'raw' = makes above expected; 'shrunk' = empirical-Bayes; 'rel' = reliability weight (0-1)")

    full = players.sort_values("smoke_total_shrunk", ascending=False)
    full.to_csv(OUT_DIR / "player_smoke_with_error_bars.csv", index=False)
    rel.to_csv(OUT_DIR / "split_half_reliability.csv", index=False)
    print(f"\n  wrote {OUT_DIR / 'player_smoke_with_error_bars.csv'}  ({len(full)} players)")
    print(f"  wrote {OUT_DIR / 'split_half_reliability.csv'}")


if __name__ == "__main__":
    main()
