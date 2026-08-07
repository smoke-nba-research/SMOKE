"""Phase 2.1-2.3 — error bars, shrinkage, and within-season reliability for SMOKE.

All three answer one question a team's statistician asks first: *how much do we trust a
single player's number?* Built on the 2014-15 Kaggle season — the full-season, richest-
feature, IP-safe dataset that v1's published leaderboard comes from. Uses v1's exact model
(`build_boosted_model`, `MODEL_COLUMNS`) so every SMOKE value stays consistent with what
was already shown.

2.1 Error bars.  Bootstrap each player's shots ~2,000x → 95% interval on makes-above-
    expected. Any player whose interval crosses zero is flagged "not distinguishable from
    expectation" — the leaderboard becomes "Chris Paul +51 [+37, +65]" instead of a bare
    number.

2.2 Shrinkage.  Empirical-Bayes (normal-normal) shrink each player's per-shot SMOKE rate
    toward the league mean, weighted by how precisely it's measured (the bootstrap SE from
    2.1). Kills the classic embarrassment where a 40-shot bench player tops the board.
    Publish both raw and shrunk; lead with shrunk.

2.3 Reliability.  Split each player's shots into odd/even halves, score each half, correlate
    across players, Spearman-Brown correct. Measures how much of SMOKE is repeatable skill vs.
    *within* a single season (complements 2.4's across-season stability).

Run:
    .venv/Scripts/python.exe -m src.validate.reliability
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.features.kaggle_shot_quality import (
    MODEL_COLUMNS,
    PLAYER_NAME_COLUMN,
    TARGET_COLUMN,
    build_modeling_frame,
    load_kaggle_shot_logs,
)
from src.models.build_model_outputs import build_boosted_model
from src.pulls._paths import REPO_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = REPO_ROOT / "data" / "v2" / "validation"
MIN_SHOTS = 150          # matches v1's PLAYER_MIN_SHOTS
N_BOOT = 2000
RNG = np.random.default_rng(42)


def score_all_shots() -> pd.DataFrame:
    """Every 2014-15 shot with its model expected-make probability (v1's model)."""
    shots = build_modeling_frame(load_kaggle_shot_logs())
    model = build_boosted_model()
    model.fit(shots[MODEL_COLUMNS], shots[TARGET_COLUMN])
    scored = shots[[PLAYER_NAME_COLUMN, TARGET_COLUMN]].copy()
    scored["expected"] = model.predict_proba(shots[MODEL_COLUMNS])[:, 1]
    scored["residual"] = scored[TARGET_COLUMN] - scored["expected"]  # per-shot SMOKE
    return scored


def bootstrap_players(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-player point estimate + bootstrap 95% CI and SE.

    Resamples each player's shots with replacement N_BOOT times. Reports both the total
    (makes above expected) and the rate (FG% above expected), plus the SE of the rate
    (feeds shrinkage in 2.2).
    """
    rows = []
    for player, grp in scored.groupby(PLAYER_NAME_COLUMN):
        resid = grp["residual"].to_numpy()
        n = len(resid)
        if n < MIN_SHOTS:
            continue
        # (N_BOOT x n) resample indices → each row is one bootstrap replicate
        idx = RNG.integers(0, n, size=(N_BOOT, n))
        boot = resid[idx]                       # resampled residuals
        boot_rate = boot.mean(axis=1)           # FG% above expected per replicate
        boot_total = boot_rate * n              # makes above expected per replicate
        rows.append(
            {
                "player": player,
                "shots": n,
                "smoke_total": float(resid.sum()),               # makes above expected
                "smoke_total_lo": float(np.percentile(boot_total, 2.5)),
                "smoke_total_hi": float(np.percentile(boot_total, 97.5)),
                "smoke_rate": float(resid.mean()),               # FG% above expected
                "smoke_rate_se": float(boot_rate.std(ddof=1)),
                "distinguishable": bool(
                    np.percentile(boot_total, 2.5) > 0 or np.percentile(boot_total, 97.5) < 0
                ),
            }
        )
    return pd.DataFrame(rows)


def empirical_bayes_shrink(players: pd.DataFrame) -> pd.DataFrame:
    """Normal-normal empirical-Bayes shrinkage of the SMOKE rate toward the league mean.

    Model: observed rate x_i = θ_i + ε_i, ε_i ~ N(0, se_i²); true skill θ_i ~ N(μ, τ²).
    μ = mean of observed rates; τ² via method of moments (DerSimonian-Laird style):
        τ² = max(0, Var(x) − mean(se_i²)).
    Shrunk estimate: θ̂_i = μ + B_i·(x_i − μ),  B_i = τ²/(τ²+se_i²)  (reliability weight).
    High-volume players (small se) barely move; low-volume ones shrink hard toward μ.
    """
    x = players["smoke_rate"].to_numpy()
    se = players["smoke_rate_se"].to_numpy()
    mu = float(np.mean(x))
    tau2 = max(0.0, float(np.var(x, ddof=1) - np.mean(se**2)))
    B = tau2 / (tau2 + se**2)
    out = players.copy()
    out["reliability"] = B                        # 0 = all sampling variation, 1 = all repeatable skill
    out["smoke_rate_shrunk"] = mu + B * (x - mu)
    out["smoke_total_shrunk"] = out["smoke_rate_shrunk"] * out["shots"]
    out.attrs["mu"] = mu
    out.attrs["tau"] = float(np.sqrt(tau2))
    return out


def split_half_reliability(scored: pd.DataFrame) -> dict:
    """Odd vs even shots per player → correlate SMOKE rate, Spearman-Brown corrected."""
    halves = []
    for player, grp in scored.groupby(PLAYER_NAME_COLUMN):
        g = grp.reset_index(drop=True)
        if len(g) < MIN_SHOTS:
            continue
        odd = g.iloc[1::2]["residual"].mean()
        even = g.iloc[0::2]["residual"].mean()
        halves.append((player, odd, even))
    h = pd.DataFrame(halves, columns=["player", "odd", "even"])
    r = float(stats.pearsonr(h["odd"], h["even"]).statistic)
    sb = 2 * r / (1 + r)                                     # Spearman-Brown full-length
    return {"n": len(h), "half_r": round(r, 3), "spearman_brown": round(sb, 3)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("PHASE 2.1-2.3 — error bars, shrinkage, within-season reliability (2014-15)")
    print("=" * 76)

    scored = score_all_shots()
    print(f"  scored {len(scored):,} shots")

    players = bootstrap_players(scored)
    print(f"  players ≥ {MIN_SHOTS} shots: {len(players)}  |  bootstrap replicates: {N_BOOT}")

    # 2.1 — how many survive the error bars?
    disting = players["distinguishable"].sum()
    print(f"\n  [2.1] distinguishable from expectation (95% CI excludes 0): "
          f"{disting}/{len(players)} ({disting / len(players):.0%})")
    print(f"        → {len(players) - disting} players are NOT statistically separable from average.")

    # 2.2 — shrinkage
    players = empirical_bayes_shrink(players)
    print(f"\n  [2.2] empirical-Bayes shrinkage: league mean μ = {players.attrs['mu']:+.4f} FG% above exp, "
          f"between-player SD τ = {players.attrs['tau']:.4f}")
    print(f"        reliability weight ranges {players['reliability'].min():.2f}–{players['reliability'].max():.2f} "
          f"(median {players['reliability'].median():.2f})")

    # 2.3 — within-season reliability
    rel = split_half_reliability(scored)
    print(f"\n  [2.3] split-half (odd vs even): r = {rel['half_r']}, "
          f"Spearman-Brown = {rel['spearman_brown']}  (n={rel['n']})")

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
    pd.DataFrame([rel]).to_csv(OUT_DIR / "split_half_reliability.csv", index=False)
    print(f"\n  wrote {OUT_DIR / 'player_smoke_with_error_bars.csv'}  ({len(full)} players)")
    print(f"  wrote {OUT_DIR / 'split_half_reliability.csv'}")


if __name__ == "__main__":
    main()
