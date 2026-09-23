"""Phase 2.6 — predictive validity: does this season's SMOKE tell you anything about next season?

Year N = 2014-15, year N+1 = 2015-16. Efficiency comes from the Basketball-Reference
full-season totals (sumitrodatta); SMOKE_N is the published, shrunk 2014-15 value from
reliability.py; SMOKE_N+1 is the 2015-16 value from the two-season stability panel
(its own cross-fitted four-feature model, so the two sides are independently defined).

Three parts, and reporting all three is what makes the test credible.

  (a) Head to head on next-season efficiency. Does year-N SMOKE predict year-N+1 eFG%
      better than year-N eFG% does? Honest expectation: no. Past eFG% carries persistent
      shot *selection* too, which SMOKE deliberately strips out.

  (b) Decomposition. eFG%_N splits (up to the three-point bonus) into the difficulty of
      the shots taken (expected FG%) and the shot-making above it (SMOKE). Regress
      eFG%_N+1 on both parts. Each coefficient is the forward carry of that part.
      The earlier specification eFG_N+1 ~ eFG_N + SMOKE_N is reported too, with a
      warning: conditional on eFG_N, a higher SMOKE_N is exactly a lower expected FG%,
      so its coefficient measures shot diet, not mean reversion.

  (c) Incremental forecast of next-season shot-making, the quantity SMOKE measures:
      SMOKE_N+1 ~ eFG_N + SMOKE_N. The SMOKE_N coefficient is the information about
      future shot-making that past efficiency does not carry, and the simple
      correlations of each predictor with SMOKE_N+1 are reported beside it. This
      replaces an earlier version that correlated the stability panel with itself.

Run:
    .venv/Scripts/python.exe -m src.validate.predictive
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.features.kaggle_shot_quality import PLAYER_ID_COLUMN
from src.models.build_model_outputs import OUTPUT_DIR
from src.pulls._paths import REPO_ROOT
from src.validate.convergent import one_row_per_player

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAL_DIR = REPO_ROOT / "data" / "v2" / "validation"
SMOKE_CSV = VAL_DIR / "player_smoke_with_error_bars.csv"   # 2014-15, from reliability.py
PANEL = VAL_DIR / "stability_2season_players.parquet"      # from stability.py
TOTALS = REPO_ROOT / "data" / "raw" / "kaggle" / "historical_stats" / "Player Totals.csv"

YEAR_N, YEAR_N1 = 2015, 2016     # 2014-15 → 2015-16
MIN_FGA_N1 = 200                 # next-season volume floor so the outcome isn't noise
SMOKE_COL = "smoke_rate_shrunk"


def season_efg(totals: pd.DataFrame, season: int) -> pd.DataFrame:
    """One eFG% per player for a season, keyed by normalized name."""
    s = one_row_per_player(totals, season)
    return s[["key", "player", "e_fg_percent", "fga", "mp"]].rename(
        columns={"e_fg_percent": "efg", "player": "player_name"}
    )


def ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """Plain OLS with intercept via numpy; returns coefs, t-stats, p-values, R², residuals."""
    Xd = X.to_numpy(dtype=float)
    Xd = np.column_stack([np.ones(len(Xd)), Xd])
    yv = y.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    resid = yv - Xd @ beta
    n, k = Xd.shape
    dof = n - k
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * stats.t.sf(abs(tvals), dof)
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof
    return {
        "names": ["intercept", *X.columns],
        "beta": beta, "se": se, "t": tvals, "p": pvals,
        "r2": r2, "adj_r2": adj_r2, "n": n, "resid": resid,
    }


def show(fit: dict, label: str) -> None:
    print(f"      {label}:  R² {fit['r2']:.3f}   adj R² {fit['adj_r2']:.3f}   n={fit['n']}")
    for name, b, se, t, p in zip(fit["names"], fit["beta"], fit["se"], fit["t"], fit["p"], strict=True):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"        {name:18s}  β={b:+.4f}  SE={se:.4f}  t={t:+.2f}  p={p:.4f} {star}")


def main() -> None:
    from src.features.names import norm_name

    smoke = pd.read_csv(SMOKE_CSV)
    smoke["key"] = smoke["player"].map(norm_name)
    rank = pd.read_csv(OUTPUT_DIR / "rank_table.csv")[[PLAYER_ID_COLUMN, "expected_fg_pct"]]
    n_smoke = len(smoke)
    smoke = smoke.merge(rank, on=PLAYER_ID_COLUMN, how="inner")
    assert len(smoke) == n_smoke, "every SMOKE player must have a rank-table row"
    smoke = smoke[["key", PLAYER_ID_COLUMN, "player", SMOKE_COL, "expected_fg_pct"]].rename(
        columns={SMOKE_COL: "smoke_N", "expected_fg_pct": "expected_fg_N"}
    )

    totals = pd.read_csv(TOTALS)
    efg_n = season_efg(totals, YEAR_N).rename(columns={"efg": "efg_N", "fga": "fga_N"})
    efg_n1 = season_efg(totals, YEAR_N1).rename(columns={"efg": "efg_N1", "fga": "fga_N1"})

    df = (
        smoke.merge(efg_n[["key", "efg_N", "fga_N"]], on="key", how="inner")
        .merge(efg_n1[["key", "efg_N1", "fga_N1"]], on="key", how="inner")
    )
    assert len(df) <= n_smoke and df["key"].is_unique, "season merges must not duplicate players"
    df = df[df["fga_N1"] >= MIN_FGA_N1].dropna(subset=["smoke_N", "efg_N", "efg_N1"])

    print("=" * 78)
    print(f"PHASE 2.6 — predictive validity  (year N = {YEAR_N - 1}-{str(YEAR_N)[-2:]} → "
          f"N+1 = {YEAR_N1 - 1}-{str(YEAR_N1)[-2:]})")
    print("=" * 78)
    print(f"  players with SMOKE_N, eFG_N, and eFG_N+1 (≥{MIN_FGA_N1} FGA next yr): {len(df)}")

    # (a) head-to-head: which single predictor better forecasts next-year eFG%?
    r_smoke = float(stats.pearsonr(df["smoke_N"], df["efg_N1"]).statistic)
    r_efg = float(stats.pearsonr(df["efg_N"], df["efg_N1"]).statistic)
    print("\n  [a] single-predictor correlation with next-season eFG%:")
    print(f"        SMOKE_N   → eFG_N+1 :  r = {r_smoke:+.3f}")
    print(f"        eFG_N     → eFG_N+1 :  r = {r_efg:+.3f}   (the baseline to beat)")
    print("        → past eFG% predicts next-season eFG% better (expected: it carries persistent")
    print("          shot *selection*, which SMOKE deliberately strips out).")

    # (b) decomposition: what carries forward, the shot diet or the shot-making?
    decomp = ols(df["efg_N1"], df[["expected_fg_N", "smoke_N"]])
    legacy = ols(df["efg_N1"], df[["efg_N", "smoke_N"]])
    print("\n  [b] decomposition — OLS  eFG_N+1 ~ expectedFG_N + SMOKE_N:")
    show(decomp, "decomposition")
    print("      (earlier specification, reported for transparency; conditional on eFG_N a higher")
    print("       SMOKE_N is a lower expected FG%, so this coefficient measures shot diet, not reversion)")
    show(legacy, "eFG_N+1 ~ eFG_N + SMOKE_N")

    # (c) the fair test: does SMOKE_N add information about NEXT-season shot-making
    #     beyond past efficiency? SMOKE_N+1 comes from the independently fitted 2015-16 model.
    panel = pd.read_parquet(PANEL)[[PLAYER_ID_COLUMN, "smoke_16", "shots_16"]]
    dc = df.merge(panel, on=PLAYER_ID_COLUMN, how="inner").dropna(subset=["smoke_16"])
    assert len(dc) <= len(df) and dc[PLAYER_ID_COLUMN].is_unique, "panel merge must not duplicate players"
    r_ss = float(stats.pearsonr(dc["smoke_N"], dc["smoke_16"]).statistic)
    r_es = float(stats.pearsonr(dc["efg_N"], dc["smoke_16"]).statistic)
    inc = ols(dc["smoke_16"], dc[["efg_N", "smoke_N"]])
    base = ols(dc["smoke_16"], dc[["efg_N"]])
    print("\n  [c] forecasting next-season SHOT-MAKING (SMOKE_N+1 from the 2015-16 model):")
    print(f"        SMOKE_N   → SMOKE_N+1 :  r = {r_ss:+.3f}")
    print(f"        eFG_N     → SMOKE_N+1 :  r = {r_es:+.3f}      (n = {len(dc)})")
    show(base, "eFG_N only")
    show(inc, "eFG_N + SMOKE_N")
    dr2 = inc["adj_r2"] - base["adj_r2"]
    p_inc = inc["p"][inc["names"].index("smoke_N")]
    print(f"      → SMOKE_N adds adj-R² of {dr2:+.3f} on top of eFG_N; coefficient "
          f"{'SIGNIFICANT' if p_inc < 0.05 else 'not significant'} (p={p_inc:.4f}).")
    if p_inc < 0.05 and inc["beta"][inc["names"].index("smoke_N")] > 0:
        print("      → SMOKE carries information about FUTURE shot-making that past efficiency misses.")

    out = df[["player", PLAYER_ID_COLUMN, "smoke_N", "expected_fg_N", "efg_N", "efg_N1", "fga_N1"]].copy()
    out.to_csv(VAL_DIR / "predictive_validity.csv", index=False)
    rows = [
        {"test": "a_corr_smoke_vs_futureEFG", "value": round(r_smoke, 3)},
        {"test": "a_corr_pastEFG_vs_futureEFG", "value": round(r_efg, 3)},
        {"test": "a_n_players", "value": len(df)},
        {"test": "b_decomp_adjR2", "value": round(decomp["adj_r2"], 3)},
        {"test": "b_decomp_beta_expectedFG", "value": round(decomp["beta"][1], 4)},
        {"test": "b_decomp_p_expectedFG", "value": round(decomp["p"][1], 4)},
        {"test": "b_decomp_beta_smoke", "value": round(decomp["beta"][2], 4)},
        {"test": "b_decomp_p_smoke", "value": round(decomp["p"][2], 4)},
        {"test": "b_legacy_beta_smoke", "value": round(legacy["beta"][2], 4)},
        {"test": "b_legacy_p_smoke", "value": round(legacy["p"][2], 4)},
        {"test": "c_corr_smoke_vs_futureSMOKE", "value": round(r_ss, 3)},
        {"test": "c_corr_pastEFG_vs_futureSMOKE", "value": round(r_es, 3)},
        {"test": "c_adjR2_efg_only", "value": round(base["adj_r2"], 3)},
        {"test": "c_adjR2_efg_plus_smoke", "value": round(inc["adj_r2"], 3)},
        {"test": "c_beta_smoke", "value": round(inc["beta"][2], 4)},
        {"test": "c_p_smoke", "value": round(p_inc, 4)},
        {"test": "c_n_players", "value": len(dc)},
    ]
    pd.DataFrame(rows).to_csv(VAL_DIR / "predictive_summary.csv", index=False)
    print(f"\n  wrote {VAL_DIR / 'predictive_validity.csv'}  ({len(out)} players)")
    print(f"  wrote {VAL_DIR / 'predictive_summary.csv'}")


if __name__ == "__main__":
    main()
