"""Phase 2.6 — predictive validity: the money test.

The single strongest sentence the eventual paper can contain would be: "a player's
shot-quality-adjusted shot-making this season tells you something about their shooting
efficiency next season that this season's efficiency alone does not."

Three regressions (year N = 2014-15, year N+1 = 2015-16 — the two seasons we have
tracking-grounded SMOKE for; efficiency from the IP-safe sumitrodatta full-season totals):

  (a) Does year-N SMOKE predict year-N+1 eFG% *better than* year-N eFG% does?
      Honest expectation: probably NOT on its own — past eFG% carries persistent shot
      *selection* too, which SMOKE deliberately strips out. Report it either way.

  (b) THE key test — does SMOKE add predictive power *incrementally*, on top of eFG%?
      OLS: eFG_{N+1} ~ eFG_N + SMOKE_N. A significant positive SMOKE coefficient means
      SMOKE carries information about future shooting that past efficiency alone misses. This
      is the defensible "money" claim.

  (c) Contract value — deferred (exploratory, needs salary data not yet pulled; the free
      Basketball-Reference/Spotrac web tables are enough when we come back to it).

Run:
    .venv/Scripts/python.exe -m src.validate.predictive
"""

from __future__ import annotations

import sys

import pandas as pd
from scipy import stats

from src.pulls._paths import REPO_ROOT
from src.validate.convergent import norm_name  # shared name normalizer (100% match rate)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAL_DIR = REPO_ROOT / "data" / "v2" / "validation"
SMOKE_CSV = VAL_DIR / "player_smoke_with_error_bars.csv"   # 2014-15, from reliability.py
TOTALS = REPO_ROOT / "data" / "raw" / "kaggle" / "historical_stats" / "Player Totals.csv"

YEAR_N, YEAR_N1 = 2015, 2016     # 2014-15 → 2015-16
MIN_FGA_N1 = 200                 # next-season volume floor so the outcome isn't noise


def season_efg(totals: pd.DataFrame, season: int) -> pd.DataFrame:
    """One eFG% per player for a season (most-minutes row if traded), keyed by norm name."""
    s = totals[totals["season"] == season].copy()
    s = s.sort_values("mp", ascending=False).drop_duplicates("player_id", keep="first")
    s["key"] = s["player"].map(norm_name)
    return s[["key", "player", "e_fg_percent", "fga", "mp"]].rename(
        columns={"e_fg_percent": "efg", "player": "player_name"}
    )


def ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """Plain OLS with intercept via numpy; returns coefs, t-stats, p-values, R²."""
    import numpy as np

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
    names = ["intercept", *X.columns]
    return {
        "names": names,
        "beta": beta,
        "se": se,
        "t": tvals,
        "p": pvals,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
    }


def main() -> None:
    smoke = pd.read_csv(SMOKE_CSV)
    smoke["key"] = smoke["player"].map(norm_name)
    smoke_col = "smoke_rate_shrunk" if "smoke_rate_shrunk" in smoke.columns else "smoke_rate"
    smoke = smoke[["key", "player", smoke_col]].rename(columns={smoke_col: "smoke_N"})

    totals = pd.read_csv(TOTALS)
    efg_n = season_efg(totals, YEAR_N).rename(columns={"efg": "efg_N", "fga": "fga_N"})
    efg_n1 = season_efg(totals, YEAR_N1).rename(columns={"efg": "efg_N1", "fga": "fga_N1"})

    df = (
        smoke.merge(efg_n[["key", "efg_N", "fga_N"]], on="key", how="inner")
        .merge(efg_n1[["key", "efg_N1", "fga_N1"]], on="key", how="inner")
    )
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
    if r_smoke > r_efg:
        print("        → SMOKE alone predicts next-year efficiency better than past efficiency does.")
    else:
        print("        → past eFG% alone predicts better (expected — it also carries persistent")
        print("          shot *selection*, which SMOKE deliberately strips out). The real test is [b].")

    # (b) THE test: does SMOKE add incremental predictive power on top of eFG%?
    base = ols(df["efg_N1"], df[["efg_N"]])
    full = ols(df["efg_N1"], df[["efg_N", "smoke_N"]])
    print("\n  [b] incremental value — OLS  eFG_N+1 ~ eFG_N + SMOKE_N:")
    print("      model                         R²      adj R²")
    print(f"      eFG_N only                    {base['r2']:.3f}   {base['adj_r2']:.3f}")
    print(f"      eFG_N + SMOKE_N               {full['r2']:.3f}   {full['adj_r2']:.3f}")
    print("\n      coefficients (full model):")
    for name, b, se, t, p in zip(full["names"], full["beta"], full["se"], full["t"], full["p"], strict=True):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"        {name:12s}  β={b:+.4f}  SE={se:.4f}  t={t:+.2f}  p={p:.4f} {star}")
    smoke_p = full["p"][full["names"].index("smoke_N")]
    dr2 = full["adj_r2"] - base["adj_r2"]
    print(f"\n      → SMOKE_N adds adj-R² of {dr2:+.3f}; its coefficient is "
          f"{'SIGNIFICANT' if smoke_p < 0.05 else 'not significant'} (p={smoke_p:.4f}).")
    if smoke_p < 0.05 and full["beta"][full["names"].index("smoke_N")] > 0:
        print("      → SMOKE carries real information about FUTURE shooting that past efficiency misses.")
        print("        This is the paper's money sentence.")

    # (b2) THE FAIR TEST — for a shot-making metric, the right target is future
    # shot-MAKING, not future efficiency (which is half shot-selection SMOKE strips out).
    # Uses both seasons' SMOKE from the 2.4 stability panel.
    panel = pd.read_parquet(VAL_DIR / "stability_2season_players.parquet")
    panel = panel.dropna(subset=["smoke_14", "smoke_16", "efg_pct_14"])
    r_smoke_smoke = float(stats.pearsonr(panel["smoke_14"], panel["smoke_16"]).statistic)
    r_efg_smoke = float(stats.pearsonr(panel["efg_pct_14"], panel["smoke_16"]).statistic)
    print("\n  [b2] THE FAIR TEST — predicting next-season SHOT-MAKING (SMOKE_N+1),")
    print("       the outcome a shot-making metric should actually forecast:")
    print(f"        SMOKE_N   → SMOKE_N+1 :  r = {r_smoke_smoke:+.3f}")
    print(f"        eFG_N     → SMOKE_N+1 :  r = {r_efg_smoke:+.3f}")
    print(f"       (n = {len(panel)})")
    if r_smoke_smoke > r_efg_smoke:
        print("        → SMOKE forecasts future shot-making BETTER than efficiency does — the")
        print("          honest 'money' claim: for the thing SMOKE measures, it out-predicts eFG%.")

    print("\n  [c] contract-value test — deferred (exploratory; needs salary data).")

    out = df[["player", "smoke_N", "efg_N", "efg_N1", "fga_N1"]].copy()
    out.to_csv(VAL_DIR / "predictive_validity.csv", index=False)
    pd.DataFrame(
        [
            {"test": "a_corr_smoke_vs_futureEFG", "value": round(r_smoke, 3)},
            {"test": "a_corr_pastEFG_vs_futureEFG", "value": round(r_efg, 3)},
            {"test": "b_adjR2_efg_only", "value": round(base["adj_r2"], 3)},
            {"test": "b_adjR2_efg_plus_smoke", "value": round(full["adj_r2"], 3)},
            {"test": "b_smoke_coef_pvalue", "value": round(smoke_p, 4)},
            {"test": "b2_corr_smoke_vs_futureSMOKE", "value": round(r_smoke_smoke, 3)},
            {"test": "b2_corr_pastEFG_vs_futureSMOKE", "value": round(r_efg_smoke, 3)},
            {"test": "n_players", "value": len(df)},
        ]
    ).to_csv(VAL_DIR / "predictive_summary.csv", index=False)
    print(f"\n  wrote {VAL_DIR / 'predictive_validity.csv'}  ({len(out)} players)")
    print(f"  wrote {VAL_DIR / 'predictive_summary.csv'}")


if __name__ == "__main__":
    main()
