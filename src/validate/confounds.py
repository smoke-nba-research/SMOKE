"""Phase 2.7 — confound checks: is the SMOKE leaderboard an artifact of context?

A skeptic's first attack on any player metric: "he only looks good because of *who* and
*when* he shot — easy defenses, home games, garbage time — not real skill." This test
partials those out and shows the leaderboard barely moves.

What the base per-shot model already controls (so these are NOT open confounds): home/away
(LOCATION), game margin (FINAL_MARGIN), period, game clock, shot clock — all model features.
The one aggregate confound it does *not* see is **opponent defensive quality**: the model
knows the defender's distance on a shot, but not that the defender plays for a good defense.

Method: for each player, compute the strength of the defenses they shot against (opponent
FG%-allowed, derived from the logs themselves — self-contained, no external join) and their
home-shot share. Regress player SMOKE on those; the residual is context-adjusted SMOKE.
Then compare the leaderboard before vs after. If it barely reorders, SMOKE is skill, not
schedule.

Deferred: rest days (not in the Kaggle data) and per-possession opponent DRtg (would need an
external team-ratings join with abbreviation mapping); opponent FG%-allowed is the honest,
self-contained proxy here — noted as a limitation.

Run:
    .venv/Scripts/python.exe -m src.validate.confounds
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.features.kaggle_shot_quality import (
    MATCHUP_PATTERN,
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


def opponent_team(matchup: str, location: str) -> str | None:
    """The defending team for a shot (the one the shooter is NOT on)."""
    if pd.isna(matchup) or pd.isna(location):
        return None
    m = MATCHUP_PATTERN.search(str(matchup))
    if not m:
        return None
    away, home = m.groups()
    return away if location == "H" else home     # opposite of SHOOTING_TEAM


def main() -> None:
    shots = add_derived_features(load_kaggle_shot_logs())
    shots = shots.dropna(subset=["SHOT_RESULT", PLAYER_NAME_COLUMN]).copy()
    shots["OPP_TEAM"] = [
        opponent_team(mu, loc)
        for mu, loc in zip(shots["MATCHUP"], shots["LOCATION"], strict=False)
    ]
    shots["is_home"] = (shots["LOCATION"] == "H").astype(int)

    # opponent defensive quality = FG% allowed by that team, from the logs themselves
    opp_fg_allowed = shots.groupby("OPP_TEAM")[TARGET_COLUMN].mean().rename("opp_fg_allowed")
    shots = shots.join(opp_fg_allowed, on="OPP_TEAM")

    # per-player context: schedule strength faced + home share
    ctx = (
        shots.groupby(PLAYER_NAME_COLUMN)
        .agg(
            shots=(TARGET_COLUMN, "size"),
            opp_fg_allowed=("opp_fg_allowed", "mean"),   # lower = tougher defenses faced
            home_share=("is_home", "mean"),
        )
        .reset_index()
    )
    ctx = ctx[ctx["shots"] >= MIN_SHOTS].copy()
    ctx["key"] = ctx[PLAYER_NAME_COLUMN].map(norm_name)

    smoke = pd.read_csv(SMOKE_CSV)
    smoke["key"] = smoke["player"].map(norm_name)
    smoke_col = "smoke_rate_shrunk" if "smoke_rate_shrunk" in smoke.columns else "smoke_rate"
    df = smoke[["key", "player", smoke_col, "smoke_total_shrunk"]].merge(
        ctx[["key", "opp_fg_allowed", "home_share"]], on="key", how="inner"
    )
    df = df.rename(columns={smoke_col: "smoke"})

    print("=" * 78)
    print("PHASE 2.7 — confound checks: does context reorder the SMOKE leaderboard? (2014-15)")
    print("=" * 78)
    print(f"  players: {len(df)}")

    # how much does context even correlate with SMOKE? (small = not a confound)
    r_opp = float(stats.pearsonr(df["smoke"], df["opp_fg_allowed"]).statistic)
    r_home = float(stats.pearsonr(df["smoke"], df["home_share"]).statistic)
    print("\n  context ↔ SMOKE correlations (want these near zero):")
    print(f"    opponent FG%-allowed faced  r = {r_opp:+.3f}   "
          f"(range of schedule strength: {df['opp_fg_allowed'].min():.3f}–{df['opp_fg_allowed'].max():.3f})")
    print(f"    home-shot share             r = {r_home:+.3f}   "
          f"(range: {df['home_share'].min():.2f}–{df['home_share'].max():.2f})")

    # residualize SMOKE on the context controls → context-adjusted SMOKE
    X = np.column_stack([np.ones(len(df)), df["opp_fg_allowed"], df["home_share"]])
    y = df["smoke"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - fitted) ** 2).sum() / ss_tot
    df["smoke_adj"] = (y - fitted) + y.mean()     # residual, recentered onto SMOKE's scale
    print(f"\n  context explains only R² = {r2:.3f} of SMOKE variance "
          f"({r2:.1%}) → context is not driving the metric.")

    # before/after leaderboard comparison
    df["rank_before"] = df["smoke"].rank(ascending=False, method="min")
    df["rank_after"] = df["smoke_adj"].rank(ascending=False, method="min")
    df["shift"] = (df["rank_before"] - df["rank_after"]).astype(int)
    rho = float(stats.spearmanr(df["smoke"], df["smoke_adj"]).statistic)
    top20 = df.nsmallest(20, "rank_before")
    print(f"\n  ranking correlation before vs after controls: Spearman ρ = {rho:.4f}")
    print(f"  top-20 mean |rank shift| = {top20['shift'].abs().mean():.1f}  "
          f"(max {top20['shift'].abs().max()})  |  full-board mean |shift| = {df['shift'].abs().mean():.1f}")

    print("\n  TOP 20 — before vs after context adjustment")
    print("  " + "-" * 62)
    print(f"  {'player':22s} {'rank→adj':>10} {'shift':>6}   {'SMOKE→adj':>16}")
    print("  " + "-" * 62)
    for _, r in top20.sort_values("rank_before").iterrows():
        arrow = f"{r['rank_before']:.0f}→{r['rank_after']:.0f}"
        sh = f"{r['shift']:+d}" if r["shift"] else "0"
        vals = f"{r['smoke']:+.4f}→{r['smoke_adj']:+.4f}"
        print(f"  {r['player'][:22]:22s} {arrow:>10} {sh:>6}   {vals:>16}")
    print("  " + "-" * 62)

    df.sort_values("rank_before").to_csv(VAL_DIR / "confound_check.csv", index=False)
    print(f"\n  wrote {VAL_DIR / 'confound_check.csv'}  ({len(df)} players)")


if __name__ == "__main__":
    main()
