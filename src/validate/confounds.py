"""Phase 2.7 — confound checks: is the SMOKE leaderboard an artifact of context?

A skeptic's first attack on any player metric: "he only looks good because of *who* and
*when* he shot — easy defenses, home games, garbage time — not real skill." This test
partials those out and shows the leaderboard barely moves.

What the base per-shot model already controls (so these are NOT open confounds): home/away
(LOCATION), period, game clock, shot clock — all model features. The one aggregate
confound it does *not* see is **opponent defensive quality**: the model knows the
defender's distance on a shot, but not that the defender plays for a good defense.

Method: for each player, compute the strength of the defenses they shot against (opponent
FG%-allowed, derived from the scored shots themselves — self-contained, no external join)
and their home-shot share. Regress player SMOKE on those; the residual is context-adjusted
SMOKE. Then compare the leaderboard (shrunk makes above expectation, the same ordering the
leaderboard figure and the dashboard use) before vs after. If it barely reorders, SMOKE is
skill, not schedule.

Deferred: rest days (not in the Kaggle data) and per-possession opponent DRtg (would need an
external team-ratings join with abbreviation mapping); opponent FG%-allowed is the honest,
self-contained proxy here — noted as a limitation.

Run:
    .venv/Scripts/python.exe -m src.validate.confounds
"""

from __future__ import annotations

import sys

import pandas as pd
from scipy import stats

from src.features.kaggle_shot_quality import OPPONENT_COLUMN, PLAYER_ID_COLUMN, TARGET_COLUMN
from src.models.build_model_outputs import PLAYER_MIN_SHOTS, load_scored_shots
from src.pulls._paths import REPO_ROOT
from src.validate.predictive import ols

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAL_DIR = REPO_ROOT / "data" / "v2" / "validation"
SMOKE_CSV = VAL_DIR / "player_smoke_with_error_bars.csv"
MIN_SHOTS = PLAYER_MIN_SHOTS
SMOKE_COL = "smoke_rate_shrunk"


def main() -> None:
    shots = load_scored_shots()
    shots["is_home"] = (shots["LOCATION"] == "H").astype(int)

    # opponent defensive quality = FG% allowed by that team, from the scored shots.
    # OPP_TEAM is the second abbreviation of the MATCHUP string on every row (the
    # string is written from the shooter's side), so no home/away logic is needed.
    n_before = len(shots)
    opp_fg_allowed = shots.groupby(OPPONENT_COLUMN)[TARGET_COLUMN].mean().rename("opp_fg_allowed")
    shots = shots.join(opp_fg_allowed, on=OPPONENT_COLUMN)
    assert len(shots) == n_before, "opponent join changed the row count"

    # per-player context: schedule strength faced + home share
    ctx = (
        shots.groupby(PLAYER_ID_COLUMN)
        .agg(
            shots=(TARGET_COLUMN, "size"),
            opp_fg_allowed=("opp_fg_allowed", "mean"),   # lower = tougher defenses faced
            home_share=("is_home", "mean"),
        )
        .reset_index()
    )
    ctx = ctx[ctx["shots"] >= MIN_SHOTS].copy()

    smoke = pd.read_csv(SMOKE_CSV)
    df = smoke[[PLAYER_ID_COLUMN, "player", "shots", SMOKE_COL, "smoke_total_shrunk"]].merge(
        ctx[[PLAYER_ID_COLUMN, "opp_fg_allowed", "home_share"]], on=PLAYER_ID_COLUMN, how="inner"
    )
    assert len(df) == len(smoke), "every SMOKE player must have a context row"
    df = df.rename(columns={SMOKE_COL: "smoke"})

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
    fit = ols(df["smoke"], df[["opp_fg_allowed", "home_share"]])
    r2 = fit["r2"]
    df["smoke_adj"] = fit["resid"] + df["smoke"].mean()     # residual, recentered onto SMOKE's scale
    df["smoke_total_adj"] = df["smoke_adj"] * df["shots"]
    print(f"\n  context explains only R² = {r2:.4f} of SMOKE variance "
          f"({r2:.1%}) → context is not driving the metric.")

    # before/after leaderboard comparison, ranked by shrunk makes above expectation
    df["rank_before"] = df["smoke_total_shrunk"].rank(ascending=False, method="min")
    df["rank_after"] = df["smoke_total_adj"].rank(ascending=False, method="min")
    df["shift"] = (df["rank_before"] - df["rank_after"]).astype(int)
    rho = float(stats.spearmanr(df["smoke_total_shrunk"], df["smoke_total_adj"]).statistic)
    top20 = df.nsmallest(20, "rank_before")
    print(f"\n  ranking correlation before vs after controls: Spearman ρ = {rho:.4f}")
    print(f"  top-20 mean |rank shift| = {top20['shift'].abs().mean():.1f}  "
          f"(max {top20['shift'].abs().max()})  |  full-board mean |shift| = {df['shift'].abs().mean():.1f}")

    print("\n  TOP 20 — before vs after context adjustment (shrunk makes above expectation)")
    print("  " + "-" * 62)
    print(f"  {'player':22s} {'rank→adj':>10} {'shift':>6}   {'SMOKE→adj':>16}")
    print("  " + "-" * 62)
    for _, r in top20.sort_values("rank_before").iterrows():
        arrow = f"{r['rank_before']:.0f}→{r['rank_after']:.0f}"
        sh = f"{r['shift']:+d}" if r["shift"] else "0"
        vals = f"{r['smoke_total_shrunk']:+.1f}→{r['smoke_total_adj']:+.1f}"
        print(f"  {r['player'][:22]:22s} {arrow:>10} {sh:>6}   {vals:>16}")
    print("  " + "-" * 62)

    df.sort_values("rank_before").to_csv(VAL_DIR / "confound_check.csv", index=False)
    pd.DataFrame(
        [
            {"stat": "corr_smoke_vs_opp_fg_allowed", "value": round(r_opp, 4)},
            {"stat": "corr_smoke_vs_home_share", "value": round(r_home, 4)},
            {"stat": "r2_context", "value": round(r2, 4)},
            {"stat": "spearman_before_after", "value": round(rho, 4)},
            {"stat": "top20_mean_abs_shift", "value": round(float(top20["shift"].abs().mean()), 2)},
            {"stat": "top20_max_abs_shift", "value": int(top20["shift"].abs().max())},
            {"stat": "top6_unchanged", "value": int((df.nsmallest(6, "rank_before")["shift"] == 0).all())},
            {"stat": "n_players", "value": len(df)},
        ]
    ).to_csv(VAL_DIR / "confound_summary.csv", index=False)
    print(f"\n  wrote {VAL_DIR / 'confound_check.csv'}  ({len(df)} players)")
    print(f"  wrote {VAL_DIR / 'confound_summary.csv'}")


if __name__ == "__main__":
    main()
