"""Integrity / cross-validation for the 2015-16 derived tracking features.

The single-game proof in `tracking_1516.py` checked our computed shot distance against
the `shotDistance` field of PlayByPlayV3 — but that is the *same* endpoint we join on,
so it is not a fully independent test. This module cross-checks against two genuinely
independent NBA sources:

  CHECK A — ShotChartDetail (per-shot court coordinates)
      A different endpoint that reports the NBA's own (LOC_X, LOC_Y) for every shot.
      Validates our full 2D geometry (radial distance, lateral offset, baseline depth),
      not just a scalar distance. Also cross-checks made/missed.

  CHECK B — LeagueDashPlayerPtShot (the NBA's OWN defender-distance buckets)
      The NBA publishes, for 2015-16, league shooting split by closest-defender distance
      (0-2 / 2-4 / 4-6 / 6+ ft), derived by them from the same optical feed using their
      proprietary method. If our independently-derived defender distances reproduce that
      distribution — both the share of shots per bucket and the FG% within each bucket —
      the extraction is sound. This is the strongest available test.

Run:
    .venv/Scripts/python.exe -m src.validate.cross_check_tracking            # 1 game (A only)
    .venv/Scripts/python.exe -m src.validate.cross_check_tracking --games 8  # A + B
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.pulls._net import call
from src.pulls.build_tracking_season import cached_pbp
from src.pulls.tracking_1516 import (
    HOOPS,
    download_game,
    extract_game_shots,
    list_archive_games,
    load_game,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEASON = "2015-16"
# NBA's published bucket labels -> our numeric edges (feet)
DEF_BUCKETS = [
    ("0-2 Feet - Very Tight", 0.0, 2.0),
    ("2-4 Feet - Tight", 2.0, 4.0),
    ("4-6 Feet - Open", 4.0, 6.0),
    ("6+ Feet - Wide Open", 6.0, 999.0),
]


# ---------------------------------------------------------------- CHECK A
def shotchart_for_game(game_id: str) -> pd.DataFrame:
    """NBA's own per-shot coordinates for one game (independent of PBP)."""
    from nba_api.stats.endpoints import shotchartdetail

    df = call(
        shotchartdetail.ShotChartDetail,
        team_id=0,
        player_id=0,
        season_nullable=SEASON,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
        game_id_nullable=game_id,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["clock_s"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]
    # LOC_X/LOC_Y are in 0.1 ft, hoop-relative: x lateral, y from baseline toward court
    df["nba_lateral_ft"] = df["LOC_X"].abs() / 10.0
    df["nba_depth_ft"] = df["LOC_Y"] / 10.0
    return df


def our_court_components(row) -> tuple[float, float]:
    """Convert our tracking (x,y) to hoop-relative (|lateral|, depth) in feet."""
    sx, sy = row["shooter_x"], row["shooter_y"]
    hx, hy = min(HOOPS, key=lambda h: (sx - h[0]) ** 2 + (sy - h[1]) ** 2)
    lateral = abs(sy - hy)
    depth = abs(sx - hx)
    return lateral, depth


def check_a(ours: pd.DataFrame, game_id: str) -> None:
    print("\n" + "=" * 74)
    print("CHECK A — vs ShotChartDetail (independent per-shot NBA coordinates)")
    print("=" * 74)
    nba = shotchart_for_game(game_id)
    if nba.empty:
        print("  !! ShotChartDetail returned nothing — cannot run check A")
        return
    print(f"  NBA shotchart rows: {len(nba)}   |   our extracted shots: {len(ours)}")

    # match on player + period + NEAREST game clock (each source logs the clock at a
    # slightly different instant, so exact equality matches almost nothing)
    TOL_S = 25.0
    n = nba.copy()
    rows = []
    used: set = set()
    for _, sh in ours.sort_values(["quarter", "game_clock"]).iterrows():
        cand = n[(n["PLAYER_ID"] == sh["player_id"]) & (n["PERIOD"] == sh["quarter"])]
        cand = cand[~cand.index.isin(used)]
        if cand.empty:
            continue
        gaps = (cand["clock_s"] - sh["game_clock"]).abs()
        j = gaps.idxmin()
        if gaps.loc[j] > TOL_S:
            continue
        used.add(j)
        r = sh.to_dict()
        for col in ["SHOT_DISTANCE", "nba_lateral_ft", "nba_depth_ft", "SHOT_MADE_FLAG"]:
            r[col] = n.loc[j, col]
        rows.append(r)
    merged = pd.DataFrame(rows)
    print(f"  matched on player+period+nearest-clock (tol {TOL_S:.0f}s): {len(merged)}")
    if merged.empty:
        print("  !! no matches — clock alignment differs between sources")
        return

    hi = merged[merged["release_confidence"] == "high"]
    for label, sub in [("ALL matched", merged), ("high-confidence", hi)]:
        if sub.empty:
            continue
        comp = sub.apply(our_court_components, axis=1, result_type="expand")
        sub = sub.assign(our_lateral=comp[0], our_depth=comp[1])
        d_err = (sub["shot_dist_ft"] - sub["SHOT_DISTANCE"]).abs()
        lat_err = (sub["our_lateral"] - sub["nba_lateral_ft"]).abs()
        dep_err = (sub["our_depth"] - sub["nba_depth_ft"]).abs()
        made_agree = (sub["made"] == sub["SHOT_MADE_FLAG"]).mean()
        print(f"\n  [{label}]  n={len(sub)}")
        print(f"    radial distance : MAE {d_err.mean():5.2f} ft | corr "
              f"{sub['shot_dist_ft'].corr(sub['SHOT_DISTANCE']):.3f} | within 3ft {(d_err <= 3).mean():.0%}")
        print(f"    lateral offset  : MAE {lat_err.mean():5.2f} ft | corr "
              f"{sub['our_lateral'].corr(sub['nba_lateral_ft']):.3f}")
        print(f"    baseline depth  : MAE {dep_err.mean():5.2f} ft | corr "
              f"{sub['our_depth'].corr(sub['nba_depth_ft']):.3f}")
        print(f"    made/missed agreement: {made_agree:.1%}")


# ---------------------------------------------------------------- CHECK B
def nba_defender_buckets() -> pd.DataFrame:
    """The NBA's OWN league-wide 2015-16 shooting split by closest-defender distance."""
    from nba_api.stats.endpoints import leaguedashplayerptshot

    rows = []
    for label, _lo, _hi in DEF_BUCKETS:
        df = call(
            leaguedashplayerptshot.LeagueDashPlayerPtShot,
            league_id="00",
            per_mode_simple="Totals",
            season=SEASON,
            season_type_all_star="Regular Season",
            close_def_dist_range_nullable=label,
        )
        if df is None or df.empty:
            continue
        rows.append(
            {"bucket": label, "FGA": int(df["FGA"].sum()), "FGM": int(df["FGM"].sum())}
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fg_pct"] = (out["FGM"] / out["FGA"] * 100).round(1)
        out["share"] = (out["FGA"] / out["FGA"].sum() * 100).round(1)
    return out


def our_defender_buckets(df: pd.DataFrame) -> pd.DataFrame:
    edges = [b[1] for b in DEF_BUCKETS] + [999.0]
    labels = [b[0] for b in DEF_BUCKETS]
    d = df.copy()
    d["bucket"] = pd.cut(d["closest_def_dist_ft"], bins=edges, labels=labels,
                         include_lowest=True, right=False)
    g = d.groupby("bucket", observed=True).agg(FGA=("made", "size"), FGM=("made", "sum"))
    g = g.reset_index()
    g["fg_pct"] = (g["FGM"] / g["FGA"] * 100).round(1)
    g["share"] = (g["FGA"] / g["FGA"].sum() * 100).round(1)
    return g


def check_b(ours: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("CHECK B — our derived defender distances vs the NBA's OWN published buckets")
    print("=" * 74)
    nba = nba_defender_buckets()
    if nba.empty:
        print("  !! could not fetch NBA buckets")
        return
    high = ours[ours["release_confidence"] == "high"]
    mine = our_defender_buckets(high)
    m = nba.merge(mine, on="bucket", how="outer", suffixes=("_nba", "_ours"))
    print(f"\n  NBA = full 2015-16 season ({nba['FGA'].sum():,} FGA)")
    print(f"  ours = {len(high):,} high-confidence shots (of {len(ours):,}) from our sampled games\n")
    print(f"  {'bucket':24s} {'share_NBA':>10} {'share_ours':>11} {'FG%_NBA':>9} {'FG%_ours':>9}")
    for _, r in m.iterrows():
        print(f"  {str(r['bucket']):24s} {r.get('share_nba', float('nan')):9.1f}% "
              f"{r.get('share_ours', float('nan')):10.1f}% "
              f"{r.get('fg_pct_nba', float('nan')):8.1f}% {r.get('fg_pct_ours', float('nan')):8.1f}%")
    if {"share_nba", "share_ours"} <= set(m.columns):
        gap = (m["share_nba"] - m["share_ours"]).abs().mean()
        print(f"\n  mean absolute gap in bucket share: {gap:.1f} percentage points")
        print("  (small gap => our derived defender distance reproduces the NBA's own metric)")


# ---------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1, help="how many games to extract")
    args = ap.parse_args()

    files = list_archive_games()[: args.games]
    print(f"Extracting {len(files)} game(s) for validation...")
    frames, first_gid = [], None
    for i, fn in enumerate(files, 1):
        path = download_game(fn)
        game = load_game(path)
        gid = game["gameid"]
        pbp = cached_pbp(gid)
        if pbp.empty:
            print(f"  [{i}/{len(files)}] {fn}: no PBP, skipped")
            continue
        df = extract_game_shots(game, pbp)
        if df.empty:
            continue
        frames.append(df)
        if first_gid is None:
            first_gid, first_df = gid, df
        print(f"  [{i}/{len(files)}] {fn} -> {len(df)} shots")

    if not frames:
        print("No shots extracted; aborting.")
        return
    allshots = pd.concat(frames, ignore_index=True)
    print(f"\nTotal extracted: {len(allshots):,} shots "
          f"({(allshots['release_confidence'] == 'high').mean():.0%} high-confidence)")

    check_a(first_df, first_gid)
    if len(allshots) >= 300:
        check_b(allshots)
    else:
        print("\n(Skipping CHECK B — need >=300 shots for a stable distribution; "
              "re-run with --games 8)")


if __name__ == "__main__":
    main()
