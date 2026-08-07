"""Phase 1.1 — endpoint reconnaissance.

Before building the multi-season pull, confirm empirically (not from docs) what each
candidate stats.nba.com endpoint actually returns and how far back it goes.

Answers three questions per endpoint:
  1. Does it respond at all, and with how many rows?
  2. What columns come back? (Per-shot records, or season aggregates?)
  3. What is the earliest season with real data?

Run:  .venv/Scripts/python.exe -m src.pulls.recon_endpoints
"""

from __future__ import annotations

import sys
import traceback

import pandas as pd

from src.pulls._net import call

# player names contain non-cp1252 characters (Vučević, Jokić) — don't die on the console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def season_str(start_year: int) -> str:
    """2013 -> '2013-14'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def show(df: pd.DataFrame | None, label: str, max_cols: int = 26) -> None:
    if df is None:
        print(f"    {label}: FAILED")
        return
    if df.empty:
        print(f"    {label}: EMPTY (0 rows)")
        return
    cols = list(df.columns)
    print(f"    {label}: {len(df):,} rows x {len(cols)} cols")
    shown = ", ".join(cols[:max_cols])
    print(f"      cols: {shown}{' ...' if len(cols) > max_cols else ''}")


def main() -> None:
    from nba_api.stats.endpoints import (
        leaguedashplayerclutch,
        leaguedashplayerptshot,
        leaguedashptstats,
        shotchartdetail,
    )

    TEST = season_str(2023)
    print("=" * 78)
    print(f"PHASE 1.1 ENDPOINT RECONNAISSANCE   (test season {TEST})")
    print("=" * 78)

    # ------------------------------------------------------------------ [1]
    print("\n[1] LeagueDashPlayerPtShot  <- THE core endpoint for Phase 1")
    print("    (shooting split by defender distance / touch time / dribbles / shot clock)")
    df = call(
        leaguedashplayerptshot.LeagueDashPlayerPtShot,
        league_id="00",
        per_mode_simple="Totals",
        season=TEST,
        season_type_all_star="Regular Season",
    )
    show(df, "unfiltered")
    if df is not None and not df.empty:
        print("\n      --- one player's row (is this per-shot or aggregate?) ---")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("      " + df.head(2).to_string(index=False).replace("\n", "\n      "))

    print("\n    filter check — does CloseDefDistRange actually partition the data?")
    for rng in ["0-2 Feet - Very Tight", "4-6 Feet - Open", "6+ Feet - Wide Open"]:
        d = call(
            leaguedashplayerptshot.LeagueDashPlayerPtShot,
            league_id="00",
            per_mode_simple="Totals",
            season=TEST,
            season_type_all_star="Regular Season",
            close_def_dist_range_nullable=rng,
        )
        if d is None:
            msg = "FAILED"
        elif d.empty:
            msg = "empty"
        else:
            fga = d["FGA"].sum() if "FGA" in d.columns else float("nan")
            msg = f"{len(d):,} players, {fga:,.0f} FGA"
        print(f"      {rng:26s} -> {msg}")

    print("\n  -- history probe (how far back does it go?)")
    for y in [2012, 2013, 2014, 2024, 2025]:
        d = call(
            leaguedashplayerptshot.LeagueDashPlayerPtShot,
            league_id="00",
            per_mode_simple="Totals",
            season=season_str(y),
            season_type_all_star="Regular Season",
        )
        status = "FAILED" if d is None else (f"{len(d):,} rows" if not d.empty else "EMPTY")
        print(f"      {season_str(y)}: {status}")

    # ------------------------------------------------------------------ [2]
    print("\n[2] LeagueDashPtStats  <- tracking categories (PtMeasureType)")
    for mt in ["CatchShoot", "PullUpShot", "Drives", "Defense", "SpeedDistance", "Passing"]:
        d = call(
            leaguedashptstats.LeagueDashPtStats,
            season=TEST,
            season_type_all_star="Regular Season",
            per_mode_simple="Totals",
            player_or_team="Player",
            pt_measure_type=mt,
        )
        show(d, f"{mt:14s}", max_cols=12)

    # ------------------------------------------------------------------ [3]
    print("\n[3] LeagueDashPlayerClutch  <- clutch context (feeds Phase 1.6)")
    d = call(
        leaguedashplayerclutch.LeagueDashPlayerClutch,
        season=TEST,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        measure_type_detailed_defense="Base",
    )
    show(d, "clutch (Base)", max_cols=14)

    # ------------------------------------------------------------------ [4]
    print("\n[4] ShotChartDetail  <- per-shot location (already used in v1)")
    d = call(
        shotchartdetail.ShotChartDetail,
        team_id=0,
        player_id=0,
        season_nullable=TEST,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
    )
    show(d, "all players", max_cols=24)

    print("\n" + "=" * 78)
    print("VERDICT to record — per-shot rows anywhere, or aggregates only?")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
