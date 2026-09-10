"""Phase 1.1a (scale-up) — extract per-shot tracking features for the whole 2015-16 archive.

Runs `tracking_1516`'s single-game extractor across all 636 archived games and writes one
parquet per game. Designed to be interrupted and re-run: anything already written is
skipped, so progress is never lost.

Scale notes
-----------
* 636 games, ~6 MB compressed each (~3.8 GB total download, cached on disk for resume).
* Each game inflates to ~104 MB of JSON, which becomes a large Python object graph, so we
  process strictly one game at a time and free it before the next.
* Play-by-play is cached per game so a resume never re-hits stats.nba.com.
* COVERAGE CAVEAT: the archive spans 2015-10-27 to 2016-01-23 only — the first half of
  the season (52% of its 1,230 games). Player aggregates from it are half-season figures.

Run:
    .venv/Scripts/python.exe -m src.pulls.build_tracking_season                # everything
    .venv/Scripts/python.exe -m src.pulls.build_tracking_season --max-games 25 # a slice
    .venv/Scripts/python.exe -m src.pulls.build_tracking_season --status       # progress
    .venv/Scripts/python.exe -m src.pulls.build_tracking_season --combine      # season file only

When every game is accounted for, the per-game parquets are concatenated into
data/v2/processed/shots_tracking_1516_season.parquet, which src/validate/stability.py
reads. A partial run (time or game limit) does not write the season file.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback

import pandas as pd

from src.pulls._paths import REPO_ROOT
from src.pulls.tracking_1516 import (
    EmptyArchiveError,
    download_game,
    extract_game_shots,
    get_pbp_shots,
    list_archive_games,
    load_game,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROCESSED = REPO_ROOT / "data" / "v2" / "processed"
SHOTS_DIR = str(_PROCESSED / "shots")
PBP_DIR = str(REPO_ROOT / "data" / "v2" / "raw" / "pbp")
FAIL_LOG = str(_PROCESSED / "_failures.csv")
EMPTY_LOG = str(_PROCESSED / "_empty_upstream.csv")
SEASON_FILE = str(_PROCESSED / "shots_tracking_1516_season.parquet")


def stem(filename: str) -> str:
    return filename[:-3] if filename.endswith(".7z") else filename


def out_path(filename: str) -> str:
    return os.path.join(SHOTS_DIR, f"{stem(filename)}.parquet")


def cached_pbp(game_id: str) -> pd.DataFrame:
    """Play-by-play shots for a game, cached to disk (resume never refetches)."""
    os.makedirs(PBP_DIR, exist_ok=True)
    p = os.path.join(PBP_DIR, f"{game_id}.parquet")
    if os.path.exists(p):
        try:
            return pd.read_parquet(p)
        except Exception:  # noqa: BLE001 - corrupt cache, just refetch
            pass
    df = get_pbp_shots(game_id)
    if df is not None and not df.empty:
        df.to_parquet(p, index=False)
    return df


def known_empty_games() -> set[str]:
    """Games confirmed to have zero tracking data upstream — never retry these."""
    if not os.path.exists(EMPTY_LOG):
        return set()
    return set(pd.read_csv(EMPTY_LOG)["file"])


def status() -> None:
    games = list_archive_games()
    done = [g for g in games if os.path.exists(out_path(g))]
    empty = known_empty_games()
    accounted = len(done) + len(empty)
    print(f"archive games      : {len(games)}")
    print(f"extracted          : {len(done)}  ({len(done) / len(games):.0%})")
    print(f"empty upstream     : {len(empty)}  (confirmed no tracking data in source — not failures)")
    print(f"remaining          : {len(games) - accounted}")
    if done:
        files = [out_path(g) for g in done]
        n = sum(len(pd.read_parquet(f, columns=["made"])) for f in files[:9999])
        print(f"shots so far       : {n:,}")
        size = sum(os.path.getsize(f) for f in files) / 1e6
        print(f"output size        : {size:.1f} MB")
    if os.path.exists(FAIL_LOG):
        print(f"open failures      : {len(pd.read_csv(FAIL_LOG))} (see {FAIL_LOG})")


def combine_season() -> pd.DataFrame:
    """Concatenate every per-game parquet into the season file stability.py reads."""
    files = sorted(f for f in os.listdir(SHOTS_DIR) if f.endswith(".parquet"))
    frames = [pd.read_parquet(os.path.join(SHOTS_DIR, f)) for f in files]
    season = pd.concat(frames, ignore_index=True)
    season.to_parquet(SEASON_FILE, index=False)
    hi = (season["release_confidence"] == "high").mean()
    print(f"season file: {len(season):,} shots from {len(files)} games "
          f"({hi:.1%} high-confidence) -> {SEASON_FILE}")
    return season


def record_failure(filename: str, reason: str) -> None:
    os.makedirs(os.path.dirname(FAIL_LOG), exist_ok=True)
    row = pd.DataFrame([{"file": filename, "reason": reason[:300], "ts": time.strftime("%F %T")}])
    row.to_csv(FAIL_LOG, mode="a", header=not os.path.exists(FAIL_LOG), index=False)


def record_empty_upstream(filename: str, reason: str) -> None:
    os.makedirs(os.path.dirname(EMPTY_LOG), exist_ok=True)
    row = pd.DataFrame([{"file": filename, "reason": reason[:300], "ts": time.strftime("%F %T")}])
    row.to_csv(EMPTY_LOG, mode="a", header=not os.path.exists(EMPTY_LOG), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--max-minutes", type=float, default=0, help="0 = no time limit")
    ap.add_argument("--purge-raw", action="store_true", help="delete each .7z after use")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--combine", action="store_true", help="only rebuild the season file")
    args = ap.parse_args()

    if args.status:
        status()
        return
    if args.combine:
        combine_season()
        return

    os.makedirs(SHOTS_DIR, exist_ok=True)
    games = list_archive_games()
    empty = known_empty_games()
    todo = [g for g in games if not os.path.exists(out_path(g)) and g not in empty]
    limited = bool(args.max_games or args.max_minutes)
    if args.max_games:
        todo = todo[: args.max_games]

    already = len([g for g in games if os.path.exists(out_path(g))])
    print(f"archive: {len(games)} games | already extracted: {already} | "
          f"known-empty (skipped): {len(empty)} | this run: {len(todo)}")
    if not todo:
        print("nothing to do — season already extracted.")
        status()
        combine_season()
        return

    started = time.time()
    ok = failed = shots_total = 0

    empty_upstream = 0

    for i, fn in enumerate(todo, 1):
        if args.max_minutes and (time.time() - started) / 60 >= args.max_minutes:
            print(f"\n[time limit {args.max_minutes} min reached — stopping cleanly]")
            break
        t0 = time.time()
        try:
            path = download_game(fn)
            try:
                game = load_game(path)
            except EmptyArchiveError:
                # genuine upstream gap (confirmed 2026-07-20: valid 7z, 0 files inside,
                # reproducible across re-downloads) — not a bug, log distinctly and move on
                empty_upstream += 1
                record_empty_upstream(fn, "archive has 0 files")
                print(f"  [{i}/{len(todo)}] {fn}: EMPTY UPSTREAM ARCHIVE — skipped (not a failure)")
                continue

            n_moments = sum(len(e.get("moments") or []) for e in game.get("events", []))
            if n_moments == 0:
                # archive loads but every event has zero tracking moments — same class of
                # gap as EmptyArchiveError, just a different shape (seen: CLE 01-23-2016)
                empty_upstream += 1
                record_empty_upstream(fn, "archive loads but 0 moments in any event")
                print(f"  [{i}/{len(todo)}] {fn}: NO TRACKING MOMENTS — skipped (not a failure)")
                del game
                gc.collect()
                continue

            gid = game.get("gameid")
            pbp = cached_pbp(gid)
            if pbp is None or pbp.empty:
                record_failure(fn, "no play-by-play")
                failed += 1
                print(f"  [{i}/{len(todo)}] {fn}: NO PBP — skipped")
                del game
                gc.collect()
                continue
            df = extract_game_shots(game, pbp)
            del game
            gc.collect()
            if df.empty:
                record_failure(fn, "no shots extracted")
                failed += 1
                print(f"  [{i}/{len(todo)}] {fn}: 0 shots — skipped")
                continue
            df.to_parquet(out_path(fn), index=False)
            if args.purge_raw and os.path.exists(path):
                os.remove(path)
            ok += 1
            shots_total += len(df)
            hi = (df["release_confidence"] == "high").mean()
            elapsed = time.time() - t0
            rate = (time.time() - started) / i
            eta_min = rate * (len(todo) - i) / 60
            print(f"  [{i}/{len(todo)}] {fn}: {len(df):3d} shots "
                  f"({hi:.0%} high-conf) {elapsed:4.1f}s | ETA {eta_min:5.1f} min", flush=True)
        except KeyboardInterrupt:
            print("\n[interrupted — progress saved, re-run to resume]")
            break
        except Exception as exc:  # noqa: BLE001 - keep going through bad games
            failed += 1
            record_failure(fn, f"{type(exc).__name__}: {exc}")
            print(f"  [{i}/{len(todo)}] {fn}: FAILED {type(exc).__name__}: {str(exc)[:90]}")
            traceback.print_exc(limit=1)
            gc.collect()

    mins = (time.time() - started) / 60
    print(f"\ndone this run: {ok} ok, {failed} failed, {empty_upstream} empty-upstream "
          f"(genuine gaps), {shots_total:,} shots, {mins:.1f} min")
    status()
    remaining = [g for g in games if not os.path.exists(out_path(g)) and g not in known_empty_games()]
    if not remaining and not limited:
        combine_season()
    elif not remaining:
        print("every game extracted; run with --combine to write the season file")


if __name__ == "__main__":
    main()
