"""Pull the league-wide game logs, season by season.

Uses LeagueGameLog, which gives us one row per (team, game). We grab both
Regular Season and Playoffs for each season. Idempotent — already-pulled
seasons get skipped.

Output: data/raw/games/games_{SEASON}.parquet  (e.g. games_2023-24.parquet)
"""

from __future__ import annotations
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from src.pulls._paths import GAMES_DIR, all_seasons, ensure_dirs

# Sleep between API calls so we don't get rate-limited — NBA.com is strict
SLEEP_SECONDS = 0.6


def fetch_season_games(season: str) -> pd.DataFrame:
    """Pull all the games (regular + playoffs) for one season."""
    frames = []
    for season_type in ("Regular Season", "Playoffs"):
        try:
            endpoint = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                league_id="00",
            )
            df = endpoint.get_data_frames()[0]
            df["SEASON_TYPE"] = season_type
            frames.append(df)
        except Exception as exc:  # broad catch — the API can fail in weird ways
            print(f"  Warning: {season} {season_type} failed: {exc}")
        time.sleep(SLEEP_SECONDS)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(seasons: list[str] | None = None) -> None:
    ensure_dirs()
    seasons = seasons or all_seasons()
    for season in seasons:
        out = GAMES_DIR / f"games_{season}.parquet"
        if out.exists():
            print(f"Skip {season} (already cached at {out.name})")
            continue
        print(f"Fetching games for {season}...")
        df = fetch_season_games(season)
        if df.empty:
            print(f"  No data returned for {season}; skipping write")
            continue
        df.to_parquet(out, index=False)
        print(f"  Wrote {len(df)} rows to {out.name}")


if __name__ == "__main__":
    main()
