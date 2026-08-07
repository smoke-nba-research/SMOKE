"""Pull the shot-detail data, season by season (this is our main dataset).

How it works: we walk every team within every season and grab that team's
shots for both Regular Season and Playoffs. ShotChartDetail gives us
shot-level rows with court coordinates, shot zone, distance, and outcome.

This is the slowest pull we have — about 30 teams x 2 season types per
season, with a polite sleep between each call. Figure ~30-40 seconds per
season, and the full 2013-present pull takes around 8-10 minutes.

Idempotent — already-pulled seasons get skipped.

Output: data/raw/shots/shots_{SEASON}.parquet  (e.g. shots_2023-24.parquet)
"""

from __future__ import annotations
import time
import pandas as pd
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import shotchartdetail

from src.pulls._paths import SHOTS_DIR, all_seasons, ensure_dirs

SLEEP_SECONDS = 0.6


def fetch_team_season_shots(
    team_id: int, season: str, season_type: str
) -> pd.DataFrame:
    """Grab one team's shots for one season + season type."""
    endpoint = shotchartdetail.ShotChartDetail(
        team_id=team_id,
        player_id=0,  # 0 means all players on the team
        season_nullable=season,
        season_type_all_star=season_type,
        context_measure_simple="FGA",
    )
    return endpoint.get_data_frames()[0]


def fetch_season_shots(season: str) -> pd.DataFrame:
    """Pull every shot for one season by walking through the teams."""
    team_ids = [t["id"] for t in static_teams.get_teams()]
    frames = []
    for i, team_id in enumerate(team_ids, 1):
        for season_type in ("Regular Season", "Playoffs"):
            try:
                df = fetch_team_season_shots(team_id, season, season_type)
                if not df.empty:
                    df["SEASON_TYPE"] = season_type
                    frames.append(df)
            except Exception as exc:
                print(f"  Warning: team {team_id} {season} {season_type} failed: {exc}")
            time.sleep(SLEEP_SECONDS)
        print(f"  [{i}/{len(team_ids)}] team {team_id} done")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(seasons: list[str] | None = None) -> None:
    ensure_dirs()
    seasons = seasons or all_seasons()
    for season in seasons:
        out = SHOTS_DIR / f"shots_{season}.parquet"
        if out.exists():
            print(f"Skip {season} (already cached at {out.name})")
            continue
        print(f"Fetching shots for {season}...")
        df = fetch_season_shots(season)
        if df.empty:
            print(f"  No data returned for {season}; skipping write")
            continue
        df.to_parquet(out, index=False)
        print(f"  Wrote {len(df)} rows to {out.name}")


if __name__ == "__main__":
    main()
