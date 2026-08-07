"""Pull every NBA player, historical + current.

Uses CommonAllPlayers with is_only_current_season=0 so we get every player
who's ever played in an NBA game. Output: data/raw/players.parquet
"""

from __future__ import annotations
import pandas as pd
from nba_api.stats.endpoints import commonallplayers

from src.pulls._paths import PLAYERS_FILE, ensure_dirs


def fetch_players() -> pd.DataFrame:
    """Give back a DataFrame of every NBA player, past and present."""
    endpoint = commonallplayers.CommonAllPlayers(
        is_only_current_season=0,
        league_id="00",  # NBA
    )
    return endpoint.get_data_frames()[0]


def main() -> None:
    ensure_dirs()
    print("Fetching all NBA players (historical + current)...")
    df = fetch_players()
    df.to_parquet(PLAYERS_FILE, index=False)
    print(f"Wrote {len(df)} players to {PLAYERS_FILE}")


if __name__ == "__main__":
    main()
