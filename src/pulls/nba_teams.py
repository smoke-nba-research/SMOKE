"""Pull the list of NBA teams.

Teams are static metadata that comes bundled with nba_api, so this doesn't
actually hit the API — it just reads from memory. Output: data/raw/teams.parquet
"""

from __future__ import annotations
import pandas as pd
from nba_api.stats.static import teams

from src.pulls._paths import TEAMS_FILE, ensure_dirs


def fetch_teams() -> pd.DataFrame:
    """Give back a DataFrame of every NBA team."""
    return pd.DataFrame(teams.get_teams())


def main() -> None:
    ensure_dirs()
    df = fetch_teams()
    df.to_parquet(TEAMS_FILE, index=False)
    print(f"Wrote {len(df)} teams to {TEAMS_FILE}")


if __name__ == "__main__":
    main()
