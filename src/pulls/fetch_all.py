"""Run every pull in the right order.

Usage:
    python -m src.pulls.fetch_all

Each pull is idempotent, so re-running is safe. If you want to force a
re-pull for a specific season, delete that season's parquet file first.

Total runtime on a fresh machine: ~10-15 minutes (most of it is the shot pull).
"""

from __future__ import annotations
import time

from src.pulls import nba_teams, nba_players, nba_games, nba_shots, kaggle_datasets


def main() -> None:
    start = time.time()
    print("=" * 60)
    print("Step 1/5: NBA teams (static metadata, instant)")
    print("=" * 60)
    nba_teams.main()

    print("\n" + "=" * 60)
    print("Step 2/5: NBA players (one API call)")
    print("=" * 60)
    nba_players.main()

    print("\n" + "=" * 60)
    print("Step 3/5: Game logs (~30 seconds per season)")
    print("=" * 60)
    nba_games.main()

    print("\n" + "=" * 60)
    print("Step 4/5: Shot detail (~30-40 seconds per season — slowest step)")
    print("=" * 60)
    nba_shots.main()

    print("\n" + "=" * 60)
    print("Step 5/5: Kaggle datasets (needs ~/.kaggle/kaggle.json)")
    print("=" * 60)
    kaggle_datasets.main()

    elapsed = time.time() - start
    print(f"\nAll pulls done in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
