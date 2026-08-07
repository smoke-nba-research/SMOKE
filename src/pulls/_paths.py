"""Shared paths for the pull scripts.

One place to set where our raw data ends up, so the individual modules
don't each reinvent it.
"""

from __future__ import annotations
from pathlib import Path

# Repo root (this file is at src/pulls/_paths.py, so go up 3 levels)
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = REPO_ROOT / "data"
RAW = DATA_ROOT / "raw"

SHOTS_DIR = RAW / "shots"
GAMES_DIR = RAW / "games"
PLAYERS_FILE = RAW / "players.parquet"
TEAMS_FILE = RAW / "teams.parquet"
KAGGLE_DIR = RAW / "kaggle"


def ensure_dirs() -> None:
    """Make sure all the raw data subdirectories exist."""
    for d in (SHOTS_DIR, GAMES_DIR, KAGGLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Default season range: 2013-14 (start of the tracking-data era) through current.
# Bump DEFAULT_END_SEASON every year as new seasons wrap up.
DEFAULT_START_SEASON = 2013
DEFAULT_END_SEASON = 2025  # the 2025-26 season


def season_str(start_year: int) -> str:
    """Format a season the NBA way: 'YYYY-YY'.

    >>> season_str(2023)
    '2023-24'
    """
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def all_seasons(
    start: int = DEFAULT_START_SEASON, end: int = DEFAULT_END_SEASON
) -> list[str]:
    """Give back all season strings from start to end, inclusive."""
    return [season_str(y) for y in range(start, end + 1)]
