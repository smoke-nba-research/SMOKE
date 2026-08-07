"""Reusable helpers for the Kaggle 2014-15 shot-quality models.

The point of this module is to keep the modeling notebook focused on
analysis instead of repeating the same cleanup and feature logic.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.pulls._paths import REPO_ROOT

KAGGLE_SHOT_LOG = (
    REPO_ROOT / "data" / "raw" / "kaggle" / "shot_logs_2014_15" / "shot_logs.csv"
)

TARGET_COLUMN = "SHOT_MADE"
PLAYER_NAME_COLUMN = "player_name"
TEAM_COLUMN = "SHOOTING_TEAM"

NUMERIC_FEATURES = [
    "FINAL_MARGIN",
    "SHOT_NUMBER",
    "PERIOD",
    "SHOT_CLOCK",
    "DRIBBLES",
    "TOUCH_TIME",
    "SHOT_DIST",
    "CLOSE_DEF_DIST",
    "GAME_CLOCK_SECONDS",
]

CATEGORICAL_FEATURES = [
    "LOCATION",
    "W",
    "PTS_TYPE",
    "SHOT_DIST_ZONE",
    "TOUCH_TIME_BUCKET",
    "LATE_CLOCK",
]

MODEL_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MATCHUP_PATTERN = re.compile(
    r"\w{3} \d{2}, \d{4} - ([A-Z]{2,3}) (?:@|vs\.) ([A-Z]{2,3})"
)


def load_kaggle_shot_logs(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the 2014-15 Kaggle shot log dataset."""
    path = csv_path or KAGGLE_SHOT_LOG
    return pd.read_csv(path)


def game_clock_to_seconds(clock_value: str) -> float:
    """Convert MM:SS game clock strings into seconds remaining in the period."""
    if pd.isna(clock_value):
        return np.nan

    minutes, seconds = str(clock_value).split(":")
    return int(minutes) * 60 + int(seconds)


def extract_shooting_team(matchup: str, location: str) -> str | None:
    """Recover the shooting team abbreviation from the matchup string.

    Kaggle stores the matchup as text like ``CHA @ BKN`` or ``CHA vs. LAL``.
    We use the home/away flag to back out which team took the shot.
    """
    if pd.isna(matchup) or pd.isna(location):
        return None

    match = MATCHUP_PATTERN.search(str(matchup))
    if not match:
        return None

    away_team, home_team = match.groups()
    return home_team if location == "H" else away_team


def add_derived_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Add the reusable shot-quality features used across the models."""
    enriched = shots.copy()

    enriched["GAME_CLOCK_SECONDS"] = enriched["GAME_CLOCK"].map(game_clock_to_seconds)
    enriched[TARGET_COLUMN] = (enriched["SHOT_RESULT"] == "made").astype(int)
    enriched[TEAM_COLUMN] = [
        extract_shooting_team(matchup, location)
        for matchup, location in zip(
            enriched["MATCHUP"], enriched["LOCATION"], strict=False
        )
    ]

    # A few human-readable buckets make the residual analysis easier to explain.
    enriched["SHOT_DIST_ZONE"] = pd.cut(
        enriched["SHOT_DIST"],
        bins=[-np.inf, 4, 14, 23.75, np.inf],
        labels=["at_rim", "short_midrange", "long_midrange", "three_plus"],
    ).astype("object")

    enriched["TOUCH_TIME_BUCKET"] = pd.cut(
        enriched["TOUCH_TIME"],
        bins=[-np.inf, 2, 6, np.inf],
        labels=["quick", "balanced", "hold"],
    ).astype("object")

    enriched["LATE_CLOCK"] = np.where(
        enriched["SHOT_CLOCK"].fillna(24) <= 4, "late_clock", "normal_clock"
    )

    return enriched


def build_modeling_frame(shots: pd.DataFrame) -> pd.DataFrame:
    """Return the feature-ready table used by the notebook models.

    We keep the row filter intentionally light so the sample stays above
    the course's 100K observation floor.
    """
    modeling_frame = add_derived_features(shots)
    modeling_frame = modeling_frame.dropna(subset=["SHOT_RESULT", PLAYER_NAME_COLUMN])
    return modeling_frame


def build_residual_table(
    scored_shots: pd.DataFrame,
    group_column: str,
    min_shots: int,
) -> pd.DataFrame:
    """Aggregate actual vs expected shooting results for players or teams."""
    summary = (
        scored_shots.groupby(group_column, dropna=False)
        .agg(
            shots=(TARGET_COLUMN, "size"),
            actual_makes=(TARGET_COLUMN, "sum"),
            expected_makes=("EXPECTED_MAKE_PROB", "sum"),
            actual_fg_pct=(TARGET_COLUMN, "mean"),
            expected_fg_pct=("EXPECTED_MAKE_PROB", "mean"),
        )
        .reset_index()
    )

    # Small samples can jump around a lot, so trim them before ranking.
    summary = summary[summary["shots"] >= min_shots].copy()
    summary["makes_above_expected"] = (
        summary["actual_makes"] - summary["expected_makes"]
    )
    summary["fg_pct_above_expected"] = (
        summary["actual_fg_pct"] - summary["expected_fg_pct"]
    )
    summary = summary.sort_values("makes_above_expected", ascending=False).reset_index(
        drop=True
    )

    return summary
