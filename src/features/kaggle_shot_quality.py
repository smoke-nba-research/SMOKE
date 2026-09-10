"""Feature construction for the 2014-15 Kaggle shot logs.

Everything the expected-make model sees is defined here, in one place, so the paper's
feature list and the code cannot drift apart.

Feature policy
--------------
Only quantities knowable at the moment of release enter the model. The Kaggle file
also carries FINAL_MARGIN and W (the game's final score margin and result); those are
outcomes shared by every shot a team takes that night, and the v1 capstone model used
them. They were removed on 2026-09-09: alone they reach a hold-out AUC of 0.53, and
they raised expectations for players on winning teams, which is not shot difficulty.

Data corrections applied before modeling
----------------------------------------
* SHOT_CLOCK is missing on 5,567 shots; 3,554 of them have 24 s or less on the game
  clock, where the shot clock is switched off and the game clock is the shot clock.
  Those are filled with the game clock and flagged (SHOT_CLOCK_OFF); the remaining
  gaps are left for median imputation inside the model pipeline.
* TOUCH_TIME is negative on 312 rows (a known artifact of this dataset, minimum
  -163.6 s). Negative values are set to missing rather than fed to the model.
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
PLAYER_ID_COLUMN = "player_id"
TEAM_COLUMN = "SHOOTING_TEAM"
OPPONENT_COLUMN = "OPP_TEAM"
DATE_COLUMN = "GAME_DATE"

# Shot difficulty and in-game context, all known at release.
NUMERIC_FEATURES = [
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
    "PTS_TYPE",
    "SHOT_DIST_ZONE",
    "TOUCH_TIME_BUCKET",
    "LATE_CLOCK",
    "SHOT_CLOCK_OFF",
]

MODEL_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns from the source file that describe the game's outcome, never features.
OUTCOME_COLUMNS = ["FINAL_MARGIN", "W"]

# The Kaggle MATCHUP string is written from the shooter's perspective: the first
# abbreviation is always the shooter's team, whether the game reads "CHA @ BKN"
# (away) or "CHA vs. LAL" (home). The second is always the opponent. LOCATION is
# not needed to tell the two apart.
MATCHUP_PATTERN = re.compile(
    r"\w{3} \d{2}, \d{4} - ([A-Z]{2,3}) (?:@|vs\.) ([A-Z]{2,3})"
)
MATCHUP_DATE_PATTERN = re.compile(r"^(\w{3} \d{2}, \d{4})")

# the shot clock is off, and the game clock governs, at or under this many seconds
SHOT_CLOCK_LENGTH_S = 24.0


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


def parse_matchup(matchup: str) -> tuple[str, str] | None:
    """Return ``(shooting team, opponent)`` from a Kaggle MATCHUP string."""
    if pd.isna(matchup):
        return None
    match = MATCHUP_PATTERN.search(str(matchup))
    if not match:
        return None
    return match.group(1), match.group(2)


def extract_shooting_team(matchup: str) -> str | None:
    """The team that took the shot: the first abbreviation in the matchup string."""
    parsed = parse_matchup(matchup)
    return parsed[0] if parsed else None


def extract_opponent_team(matchup: str) -> str | None:
    """The team defending the shot: the second abbreviation in the matchup string."""
    parsed = parse_matchup(matchup)
    return parsed[1] if parsed else None


def fill_shot_clock(shot_clock: pd.Series, game_clock_seconds: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Apply the end-of-period rule. Returns (filled shot clock, shot-clock-off flag)."""
    off = shot_clock.isna() & (game_clock_seconds <= SHOT_CLOCK_LENGTH_S)
    filled = shot_clock.where(~off, game_clock_seconds)
    return filled, off


def add_derived_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Add the reusable shot-quality features used across the models."""
    enriched = shots.copy()

    enriched["GAME_CLOCK_SECONDS"] = enriched["GAME_CLOCK"].map(game_clock_to_seconds)
    enriched[TARGET_COLUMN] = (enriched["SHOT_RESULT"] == "made").astype(int)
    teams = enriched["MATCHUP"].str.extract(MATCHUP_PATTERN)
    enriched[TEAM_COLUMN] = teams[0]
    enriched[OPPONENT_COLUMN] = teams[1]
    enriched[DATE_COLUMN] = pd.to_datetime(
        enriched["MATCHUP"].str.extract(MATCHUP_DATE_PATTERN)[0], format="%b %d, %Y"
    )

    enriched["SHOT_CLOCK"], off = fill_shot_clock(
        enriched["SHOT_CLOCK"], enriched["GAME_CLOCK_SECONDS"]
    )
    enriched["SHOT_CLOCK_OFF"] = np.where(off, "clock_off", "clock_on")
    enriched["TOUCH_TIME"] = enriched["TOUCH_TIME"].where(enriched["TOUCH_TIME"] >= 0)

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
        enriched["SHOT_CLOCK"].fillna(SHOT_CLOCK_LENGTH_S) <= 4, "late_clock", "normal_clock"
    )

    return enriched


def build_modeling_frame(shots: pd.DataFrame) -> pd.DataFrame:
    """Return the feature-ready table used by the models."""
    modeling_frame = add_derived_features(shots)
    modeling_frame = modeling_frame.dropna(subset=["SHOT_RESULT", PLAYER_NAME_COLUMN])
    return modeling_frame


def build_residual_table(
    scored_shots: pd.DataFrame,
    group_columns: str | list[str],
    min_shots: int,
    expected_column: str = "expected",
) -> pd.DataFrame:
    """Aggregate actual versus expected makes for players or teams.

    ``smoke_total`` is makes above expectation and ``smoke_rate`` is field goal
    percentage above expectation; both are the raw (unshrunk) quantities.
    """
    keys = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    summary = (
        scored_shots.groupby(keys, dropna=False)
        .agg(
            shots=(TARGET_COLUMN, "size"),
            actual_makes=(TARGET_COLUMN, "sum"),
            expected_makes=(expected_column, "sum"),
            actual_fg_pct=(TARGET_COLUMN, "mean"),
            expected_fg_pct=(expected_column, "mean"),
        )
        .reset_index()
    )

    # Small samples can jump around a lot, so trim them before ranking.
    summary = summary[summary["shots"] >= min_shots].copy()
    summary["smoke_total"] = summary["actual_makes"] - summary["expected_makes"]
    summary["smoke_rate"] = summary["actual_fg_pct"] - summary["expected_fg_pct"]
    summary = summary.sort_values("smoke_total", ascending=False).reset_index(drop=True)

    return summary
