"""The Kaggle MATCHUP string leads with the shooter's team on every row."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.kaggle_shot_quality import (
    KAGGLE_SHOT_LOG,
    OPPONENT_COLUMN,
    TEAM_COLUMN,
    add_derived_features,
    extract_opponent_team,
    extract_shooting_team,
    load_kaggle_shot_logs,
    parse_matchup,
)


def test_away_game_first_token_is_shooter():
    assert parse_matchup("MAR 04, 2015 - CHA @ BKN") == ("CHA", "BKN")
    assert extract_shooting_team("MAR 04, 2015 - CHA @ BKN") == "CHA"
    assert extract_opponent_team("MAR 04, 2015 - CHA @ BKN") == "BKN"


def test_home_game_first_token_is_still_shooter():
    assert parse_matchup("MAR 02, 2015 - CHA vs. LAL") == ("CHA", "LAL")
    assert extract_shooting_team("MAR 02, 2015 - CHA vs. LAL") == "CHA"
    assert extract_opponent_team("MAR 02, 2015 - CHA vs. LAL") == "LAL"


def test_unparseable_returns_none():
    assert parse_matchup(float("nan")) is None
    assert parse_matchup("garbage") is None
    assert extract_shooting_team("garbage") is None


@pytest.mark.skipif(not KAGGLE_SHOT_LOG.exists(), reason="Kaggle shot logs not present")
def test_real_data_one_team_per_player_game():
    shots = add_derived_features(load_kaggle_shot_logs())
    assert shots[TEAM_COLUMN].isna().sum() == 0
    assert shots[OPPONENT_COLUMN].isna().sum() == 0
    # a player shoots for exactly one team in a game, on both home and away rows
    per_game = shots.groupby(["GAME_ID", "player_id"])[TEAM_COLUMN].nunique()
    assert per_game.max() == 1
    # over a season a player has at most a handful of teams (trades), never dozens
    per_season = shots.groupby("player_id")[TEAM_COLUMN].nunique()
    assert per_season.max() <= 3
    # the opponent is never the shooter's own team
    assert (shots[TEAM_COLUMN] == shots[OPPONENT_COLUMN]).sum() == 0
    # home rows: the shooter's team is the home team, so the location flag agrees
    home = shots[shots["LOCATION"] == "H"]
    assert home["MATCHUP"].str.contains("vs.").all()
    assert isinstance(home, pd.DataFrame)
