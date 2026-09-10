"""Feature policy: no game outcomes, end-of-period shot clocks, no negative touch times."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.kaggle_shot_quality import (
    KAGGLE_SHOT_LOG,
    MODEL_COLUMNS,
    OUTCOME_COLUMNS,
    add_derived_features,
    build_modeling_frame,
    fill_shot_clock,
    load_kaggle_shot_logs,
)


def test_game_outcomes_are_not_features():
    for col in OUTCOME_COLUMNS:
        assert col not in MODEL_COLUMNS


def test_fill_shot_clock_uses_game_clock_when_off():
    shot_clock = pd.Series([10.0, np.nan, np.nan, np.nan])
    game_clock = pd.Series([300.0, 12.0, 24.0, 90.0])
    filled, off = fill_shot_clock(shot_clock, game_clock)
    assert filled.tolist()[:3] == [10.0, 12.0, 24.0]
    assert np.isnan(filled.iloc[3])            # a genuine gap stays missing
    assert off.tolist() == [False, True, True, False]


def _frame(**overrides):
    base = {
        "MATCHUP": ["MAR 04, 2015 - CHA @ BKN"],
        "LOCATION": ["A"],
        "GAME_CLOCK": ["0:14"],
        "SHOT_CLOCK": [np.nan],
        "TOUCH_TIME": [-5.0],
        "SHOT_DIST": [10.0],
        "SHOT_RESULT": ["made"],
        "player_name": ["x"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_add_derived_features_applies_corrections():
    out = add_derived_features(_frame())
    assert out["SHOT_CLOCK"].iloc[0] == 14.0
    assert out["SHOT_CLOCK_OFF"].iloc[0] == "clock_off"
    assert out["LATE_CLOCK"].iloc[0] == "normal_clock"
    assert np.isnan(out["TOUCH_TIME"].iloc[0])
    assert pd.isna(out["TOUCH_TIME_BUCKET"].iloc[0])
    assert out["GAME_DATE"].iloc[0] == pd.Timestamp("2015-03-04")


@pytest.mark.skipif(not KAGGLE_SHOT_LOG.exists(), reason="Kaggle shot logs not present")
def test_real_data_corrections():
    shots = build_modeling_frame(load_kaggle_shot_logs())
    assert (shots["TOUCH_TIME"] < 0).sum() == 0
    assert (shots["SHOT_CLOCK_OFF"] == "clock_off").sum() == 3554
    assert shots["SHOT_CLOCK"].isna().sum() == 5567 - 3554
    assert shots["GAME_DATE"].min() == pd.Timestamp("2014-10-28")
    assert shots["GAME_DATE"].max() == pd.Timestamp("2015-03-04")
    assert shots["GAME_ID"].nunique() == 904
