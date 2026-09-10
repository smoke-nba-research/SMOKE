"""Release-frame detection: one frame per shot, and a distance sanity check."""

from __future__ import annotations

import pandas as pd

from src.pulls.tracking_1516 import (
    MAX_DIST_ERROR_FT,
    build_moment_index,
    extract_game_shots,
    find_release,
)

SHOOTER, DEFENDER, BALL = 100, 200, -1
TEAM_A, TEAM_B = 1, 2


def moment(period: int, ts_ms: int, clock: float, ball_z: float, ball_dx: float = 0.5):
    """One tracking frame: shooter at (20, 25), a defender 4 ft away, ball near the shooter."""
    return [
        period, ts_ms, clock, 14.0, None,
        [
            [-1, BALL, 20.0 + ball_dx, 25.0, ball_z],
            [TEAM_A, SHOOTER, 20.0, 25.0, 0.0],
            [TEAM_B, DEFENDER, 24.0, 25.0, 0.0],
        ],
    ]


def shooting_motion(period: int, clock_at_apex: float, start_ts: int):
    """Twelve frames where the ball rises to an apex in the shooter's hands then leaves."""
    frames = []
    for k in range(12):
        clock = clock_at_apex + (6 - k) * 0.04
        z = 6.0 + min(k, 6) * 0.5            # apex (9.0) at k == 6
        dx = 0.5 if k <= 6 else 0.5 + (k - 6) * 3.0   # ball leaves the hands after the apex
        frames.append(moment(period, start_ts + k * 40, round(clock, 2), z, dx))
    return frames


def test_find_release_picks_apex_and_honours_exclusion():
    moments = shooting_motion(1, 100.0, 1_000)
    idx, m, conf = find_release(moments, SHOOTER, period=1, clock_s=100.0)
    assert conf == "high"
    assert m[5][0][4] == 9.0                       # the apex frame
    idx2, m2, conf2 = find_release(moments, SHOOTER, period=1, clock_s=100.0, exclude={idx})
    assert idx2 != idx and m2[5][0][4] < 9.0       # the next-best held frame, not the same one


def test_two_shots_get_different_frames_and_bad_distance_is_low_confidence():
    # first motion at clock 100 s (apex 9 ft), second by the same player at 97 s (apex 8.5 ft)
    motion1 = shooting_motion(1, 100.0, 1_000)
    motion2 = [moment(1, 2_000 + k * 40, round(97.0 + (6 - k) * 0.04, 2), 5.5 + min(k, 6) * 0.5,
                      0.5 if k <= 6 else 0.5 + (k - 6) * 3.0) for k in range(12)]
    game = {"gameid": "g", "gamedate": "d", "events": [{"moments": motion1 + motion2}]}
    index = build_moment_index(game)
    assert len(index[1]) == 24

    # shooter at (20,25): 14.75 ft from the left hoop (5.25, 25)
    pbp = pd.DataFrame(
        {
            "EVENTNUM": [1, 2],
            "PERIOD": [1, 1],
            "PLAYER1_ID": [SHOOTER, SHOOTER],
            "PLAYER1_NAME": ["Test", "Test"],
            "made": [0, 1],
            "is_3pt": [0, 0],
            "nba_shot_dist_ft": [15, 15 + MAX_DIST_ERROR_FT + 3],   # second is deliberately wrong
            "clock_s": [100.0, 97.0],
        }
    )
    out = extract_game_shots(game, pbp)
    assert len(out) == 2
    assert out["game_clock"].nunique() == 2                    # no shared release frame
    assert out.loc[out.event_num == 1, "release_confidence"].item() == "high"
    assert out.loc[out.event_num == 2, "release_confidence"].item() == "low"
    assert out.loc[out.event_num == 1, "dist_error_ft"].item() < 1.0
