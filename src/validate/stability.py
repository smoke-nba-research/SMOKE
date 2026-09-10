"""Phase 2.4 — year-over-year stability of SMOKE (2014-15 -> 2015-16).

The single most important validation test in the whole project. The claim we need to
support: a player's shot-making-above-expectation is a real, repeatable skill, not
single-season noise — and specifically, that it is *at least as stable* as effective
field goal percentage, the efficiency metric it refines. Raw field goal percentage is
expected to be more stable still, because it inherits shot selection (a rim-runner
posts a high percentage every year), which SMOKE removes on purpose.

Published benchmark to land near: BBall Index reports year-over-year correlation of
~0.66 for their Shot Making metric vs ~0.28 for eFG% (Wyman, 2025), on full seasons.
Both of our seasons are partial (2014-15: 904 of 1,230 games; 2015-16: 631), which
depresses every correlation here relative to that benchmark.

Design
------
1. **Feature set = the aligned difficulty features only.** Shot distance, defender
   distance, shot clock, and shot type line up across the two extractions. Touch time
   and dribbles are excluded: the 2015-16 extraction measures the shooter's final
   touch while the 2014-15 file measures full-possession touch time, a definitional
   mismatch that would leak season identity into the model.

2. **Identical cleaning on both seasons.** The same shot-clock rule (game clock under
   24 s means the shot clock is off) and the same physical bounds are applied to both.

3. **Primary: per-season cross-fitted models.** Each season gets its own model on the
   four shared features, scored out of fold (five folds by game). Both seasons are
   then treated symmetrically and neither carries calibration drift from the other.

4. **Robustness: one model, both seasons.** The original design, a 2014-15 model
   applied unchanged to 2015-16. Reported alongside the primary, together with the
   calibration gap it produces on 2015-16 (the 2015-16 defender distances read about
   half a foot tighter than the league's, so expectations come out low).

5. **Only players who clear a shot floor in BOTH seasons** enter the correlation, and
   the result is re-run at every floor from 50 to 250 shots.

Run:
    .venv/Scripts/python.exe -m src.validate.stability
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.features.kaggle_shot_quality import (
    PLAYER_ID_COLUMN,
    PLAYER_NAME_COLUMN,
    SHOT_CLOCK_LENGTH_S,
    TARGET_COLUMN,
)
from src.models.build_model_outputs import SEED, build_boosted_model, load_scored_shots
from src.pulls._paths import REPO_ROOT
from src.pulls.build_tracking_season import SEASON_FILE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRACKING_1516 = REPO_ROOT / "data" / "v2" / "processed" / "shots_tracking_1516_season.parquet"
assert str(TRACKING_1516) == SEASON_FILE, "stability.py must read the file build_tracking_season writes"
OUT_DIR = REPO_ROOT / "data" / "v2" / "validation"

# aligned shared features only (see docstring point 1)
NUM_FEATURES = ["shot_dist_ft", "closest_def_dist_ft", "shot_clock"]
CAT_FEATURES = ["is_3pt"]
MODEL_COLS = NUM_FEATURES + CAT_FEATURES
TARGET = "made"

# physical bounds applied to both seasons: a shot is taken inside the court, and no
# defender is farther away than the court is long
MAX_DEF_DIST_FT = 40.0
MAX_SHOT_DIST_FT = 47.0

# player must clear this many shots in a season to be scored (both seasons are partial)
MIN_SHOTS = 100
THRESHOLDS = [50, 100, 150, 200, 250]
N_FOLDS = 5


def _shared_frame(
    player_id: pd.Series, player: pd.Series, shot_dist: pd.Series, def_dist: pd.Series,
    shot_clock: pd.Series, is_3pt: pd.Series, made: pd.Series, game_id: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "player_id": player_id.astype(int).to_numpy(),
            "player": player.astype(str).str.strip().str.lower().to_numpy(),
            "shot_dist_ft": shot_dist.to_numpy(dtype=float),
            "closest_def_dist_ft": def_dist.to_numpy(dtype=float),
            "shot_clock": shot_clock.to_numpy(dtype=float),
            "is_3pt": is_3pt.astype(int).to_numpy(),
            "made": made.astype(int).to_numpy(),
            "game_id": game_id.astype(str).to_numpy(),
        }
    )
    keep = (out["closest_def_dist_ft"] <= MAX_DEF_DIST_FT) & (out["shot_dist_ft"] <= MAX_SHOT_DIST_FT)
    return out[keep].reset_index(drop=True)


def load_1415_shared() -> pd.DataFrame:
    """2014-15 shots from the scored-shots artifact, in the shared schema.

    The artifact already carries the shot-clock rule and the cleaned touch times; the
    join key is NBA.com ``player_id``, which both sources carry (verified 2026-07-21).
    Names are not used as keys: the 2015-16 extraction stores last-name-only.
    """
    k = load_scored_shots()
    return _shared_frame(
        k[PLAYER_ID_COLUMN], k[PLAYER_NAME_COLUMN], k["SHOT_DIST"], k["CLOSE_DEF_DIST"],
        k["SHOT_CLOCK"], (k["PTS_TYPE"] == 3), k[TARGET_COLUMN], k["GAME_ID"],
    )


def load_1516_shared() -> pd.DataFrame:
    """2015-16 extracted shots, high-confidence releases only, in the shared schema."""
    if not TRACKING_1516.exists():
        raise FileNotFoundError(
            f"{TRACKING_1516} not found. Run `python -m src.pulls.build_tracking_season` "
            "(it writes the season file when every game is extracted)."
        )
    t = pd.read_parquet(TRACKING_1516)
    t = t[t["release_confidence"] == "high"].copy()
    shot_clock = pd.to_numeric(t["shot_clock"], errors="coerce")
    game_clock = pd.to_numeric(t["game_clock"], errors="coerce")
    off = shot_clock.isna() & (game_clock <= SHOT_CLOCK_LENGTH_S)
    shot_clock = shot_clock.where(~off, game_clock)
    return _shared_frame(
        t["player_id"], t["player_name"], t["shot_dist_ft"], t["closest_def_dist_ft"],
        shot_clock, t["is_3pt"], t["made"], t["game_id"],
    )


def build_model():
    """Expected-make model on the shared difficulty features (same estimator as v1)."""
    return build_boosted_model(NUM_FEATURES, CAT_FEATURES)


def cross_fit_season(shots: pd.DataFrame) -> np.ndarray:
    """Out-of-fold expected probability for one season, folds grouped by game."""
    expected = np.full(len(shots), np.nan)
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in folds.split(shots, shots[TARGET], shots["game_id"]):
        model = build_model().fit(shots.iloc[train_idx][MODEL_COLS], shots.iloc[train_idx][TARGET])
        expected[test_idx] = model.predict_proba(shots.iloc[test_idx][MODEL_COLS])[:, 1]
    assert not np.isnan(expected).any()
    return expected


def player_scores(shots: pd.DataFrame, expected_prob: np.ndarray, min_shots: int) -> pd.DataFrame:
    """Aggregate to per-player SMOKE (actual FG% minus expected FG%) + FG% and eFG%."""
    df = shots.copy()
    df["expected"] = expected_prob
    df["points"] = np.where(df["is_3pt"] == 1, df["made"] * 3, df["made"] * 2)
    g = (
        df.groupby("player_id")
        .agg(
            player=("player", "first"),
            shots=("made", "size"),
            fg_pct=("made", "mean"),
            expected_fg_pct=("expected", "mean"),
            points=("points", "sum"),
        )
        .reset_index()
    )
    g = g[g["shots"] >= min_shots].copy()
    g["efg_pct"] = (g["points"] / 2) / g["shots"]
    g["smoke"] = g["fg_pct"] - g["expected_fg_pct"]  # actual minus expected FG%
    return g


def two_season_panel(s14, e14, s16, e16, min_shots: int) -> pd.DataFrame:
    """Players clearing the floor in both seasons, one row each."""
    p14 = player_scores(s14, e14, min_shots).rename(columns=lambda c: c if c in ("player_id", "player") else f"{c}_14")
    p16 = player_scores(s16, e16, min_shots).drop(columns=["player"]).rename(
        columns=lambda c: c if c == "player_id" else f"{c}_16"
    )
    return p14.merge(p16, on="player_id", how="inner")


def corr_rows(panel: pd.DataFrame, design: str) -> list[dict]:
    rows = []
    for label, a, b in [
        ("SMOKE", "smoke_14", "smoke_16"),
        ("raw FG%", "fg_pct_14", "fg_pct_16"),
        ("eFG%", "efg_pct_14", "efg_pct_16"),
    ]:
        pear = stats.pearsonr(panel[a], panel[b])
        spear = stats.spearmanr(panel[a], panel[b])
        rows.append(
            {
                "design": design,
                "metric": label,
                "n": len(panel),
                "pearson_r": round(float(pear.statistic), 3),
                "pearson_p": float(pear.pvalue),
                "spearman_rho": round(float(spear.statistic), 3),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("PHASE 2.4 — SMOKE year-over-year stability, 2014-15 -> 2015-16")
    print("=" * 74)

    s14, s16 = load_1415_shared(), load_1516_shared()
    for label, s in [("2014-15", s14), ("2015-16", s16)]:
        print(f"  {label}: {len(s):,} shots, {s['game_id'].nunique()} games | mean shot dist {s['shot_dist_ft'].mean():.2f} ft, "
              f"defender {s['closest_def_dist_ft'].mean():.2f} ft, shot clock {s['shot_clock'].mean():.2f} s, "
              f"3PT share {s['is_3pt'].mean():.3f}, FG% {s['made'].mean():.4f}")

    # primary: per-season cross-fitted expectations
    e14, e16 = cross_fit_season(s14), cross_fit_season(s16)
    # robustness: the 2014-15 model applied to both seasons
    one_model = build_model().fit(s14[MODEL_COLS], s14[TARGET])
    e14_one, e16_one = one_model.predict_proba(s14[MODEL_COLS])[:, 1], one_model.predict_proba(s16[MODEL_COLS])[:, 1]

    calibration = pd.DataFrame(
        [
            {"design": "per-season cross-fit", "season": "2014-15", "mean_expected": e14.mean(), "mean_actual": s14[TARGET].mean(), "auc": roc_auc_score(s14[TARGET], e14)},
            {"design": "per-season cross-fit", "season": "2015-16", "mean_expected": e16.mean(), "mean_actual": s16[TARGET].mean(), "auc": roc_auc_score(s16[TARGET], e16)},
            {"design": "one model (2014-15)", "season": "2014-15 (in-sample)", "mean_expected": e14_one.mean(), "mean_actual": s14[TARGET].mean(), "auc": roc_auc_score(s14[TARGET], e14_one)},
            {"design": "one model (2014-15)", "season": "2015-16", "mean_expected": e16_one.mean(), "mean_actual": s16[TARGET].mean(), "auc": roc_auc_score(s16[TARGET], e16_one)},
        ]
    )
    calibration["gap"] = calibration["mean_expected"] - calibration["mean_actual"]
    print("\n  calibration by design (gap = mean expected - mean actual; a nonzero gap shifts every SMOKE in that season):")
    for r in calibration.itertuples():
        print(f"    {r.design:22s} {r.season:20s} expected {r.mean_expected:.4f} actual {r.mean_actual:.4f} gap {r.gap:+.4f}  AUC {r.auc:.4f}")

    panel = two_season_panel(s14, e14, s16, e16, MIN_SHOTS)
    panel_one = two_season_panel(s14, e14_one, s16, e16_one, MIN_SHOTS)
    panel = panel.merge(panel_one[["player_id", "smoke_14", "smoke_16"]].rename(
        columns={"smoke_14": "smoke_14_onemodel", "smoke_16": "smoke_16_onemodel"}), on="player_id")
    print(f"\n  players clearing {MIN_SHOTS} shots in BOTH seasons (the stability sample): {len(panel)}")
    if len(panel) < 30:
        print("  !! sample too small to trust — investigate")

    rows = corr_rows(panel, "per-season cross-fit") + corr_rows(panel_one, "one model (2014-15)")
    res = pd.DataFrame(rows)
    print("\n  YEAR-OVER-YEAR STABILITY (higher = more repeatable skill)")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['design']:22s} {r['metric']:8s}  r = {r['pearson_r']:+.3f}   rho = {r['spearman_rho']:+.3f}   (n={r['n']})")
    print("  " + "-" * 70)

    primary = res[res["design"] == "per-season cross-fit"].set_index("metric")["pearson_r"]
    smoke_r, fg_r, efg_r = primary["SMOKE"], primary["raw FG%"], primary["eFG%"]
    print("\n  READ (primary design):")
    print(f"    SMOKE YoY r = {smoke_r:+.3f}  vs  eFG% {efg_r:+.3f}  /  raw FG% {fg_r:+.3f}")
    if smoke_r >= efg_r:
        print("    -> SMOKE is AT LEAST AS STABLE as eFG%, the metric it refines. Claim holds.")
    elif smoke_r >= 0.4:
        print("    -> SMOKE is stable and in a defensible range, though below eFG%. Nuance needed.")
    else:
        print("    -> SMOKE stability is weak. Red flag — investigate before publishing.")
    print("    Raw FG% is expected to be more stable than both: it carries shot selection, i.e. role")
    print("    consistency, which SMOKE strips out by design.")

    sweep = []
    for floor in THRESHOLDS:
        p = two_season_panel(s14, e14, s16, e16, floor)
        for row in corr_rows(p, "per-season cross-fit"):
            sweep.append({"min_shots": floor, **row})
    sweep = pd.DataFrame(sweep)
    print("\n  threshold sweep (primary design):")
    for floor in THRESHOLDS:
        sub = sweep[sweep["min_shots"] == floor].set_index("metric")
        print(f"    floor {floor:3d}: SMOKE {sub.loc['SMOKE', 'pearson_r']:.3f} | eFG% {sub.loc['eFG%', 'pearson_r']:.3f} "
              f"| FG% {sub.loc['raw FG%', 'pearson_r']:.3f} | n={sub.loc['SMOKE', 'n']}")

    panel.to_parquet(OUT_DIR / "stability_2season_players.parquet", index=False)
    res.to_csv(OUT_DIR / "stability_correlations.csv", index=False)
    sweep.to_csv(OUT_DIR / "stability_threshold_sweep.csv", index=False)
    calibration.to_csv(OUT_DIR / "stability_calibration.csv", index=False)
    print(f"\n  wrote {OUT_DIR / 'stability_correlations.csv'}")
    print(f"  wrote {OUT_DIR / 'stability_threshold_sweep.csv'}")
    print(f"  wrote {OUT_DIR / 'stability_calibration.csv'}")
    print(f"  wrote {OUT_DIR / 'stability_2season_players.parquet'}  ({len(panel)} players)")


if __name__ == "__main__":
    main()
