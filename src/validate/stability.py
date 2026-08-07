"""Phase 2.4 — year-over-year stability of SMOKE (2014-15 -> 2015-16).

The single most important validation test in the whole project. The claim we need to
support: a player's shot-making-above-expectation is a real, repeatable skill, not
single-season noise — and specifically, that it is *at least as stable* as the
efficiency metrics teams already use (FG%, eFG%), while measuring something cleaner.

Published benchmark to land near: BBall Index reports year-over-year
correlation of ~0.66 for their Shot Making metric vs ~0.28 for raw eFG%.

Design decisions that make this an honest test
----------------------------------------------
1. **One model, both seasons, identical features.** A low YoY correlation must mean
   "the skill isn't stable," never "the two feature pipelines differ." So we train a
   single expected-make model on 2014-15 and apply the *same* model to both seasons,
   using only features whose distributions we've verified match across seasons.

2. **Feature set = the aligned difficulty features only.** Verified 2026-07-21: shot
   distance, defender distance, shot clock, and shot type line up almost exactly across
   the two seasons (means within ~0.1-0.4). We deliberately EXCLUDE touch_time and
   dribbles: our 2015-16 extraction measures the shooter's *final* touch, while the
   Kaggle 2014-15 set measured full-possession touch time — a genuine definitional
   mismatch (means 1.7 vs 2.8s) that would leak season identity into the model. Better a
   slightly weaker model on trustworthy shared features than a stronger one that cheats.

3. **SMOKE per player-season = actual FG% minus model-expected FG%.** Same construction
   as v1 (fg_pct_above_expected), just applied consistently to both seasons.

4. **Only players who clear a shot floor in BOTH seasons** enter the correlation, so we
   compare the same people to themselves. Half-season caveat noted: 2015-16 is ~52% of
   the season, so its per-player samples are smaller — we report n and set the floor
   accordingly rather than pretending both seasons are full.

Run:
    .venv/Scripts/python.exe -m src.validate.stability
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.features.kaggle_shot_quality import add_derived_features, load_kaggle_shot_logs
from src.pulls._paths import REPO_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRACKING_1516 = REPO_ROOT / "data" / "v2" / "processed" / "shots_tracking_1516_season.parquet"
OUT_DIR = REPO_ROOT / "data" / "v2" / "validation"

# aligned shared features only (see docstring point 2)
NUM_FEATURES = ["shot_dist_ft", "closest_def_dist_ft", "shot_clock"]
CAT_FEATURES = ["is_3pt"]
MODEL_COLS = NUM_FEATURES + CAT_FEATURES
TARGET = "made"

# player must clear this many shots in a season to be scored (2015-16 is a half season)
MIN_SHOTS = 100


def load_1415_shared() -> pd.DataFrame:
    """2014-15 Kaggle shots, renamed to the shared feature schema.

    Join key is NBA.com ``player_id`` — both sources carry it and their ID spaces match
    (verified 2026-07-21). We deliberately do NOT join on name: the 2015-16 tracking
    extraction stores last-name-only (from PlayByPlayV3), which collides ("Paul", "James")
    and matched zero players against the Kaggle full names on the first attempt.
    """
    k = add_derived_features(load_kaggle_shot_logs())
    k = k.dropna(subset=["SHOT_RESULT", "player_id"]).copy()
    out = pd.DataFrame(
        {
            "player_id": k["player_id"].astype(int),
            "player": k["player_name"].str.strip().str.lower(),
            "shot_dist_ft": k["SHOT_DIST"],
            "closest_def_dist_ft": k["CLOSE_DEF_DIST"],
            "shot_clock": k["SHOT_CLOCK"],
            "is_3pt": (k["PTS_TYPE"] == 3).astype(int),
            "made": k["SHOT_MADE"].astype(int),
        }
    )
    return out


def load_1516_shared() -> pd.DataFrame:
    """2015-16 extracted shots, cleaned to the shared feature schema.

    High-confidence releases only, with physically implausible extraction artifacts
    removed (<0.4% of shots each): touch/dribble/def-distance outliers.
    """
    t = pd.read_parquet(TRACKING_1516)
    t = t[t["release_confidence"] == "high"].copy()
    t = t[(t["closest_def_dist_ft"] <= 40) & (t["shot_dist_ft"] <= 47)]
    out = pd.DataFrame(
        {
            "player_id": t["player_id"].astype(int),
            "player": t["player_name"].astype(str).str.strip().str.lower(),
            "shot_dist_ft": t["shot_dist_ft"],
            "closest_def_dist_ft": t["closest_def_dist_ft"],
            "shot_clock": t["shot_clock"],
            "is_3pt": t["is_3pt"].astype(int),
            "made": t["made"].astype(int),
        }
    )
    return out


def build_model() -> Pipeline:
    """Expected-make model on the shared difficulty features (same family as v1's GBT)."""
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_lambda=1.0,
                    random_state=42,
                    tree_method="hist",
                    n_jobs=4,
                ),
            ),
        ]
    )


def player_scores(shots: pd.DataFrame, expected_prob: np.ndarray, min_shots: int) -> pd.DataFrame:
    """Aggregate to per-player SMOKE (actual FG% minus expected FG%) + eFG%."""
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


def corr_block(label: str, x: pd.Series, y: pd.Series) -> dict:
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    return {
        "metric": label,
        "n": len(x),
        "pearson_r": round(float(pear.statistic), 3),
        "pearson_p": float(pear.pvalue),
        "spearman_rho": round(float(spear.statistic), 3),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("PHASE 2.4 — SMOKE year-over-year stability, 2014-15 -> 2015-16")
    print("=" * 74)

    s14 = load_1415_shared()
    s16 = load_1516_shared()
    print(f"  2014-15 shots (shared schema): {len(s14):,}")
    print(f"  2015-16 shots (shared schema): {len(s16):,}")

    # one model, trained on 2014-15, applied to BOTH seasons identically
    model = build_model()
    model.fit(s14[MODEL_COLS], s14[TARGET])
    print("  trained expected-make model on 2014-15 shared features:", MODEL_COLS)

    exp14 = model.predict_proba(s14[MODEL_COLS])[:, 1]
    exp16 = model.predict_proba(s16[MODEL_COLS])[:, 1]

    keep = lambda c: c in ("player_id", "player")  # noqa: E731
    p14 = player_scores(s14, exp14, MIN_SHOTS).rename(columns=lambda c: c if keep(c) else f"{c}_14")
    p16 = player_scores(s16, exp16, MIN_SHOTS).rename(columns=lambda c: c if keep(c) else f"{c}_16")
    print(f"  players clearing {MIN_SHOTS} shots — 2014-15: {len(p14)}, 2015-16: {len(p16)}")

    merged = p14.merge(p16, on="player_id", how="inner", suffixes=("", "_16name"))
    print(f"  players in BOTH seasons (the stability sample): {len(merged)}")
    if len(merged) < 30:
        print("  !! sample too small to trust — investigate")

    # THE test: is SMOKE at least as stable YoY as FG% and eFG%?
    results = [
        corr_block("SMOKE (shot-quality-adjusted)", merged["smoke_14"], merged["smoke_16"]),
        corr_block("raw FG%", merged["fg_pct_14"], merged["fg_pct_16"]),
        corr_block("eFG%", merged["efg_pct_14"], merged["efg_pct_16"]),
    ]
    res = pd.DataFrame(results)

    print("\n  YEAR-OVER-YEAR STABILITY (higher = more repeatable skill)")
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['metric']:32s}  r = {r['pearson_r']:+.3f}   rho = {r['spearman_rho']:+.3f}   (n={r['n']})")
    print("  " + "-" * 66)

    smoke_r = res.loc[res["metric"].str.startswith("SMOKE"), "pearson_r"].iloc[0]
    fg_r = res.loc[res["metric"] == "raw FG%", "pearson_r"].iloc[0]
    efg_r = res.loc[res["metric"] == "eFG%", "pearson_r"].iloc[0]
    print("\n  READ:")
    print(f"    SMOKE YoY r = {smoke_r:+.3f}  vs  FG% {fg_r:+.3f}  /  eFG% {efg_r:+.3f}")
    if smoke_r >= max(fg_r, efg_r):
        print("    -> SMOKE is AT LEAST AS STABLE as the metrics teams already use. Claim holds.")
    elif smoke_r >= 0.4:
        print("    -> SMOKE is stable and in a defensible range, though not above eFG%. Nuance needed.")
    else:
        print("    -> SMOKE stability is weak. Red flag — investigate before publishing.")
    print("    Benchmark (BBall Index Shot Making ~0.66, eFG% ~0.28) is a full-season figure;")
    print("    ours uses a half-season 2015-16, so expect somewhat lower and note it honestly.")

    merged.to_parquet(OUT_DIR / "stability_2season_players.parquet", index=False)
    res.to_csv(OUT_DIR / "stability_correlations.csv", index=False)
    print(f"\n  wrote {OUT_DIR / 'stability_correlations.csv'}")
    print(f"  wrote {OUT_DIR / 'stability_2season_players.parquet'}  ({len(merged)} players)")


if __name__ == "__main__":
    main()
