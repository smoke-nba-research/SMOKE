"""Fit the expected-make model once and write everything downstream reads.

This is the only script that trains the SMOKE model on the 2014-15 shot logs. It
produces one scored-shots artifact, ``data/v2/processed/scored_shots_1415.parquet``,
holding every shot's features, outcome, and cross-fitted expected make probability.
The validation battery, the figure suite, the one-pager, and the dashboard all read
that file, so every published number rests on one identical set of probabilities.

Expected probabilities are out-of-fold: the shots are split into five folds by game
(StratifiedGroupKFold on GAME_ID), and each shot is scored by a model that never saw
its game. Residuals are therefore not flattered by fitting noise, and the model's
hold-out metrics (accuracy, area under the ROC curve, log loss, Brier score) come
from a game-grouped 80/20 split with the same discipline.

Also written, all derived from the scored shots:
    data/model_outputs/player_residuals.csv   raw SMOKE per player (150-shot floor)
    data/model_outputs/team_residuals.csv     raw SMOKE per team
    data/model_outputs/rank_table.csv         FG%, eFG%, and SMOKE ranks per player
    data/model_outputs/model_metrics.csv      hold-out scorecard, with a base-rate row
    data/model_outputs/model_card.json        features, hyperparameters, versions

Usage:
    python -m src.models.build_model_outputs
"""

from __future__ import annotations

import json
import platform
from datetime import date

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features.kaggle_shot_quality import (
    CATEGORICAL_FEATURES,
    DATE_COLUMN,
    MODEL_COLUMNS,
    NUMERIC_FEATURES,
    OPPONENT_COLUMN,
    PLAYER_ID_COLUMN,
    PLAYER_NAME_COLUMN,
    TARGET_COLUMN,
    TEAM_COLUMN,
    build_modeling_frame,
    build_residual_table,
    load_kaggle_shot_logs,
)
from src.pulls._paths import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "data" / "model_outputs"
SCORED_SHOTS = REPO_ROOT / "data" / "v2" / "processed" / "scored_shots_1415.parquet"
MODEL_CARD = OUTPUT_DIR / "model_card.json"

PLAYER_MIN_SHOTS = 150
TEAM_MIN_SHOTS = 2000
N_FOLDS = 5
SEED = 42
GROUP_COLUMN = "GAME_ID"

# One hyperparameter block, shared with the two-season stability model.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "tree_method": "hist",
    "n_jobs": 4,
}

# Columns carried into the scored-shots artifact alongside the model features.
CONTEXT_COLUMNS = [
    GROUP_COLUMN,
    DATE_COLUMN,
    PLAYER_ID_COLUMN,
    PLAYER_NAME_COLUMN,
    TEAM_COLUMN,
    OPPONENT_COLUMN,
    "FGM",
    "PTS",
    TARGET_COLUMN,
]


def _categorical_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def build_logistic_model(
    numeric: list[str] = NUMERIC_FEATURES, categorical: list[str] = CATEGORICAL_FEATURES
) -> Pipeline:
    """Logistic regression with scaled numeric and one-hot categorical inputs."""
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("cat", _categorical_pipeline(), categorical),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )


def build_boosted_model(
    numeric: list[str] = NUMERIC_FEATURES, categorical: list[str] = CATEGORICAL_FEATURES
) -> Pipeline:
    """Gradient-boosted trees with imputed numeric and one-hot categorical inputs."""
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", _categorical_pipeline(), categorical),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**XGB_PARAMS)),
        ]
    )


def _scorecard(name: str, y_true: pd.Series, prob: np.ndarray) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, prob > 0.5),
        "roc_auc": roc_auc_score(y_true, prob),
        "log_loss": log_loss(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
    }


def holdout_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Score both models on a game-grouped 80/20 hold-out, against the base rate."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(frame, frame[TARGET_COLUMN], frame[GROUP_COLUMN]))
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    y_train, y_test = train[TARGET_COLUMN], test[TARGET_COLUMN]

    rows = [_scorecard("Base rate", y_test, np.full(len(test), y_train.mean()))]
    for name, model in [
        ("Logistic Regression", build_logistic_model()),
        ("Gradient-Boosted Trees", build_boosted_model()),
    ]:
        model.fit(train[MODEL_COLUMNS], y_train)
        rows.append(_scorecard(name, y_test, model.predict_proba(test[MODEL_COLUMNS])[:, 1]))
    out = pd.DataFrame(rows)
    out["n_train"], out["n_test"] = len(train), len(test)
    return out


def cross_fit(
    frame: pd.DataFrame,
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    group_column: str = GROUP_COLUMN,
) -> np.ndarray:
    """Out-of-fold expected make probability for every row, folds grouped by game."""
    columns = numeric + categorical
    expected = np.full(len(frame), np.nan)
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(frame, frame[TARGET_COLUMN], frame[group_column]):
        model = build_boosted_model(numeric, categorical)
        model.fit(frame.iloc[train_idx][columns], frame.iloc[train_idx][TARGET_COLUMN])
        expected[test_idx] = model.predict_proba(frame.iloc[test_idx][columns])[:, 1]
    assert not np.isnan(expected).any(), "every shot must be scored out of fold"
    return expected


def score_shots(frame: pd.DataFrame) -> pd.DataFrame:
    """The scored-shots artifact: context, features, outcome, and OOF expectation."""
    scored = frame[CONTEXT_COLUMNS + MODEL_COLUMNS].copy()
    scored["expected"] = cross_fit(frame)
    return scored


def load_scored_shots() -> pd.DataFrame:
    """Read the artifact; every downstream script starts here."""
    if not SCORED_SHOTS.exists():
        raise FileNotFoundError(
            f"{SCORED_SHOTS} not found. Run `python -m src.models.build_model_outputs` first."
        )
    return pd.read_parquet(SCORED_SHOTS)


def build_rank_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Compare each player's traditional efficiency to the SMOKE view."""
    rank_table = (
        scored.groupby([PLAYER_NAME_COLUMN, PLAYER_ID_COLUMN])
        .agg(
            shots=(TARGET_COLUMN, "size"),
            fgm=("FGM", "sum"),
            points=("PTS", "sum"),
            expected_makes=("expected", "sum"),
        )
        .reset_index()
    )

    rank_table = rank_table[rank_table["shots"] >= PLAYER_MIN_SHOTS].copy()
    rank_table["fg_pct"] = rank_table["fgm"] / rank_table["shots"]
    rank_table["efg_pct"] = (rank_table["points"] / 2) / rank_table["shots"]
    rank_table["expected_fg_pct"] = rank_table["expected_makes"] / rank_table["shots"]
    rank_table["smoke_rate"] = rank_table["fg_pct"] - rank_table["expected_fg_pct"]

    rank_table["fg_rank"] = rank_table["fg_pct"].rank(ascending=False, method="min")
    rank_table["efg_rank"] = rank_table["efg_pct"].rank(ascending=False, method="min")
    rank_table["smoke_rank"] = rank_table["smoke_rate"].rank(ascending=False, method="min")
    rank_table["rank_shift_vs_fg"] = rank_table["fg_rank"] - rank_table["smoke_rank"]
    rank_table["rank_shift_vs_efg"] = rank_table["efg_rank"] - rank_table["smoke_rank"]

    return rank_table.sort_values("rank_shift_vs_fg", ascending=False).reset_index(drop=True)


def write_model_card(frame: pd.DataFrame, scored: pd.DataFrame, metrics: pd.DataFrame) -> None:
    card = {
        "metric": "SMOKE (Shots Made Over Known Expectation)",
        "trained_on": "Kaggle 2014-15 NBA shot logs (dansbecker/nba-shot-logs)",
        "date_built": date.today().isoformat(),
        "n_shots": int(len(scored)),
        "n_games": int(frame[GROUP_COLUMN].nunique()),
        "n_players": int(frame[PLAYER_ID_COLUMN].nunique()),
        "date_range": [str(frame[DATE_COLUMN].min().date()), str(frame[DATE_COLUMN].max().date())],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_outcome_columns": ["FINAL_MARGIN", "W"],
        "xgboost_params": XGB_PARAMS,
        "cross_fit": {"n_folds": N_FOLDS, "grouped_by": GROUP_COLUMN, "seed": SEED},
        "oof_roc_auc": float(roc_auc_score(scored[TARGET_COLUMN], scored["expected"])),
        "oof_log_loss": float(log_loss(scored[TARGET_COLUMN], scored["expected"])),
        "oof_mean_expected_minus_actual": float(scored["expected"].mean() - scored[TARGET_COLUMN].mean()),
        "holdout": metrics.to_dict(orient="records"),
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    MODEL_CARD.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCORED_SHOTS.parent.mkdir(parents=True, exist_ok=True)

    frame = build_modeling_frame(load_kaggle_shot_logs()).reset_index(drop=True)
    print(f"Modeling rows: {len(frame):,} across {frame[GROUP_COLUMN].nunique()} games, "
          f"{frame[DATE_COLUMN].min().date()} to {frame[DATE_COLUMN].max().date()}")
    print("Features:", ", ".join(MODEL_COLUMNS))

    metrics = holdout_metrics(frame)
    print("\nHold-out metrics (game-grouped 80/20 split):")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    scored = score_shots(frame)
    print(f"\nOut-of-fold ({N_FOLDS} folds by game): AUC "
          f"{roc_auc_score(scored[TARGET_COLUMN], scored['expected']):.4f}, "
          f"log loss {log_loss(scored[TARGET_COLUMN], scored['expected']):.4f}, "
          f"mean expected - actual {scored['expected'].mean() - scored[TARGET_COLUMN].mean():+.5f}")
    scored.to_parquet(SCORED_SHOTS, index=False)
    print(f"Wrote {len(scored):,} scored shots to {SCORED_SHOTS.relative_to(REPO_ROOT)}")

    player_residuals = build_residual_table(
        scored, [PLAYER_NAME_COLUMN, PLAYER_ID_COLUMN], min_shots=PLAYER_MIN_SHOTS
    )
    team_residuals = build_residual_table(scored, TEAM_COLUMN, min_shots=TEAM_MIN_SHOTS)
    rank_table = build_rank_table(scored)

    outputs = {
        "player_residuals.csv": player_residuals,
        "team_residuals.csv": team_residuals,
        "rank_table.csv": rank_table,
        "model_metrics.csv": metrics,
    }
    for filename, table in outputs.items():
        path = OUTPUT_DIR / filename
        table.to_csv(path, index=False)
        print(f"Wrote {len(table):>4} rows to {path.relative_to(REPO_ROOT)}")
    write_model_card(frame, scored, metrics)
    print(f"Wrote {MODEL_CARD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
