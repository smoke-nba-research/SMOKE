"""Fit the shot-quality models and write their output tables to disk.

This script runs the logistic and gradient-boosted models on the Kaggle
2014-15 shot logs and saves the player residuals, team residuals, rank
table, and model metrics as CSV files under ``data/model_outputs/``.

Running the models needs XGBoost, so this step is run once on a machine
where XGBoost is available. The visualization notebook then reads the
saved CSVs directly, which means the charts can be regenerated on any
machine without having to fit the models again.

Usage:
    python -m src.models.build_model_outputs
"""

from __future__ import annotations


import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features.kaggle_shot_quality import (
    CATEGORICAL_FEATURES,
    MODEL_COLUMNS,
    NUMERIC_FEATURES,
    PLAYER_NAME_COLUMN,
    TARGET_COLUMN,
    TEAM_COLUMN,
    build_modeling_frame,
    build_residual_table,
    load_kaggle_shot_logs,
)
from src.pulls._paths import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "data" / "model_outputs"

PLAYER_MIN_SHOTS = 150
TEAM_MIN_SHOTS = 2000


def build_logistic_model() -> Pipeline:
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
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )


def build_boosted_model() -> Pipeline:
    """Gradient-boosted trees with imputed numeric and one-hot categorical inputs."""
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
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


def score_models(modeling_shots: pd.DataFrame) -> pd.DataFrame:
    """Train both models on a hold-out split and return their test metrics."""
    features = modeling_shots[MODEL_COLUMNS].copy()
    outcomes = modeling_shots[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        outcomes,
        test_size=0.2,
        random_state=42,
        stratify=outcomes,
    )

    rows = []
    for name, model in [
        ("Logistic Regression", build_logistic_model()),
        ("Gradient-Boosted Trees", build_boosted_model()),
    ]:
        trained = clone(model)
        trained.fit(X_train, y_train)
        predicted_labels = trained.predict(X_test)
        predicted_probs = trained.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predicted_labels),
                "roc_auc": roc_auc_score(y_test, predicted_probs),
            }
        )

    return pd.DataFrame(rows)


def build_rank_table(scored_shots: pd.DataFrame) -> pd.DataFrame:
    """Compare each player's traditional efficiency to a shot-quality-adjusted view."""
    rank_table = (
        scored_shots.groupby(PLAYER_NAME_COLUMN)
        .agg(
            shots=(TARGET_COLUMN, "size"),
            fgm=("FGM", "sum"),
            points=("PTS", "sum"),
            expected_makes=("EXPECTED_MAKE_PROB", "sum"),
        )
        .reset_index()
    )

    rank_table = rank_table[rank_table["shots"] >= PLAYER_MIN_SHOTS].copy()
    rank_table["fg_pct"] = rank_table["fgm"] / rank_table["shots"]
    rank_table["efg_pct"] = (rank_table["points"] / 2) / rank_table["shots"]
    rank_table["expected_fg_pct"] = rank_table["expected_makes"] / rank_table["shots"]
    rank_table["shot_quality_adjusted"] = (
        rank_table["fg_pct"] - rank_table["expected_fg_pct"]
    )

    rank_table["fg_rank"] = rank_table["fg_pct"].rank(ascending=False, method="min")
    rank_table["efg_rank"] = rank_table["efg_pct"].rank(ascending=False, method="min")
    rank_table["adjusted_rank"] = rank_table["shot_quality_adjusted"].rank(
        ascending=False, method="min"
    )
    rank_table["rank_shift_vs_fg"] = rank_table["fg_rank"] - rank_table["adjusted_rank"]
    rank_table["rank_shift_vs_efg"] = (
        rank_table["efg_rank"] - rank_table["adjusted_rank"]
    )

    return rank_table.sort_values("rank_shift_vs_fg", ascending=False).reset_index(
        drop=True
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    modeling_shots = build_modeling_frame(load_kaggle_shot_logs())
    print(f"Modeling rows: {len(modeling_shots):,}")

    metrics = score_models(modeling_shots)
    print("\nModel metrics:")
    print(metrics.to_string(index=False))

    # Refit the boosted model on every shot so each one gets an expected make
    # probability we can compare against what actually happened.
    full_model = build_boosted_model()
    full_model.fit(modeling_shots[MODEL_COLUMNS], modeling_shots[TARGET_COLUMN])

    scored_shots = modeling_shots[
        [PLAYER_NAME_COLUMN, TEAM_COLUMN, TARGET_COLUMN, "FGM", "PTS"]
    ].copy()
    scored_shots["EXPECTED_MAKE_PROB"] = full_model.predict_proba(
        modeling_shots[MODEL_COLUMNS]
    )[:, 1]

    player_residuals = build_residual_table(
        scored_shots, PLAYER_NAME_COLUMN, min_shots=PLAYER_MIN_SHOTS
    )
    team_residuals = build_residual_table(
        scored_shots, TEAM_COLUMN, min_shots=TEAM_MIN_SHOTS
    )
    rank_table = build_rank_table(scored_shots)

    outputs = {
        "player_residuals.csv": player_residuals,
        "team_residuals.csv": team_residuals,
        "rank_table.csv": rank_table,
        "model_metrics.csv": metrics,
    }
    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False)
        print(f"Wrote {len(frame):>4} rows to {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
