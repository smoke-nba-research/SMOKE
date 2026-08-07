"""Build the remaining RQ1 and RQ3 analysis tables and save them to disk.

This script covers the two analysis pieces that sit on top of the shot-quality
model:

* RQ1 - group the overperforming players into playing-style archetypes using
  K-means clustering, so we can describe *what distinguishes* the players who
  beat their expected shot value.
* RQ3 - measure how much the shot-quality-adjusted ranking differs from the
  traditional efficiency metrics using Spearman rank correlation.

It reads the player residuals saved by ``build_model_outputs`` and the Kaggle
shot logs, so it does not need to refit any model and runs without XGBoost.

Usage:
    python -m src.models.build_analysis_outputs
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.features.kaggle_shot_quality import (
    PLAYER_NAME_COLUMN,
    TARGET_COLUMN,
    build_modeling_frame,
    load_kaggle_shot_logs,
)
from src.pulls._paths import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "data" / "model_outputs"

STYLE_FEATURES = [
    "avg_shot_dist",
    "avg_def_dist",
    "avg_touch_time",
    "avg_dribbles",
    "three_rate",
]
MIN_SHOTS = 150
N_ARCHETYPES = 4


def build_player_style(shots: pd.DataFrame) -> pd.DataFrame:
    """Summarize each player's shot diet into playing-style features."""
    style = (
        shots.groupby(PLAYER_NAME_COLUMN)
        .agg(
            shots=(TARGET_COLUMN, "size"),
            avg_shot_dist=("SHOT_DIST", "mean"),
            avg_def_dist=("CLOSE_DEF_DIST", "mean"),
            avg_touch_time=("TOUCH_TIME", "mean"),
            avg_dribbles=("DRIBBLES", "mean"),
            three_rate=("PTS_TYPE", lambda values: (values == 3).mean()),
        )
        .reset_index()
    )
    return style[style["shots"] >= MIN_SHOTS].copy()


def label_archetypes(profiles: pd.DataFrame) -> dict[int, str]:
    """Give each cluster a descriptive label based on its centroid profile.

    Labels are assigned by relative ranking rather than fixed thresholds so the
    descriptions stay sensible even if the underlying numbers move a little:
    the closest-to-the-rim cluster is the interior group, the most ball-dominant
    of the rest are the creators, the most three-heavy of the rest are the
    catch-and-shoot group, and the remaining cluster is the mid-range group.
    """
    remaining = list(profiles.index)
    labels: dict[int, str] = {}

    interior = profiles.loc[remaining, "avg_shot_dist"].idxmin()
    labels[interior] = "Interior Finishers"
    remaining.remove(interior)

    creators = profiles.loc[remaining, "avg_dribbles"].idxmax()
    labels[creators] = "On-Ball Creators"
    remaining.remove(creators)

    shooters = profiles.loc[remaining, "three_rate"].idxmax()
    labels[shooters] = "Catch-and-Shoot Specialists"
    remaining.remove(shooters)

    for cluster in remaining:
        labels[cluster] = "Mid-Range Scorers"

    return labels


def build_archetypes(
    shots: pd.DataFrame, player_residuals: pd.DataFrame
) -> pd.DataFrame:
    """Cluster the overperforming players into playing-style archetypes."""
    style = build_player_style(shots)
    style = style.merge(
        player_residuals[[PLAYER_NAME_COLUMN, "makes_above_expected"]],
        on=PLAYER_NAME_COLUMN,
        how="inner",
    )
    overperformers = style[style["makes_above_expected"] > 0].copy()

    scaled = StandardScaler().fit_transform(overperformers[STYLE_FEATURES])
    model = KMeans(n_clusters=N_ARCHETYPES, random_state=42, n_init=10)
    overperformers["cluster"] = model.fit_predict(scaled)

    profiles = overperformers.groupby("cluster")[STYLE_FEATURES].mean()
    overperformers["archetype"] = overperformers["cluster"].map(
        label_archetypes(profiles)
    )

    return overperformers.sort_values(
        ["archetype", "makes_above_expected"], ascending=[True, False]
    ).reset_index(drop=True)


def build_rank_correlations(rank_table: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between the adjusted metric and the traditional ones."""
    rows = []
    for metric in ["fg_pct", "efg_pct"]:
        rho, p_value = spearmanr(
            rank_table["shot_quality_adjusted"], rank_table[metric]
        )
        rows.append(
            {
                "comparison": f"shot_quality_adjusted vs {metric}",
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shots = build_modeling_frame(load_kaggle_shot_logs())
    player_residuals = pd.read_csv(OUTPUT_DIR / "player_residuals.csv")
    rank_table = pd.read_csv(OUTPUT_DIR / "rank_table.csv")

    archetypes = build_archetypes(shots, player_residuals)
    correlations = build_rank_correlations(rank_table)

    archetypes.to_csv(OUTPUT_DIR / "player_archetypes.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "rank_correlations.csv", index=False)

    print("Player archetypes:")
    print(archetypes["archetype"].value_counts().to_string())
    print("\nRank correlations:")
    print(correlations.to_string(index=False))
    print(
        f"\nWrote player_archetypes.csv and rank_correlations.csv to "
        f"{OUTPUT_DIR.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
