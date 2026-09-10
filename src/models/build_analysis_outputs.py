"""Build the RQ1 archetype table and the RQ3 rank correlations.

Reads the scored-shots artifact and the residual tables written by
``build_model_outputs`` and does not refit anything.

* RQ1: cluster the overperforming players into playing-style archetypes with the
  shared K-means implementation in ``src/features/archetypes.py``.
* RQ3: Spearman rank correlation between SMOKE and the traditional efficiency
  metrics, from the rank table.

Usage:
    python -m src.models.build_analysis_outputs
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from src.features.archetypes import build_player_style, cluster_archetypes
from src.features.kaggle_shot_quality import PLAYER_ID_COLUMN, PLAYER_NAME_COLUMN
from src.models.build_model_outputs import OUTPUT_DIR, PLAYER_MIN_SHOTS, load_scored_shots
from src.pulls._paths import REPO_ROOT


def build_archetypes(shots: pd.DataFrame, player_residuals: pd.DataFrame) -> pd.DataFrame:
    """Cluster the overperforming players into playing-style archetypes."""
    style = build_player_style(shots, min_shots=PLAYER_MIN_SHOTS)
    style = style.merge(
        player_residuals[[PLAYER_ID_COLUMN, "smoke_total"]], on=PLAYER_ID_COLUMN, how="inner"
    )
    overperformers = style[style["smoke_total"] > 0].copy()
    clustered = cluster_archetypes(overperformers)
    return clustered.sort_values(["archetype", "smoke_total"], ascending=[True, False]).reset_index(drop=True)


def build_rank_correlations(rank_table: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between SMOKE and the traditional metrics."""
    rows = []
    for metric in ["fg_pct", "efg_pct"]:
        rho, p_value = spearmanr(rank_table["smoke_rate"], rank_table[metric])
        rows.append({"comparison": f"smoke_rate vs {metric}", "spearman_rho": rho, "p_value": p_value})
    return pd.DataFrame(rows)


def main() -> None:
    shots = load_scored_shots()
    player_residuals = pd.read_csv(OUTPUT_DIR / "player_residuals.csv")
    rank_table = pd.read_csv(OUTPUT_DIR / "rank_table.csv")

    archetypes = build_archetypes(shots, player_residuals)
    correlations = build_rank_correlations(rank_table)

    archetypes.to_csv(OUTPUT_DIR / "player_archetypes.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "rank_correlations.csv", index=False)

    print("Player archetypes (overperformers only):")
    print(archetypes["archetype"].value_counts().to_string())
    print("\nRank correlations:")
    print(correlations.to_string(index=False))
    print(f"\nWrote player_archetypes.csv and rank_correlations.csv to {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    assert PLAYER_NAME_COLUMN in archetypes.columns


if __name__ == "__main__":
    main()
