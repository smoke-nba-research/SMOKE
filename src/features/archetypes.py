"""Playing-style archetypes: one implementation for the v1 analysis and the fairness test.

Players are clustered on their shot diet (average distance, defender distance, touch
time, dribbles, three-point rate) with K-means, and the four clusters are named by
rule rather than by fixed thresholds: the cluster closest to the rim is the interior
group, the most ball-dominant of the rest are the creators, the most three-heavy of
the rest are the catch-and-shoot group, and the remaining cluster is the mid-range
group. The caller decides which population to cluster (all qualified players for the
fairness test; the overperformers only for the v1 descriptive analysis).
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

STYLE_FEATURES = [
    "avg_shot_dist",
    "avg_def_dist",
    "avg_touch_time",
    "avg_dribbles",
    "three_rate",
]
N_ARCHETYPES = 4
ARCHETYPE_ORDER = ["On-Ball Creators", "Catch-and-Shoot", "Mid-Range Scorers", "Interior Finishers"]


def build_player_style(shots: pd.DataFrame, min_shots: int, player_col: str = "player_name") -> pd.DataFrame:
    """Summarize each player's shot diet into playing-style features.

    Expects per-shot columns SHOT_DIST, CLOSE_DEF_DIST, TOUCH_TIME, DRIBBLES, PTS_TYPE
    and a player id column alongside ``player_col``.
    """
    style = (
        shots.groupby([player_col, "player_id"])
        .agg(
            shots=("SHOT_DIST", "size"),
            avg_shot_dist=("SHOT_DIST", "mean"),
            avg_def_dist=("CLOSE_DEF_DIST", "mean"),
            avg_touch_time=("TOUCH_TIME", "mean"),
            avg_dribbles=("DRIBBLES", "mean"),
            three_rate=("PTS_TYPE", lambda values: (values == 3).mean()),
        )
        .reset_index()
    )
    return style[style["shots"] >= min_shots].reset_index(drop=True)


def label_archetypes(profiles: pd.DataFrame) -> dict[int, str]:
    """Name each cluster from its centroid, by relative ranking."""
    remaining = list(profiles.index)
    labels: dict[int, str] = {}

    interior = profiles.loc[remaining, "avg_shot_dist"].idxmin()
    labels[interior] = "Interior Finishers"
    remaining.remove(interior)

    creators = profiles.loc[remaining, "avg_dribbles"].idxmax()
    labels[creators] = "On-Ball Creators"
    remaining.remove(creators)

    shooters = profiles.loc[remaining, "three_rate"].idxmax()
    labels[shooters] = "Catch-and-Shoot"
    remaining.remove(shooters)

    for cluster in remaining:
        labels[cluster] = "Mid-Range Scorers"
    return labels


def cluster_archetypes(style: pd.DataFrame) -> pd.DataFrame:
    """Add ``cluster`` and ``archetype`` columns to a player-style table."""
    out = style.copy()
    scaled = StandardScaler().fit_transform(out[STYLE_FEATURES])
    model = KMeans(n_clusters=N_ARCHETYPES, random_state=42, n_init=10)
    out["cluster"] = model.fit_predict(scaled)
    profiles = out.groupby("cluster")[STYLE_FEATURES].mean()
    out["archetype"] = out["cluster"].map(label_archetypes(profiles))
    return out
