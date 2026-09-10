"""Collect every headline number the documents quote into one manifest.

The dossier, the manuscript, the abstract, the FAQ, the README, the one-pager and the
dashboard all quote the same handful of results. This module reads them from the
pipeline outputs once and writes ``data/v2/validation/headline_numbers.json``, so a
document can be checked against the outputs instead of against memory.

Run:
    .venv/Scripts/python.exe -m src.report.numbers
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from src.models.build_model_outputs import MODEL_CARD, OUTPUT_DIR, load_scored_shots
from src.pulls._paths import REPO_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAL = REPO_ROOT / "data" / "v2" / "validation"
OUT = VAL / "headline_numbers.json"


def _kv(path, key_col, value_col) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df[key_col], df[value_col], strict=True))


def collect() -> dict:
    card = json.loads(MODEL_CARD.read_text(encoding="utf-8"))
    scored = load_scored_shots()
    players = pd.read_csv(VAL / "player_smoke_with_error_bars.csv")
    top = players.sort_values("smoke_total_shrunk", ascending=False).head(5)
    half_width = (players["smoke_total_hi"] - players["smoke_total_lo"]) / 2
    split = pd.read_csv(VAL / "split_half_reliability.csv").set_index("design")
    stab = pd.read_csv(VAL / "stability_correlations.csv")
    primary = stab[stab["design"] == "per-season cross-fit"].set_index("metric")
    onemodel = stab[stab["design"] == "one model (2014-15)"].set_index("metric")
    sweep = pd.read_csv(VAL / "stability_threshold_sweep.csv")
    calib = pd.read_csv(VAL / "stability_calibration.csv")
    conv = pd.read_csv(VAL / "convergent_correlations.csv").set_index("metric")
    pred = _kv(VAL / "predictive_summary.csv", "test", "value")
    conf = _kv(VAL / "confound_summary.csv", "stat", "value")
    arch = pd.read_csv(VAL / "archetype_means.csv").set_index("archetype")
    omni = _kv(VAL / "archetype_omnibus.csv", "stat", "value")
    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv").set_index("model")
    rank = pd.read_csv(OUTPUT_DIR / "rank_table.csv")
    rank_corr = _kv(OUTPUT_DIR / "rank_correlations.csv", "comparison", "spearman_rho")
    panel = pd.read_parquet(VAL / "stability_2season_players.parquet")
    tracking = pd.read_parquet(REPO_ROOT / "data" / "v2" / "processed" / "shots_tracking_1516_season.parquet")
    tracking_hi = tracking[tracking["release_confidence"] == "high"]

    sweep_rows = {}
    for floor, g in sweep[sweep["design"] == "per-season cross-fit"].groupby("min_shots"):
        g = g.set_index("metric")
        sweep_rows[int(floor)] = {"SMOKE": float(g.loc["SMOKE", "pearson_r"]), "eFG%": float(g.loc["eFG%", "pearson_r"]),
                                  "FG%": float(g.loc["raw FG%", "pearson_r"]), "n": int(g.loc["SMOKE", "n"])}

    return {
        "data": {
            "n_shots_1415": int(card["n_shots"]),
            "n_games_1415": int(card["n_games"]),
            "season_games": 1230,
            "n_players_1415": int(card["n_players"]),
            "date_range_1415": card["date_range"],
            "n_shots_1516_extracted": int(len(tracking)),
            "n_shots_1516_high_conf": int(len(tracking_hi)),
            "share_high_conf_1516": float(len(tracking_hi) / len(tracking)),
            "n_games_1516": int(tracking["game_id"].nunique()),
            "date_range_1516": [str(tracking["game_date"].min()), str(tracking["game_date"].max())],
            "n_players_1516": int(tracking["player_id"].nunique()),
            "tracking_dist_error_mae_ft": (
                float(pd.to_numeric(tracking_hi["dist_error_ft"], errors="coerce").mean())
                if "dist_error_ft" in tracking_hi.columns else None
            ),
        },
        "model": {
            "features": card["numeric_features"] + card["categorical_features"],
            "excluded_outcome_columns": card["excluded_outcome_columns"],
            "holdout": {m: {k: float(v) for k, v in row.items() if k in ("accuracy", "roc_auc", "log_loss", "brier")}
                        for m, row in metrics.iterrows()},
            "oof_roc_auc": float(card["oof_roc_auc"]),
            "oof_log_loss": float(card["oof_log_loss"]),
            "oof_calibration_gap": float(card["oof_mean_expected_minus_actual"]),
            "cross_fit": card["cross_fit"],
        },
        "leaderboard": {
            "n_qualified": int(len(players)),
            "min_shots": 150,
            "n_distinguishable": int(players["distinguishable"].sum()),
            "share_distinguishable": float(players["distinguishable"].mean()),
            "mean_shots_per_qualified": float(players["shots"].mean()),
            "median_shots_per_qualified": float(players["shots"].median()),
            "mean_ci_half_width_makes": float(half_width.mean()),
            "league_mean_rate": float(players["smoke_rate"].mean()),
            "reliability_min": float(players["reliability"].min()),
            "reliability_max": float(players["reliability"].max()),
            "reliability_median": float(players["reliability"].median()),
            "top5_by_shrunk_total": [
                {"player": r.player, "shots": int(r.shots), "raw_total": round(float(r.smoke_total), 1),
                 "ci": [round(float(r.smoke_total_lo)), round(float(r.smoke_total_hi))],
                 "shrunk_total": round(float(r.smoke_total_shrunk), 1), "reliability": round(float(r.reliability), 2)}
                for r in top.itertuples()
            ],
            "rank_spearman_vs_fg": float(rank_corr["smoke_rate vs fg_pct"]),
            "rank_spearman_vs_efg": float(rank_corr["smoke_rate vs efg_pct"]),
            "corr_expected_fg_vs_smoke": float(rank["expected_fg_pct"].corr(rank["smoke_rate"])),
            "biggest_riser": {"player": rank.iloc[0]["player_name"], "spots": int(rank.iloc[0]["rank_shift_vs_fg"])},
            "biggest_faller": {"player": rank.iloc[-1]["player_name"], "spots": int(rank.iloc[-1]["rank_shift_vs_fg"])},
        },
        "reliability": {
            "split_half_alternating_r": float(split.loc["alternating shots", "half_r"]),
            "split_half_alternating_sb": float(split.loc["alternating shots", "spearman_brown"]),
            "split_half_date_r": float(split.loc["first vs second half by date", "half_r"]),
            "split_half_date_sb": float(split.loc["first vs second half by date", "spearman_brown"]),
        },
        "stability": {
            "n_players": int(primary.loc["SMOKE", "n"]),
            "smoke_r": float(primary.loc["SMOKE", "pearson_r"]),
            "smoke_rho": float(primary.loc["SMOKE", "spearman_rho"]),
            "fg_r": float(primary.loc["raw FG%", "pearson_r"]),
            "efg_r": float(primary.loc["eFG%", "pearson_r"]),
            "smoke_r_onemodel": float(onemodel.loc["SMOKE", "pearson_r"]),
            "onemodel_gap_1516": float(calib[(calib["design"] == "one model (2014-15)") & (calib["season"] == "2015-16")]["gap"].iloc[0]),
            "threshold_sweep": sweep_rows,
            "n_shots_1415_used": int(len(scored)),
            "mean_smoke_16_primary": float(panel["smoke_16"].mean()),
        },
        "convergent": {m: float(conv.loc[m, "pearson_r"]) for m in ["ts_percent", "obpm", "per", "bpm", "usg_percent"]},
        "predictive": {k: float(v) for k, v in pred.items()},
        "confounds": {k: float(v) for k, v in conf.items()},
        "fairness": {
            "kruskal_p": float(omni["kruskal_p"]),
            "archetypes": {
                a: {"n": int(r["n"]), "weighted_mean_raw": float(r["weighted_mean_raw"]), "p_weighted": float(r["p_weighted_vs_0"]),
                    "mean_shrunk": float(r["mean_shrunk"]), "ci_shrunk": [float(r["ci_lo_shrunk"]), float(r["ci_hi_shrunk"])],
                    "p_shrunk": float(r["p_shrunk_vs_0"]), "separable_share": float(r["separable_share"])}
                for a, r in arch.iterrows()
            },
        },
    }


def main() -> None:
    numbers = collect()
    OUT.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(numbers, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
