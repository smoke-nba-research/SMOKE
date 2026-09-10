# Data

Source datasets are not committed to this repository. They are all free and public, but the project's IP policy avoids redistributing raw source data. This file records where each input comes from and where to place it so the pipeline reproduces.

`data/model_outputs/` is committed. Everything under `data/raw/` and `data/v2/` is generated or downloaded locally.

## Inputs to place manually

### 2014-15 shot logs (the primary per-shot dataset)

- Source: Kaggle, `dansbecker/nba-shot-logs`.
- Place at: `data/raw/kaggle/shot_logs_2014_15/shot_logs.csv`.
- Used by: the expected-make model, and every test in `src/validate/`.
- Fields include per-shot defender distance, shot clock, touch time, and dribbles. This is the only season with those features published as a clean per-shot table. Coverage: 128,069 shots, 904 of 1,230 games (October 28, 2014 to March 4, 2015), 281 shooters.
- Known quirks, handled in `src/features/kaggle_shot_quality.py`: 5,567 shots have no shot clock (3,554 of them with 24 seconds or less on the game clock, where the shot clock was off); 312 rows carry a negative touch time; the `MATCHUP` string always names the shooter's team first; `FINAL_MARGIN` and `W` are game outcomes and are never used as features.

### Historical advanced and total stats (comparison metrics)

- Source: Kaggle, `sumitrodatta/nba-aba-baa-stats`.
- Place at: `data/raw/kaggle/historical_stats/`.
- Files used: `Advanced.csv` (box plus/minus, offensive box plus/minus, player efficiency rating, true shooting percentage, usage rate) and `Player Totals.csv` (effective field goal percentage per season).
- Used by: 2.5 convergent validity and 2.6 predictive validity.

## Inputs downloaded by the pipeline

### 2015-16 raw optical tracking (the second tracking-rich season)

- Source: GitHub, `sealneaward/nba-movement-data` (a community mirror of the NBA's briefly public SportVU release).
- Downloaded and cached to `data/v2/raw/tracking_1516/` and processed to one parquet per game under `data/v2/processed/shots/` by `python -m src.pulls.build_tracking_season`, which also writes the season file `data/v2/processed/shots_tracking_1516_season.parquet` that `src/validate/stability.py` reads. About 3.6 GB of downloads and two hours of processing; the run is resumable.
- Coverage: 631 games, October 27, 2015 to January 23, 2016 (about half the season). Each shot gets its own release frame; a release whose derived distance disagrees with the league's play-by-play distance by more than 5 feet is marked low confidence and excluded from analysis.
- Used by: 2.4 across-season stability and 2.6 predictive validity (part c).

## Generated outputs

- `data/v2/processed/scored_shots_1415.parquet`: every 2014-15 shot with its features, outcome, and cross-fitted expected make probability. Written once by `python -m src.models.build_model_outputs`; every validation script, figure, the one-pager and the dashboard read it. Nothing downstream refits the model.
- `data/model_outputs/`: committed tables derived from the scored shots (player and team residuals, rank table, hold-out metrics, archetypes, rank correlations) plus `model_card.json` (features, hyperparameters, versions).
- `data/v2/validation/`: every table the validation scripts write, plus `headline_numbers.json` from `python -m src.report.numbers`. Regenerate by running the scripts in the order listed in `VALIDATION.md` under Reproducibility.
