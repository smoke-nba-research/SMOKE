# Data

Source datasets are not committed to this repository. They are all free and public, but the project's IP policy avoids redistributing raw source data. This file records where each input comes from and where to place it so the pipeline reproduces.

`data/model_outputs/` is committed. Everything under `data/raw/` and `data/v2/` is generated or downloaded locally.

## Inputs to place manually

### 2014-15 shot logs (the primary per-shot dataset)

- Source: Kaggle, `dansbecker/nba-shot-logs`.
- Place at: `data/raw/kaggle/shot_logs_2014_15/shot_logs.csv`.
- Used by: the expected-make model, and every test in `src/validate/`.
- Fields include per-shot defender distance, shot clock, touch time, and dribbles. This is the only season with those features published as a clean per-shot table.

### Historical advanced and total stats (comparison metrics)

- Source: Kaggle, `sumitrodatta/nba-aba-baa-stats`.
- Place at: `data/raw/kaggle/historical_stats/`.
- Files used: `Advanced.csv` (box plus/minus, offensive box plus/minus, player efficiency rating, true shooting percentage, usage rate) and `Player Totals.csv` (effective field goal percentage per season).
- Used by: 2.5 convergent validity and 2.6 predictive validity.

## Inputs downloaded by the pipeline

### 2015-16 raw optical tracking (the second tracking-rich season)

- Source: GitHub, `sealneaward/nba-movement-data` (a community mirror of the NBA's briefly public SportVU release).
- Downloaded and cached to `data/v2/raw/tracking_1516/` by `src/pulls/tracking_1516.py`.
- Processed to one parquet per game under `data/v2/processed/` by `src/pulls/build_tracking_season.py`.
- Used by: 2.4 across-season stability and 2.6 predictive validity.

## Generated outputs

- `data/v2/validation/` holds every table the validation scripts write. Regenerate by running the scripts in the order listed in `VALIDATION.md` under Reproducibility.
