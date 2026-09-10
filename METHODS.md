# SMOKE: Methods

Technical writeup for an analytics audience. This document is the working draft of the eventual paper's methods and results narrative; the full test-by-test detail lives in `VALIDATION.md`, and the plain-language version lives in the dashboard's Methods page and the one-pager. Written by Cole Campbell.

## Problem

Field goal percentage conflates two skills: shot selection (taking makeable shots) and shot-making (converting above what a shot's difficulty predicts). Two players with identical percentages can be doing entirely different jobs. Teams evaluating shooters need the two separated, and the public toolkit does not separate them: raw and effective field goal percentage blend both, while the metrics that do separate them (Second Spectrum's quantified shot quality, ShotQuality Inc.'s expected-value models) are proprietary, publish no reliability or error bars, and are unavailable to most of the league's audience.

SMOKE (Shots Made Over Known Expectation) is a transparent, validated alternative: a player's actual field goal percentage minus their model-expected field goal percentage, aggregated per player and shrunk for sample size. The contribution is not the construction, which is deliberately simple. It is that every claim about the metric is tested and reproducible, which no competing shot-quality product offers.

## Data

Three public sources, in defined roles. Both tracking seasons are partial, and the documents say so.

1. **2014-15 Kaggle shot logs** (128,069 shots, 904 of 1,230 games, October 28 to March 4, 281 shooters). The one season with per-shot tracking features published as a clean table: defender distance, shot clock, touch time, dribbles. Carries every load-bearing claim. Two corrections are applied before modeling: a missing shot clock with 24 seconds or less on the game clock means the shot clock was off and is filled with the game clock (3,554 shots, flagged); the 312 negative touch times in the file are set to missing.
2. **2015-16 raw SportVU tracking** (104,721 shots across 631 games, October 27 through January 23, about half the season; the 90,559 high-confidence releases, 86.5%, are used). Extracted from the community-preserved archive of the NBA's briefly public tracking release; per-shot features derived from raw coordinates, one release frame per shot, with a release demoted to low confidence when its derived distance disagrees with the NBA's own reported distance by more than 5 feet. Corroborating evidence only, per the project's data-provenance policy.
3. **Basketball-Reference season aggregates** (via the sumitrodatta Kaggle set). Comparison metrics (box plus/minus family, true shooting, usage) and season efficiency totals for the convergent and predictive tests.

## Metric

A gradient-boosted tree model estimates each shot's make probability from fourteen features known at release: shot distance, defender distance, shot clock, touch time, dribbles, the shooter's shot number in the game, period, game clock, home or away, shot type, and derived buckets for distance zone, touch-time band, late clock, and shot-clock-off. The game's final margin and result are in the source file and are not features. Expected probabilities are cross-fitted (five folds by game), so each shot is scored by a model that never saw its game. On a game-grouped hold-out the model reaches 0.640 area under the ROC curve, 0.649 log loss, and 0.229 Brier score, against 0.500, 0.689, and 0.248 for the base rate.

Per player:

- SMOKE rate = actual field goal percentage minus mean expected field goal percentage.
- SMOKE total = actual makes minus expected makes.
- Published values are shrunk by normal-normal empirical Bayes toward the league mean, with the between-player variance and the league mean estimated by DerSimonian and Laird's random-effects method and each player weighted by his bootstrap standard error; a 95% bootstrap interval and a reliability weight are attached.

The feature set is deliberately small and interpretable. For a metric whose purpose is to be trusted and checked, an explainable model beats a marginally more accurate opaque one.

## Validation

Eight pre-registered tests; full detail and artifacts in `VALIDATION.md`.

| Test | Result |
|---|---|
| Error bars | 53 of 266 players separable from average in one season; the top tier separates cleanly. |
| Shrinkage | Weighted league mean near zero (calibrated); reliability weights 0.29 to 0.76. |
| Within-season reliability | Split-half 0.54 on alternating shots, 0.63 on first versus second half by date (Spearman-Brown). |
| Across-season stability | 0.48, above effective field goal percentage (0.43), below raw field goal percentage (0.67, which raw stability inherits from selection, not skill); holds at every shot floor from 50 to 250. |
| Convergent validity | True shooting 0.62 down to usage 0.19; offensive box plus/minus above total, as predicted; the largest disagreements with offensive box plus/minus are all principled. |
| Predictive validity | Shot diet and shot-making both carry forward into next-season efficiency (each p < 0.001); for next-season shot-making, SMOKE adds information beyond past efficiency (adjusted R-squared 0.09 to 0.16, p < 0.001). |
| Confounds | Opponent quality and venue explain 0.6% of variance; rankings correlate 0.996 before and after adjustment. |
| Archetype fairness | Interior finishers are not penalized (p = 0.41) and are the best-measured group (28% separable); mid-range scorers test 0.6 points below zero (p = 0.045), reported; omnibus p = 0.17. |

The reliability figures cohere (0.54 to 0.63 within season, 0.48 across seasons, intervals consistent with both), which is itself evidence the pipeline is sound.

## Limitations

- SMOKE measures shot-making only. Defense, playmaking, rebounding, and screening are out of scope by design; the metric is one input, not a verdict.
- Public per-shot difficulty data exists for two partial seasons. Stability and predictive results rest on a single season pair.
- The 2015-16 archive is a preserved historical artifact, not an ongoing source, and its provenance is fragile; no public claim depends on it alone.
- Derived defender distance in 2015-16 reads about half a foot tighter than the league's own measurement; the stability test fits each season its own model so neither inherits the other's calibration.
- Most individual players are not separable from average on one season of data. The intervals and shrinkage exist so no one over-reads a single season.
- Bootstrap intervals capture sampling uncertainty in outcomes, not model-estimation uncertainty.

## Future work

The planned v2 is a possession-value decomposition: shot creation (the pre-shot expected value of the look, credited to whoever generated it) plus shot finishing (SMOKE, unchanged), summing on every shooting possession. SMOKE is one component of that framework, not a competitor to it. Full continuous expected-possession-value modeling in the Cervone tradition remains out of scope for public data.

## Reproducibility

All code, the validation dossier, the unit tests, and data-provenance instructions are in this repository (`src/`, `tests/`, `VALIDATION.md`, `data/README.md`). The model is fit once, in `src/models/build_model_outputs.py`, and every test, figure, and dashboard page reads the scored-shots artifact it writes; `python -m src.report.numbers` writes every headline number to one manifest. The dashboard (`dashboard/app.py`) presents the same numbers with no additional processing.
