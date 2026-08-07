# SMOKE: Methods

Technical writeup for an analytics audience. This document is the working draft of the eventual paper's methods and results narrative; the full test-by-test detail lives in `VALIDATION.md`, and the plain-language version lives in the dashboard's Methods page and the one-pager. Written by Cole Campbell.

## Problem

Field goal percentage conflates two skills: shot selection (taking makeable shots) and shot-making (converting above what a shot's difficulty predicts). Two players with identical percentages can be doing entirely different jobs. Teams evaluating shooters need the two separated, and the public toolkit does not separate them: raw and effective field goal percentage blend both, while the metrics that do separate them (Second Spectrum's quantified shot quality, ShotQuality Inc.'s expected-value models) are proprietary, unvalidated in public, and unavailable to most of the league's audience.

SMOKE (Shots Made Over Known Expectation) is a transparent, validated alternative: a player's actual field goal percentage minus their model-expected field goal percentage, aggregated per player and shrunk for sample size. The contribution is not the construction, which is deliberately simple. It is that every claim about the metric is tested and reproducible, which no competing shot-quality product offers.

## Data

Three public sources, in defined roles:

1. **2014-15 Kaggle shot logs** (128,069 shots). The one season with per-shot tracking features published as a clean table: defender distance, shot clock, touch time, dribbles. Carries every load-bearing claim.
2. **2015-16 raw SportVU tracking** (104,722 shots across 631 games, October 27, 2015 through January 23, 2016). Extracted from the community-preserved archive of the NBA's briefly public tracking release; per-shot features derived from raw coordinates and cross-validated against two independent NBA sources (shot-location correlation 0.96 against the league's own shot charts; derived defender-distance distribution within 4 percentage points of the league's published buckets). Corroborating evidence only, per the project's data-provenance policy.
3. **Basketball-Reference season aggregates** (via the sumitrodatta Kaggle set). Comparison metrics (box plus/minus family, true shooting, usage) and season efficiency totals for the convergent and predictive tests.

## Metric

A gradient-boosted tree model estimates each shot's make probability from difficulty features (shot distance, defender distance, shot clock, touch time, dribbles, and derived buckets), trained on 2014-15 (test accuracy 0.619, area under the curve 0.644, against a 0.548 majority baseline). Per player:

- SMOKE rate = actual field goal percentage minus mean expected field goal percentage.
- SMOKE total = actual makes minus expected makes.
- Published values are shrunk by normal-normal empirical Bayes toward the league mean, weighted by each player's bootstrap standard error, with a 95% bootstrap interval and a reliability weight attached.

The feature set is deliberately small and interpretable. For a metric whose purpose is to be trusted and checked, an explainable model beats a marginally more accurate opaque one.

## Validation

Eight pre-registered tests; full detail and artifacts in `VALIDATION.md`.

| Test | Result |
|---|---|
| Error bars | 49 of 266 players separable from average in one season; the top tier separates cleanly. |
| Shrinkage | League mean near zero (calibrated); reliability weights 0.32 to 0.78. |
| Within-season reliability | Split-half 0.57 (Spearman-Brown). |
| Across-season stability | 0.51, above effective field goal percentage (0.47), below raw field goal percentage (0.69, which raw stability inherits from selection, not skill). |
| Convergent validity | Correlations land in the predicted order (true shooting 0.54 down to usage 0.19); the largest disagreements with offensive box plus/minus are all principled. |
| Predictive validity | Forecasts next-season shot-making at 0.51 versus 0.28 for efficiency; does not forecast future efficiency, by design, and the paper says so. |
| Confounds | Opponent quality and venue explain 2.2% of variance; rankings correlate 0.99 before and after adjustment. |
| Archetype fairness | No playing style penalized (Kruskal-Wallis p = 0.22); interior finishers are the best-measured group, not the worst. |

The three reliability figures cohere (0.57 within season, 0.51 across seasons, intervals consistent with both), which is itself evidence the pipeline is sound.

## Limitations

- SMOKE measures shot-making only. Defense, playmaking, rebounding, and screening are out of scope by design; the metric is one input, not a verdict.
- Public per-shot difficulty data exists for one full season (2014-15) and one half-season (2015-16). Stability and predictive results rest on a single season pair.
- The 2015-16 archive is a preserved historical artifact, not an ongoing source, and its provenance is fragile; no public claim depends on it alone.
- Derived defender distance in 2015-16 reads about half a foot tighter than the league's own measurement (release-frame timing); documented, monotone, and immaterial for ordinal use.
- Most individual players are not separable from average on one season of data. The intervals and shrinkage exist so no one over-reads a single season.

## Future work

The planned v2 is a possession-value decomposition: shot creation (the pre-shot expected value of the look, credited to whoever generated it) plus shot finishing (SMOKE, unchanged), summing on every shooting possession. SMOKE is one component of that framework, not a competitor to it. Full continuous expected-possession-value modeling in the Cervone tradition remains out of scope for public data.

## Reproducibility

All code, the validation dossier, and data-provenance instructions are in this repository (`src/validate/`, `VALIDATION.md`, `data/README.md`). Every figure in this document regenerates from public data. The dashboard (`dashboard/app.py`) presents the same numbers with no additional processing.
