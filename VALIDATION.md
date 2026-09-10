# SMOKE: Validation Dossier

SMOKE (Shots Made Over Known Expectation) is a player's actual field goal percentage minus their model-expected field goal percentage, where the expectation comes from a shot-difficulty model. A positive value means a player makes more shots than the difficulty of their attempts predicts. The metric isolates shot-making skill from shot selection.

This document records every validation test, its result, and its limitations. It is the results section of the eventual paper. Each test's success threshold was written down before the test was first run; the pipeline was then audited and corrected in September 2026 (game-outcome features removed from the model, a matchup-parsing bug fixed, expectations cross-fitted, one release frame per shot in the 2015-16 extraction, per-season stability models, a rebuilt predictive test, and DerSimonian-Laird shrinkage), and every test was re-run against the same thresholds. All figures below are reproducible from the code and data described in the Reproducibility section, and every number quoted here is also written to `data/v2/validation/headline_numbers.json` by `python -m src.report.numbers`.

---

## Data, stated plainly

Both seasons are partial.

- **2014-15 (primary).** The Kaggle shot logs hold 128,069 shots from 904 of the season's 1,230 games (October 28, 2014 to March 4, 2015, 73.5% of the schedule) and 281 shooters. Every load-bearing claim rests on this file. 266 of those 281 shooters clear the 150-shot floor, with a mean of 475 shots (median 435) per qualified player.
- **2015-16 (corroborating).** The community-preserved SportVU archive covers 631 games (October 27, 2015 to January 23, 2016, about half the season). Per-shot features are derived from raw coordinates for 104,721 shots; only the 90,559 releases (86.5%) the extractor is confident in are used. On those, the derived shot distance agrees with the league's play-by-play distance to a mean absolute error of 1.5 feet, derived shot locations correlate 0.99 with the league's published shot charts (mean absolute error 1.4 feet, 8-game check), and the derived defender-distance distribution puts 18.6% of shots in the league's tightest bucket against a published 18.2%, with a documented tendency to read defenders about half a foot closer than the league does. This season supports the across-season tests and nothing else.

## The model

A gradient-boosted tree model estimates each shot's make probability from fourteen features known at release: shot distance, defender distance, shot clock, touch time, dribbles, the shooter's shot number in the game, period, game clock, home or away, shot type, and derived buckets for distance zone, touch-time band, late clock, and a flag for shots taken with the shot clock off. The game's final margin and result are in the source file and are not features; they were removed in the audit because they are outcomes shared by every shot a team takes that night.

Expected probabilities are **cross-fitted**: shots are split into five folds by game and each shot is scored by a model that never saw its game. On a game-grouped 80/20 hold-out the boosted model reaches 0.640 area under the ROC curve, 0.649 log loss, and 0.229 Brier score, against 0.500, 0.689, and 0.248 for a model that always predicts the base rate (a logistic regression reaches 0.630, 0.659, and 0.234). Out of fold across all 128,069 shots, area under the curve is 0.642 and the mean expected make rate matches the actual rate to within 0.00003. Accuracy at a 0.5 threshold (0.618 against a 0.543 base rate) is reported for continuity with the capstone but says little about the probabilities the metric subtracts.

---

## Summary

| Test | Question | Headline result |
|---|---|---|
| 2.1 Error bars | Which players separate from average in one season? | 53 of 266 (20%) have a 95% interval excluding zero. The top tier separates cleanly (Stephen Curry +62, interval [+32, +91]; Chris Paul +61 [+32, +91]). |
| 2.2 Shrinkage | Do small samples distort the leaderboard? | DerSimonian-Laird empirical Bayes; reliability weights 0.29 to 0.76 (median 0.57). Weighted league mean -0.002, near zero. The top holds; low-volume outliers regress. |
| 2.3 Within-season reliability | How much of one season is repeatable? | Split-half, Spearman-Brown corrected: 0.54 on alternating shots, 0.63 on first versus second half by date. |
| 2.4 Across-season stability | Does the skill repeat year to year? | SMOKE r = 0.48, above effective field goal percentage (0.43), below raw field goal percentage (0.67, which is expected). Holds at every shot floor from 50 to 250. |
| 2.5 Convergent validity | Does it agree with established metrics without duplicating them? | True shooting 0.62 down to usage 0.19; every large disagreement with offensive box plus/minus is explainable. |
| 2.6 Predictive validity | Does it forecast the future? | Both the shot diet and the shot-making carry forward into next-season efficiency (each p < 0.001). For next-season shot-making, SMOKE adds information beyond past efficiency: adjusted R-squared 0.09 to 0.16, p < 0.001; correlations 0.40 against 0.31. |
| 2.7 Confound checks | Is the leaderboard an artifact of context? | Opponent quality and home share explain 0.6% of variance. Ranking before versus after adjustment: Spearman 0.996. |
| 2.8 Archetype fairness | Does it penalize a player type? | Interior finishers are not distinguishable from zero (p = 0.41) and are the best-measured group (28% separable). Mid-range scorers test slightly below zero (-0.6 points, p = 0.045), reported. Omnibus p = 0.17. |

Overall: SMOKE is a real but moderately noisy single-season measure of shot-making that firms up with volume, shrinkage, and multiple seasons. It agrees with established metrics where it should, diverges from them for reasons that hold up to inspection, resists the obvious confounds, and treats every playing style on equal footing.

---

## Reproducibility

From a clean clone with the source data placed as `data/README.md` describes:

1. `python -m pytest` (unit tests for the parser, feature policy, shrinkage estimator, and release detection)
2. `python -m src.models.build_model_outputs` (fits the model once, cross-fitted, and writes the scored-shots artifact every later step reads)
3. `python -m src.models.build_analysis_outputs`
4. `python -m src.pulls.build_tracking_season` (downloads and processes the 2015-16 archive, about 3.6 GB and two hours; needed for 2.4 and 2.6 part c)
5. `python -m src.validate.reliability` (2.1, 2.2, 2.3)
6. `python -m src.validate.stability` (2.4)
7. `python -m src.validate.convergent` (2.5)
8. `python -m src.validate.predictive` (2.6)
9. `python -m src.validate.confounds` (2.7)
10. `python -m src.validate.archetype_fairness` (2.8)
11. `python -m src.report.numbers`, then `python -m src.report.figures`

Outputs land in `data/v2/validation/`. Nothing downstream of step 2 refits the model, so the leaderboard, every test, every figure, and the dashboard rest on one identical set of expected probabilities.

---

## 2.1 Error bars (bootstrap intervals)

**Question.** Put a defensible interval on every player's number, and flag anyone not separable from average.

**Method.** Resample each player's shots with replacement 2,000 times, recompute makes-above-expected each time, and take the 2.5 and 97.5 percentiles. This bootstraps the outcomes with expected-make probabilities held fixed. It captures sampling uncertainty in realized shot-making, not model-estimation uncertainty, which is second-order (noted as a limitation). 266 players cleared the 150-shot floor.

**Result.** 53 of 266 players (20%) have a 95% interval that excludes zero. The other 80% are not individually separable from average at this sample size. This is expected: with a mean of 475 shots per qualified player and binary outcomes, the mean interval half-width is about 20 makes, so only genuine standouts clear the bar. The top tier does: Stephen Curry +62 [+32, +91], Chris Paul +61 [+32, +91], Kyle Korver +54 [+33, +74], LeBron James +44 [+16, +74], Klay Thompson +42 [+14, +72]. James Harden, with 1,054 shots and +28, does not (interval [-4, +60]).

**Why it matters.** No commercial shot-quality product (Second Spectrum's qSQ, ShotQuality Inc.) publishes intervals at all. Reporting that most single-season numbers are noisy, and stating which ones are not, is a point of differentiation. It also supports the ethics position that SMOKE is one input among many, not a verdict.

**Artifacts.** `player_smoke_with_error_bars.csv` (266 players: raw value, interval, shrunk value, reliability weight, distinguishable flag, NBA player ID).

## 2.2 Shrinkage (empirical Bayes)

**Method.** Normal-normal empirical-Bayes shrinkage of each player's SMOKE rate toward the league mean, weighted by the bootstrap standard error from 2.1. The between-player variance and the league mean are the DerSimonian-Laird random-effects estimates (inverse-variance weighted, from the Q statistic). The shrunk total is the shrunk rate times shot count.

**Result.** The weighted league mean is -0.002 field goal percentage above expected, near zero, consistent with the model's out-of-fold calibration. The between-player standard deviation is 0.026. Reliability weights range 0.29 to 0.76 (median 0.57): high-volume players barely move, low-volume players regress toward the mean. Stephen Curry (968 shots, reliability 0.73) stays first (+62 raw, +45 shrunk). Kyle Korver (478 shots, reliability 0.57) slides from third by raw total to fifth, because less of his observed value survives shrinkage. Report the shrunk numbers first, with raw alongside.

## 2.3 Within-season reliability (split-half)

**Method.** Two designs, both correlated across the 266 players and Spearman-Brown corrected to full length: (a) alternate shots in chronological order, so both halves share every game; (b) each player's first half of the season against the second, by date, so the halves share no games. Design (a) is the more optimistic of the two; design (b) is the like-for-like comparison with the across-season test.

**Result.** Alternating shots: half-length r = 0.37, corrected to 0.54. First versus second half by date: r = 0.46, corrected to 0.63. A little over half the variation in a single season's values is repeatable; the rest is sampling variation. The estimate is sensitive to the split, which is why both are reported.

**Why it matters.** The within-season figures and the across-season figure from 2.4 (0.48) sit close together. Three reliability measures (the 2.1 intervals, this split-half, and the 2.4 year-over-year correlation) agree at consistent magnitudes, which is itself evidence the pipeline is sound.

**Artifacts.** `split_half_reliability.csv` (both designs).

---

## 2.4 Across-season stability

**Question.** Is a player's SMOKE a repeatable skill across seasons, or single-season noise? Specifically, is it at least as stable as the efficiency metric it refines?

**Threshold (pre-registered).** The year-over-year correlation should be clearly positive and should meet or beat the effective field goal percentage correlation on the same players. Reference point: BBall Index reports about 0.66 for their Shot Making metric versus about 0.28 for effective field goal percentage (Wyman, 2025), on full seasons; both of ours are partial, so expect lower.

**Method.** Only the four features that align across the two extractions are used: shot distance, defender distance, shot clock, and shot type. Touch time and dribbles are excluded because the 2015-16 extraction measures the shooter's final touch while the 2014-15 file measures full-possession touch time, a definitional mismatch that would leak season identity into the model. The same shot-clock rule and the same physical bounds are applied to both seasons. The primary design fits a separate cross-fitted model per season (five folds by game), so the two seasons are treated symmetrically and neither carries calibration drift from the other. The original design, one model fit on 2014-15 and applied unchanged to 2015-16, is kept as a robustness variant. Players are joined on NBA.com player ID. 214 players cleared 100 shots in both seasons.

**Result (primary design).**

| Metric | Year-over-year r | Spearman |
|---|---|---|
| SMOKE | 0.48 | 0.43 |
| raw field goal percentage | 0.67 | 0.63 |
| effective field goal percentage | 0.43 | 0.39 |

The one-model variant gives 0.50 for SMOKE with the same comparators, but it expects a 46.5% make rate on 2015-16 against an actual 45.3%, a +1.3 point calibration gap that shifts every 2015-16 value; the per-season design removes it. The result holds at every shot floor tested from 50 to 250 (SMOKE 0.48 to 0.54, above effective field goal percentage at each floor, always below raw field goal percentage), so it is not an artifact of the 100-shot cutoff.

**Interpretation.** SMOKE is a repeatable skill (0.48), not noise, and it is at least as stable as the efficiency metric it improves on (0.48 versus 0.43). Raw field goal percentage is more stable (0.67), and this is expected rather than a weakness: raw field goal percentage is dominated by shot selection, so a rim-running center posts a stable-high figure every year because he always dunks. SMOKE strips selection out, so it cannot inherit that stability; it isolates the harder, noisier quantity of shot-making above difficulty. The paper must state this directly rather than let a reader stop at "field goal percentage is more stable."

**Caveats.** Both seasons are partial (2014-15 about three quarters, 2015-16 about half), which depresses every correlation against the full-season benchmark. Only two consecutive tracking-rich seasons exist publicly, so this is one season-pair, not an average over many. The comparators are computed on the same filtered shots as SMOKE, not on official season totals. The 2015-16 defender distances read about half a foot tighter than the league's own measurement; the per-season design absorbs that into each season's own model.

**Artifacts.** `stability_correlations.csv` (both designs), `stability_threshold_sweep.csv`, `stability_calibration.csv`, `stability_2season_players.parquet` (214 players, both seasons, both designs).

---

## 2.5 Convergent validity

**Question.** Does SMOKE agree with established all-in-one metrics enough to show it measures shooting, while disagreeing enough to show it is not a duplicate of what teams already have? And are the disagreements explainable?

**Threshold (pre-registered).** Moderate positive correlations (too low means noise, too high means redundant), with true shooting percentage highest (both are shooting efficiency, and SMOKE refines it), offensive box plus/minus above total box plus/minus (which is dragged down by defense, which SMOKE ignores), and usage rate near zero (a negative control, since shot-making skill should be nearly independent of how often a player shoots). The original pre-registration also placed player efficiency rating between the two box plus/minus figures.

**Comparators.** Basketball-Reference advanced stats for 2014-15, from the sumitrodatta Kaggle set (free, public, standard academic benchmark). DARKO was the roadmap's named lead comparator, but its free download is a current-season snapshot with no historical endpoint. Matched 266 of 266 players on normalized names; the key is asserted unique on both sides.

**Result.**

| Comparator | Pearson r | Spearman |
|---|---|---|
| true shooting percentage | 0.62 | 0.58 |
| offensive box plus/minus | 0.46 | 0.44 |
| box plus/minus | 0.40 | 0.39 |
| player efficiency rating | 0.38 | 0.36 |
| usage rate | 0.19 | 0.19 |

True shooting is highest and usage lowest, as predicted, and offensive box plus/minus sits above total box plus/minus, as predicted. Player efficiency rating lands 0.02 below box plus/minus rather than above it; that ordering was pre-registered and did not hold, and is reported as such. SMOKE correlates most with raw shooting efficiency (which it is built to refine) and is nearly orthogonal to usage, confirming it is not re-measuring volume or role.

**Disagreement analysis.** This is the most important part of the test. Ranking players by SMOKE percentile against offensive box plus/minus percentile and taking the largest gaps, every name on both ends is explainable in one sentence, and the explanation is always the same: the all-in-one metric credits value that SMOKE ignores by design.

- SMOKE far above offensive box plus/minus (efficient shot-makers with modest overall box impact, often low-usage role players): Andre Miller (SMOKE 97th percentile, offensive box plus/minus 33rd), Kevin Seraphin (93rd, 2nd), Shaun Livingston (93rd, 27th), Ben Gordon (90th, 17th), Pablo Prigioni (75th, 27th). SMOKE credits them for making the shots they take; offensive box plus/minus sees limited overall offense. Both are true.
- Offensive box plus/minus far above SMOKE (offensive value from playmaking, volume, drawing fouls, or rebounding): Russell Westbrook (SMOKE 24th, offensive box plus/minus 99th, his high-usage 2014-15 season), Brandon Jennings (18th, 96th), DeMarcus Cousins (14th, 90th), Kevin Love (34th, 93rd), Kobe Bryant (23rd, 83rd), Draymond Green (16th, 68th). SMOKE correctly declines to credit them for a skill they provide through other channels.

The metric does not just correlate sensibly; its errors are principled.

**Caveat.** The SMOKE-far-above list skews toward low-minutes players (Miller and Prigioni have 174 shots each), where both metrics are noisier. Shrinkage already tempers the most extreme small-sample cases, and the marquee disagreements on the other side (Westbrook, Cousins, Love) are high-volume players.

**Artifacts.** `convergent_correlations.csv`, `convergent_validity.csv` (266 players, SMOKE against all five comparators with the percentile-gap column).

---

## 2.6 Predictive validity

**Question.** Does year-N SMOKE tell you anything about year N+1? Efficiency figures come from the Basketball-Reference full-season totals; year-N SMOKE is the published shrunk 2014-15 value; year-N+1 SMOKE comes from the 2015-16 model of test 2.4, so the two sides of the forecast are independently defined. 225 players had SMOKE in 2014-15 and at least 200 field goal attempts in 2015-16; 208 of them also qualify in the two-season panel.

**Part (a): predicting next-season efficiency, head to head.** SMOKE does not beat effective field goal percentage, and should not.

| Predictor of next-season effective field goal percentage | Pearson r |
|---|---|
| past effective field goal percentage | 0.55 |
| SMOKE | 0.23 |

Effective field goal percentage is self-predictive because it is dominated by shot selection, which persists year to year, and SMOKE strips selection out.

**Part (b): decomposition.** Year-N efficiency splits, up to the three-point bonus, into the difficulty of the shots taken (expected field goal percentage) and the shot-making above it (SMOKE). Regressing next-season effective field goal percentage on both parts: expected field goal percentage carries a coefficient of +0.36 and SMOKE +0.65, each with p < 0.001, adjusted R-squared 0.22. Both parts of a player's efficiency carry forward, and a point of shot-making carries at least as much as a point of shot diet.

An earlier version of this test regressed next-season efficiency on past efficiency and SMOKE together and found a small negative SMOKE coefficient (-0.37, p = 0.02). That specification is reported for transparency but should not be read as evidence of mean reversion: conditional on a player's efficiency, a higher SMOKE is algebraically a lower expected field goal percentage, so the coefficient measures the forward carry of a harder shot diet, which is the shot-selection persistence in part (a) restated.

**Part (c): forecasting next-season shot-making, the quantity SMOKE measures.** Regressing 2015-16 SMOKE on 2014-15 effective field goal percentage and 2014-15 SMOKE (n = 208): adding SMOKE raises adjusted R-squared from 0.09 to 0.16, with a coefficient of +0.78 (p < 0.001). The simple correlations are 0.40 for SMOKE and 0.31 for past efficiency. SMOKE carries information about future shot-making that past efficiency does not.

**Why all three parts together are stronger than a single win.** They show exactly what SMOKE does and does not predict. It is not an efficiency-forecasting metric and does not claim to be; it is a shot-making metric, and it adds information about future shot-making beyond the incumbent statistic.

**Deferred.** SMOKE against next-contract value (exploratory, needs salary data, free on Basketball-Reference or Spotrac).

**Artifacts.** `predictive_validity.csv`, `predictive_summary.csv`.

---

## 2.7 Confound checks

**Question.** Does the leaderboard survive controlling for context a player does not choose: the defenses they faced and where they played? The skeptic's attack is that a player looks good only because of who and when he shot, not skill.

**What is already controlled.** The base per-shot model includes home/away, period, game clock, and shot clock. The one aggregate confound it cannot see is opponent defensive quality: it knows a defender's distance on a shot, but not that the defender plays for a good defense. That is the real thing to test.

**Method.** For each player, derive the strength of the defenses faced (opponent field goal percentage allowed, computed from the scored shots, so no external join; the opponent is the second team named in each shot's matchup string) and home-shot share. Regress SMOKE on both. The residual is context-adjusted SMOKE. Compare the leaderboard, ranked by shrunk makes above expectation as everywhere else, before and after.

**Result.** Context correlates weakly with SMOKE: opponent defense faced r = 0.06, home share r = -0.05. Together they explain 0.6% of SMOKE's variance. The ranking before versus after adjustment has Spearman 0.996. The top-20 mean rank shift is 1.1 spots (maximum 5); the only movement at the very top is Curry and Paul exchanging first and second.

**Conclusion.** SMOKE reflects shot-making skill, not schedule luck. The players who beat expectation did so against the defenses and in the venues they actually faced, and adjusting for those rescues or demotes no one at the top.

**Caveats.** Rest days are not in the Kaggle data (deferred). Opponent quality here is field goal percentage allowed, an honest self-contained proxy rather than per-possession defensive rating, which would need an external team-ratings join. Home/away is also already a base-model feature, so its player-level control is a redundancy check.

**Artifacts.** `confound_check.csv` (266 players: SMOKE, context-adjusted SMOKE, rank before and after, shift), `confound_summary.csv`.

---

## 2.8 Archetype fairness

This test is the ethics section's empirical backbone.

**Question.** Does SMOKE systematically penalize a player type? The concern: interior big men shoot easy layups and dunks that the model expects them to make, so beating expectation could be structurally near-impossible for them, penalizing centers through no fault of their own. A team must know this before using SMOKE to judge a big man.

**What the test can show.** The archetypes are K-means clusters (four clusters, on shot distance, defender distance, touch time, dribbles, and three-point rate) over all 266 qualified players, not just the overperformers, which would be circular. Because those are the model's own inputs, a calibrated model will show a near-zero mean residual within each cluster; a near-zero archetype mean is therefore partly a calibration property, and a mean that departs from zero is evidence of miscalibration in that region of the feature space. The fairness question proper is about precision: whether one role's players can be measured at all. That is the separability rate by archetype, reported last.

**Method.** Two versions of each archetype's mean: the inverse-variance-weighted mean of the raw rates (each player weighted by the precision of his own bootstrap estimate, the appropriate test) and the plain mean of the shrunk rates (what the leaderboard shows), with a one-sample test against zero for each, a Kruskal-Wallis omnibus test, and the share of each archetype whose 95% interval excludes zero.

**Result.**

| Archetype | n | Weighted mean SMOKE (raw) | p versus 0 | Mean shrunk SMOKE | Separable from average |
|---|---|---|---|---|---|
| On-Ball Creators | 66 | +0.004 | 0.13 | +0.001 | 20% |
| Catch-and-Shoot | 72 | +0.002 | 0.56 | -0.001 | 13% |
| Mid-Range Scorers | 64 | -0.006 | 0.045 | -0.006 | 20% |
| Interior Finishers | 64 | -0.002 | 0.41 | -0.003 | 28% |

The key check passes: Interior Finishers sit at -0.002, not distinguishable from zero (p = 0.41), and they have the highest separability rate of any archetype (28%, versus 13% for catch-and-shoot). Rim shots are low-variance, so a big man who consistently finishes above or below expectation separates more cleanly, not less. Big men are among the best-measured players, the opposite of the concern. The omnibus Kruskal-Wallis test across archetypes gives p = 0.17.

**The one wrinkle, reported plainly.** Mid-Range Scorers test slightly below zero (-0.006 weighted, p = 0.045; -0.006 shrunk with a 95% interval of [-0.011, -0.002]). That is the opposite of the feared bias and a small effect, about 0.6 percentage points of field goal percentage above expected. It most likely reflects mild model miscalibration in the mid-range region of the feature space, or that a mid-range-heavy diet is genuinely the hardest to beat expectation on. It is named here rather than smoothed over.

**Why this is the ethics backbone.** The paper and any pitch can state, with evidence, that SMOKE does not structurally penalize any playing style and measures the role most often assumed to be disadvantaged more precisely than any other. The low rankings of specific interior players reflect their actual shot-making relative to peers, not a structural penalty on their role.

**Caveat.** Run on 2014-15, a single partial season.

**Artifacts.** `archetype_fairness.csv` (266 players, archetype, raw and shrunk SMOKE, separability flag), `archetype_means.csv`, `archetype_omnibus.csv`.
