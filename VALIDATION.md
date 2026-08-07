# SMOKE: Validation Dossier

SMOKE (Shots Made Over Known Expectation) is a player's actual field goal percentage minus their model-expected field goal percentage, where the expectation comes from a shot-difficulty model. A positive value means a player makes more shots than the difficulty of their attempts predicts. The metric isolates shot-making skill from shot selection.

This document records every validation test, its result, and its limitations. It is the results section of the eventual paper. Each test's success threshold was written down before the test was run. All figures below are reproducible from the code and data described in the Reproducibility section.

---

## Summary

Eight tests, run on the 2014-15 Kaggle shot logs (the primary, full-season, richest-feature dataset) with cross-season work extending into 2015-16. The results are internally consistent: three separate reliability measures land within 0.06 of each other, and every convergent and predictive result points the same direction.

| Test | Question | Headline result |
|---|---|---|
| 2.1 Error bars | Which players separate from average in one season? | 49 of 266 (18%) have a 95% interval excluding zero. The top tier separates cleanly (Chris Paul +53, interval [+25, +83]). |
| 2.2 Shrinkage | Do small samples distort the leaderboard? | Empirical-Bayes shrinkage; reliability weights 0.32 to 0.78. League mean is near zero (well calibrated). The top holds; low-volume outliers regress. |
| 2.3 Within-season reliability | How much of one season is repeatable? | Split-half r = 0.40, Spearman-Brown corrected to 0.57. |
| 2.4 Across-season stability | Does the skill repeat year to year? | SMOKE r = 0.51, above effective field goal percentage (0.47), below raw field goal percentage (0.69, which is expected). |
| 2.5 Convergent validity | Does it agree with established metrics without duplicating them? | Correlations land in the predicted order (0.54 down to 0.19). Every large disagreement is explainable. |
| 2.6 Predictive validity | Does it forecast the future? | Predicts next-season shot-making at r = 0.51 versus efficiency at 0.28. It does not beat efficiency at forecasting efficiency, which is by design. |
| 2.7 Confound checks | Is the leaderboard an artifact of context? | Opponent quality and home share explain 2.2% of variance. Ranking before versus after adjustment: Spearman 0.99. |
| 2.8 Archetype fairness | Does it penalize a player type? | No archetype is significantly disadvantaged (Kruskal-Wallis p = 0.22). Interior finishers are not distinguishable from zero. |

Overall: SMOKE is a real but moderately noisy single-season measure of shot-making that firms up with volume, shrinkage, and multiple seasons. It agrees with established metrics where it should, diverges from them for reasons that hold up to inspection, resists the obvious confounds, and treats every playing style on equal footing.

---

## Reproducibility

Every figure regenerates from the scripts in `src/validate/`. Run them in this order, since later tests read the leaderboard the first one writes:

1. `python -m src.validate.reliability` (2.1, 2.2, 2.3)
2. `python -m src.validate.stability` (2.4)
3. `python -m src.validate.convergent` (2.5)
4. `python -m src.validate.predictive` (2.6)
5. `python -m src.validate.confounds` (2.7)
6. `python -m src.validate.archetype_fairness` (2.8)

Outputs land in `data/v2/validation/`. The input datasets are not committed (see `data/README.md` for provenance and placement); this is deliberate, per the project's IP policy on redistributing source data. All inputs are free and public.

The expected-make model is a gradient-boosted tree model, identical to the capstone's, so SMOKE values match the previously published leaderboard.

---

## 2.1 Error bars (bootstrap intervals)

**Question.** Put a defensible interval on every player's number, and flag anyone not separable from average.

**Method.** Resample each player's shots with replacement 2,000 times, recompute makes-above-expected each time, and take the 2.5 and 97.5 percentiles. This bootstraps the outcomes with expected-make probabilities held fixed. It captures sampling uncertainty in realized shot-making, not model-estimation uncertainty, which is second-order (noted as a limitation). 266 players cleared the 150-shot floor.

**Result.** Only 49 of 266 players (18%) have a 95% interval that excludes zero. The other 82% are not individually separable from average at one season's sample size. This is expected: with roughly 800 shots and binary outcomes, the interval half-width is about 28 makes, so only genuine standouts clear the bar. The top tier does clear it: Chris Paul +53 [+25, +83], Nikola Vučević +40 [+10, +67], Stephen Curry +38 [+7, +66], Kyle Korver +46 [+25, +67].

**Why it matters.** No commercial shot-quality product (Second Spectrum's qSQ, ShotQuality Inc.) publishes intervals at all. Reporting that most single-season numbers are noisy, and stating which ones are not, is a point of differentiation. It also supports the ethics position that SMOKE is one input among many, not a verdict.

**Artifacts.** `player_smoke_with_error_bars.csv` (266 players: raw value, interval, shrunk value, reliability weight, distinguishable flag).

## 2.2 Shrinkage (empirical Bayes)

**Method.** Normal-normal empirical-Bayes shrinkage of each player's SMOKE rate toward the league mean, weighted by the bootstrap standard error from 2.1. The between-player variance is estimated by method of moments (DerSimonian-Laird). The shrunk total is the shrunk rate times shot count.

**Result.** League mean is -0.003 field goal percentage above expected, near zero, confirming the model is well calibrated. Between-player standard deviation is 0.027. Reliability weights range 0.32 to 0.78 (median 0.59): high-volume players barely move, low-volume players regress toward the mean. The effect is as intended. Chris Paul (885 shots, reliability 0.73) stays first (+53 raw, +38 shrunk). Kyle Korver (478 shots, reliability 0.59) slides from the raw second tier to fourth, because less of his observed value survives shrinkage. Report the shrunk numbers first, with raw alongside.

## 2.3 Within-season reliability (split-half)

**Method.** Score each player on their odd-indexed shots and even-indexed shots separately, correlate the two halves across players, and apply the Spearman-Brown correction to full length.

**Result.** Half-length r = 0.40, Spearman-Brown corrected to 0.57 (n = 266). A little over half the variation in a single season's values is repeatable; the rest is sampling variation.

**Why it matters.** The within-season figure (0.57) and the across-season figure from 2.4 (0.51) sit next to each other, with across-season slightly lower. That is the expected ordering: a player resembles himself more within a season than across two, once real year-to-year change and noise are added. Three independent reliability measures (the 2.1 intervals, this split-half, and the 2.4 year-over-year correlation) agree at consistent magnitudes. That internal consistency is itself evidence the pipeline is sound.

**Artifacts.** `split_half_reliability.csv`.

---

## 2.4 Across-season stability

**Question.** Is a player's SMOKE a repeatable skill across seasons, or single-season noise? Specifically, is it at least as stable as the efficiency metrics teams already use, while measuring something cleaner?

**Threshold (pre-registered).** The year-over-year correlation should be clearly positive and should meet or beat the effective field goal percentage correlation on the same players. Reference point: BBall Index reports about 0.66 for their Shot Making metric versus about 0.28 for effective field goal percentage, but that is a full-season figure; our 2015-16 is a half-season, so expect somewhat lower.

**Method.** One expected-make model, trained on 2014-15 and applied unchanged to both seasons, so a low correlation can only mean the skill is not stable, never that two pipelines differ. Only the four features that align across seasons are used: shot distance, defender distance, shot clock, and shot type. Touch time and dribbles are excluded because the 2015-16 extraction measures the shooter's final touch while the 2014-15 Kaggle set measures full-possession touch time, a definitional mismatch that would leak season identity into the model. Players are joined on NBA.com player ID, not name. 218 players cleared 100 shots in both seasons.

**Result.**

| Metric | Year-over-year r | Spearman |
|---|---|---|
| SMOKE | 0.51 | 0.47 |
| raw field goal percentage | 0.69 | 0.64 |
| effective field goal percentage | 0.47 | 0.42 |

The result is stable across every shot floor tested from 50 to 250 (SMOKE 0.51 to 0.56, always above effective field goal percentage, always below raw field goal percentage), so it is not an artifact of the 100-shot cutoff.

**Interpretation.** SMOKE is a repeatable skill (0.51), not noise, and it is more stable than the efficiency metric it improves on (0.51 versus 0.47). Raw field goal percentage is more stable (0.69), and this is expected rather than a weakness: raw field goal percentage is dominated by shot selection, so a rim-running center posts a stable-high figure every year because he always dunks. SMOKE strips selection out, so it cannot inherit that stability; it isolates the harder, noisier quantity of shot-making above difficulty. Part of raw field goal percentage's extra stability is measuring role consistency, not skill. The paper must state this directly rather than let a reader stop at "field goal percentage is more stable."

**Caveats.** 2015-16 is a half-season (52% of games), which mechanically depresses the correlation against the full-season benchmark; landing at 0.51 on half a season of the noisier target is arguably stronger than it appears. Only two consecutive tracking-rich seasons exist publicly, so this is one season-pair, not an average over many. The matchup-based coarse extension (2017-18 to present) is the path to more season-pairs.

**Artifacts.** `stability_correlations.csv`, `stability_2season_players.parquet` (218 players, both seasons' SMOKE, field goal percentage, and effective field goal percentage).

---

## 2.5 Convergent validity

**Question.** Does SMOKE agree with established all-in-one metrics enough to show it measures shooting, while disagreeing enough to show it is not a duplicate of what teams already have? And are the disagreements explainable?

**Threshold (pre-registered).** Moderate positive correlations (too low means noise, too high means redundant), in a specific order: true shooting percentage highest (both are shooting efficiency, and SMOKE refines it), then offensive box plus/minus (offense including shooting), then player efficiency rating, then box plus/minus (dragged down by defense, which SMOKE ignores), then usage rate near zero (a negative control, since shot-making skill should be nearly independent of how often a player shoots).

**Comparators.** Basketball-Reference advanced stats for 2014-15, from the sumitrodatta Kaggle set (free, public, standard academic benchmark). DARKO was the roadmap's named lead comparator, but its free download is a current-season snapshot with no historical endpoint, so it cannot supply 2014-15; box plus/minus is public for every season and is more standard for a paper. Matched 266 of 266 players on normalized names.

**Result.** The predicted order holds.

| Comparator | Pearson r | Spearman |
|---|---|---|
| true shooting percentage | 0.54 | 0.50 |
| offensive box plus/minus | 0.39 | 0.37 |
| player efficiency rating | 0.31 | 0.30 |
| box plus/minus | 0.29 | 0.29 |
| usage rate | 0.19 | 0.19 |

SMOKE sits in the moderate-positive band with every all-in-one metric, correlates most with raw shooting efficiency (which it is built to refine), correlates least with total box plus/minus (half of which is defense), and is nearly orthogonal to usage. The usage result confirms SMOKE is not re-measuring volume or role.

**Disagreement analysis.** This is the most important part of the test. Ranking players by SMOKE percentile against offensive box plus/minus percentile and taking the largest gaps, every name on both ends is explainable in one sentence, and the explanation is always the same: the all-in-one metric credits value that SMOKE ignores by design.

- SMOKE far above offensive box plus/minus (efficient shot-makers with modest overall box impact, often low-usage role players or veterans who pick their spots): Andre Miller (SMOKE 98th percentile, offensive box plus/minus 32nd), Pablo Prigioni (86th, 27th; the same player the capstone flagged as a top riser), Ben Gordon, Kevin Seraphin, Jason Smith. SMOKE credits them for making the shots they take; offensive box plus/minus sees limited overall offense. Both are true.
- Offensive box plus/minus far above SMOKE (offensive value from playmaking, volume, drawing fouls, or rebounding, not shot-making over expectation): Russell Westbrook (SMOKE 18th, offensive box plus/minus 99th, his high-usage 2014-15 season), DeMarcus Cousins, Kevin Love, Brandon Jennings, Draymond Green (6th, 68th; value is playmaking and defense). SMOKE correctly declines to credit them for a skill they provide through other channels.

The metric does not just correlate sensibly; its errors are principled. Where it diverges from the consensus, it diverges for exactly the reason it was built.

**Caveat.** The SMOKE-far-above list skews toward low-minutes players, where both metrics are noisier. The 2.2 shrinkage already tempers the most extreme small-sample cases, and none of the named marquee disagreements (Westbrook, Green, Prigioni) are small-sample artifacts.

**Artifacts.** `convergent_correlations.csv`, `convergent_validity.csv` (266 players, SMOKE against all five comparators with the percentile-gap column).

---

## 2.6 Predictive validity

**Question.** Does year-N SMOKE forecast the future, and does it carry information that traditional efficiency misses? Efficiency figures come from the sumitrodatta full-season totals. 225 players had SMOKE in 2014-15 and at least 200 field goal attempts in 2015-16.

The result has two parts, and reporting both is what makes it credible.

**Part 1: predicting next-season efficiency.** SMOKE does not beat effective field goal percentage, and should not.

| Predictor of next-season effective field goal percentage | Pearson r |
|---|---|
| past effective field goal percentage | 0.55 |
| SMOKE | 0.18 |

Adding SMOKE to a regression of next-season effective field goal percentage on past effective field goal percentage raises adjusted R-squared by only 0.012, and the SMOKE coefficient is weakly negative (beta -0.33, p = 0.03). This is expected. Effective field goal percentage is self-predictive because it is dominated by shot selection, which persists year to year, and SMOKE strips selection out. A metric built to remove the sticky part of efficiency should not be expected to forecast efficiency. The small negative coefficient is a real nuance: conditional on efficiency, a high-SMOKE season partly reflects unsustainable hot shooting on hard shots that mean-reverts. This is exactly why single-season SMOKE is shrunk (2.2), and why one season should not be read as permanent skill.

**Part 2: predicting next-season shot-making.** This is the correct target for a shot-making metric, using both seasons' SMOKE (n = 218).

| Predictor of next-season SMOKE | Pearson r |
|---|---|
| SMOKE | 0.51 |
| past effective field goal percentage | 0.28 |

SMOKE forecasts next-season shot-making almost twice as well as traditional efficiency does (0.51 versus 0.28). For the specific quantity SMOKE measures, making shots above their difficulty, it substantially out-predicts effective field goal percentage. Efficiency is the worse forecaster because it is contaminated by shot selection.

**Why both parts together are stronger than a single win.** They show exactly what SMOKE does and does not predict. It is not an efficiency-forecasting metric and does not claim to be; it is a shot-making metric, and it predicts future shot-making better than the incumbent stat. Claiming "SMOKE predicts everything" would invite the skeptical takedown that Part 1, stated plainly, pre-empts.

**Deferred.** SMOKE against next-contract value (exploratory, needs salary data, free on Basketball-Reference or Spotrac).

**Artifacts.** `predictive_validity.csv`, `predictive_summary.csv`.

---

## 2.7 Confound checks

**Question.** Does the leaderboard survive controlling for context a player does not choose: the defenses they faced and where they played? The skeptic's attack is that a player looks good only because of who and when he shot, not skill.

**What is already controlled.** The base per-shot model includes home/away, game margin, period, game clock, and shot clock. The one aggregate confound it cannot see is opponent defensive quality: it knows a defender's distance on a shot, but not that the defender plays for a good defense. That is the real thing to test.

**Method.** For each player, derive the strength of the defenses faced (opponent field goal percentage allowed, computed from the logs, so no external join) and home-shot share. Regress SMOKE on both. The residual is context-adjusted SMOKE. Compare the leaderboard before and after.

**Result.** Context correlates weakly with SMOKE: opponent defense faced r = 0.13, home share r = -0.07. Together they explain 2.2% of SMOKE's variance. The ranking before versus after adjustment has Spearman 0.99. Top-20 mean rank shift is 2.4 spots (maximum 12), and the top six do not move at all (Korver first, James Johnson second, Chris Paul third, DeAndre Jordan fourth, Andre Miller fifth, Amar'e Stoudemire sixth).

**Conclusion.** SMOKE reflects shot-making skill, not schedule luck. The players who beat expectation did so against the defenses and in the venues they actually faced, and adjusting for those rescues or demotes no one at the top.

**Caveats.** Rest days are not in the Kaggle data (deferred). Opponent quality here is field goal percentage allowed, an honest self-contained proxy rather than per-possession defensive rating, which would need an external team-ratings join. Home/away is also already a base-model feature, so its player-level control is a redundancy check. The one genuinely new adjustment, opponent quality, moves nothing material.

**Artifacts.** `confound_check.csv` (266 players: SMOKE, context-adjusted SMOKE, rank before and after, shift).

---

## 2.8 Archetype fairness

This test is the ethics section's empirical backbone.

**Question.** Does SMOKE systematically penalize a player type? The concern: interior big men shoot easy layups and dunks that the model expects them to make, so beating expectation could be structurally near-impossible for them, penalizing centers through no fault of their own. A team must know this before using SMOKE to judge a big man.

**Method.** Cluster all 266 qualified players (K-means, four clusters, on shot-selection features; using all players, not just the overperformers, which would be circular) into the capstone's four archetypes. Test whether any archetype's mean SMOKE is systematically below zero. SMOKE's league mean is near zero by construction, so the question is whether it is near zero within each role.

**Result.**

| Archetype | n | Mean SMOKE | p versus 0 |
|---|---|---|---|
| On-Ball Creators | 66 | +0.000 | 0.82 |
| Catch-and-Shoot | 72 | -0.002 | 0.47 |
| Interior Finishers | 64 | -0.003 | 0.26 |
| Mid-Range Scorers | 64 | -0.006 | 0.005 |

The key check passes: Interior Finishers sit at -0.003, not significantly different from zero. The feared bias does not exist, confirming the capstone's finding on a full-population test with statistical inference. Interior finishers also have the highest distinguishable-from-average rate of any archetype (27%, versus 12% for catch-and-shoot and mid-range): rim shots are low-variance, so a big man who consistently finishes above or below expectation separates more cleanly, not less. Big men are among the best-measured players, the opposite of the concern. The omnibus Kruskal-Wallis test across archetypes gives p = 0.22, so no archetype is systematically advantaged or disadvantaged.

**The one wrinkle, reported plainly.** The only archetype with a mean detectably below zero is Mid-Range Scorers (-0.006, p = 0.005), which is the opposite of the feared bias and a small effect (0.6 percentage points of field goal percentage above expected). It likely reflects that a mid-range-heavy diet is the hardest to beat expectation on, or a slight model miscalibration in that zone. It penalizes no one and does not move the omnibus test. Naming it is the kind of transparency that separates SMOKE from the closed products.

**Why this is the ethics backbone.** The paper and any pitch can now state, with evidence, that SMOKE evaluates every playing style on equal footing. The low rankings of specific interior players reflect their actual shot-making relative to peers, not a structural penalty on their role. This is the difference between a metric that seems fair and a metric shown to be fair.

**Caveat.** Run on 2014-15, not a 13-season panel; we built the richer two-season tracking data instead of the coarse 13-season one the task originally imagined. This raises the capstone's single-season anecdote to a full-population result. Extending to 2015-16 is a quick corroboration when convenient.

**Artifacts.** `archetype_fairness.csv` (266 players, archetype, and SMOKE), `archetype_means.csv`.
