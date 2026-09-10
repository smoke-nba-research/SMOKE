# Anticipated Questions

Every question a reviewer, an analytics staffer, or a skeptical reader is likely to raise about SMOKE, with the short answer and the artifact that supports it. Written to be used two ways: as preparation before a conversation, and as a checklist confirming nothing important is unanswered.

Figures are in `artifacts/`. Test detail is in `VALIDATION.md`. Every number below is also in `data/v2/validation/headline_numbers.json`.

---

## About the model

**Does the model actually work, or is it a black box producing plausible numbers?**
It is calibrated: binned out-of-fold predicted make probabilities track observed make rates along the diagonal across the full range, and the mean expected make rate matches the actual rate to within 0.00003. See Figure 1, left panel.

**Did it learn real basketball, or fit noise?**
Hold shot distance fixed and the effect of defender distance is large and monotone in every band: at the rim, field goal percentage rises from about 53% when tightly guarded to about 94% when wide open; on three-pointers, from about 21% to 37%. See Figure 1, right panel.

**Why is a model needed? Can't you just look at defender distance?**
No, and this is a useful demonstration. Raw field goal percentage by defender distance is nearly flat (about 45% in every bucket) because tightly guarded shots are mostly layups while wide-open shots are mostly threes. The two effects cancel. Only a model that conditions on both recovers the real relationship.

**Are the expected probabilities fit in-sample?**
No. Shots are split into five folds by game and each shot is scored by a model that never saw its game. Nothing downstream refits the model; every test, figure and dashboard page reads the same scored-shots artifact.

**What is in the model, exactly?**
Fourteen features known at release: shot distance, defender distance, shot clock, touch time, dribbles, the shooter's shot number in the game, period, game clock, home or away, shot type, and derived buckets for distance zone, touch-time band, late clock and shot-clock-off. The game's final margin and result are in the source file and are deliberately excluded: they are outcomes shared by every shot a team takes that night, and a model that conditions on them credits part of a team's shooting to the fact that it won.

**Why gradient-boosted trees and not something more sophisticated?**
The feature set is small and interpretable on purpose. For a metric whose value proposition is that it can be checked, an explainable model beats a marginally more accurate opaque one. On held-out games the model reaches 0.640 area under the curve, 0.649 log loss and 0.229 Brier score against 0.500, 0.689 and 0.248 for the base rate.

**Why report log loss and Brier score rather than accuracy?**
Because SMOKE subtracts probabilities, not labels. Accuracy at a 0.5 threshold says nothing about how good the probabilities are. It is reported for continuity (0.618 against a 0.543 base rate) but it is not the evidence.

---

## About the premise

**Are shot selection and shot-making really separate, or is this a distinction without a difference?**
They are close to independent in the data. Plotting shot difficulty faced against SMOKE gives a correlation of −0.19: knowing how easy a player's shots are tells you almost nothing about whether he beats them. See Figure 2.

**Isn't this just effective field goal percentage with extra steps?**
No. SMOKE correlates 0.62 with true shooting percentage, which is high enough to confirm both measure shooting and far too low to be a repackage. Its rank correlation with raw field goal percentage is 0.44, and the top movers shift by well over 150 positions. See Figures 2 and 5, and the movers chart.

---

## About trusting a number

**How confident can I be in one player's value?**
Depends on the player, and we publish the answer for each. Every value ships with a 95% bootstrap interval and a reliability weight. 53 of 266 qualified players have intervals excluding zero. See Figure 3.

**Isn't 20% separability a damning result?**
It is an honest one. With a mean of 475 shots per qualified player and binary outcomes, the interval half-width is about 20 makes, so only genuine outliers clear it. This is true of any single-season shooting metric; the difference is that we say so. Presenting 266 confident-looking numbers when 213 are indistinguishable from average would be the actual failure.

**What stops a 200-shot hot streak from topping the leaderboard?**
Empirical Bayes shrinkage, weighted by each player's bootstrap standard error, with the between-player variance from the DerSimonian-Laird estimator. Reliability weights run 0.29 to 0.76. Published values are shrunk; raw values are reported alongside.

---

## About repeatability

**Is this a repeatable skill or one season of noise?**
Repeatable. Year-over-year correlation is 0.48 across 214 players who qualified in both seasons, above effective field goal percentage at 0.43, and the result holds at every minimum-shot threshold from 50 to 250. See Figure 4.

**Raw field goal percentage is more stable than SMOKE (0.67 against 0.48). Doesn't that make it the better metric?**
No, and the gap is expected. Raw percentage is dominated by shot selection, which persists because a player's role persists. A rim-runner posts a stable high percentage every year because he always shoots at the rim. Part of that apparent reliability is measuring role consistency, not shooting skill. SMOKE removes the selection component deliberately, so it cannot inherit the stability that comes with it. The fair comparison is against effective field goal percentage, which SMOKE meets.

**Why fit a separate model to each season instead of one model applied to both?**
Because the 2015-16 features are derived from raw coordinates with a documented half-foot bias in defender distance. A 2014-15 model applied to 2015-16 expects a 46.5% make rate against an actual 45.3%, shifting every 2015-16 value. Fitting each season its own cross-fitted model removes that drift; the one-model design is reported as a robustness check (0.50) and lands in the same place.

**Several reliability numbers appear in the paper. Do they agree?**
Yes, and that coherence is itself evidence. Within-season split-half is 0.54 (alternating shots) to 0.63 (first versus second half by date), across-season is 0.48, and bootstrap intervals are consistent with both. The ordering is what theory predicts: a player resembles himself more within a season than across two.

---

## About the metric's relationship to existing work

**Does it agree with metrics teams already trust?**
True shooting 0.62, offensive box plus/minus 0.46, box plus/minus 0.40, player efficiency rating 0.38, usage rate 0.19. Highest against shooting efficiency and lowest against usage, as predicted, and offensive box plus/minus above total box plus/minus, as predicted. See Figure 5.

**Did every pre-registered ordering hold?**
No. Player efficiency rating was expected to sit between offensive and total box plus/minus and landed 0.02 below total box plus/minus. It is reported as a prediction that did not hold.

**Where it disagrees, is it wrong?**
Every large disagreement resolves to the same explanation: the all-in-one metric credits value SMOKE excludes by design. SMOKE rates Andre Miller and Shaun Livingston far above their offensive box plus/minus because they made difficult shots without carrying an offense. Offensive box plus/minus rates Russell Westbrook, DeMarcus Cousins and Kevin Love far above their SMOKE because their value comes from volume, playmaking and drawing fouls. Both readings are correct about different things.

**Isn't near-zero correlation with usage a problem? Shouldn't good shooters shoot more?**
That correlation is a deliberate negative control. If SMOKE tracked usage closely, it would be measuring role rather than skill. Independence is the desired result.

---

## About prediction

**Does it predict anything?**
For next-season shot-making, adding SMOKE to past effective field goal percentage raises adjusted R² from 0.09 to 0.16 (p < 0.001); the simple correlations are 0.40 for SMOKE and 0.31 for past efficiency. See Figure 6.

**But it does not beat efficiency at predicting future efficiency. Isn't that a failure?**
It is the expected result, and the paper reports it first (0.23 against 0.55). Effective field goal percentage is self-predictive because it carries persistent shot selection, which SMOKE strips out. A metric built to remove the sticky component of efficiency should not be expected to forecast efficiency. Claiming otherwise would be the overreach.

**Does shot-making carry forward at all, then?**
Yes. Decomposing this season's efficiency into the difficulty of the shots taken and the shot-making above it, both parts predict next season's efficiency (coefficients +0.36 and +0.65, each p < 0.001).

**Isn't "predicts next-season shot-making" just the stability correlation again?**
An earlier draft made exactly that mistake and it was corrected. The forecast now regresses next-season SMOKE, from the independently fitted 2015-16 model, on past efficiency and past SMOKE together; the incremental adjusted R² is the claim.

---

## About fairness and confounds

**Does SMOKE penalize interior big men, whose shots are easy by definition?**
No. Interior finishers average −0.002 weighted by precision, which is not distinguishable from zero (p = 0.41). The omnibus test across archetypes returns p = 0.17. See Figure 7.

**Are big men at least measured less precisely?**
The opposite. Interior finishers have the highest separability rate of any archetype (28%, against 13% for catch-and-shoot shooters), because rim attempts are low-variance. Big men are the best-measured group.

**Isn't a near-zero mean within each archetype guaranteed by construction?**
Partly. The archetypes are clusters on the model's own inputs, and a calibrated model has near-zero mean residual within them, so the mean-zero rows are a calibration check. A mean that departs from zero flags miscalibration in that region. The fairness statistic proper is separability, which is why it is reported beside the means.

**Is any archetype penalized?**
Mid-range scorers test slightly below zero (−0.006, p = 0.045). This is the opposite of the anticipated bias and a small effect, roughly 0.6 percentage points. It most likely reflects mild miscalibration in the mid-range region of the feature space, or that a mid-range diet is genuinely hardest to beat expectation on. We report it because a fairness analysis that surfaces only convenient results is not one.

**Does the leaderboard just reflect easy schedules or home cooking?**
No. Opponent quality faced and home-shot share together explain 0.6% of SMOKE's variance. Rankings before and after adjustment correlate at 0.996, the top-20 mean shift is 1.1 places, and the only movement at the very top is Curry and Paul exchanging first and second.

---

## About the data

**Is 2014-15 a full season?**
No. The Kaggle file covers 904 of 1,230 games (October 28 to March 4) and 281 shooters. Every load-bearing claim rests on it, and the paper describes it as partial.

**Where did the second season come from, and can it be trusted?**
Derived from raw 2015-16 optical tracking for 631 games (about half the season; 104,721 shots, of which 90,559 high-confidence releases are used), one release frame per shot, with a release discarded when its derived distance disagrees with the league's play-by-play distance by more than 5 feet. On an eight-game check the derived shot locations correlate 0.99 with the league's own shot charts (mean absolute error 1.4 feet), and the derived defender-distance distribution puts 18.6% of shots in the league's tightest bucket against a published 18.2%. We also document a systematic bias, roughly half a foot tighter than the league's method, and fit that season its own model so it does not inherit the other season's calibration.

**Isn't a half-season of 2015-16 a weakness for the stability test?**
It works against us rather than for us. A half-season is noisier, which depresses the correlation. Landing at 0.48 on that basis is a conservative estimate, not an inflated one.

**Can someone reproduce this?**
Yes, and that is the point. All code and unit tests are public, dataset provenance and placement are documented, the model is fit once and every figure regenerates from public data.

---

## About scope and use

**Why not build full expected possession value instead?**
Full EPV requires continuous optical tracking for current seasons, which is not publicly available. SMOKE is best understood as one component of possession value, specifically the finishing term. The planned next step is a creation-versus-finishing decomposition in which SMOKE is unchanged and the creation half is added.

**How should a team actually use this?**
As one input among several, alongside defense, playmaking and rebounding, and always with the interval attached. It answers one narrow question well and says nothing about the rest of a player's game.

**What is genuinely novel here, given that similar metrics exist?**
Not the estimator. Similar constructions exist, notably BBall Index's Shot Making. What does not exist publicly is a shot-quality metric with published stability, confidence intervals, a fairness analysis, a confound check, cross-fitted expectations and a reproducible pipeline. The contribution is the evidence, not the arithmetic.
