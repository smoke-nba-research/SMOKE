# Anticipated Questions

Every question a reviewer, an analytics staffer, or a skeptical reader is likely to raise about SMOKE, with the short answer and the artifact that supports it. Written to be used two ways: as preparation before a conversation, and as a checklist confirming nothing important is unanswered.

Figures are in `artifacts/`. Test detail is in `VALIDATION.md`.

---

## About the model

**Does the model actually work, or is it a black box producing plausible numbers?**
It is calibrated: binned predicted make probabilities track observed make rates along the diagonal across the full range. See Figure 1, left panel.

**Did it learn real basketball, or fit noise?**
Hold shot distance fixed and the effect of defender distance is large and monotone in every band: at the rim, field goal percentage rises from 53% when tightly guarded to 94% when wide open; on three-pointers, from 22% to 37%. See Figure 1, right panel.

**Why is a model needed? Can't you just look at defender distance?**
No, and this is a useful demonstration. Raw field goal percentage by defender distance is nearly flat (45%, 47%, 43%, 45%) because tightly guarded shots are mostly layups (5.9 feet average) while wide-open shots are mostly threes (20.9 feet average, 59% of them threes). The two effects cancel. Only a model that conditions on both recovers the real relationship.

**Why gradient-boosted trees and not something more sophisticated?**
The feature set is small and interpretable on purpose. For a metric whose value proposition is that it can be checked, an explainable model beats a marginally more accurate opaque one. Model accuracy is 61.9% against a 54.8% majority baseline, with area under the curve 0.644.

---

## About the premise

**Are shot selection and shot-making really separate, or is this a distinction without a difference?**
They are close to independent in the data. Plotting shot difficulty faced against SMOKE gives a correlation of −0.21: knowing how easy a player's shots are tells you almost nothing about whether he beats them. DeAndre Jordan and Omer Asik took similarly easy shots to very different effect. See Figure 2.

**Isn't this just effective field goal percentage with extra steps?**
No. SMOKE correlates 0.54 with true shooting percentage, which is high enough to confirm both measure shooting and far too low to be a repackage. It ranks players differently enough that the top movers shift by nearly 200 positions. See Figures 2 and 5, and the movers chart.

---

## About trusting a number

**How confident can I be in one player's value?**
Depends on the player, and we publish the answer for each. Every value ships with a 95% bootstrap interval and a reliability weight. Only 49 of 266 qualified players have intervals excluding zero. See Figure 3.

**Isn't 18% separability a damning result?**
It is an honest one. With roughly 800 shots and binary outcomes, the interval half-width is about 28 makes, so only genuine outliers clear it. This is true of any single-season shooting metric; the difference is that we say so. Presenting 266 confident-looking numbers when 217 are indistinguishable from average would be the actual failure.

**What stops a 200-shot hot streak from topping the leaderboard?**
Empirical Bayes shrinkage, weighted by each player's bootstrap standard error. Reliability weights run 0.32 to 0.78. Published values are shrunk; raw values are reported alongside.

---

## About repeatability

**Is this a repeatable skill or one season of noise?**
Repeatable. Year-over-year correlation is 0.51 across 218 players who qualified in both seasons, and the result holds at every minimum-shot threshold from 50 to 250. See Figure 4.

**Raw field goal percentage is more stable than SMOKE (0.69 against 0.51). Doesn't that make it the better metric?**
No, and the gap is expected. Raw percentage is dominated by shot selection, which persists because a player's role persists. A rim-runner posts a stable high percentage every year because he always shoots at the rim. Part of that apparent reliability is measuring role consistency, not shooting skill. SMOKE removes the selection component deliberately, so it cannot inherit the stability that comes with it. The fair comparison is against effective field goal percentage, which SMOKE beats.

**Three reliability numbers appear in the paper. Do they agree?**
Yes, and that coherence is itself evidence. Within-season split-half is 0.566, across-season is 0.506, and bootstrap intervals are consistent with both. The ordering is what theory predicts: a player resembles himself more within a season than across two.

---

## About the metric's relationship to existing work

**Does it agree with metrics teams already trust?**
In the order predicted before running the test: true shooting 0.54, offensive box plus/minus 0.39, player efficiency rating 0.31, box plus/minus 0.29, usage rate 0.19. Highest against shooting efficiency, lowest against the composite that is half defense, near zero against usage. See Figure 5.

**Where it disagrees, is it wrong?**
Every large disagreement resolves to the same explanation: the all-in-one metric credits value SMOKE excludes by design. SMOKE rates Andre Miller and Pablo Prigioni far above their offensive box plus/minus because they made difficult shots without carrying an offense. Offensive box plus/minus rates Russell Westbrook and Draymond Green far above their SMOKE because their value comes from volume, playmaking, and defense. Both readings are correct about different things.

**Isn't near-zero correlation with usage a problem? Shouldn't good shooters shoot more?**
That correlation is a deliberate negative control. If SMOKE tracked usage closely, it would be measuring role rather than skill. Independence is the desired result.

---

## About prediction

**Does it predict anything?**
It predicts next-season shot-making at 0.51, against 0.28 for effective field goal percentage. For the quantity it measures, it nearly doubles the incumbent statistic. See Figure 6.

**But it does not beat efficiency at predicting future efficiency. Isn't that a failure?**
It is the expected result, and the paper reports it before anyone asks. Effective field goal percentage is self-predictive because it carries persistent shot selection, which SMOKE strips out. A metric built to remove the sticky component of efficiency should not be expected to forecast efficiency. Claiming otherwise would be the overreach.

**The incremental coefficient on SMOKE is slightly negative. What does that mean?**
Conditional on efficiency, an unusually high SMOKE season partly reflects unsustainable hot shooting that mean-reverts. That is a real finding, it is consistent with why single-season values are shrunk, and it is reported rather than omitted.

---

## About fairness and confounds

**Does SMOKE penalize interior big men, whose shots are easy by definition?**
No. Interior finishers average −0.003, which is not distinguishable from zero (p = 0.26). The omnibus test across archetypes returns p = 0.22. See Figure 7.

**Are big men at least measured less precisely?**
The opposite. Interior finishers have the highest separability rate of any archetype (27%, against 12% for catch-and-shoot and mid-range shooters), because rim attempts are low-variance. Big men are the best-measured group.

**Is any archetype penalized?**
Mid-range scorers test slightly below zero (−0.006, p = 0.005). This is the opposite of the anticipated bias and a small effect, roughly 0.6 percentage points. It may reflect that a mid-range diet is genuinely hardest to beat expectation on, or mild model miscalibration in that zone. We report it because a fairness analysis that surfaces only convenient results is not one.

**Does the leaderboard just reflect easy schedules or home cooking?**
No. Opponent quality faced and home-shot share together explain 2.2% of SMOKE's variance. Rankings before and after adjustment correlate at 0.99, the top-20 mean shift is 2.4 places, and the top six positions do not move.

---

## About the data

**Why only 2014-15 for the main results?**
It is the only season with per-shot tracking detail published as a clean table. Every load-bearing claim rests on it.

**Where did the second season come from, and can it be trusted?**
Derived from raw 2015-16 optical tracking, then cross-validated against two independent NBA sources: our shot locations correlate 0.96 with the league's own shot charts, and our derived defender-distance distribution reproduces the league's published bucket shares (18.1% against 18.2% in the tightest bucket). We also document a systematic bias, roughly half a foot to one foot tighter than the league's method, arising from release-frame timing.

**Isn't a half-season of 2015-16 a weakness for the stability test?**
It works against us rather than for us. A half-season is noisier, which depresses the correlation. Landing at 0.51 on that basis is a conservative estimate, not an inflated one.

**Can someone reproduce this?**
Yes, and that is the point. All code is public, dataset provenance and placement are documented, and every figure regenerates from public data.

---

## About scope and use

**Why not build full expected possession value instead?**
Full EPV requires continuous optical tracking for current seasons, which is not publicly available. SMOKE is best understood as one component of possession value, specifically the finishing term. The planned next step is a creation-versus-finishing decomposition in which SMOKE is unchanged and the creation half is added.

**How should a team actually use this?**
As one input among several, alongside defense, playmaking, and rebounding, and always with the interval attached. It answers one narrow question well and says nothing about the rest of a player's game.

**What is genuinely novel here, given that similar metrics exist?**
Not the estimator. Similar constructions exist, notably BBall Index's Shot Making. What does not exist publicly is a shot-quality metric with published stability, confidence intervals, a fairness analysis, a confound check, and a reproducible pipeline. The contribution is the evidence, not the arithmetic.
