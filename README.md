# SMOKE

**Shots Made Over Known Expectation.** Shot-making, separated from shot selection.

SMOKE is a player's actual field goal percentage minus the percentage a shot-difficulty
model expected, given where and how they shot. Positive means a player makes more shots
than the difficulty of their attempts predicts. The construction is deliberately simple.
The contribution is that every claim about the metric is tested and reproducible.

## The question

How much of a player's field goal percentage reflects his ability, and how much reflects
the shots he gets? Field goal percentage is makes over attempts; it cannot tell a player
who converts hard shots from one who takes easy ones. Across qualified players in the
2014-15 tracking data, shot difficulty alone explains 59% of the variation in field
goal percentage. Dirk Nowitzki and DeMarcus Cousins shot 46.2% and 46.4% on tracked shots, yet the model expected 40.9% from Nowitzki's attempts and 49.8% from Cousins's: Nowitzki made 43 more shots than his attempts predicted, Cousins 26 fewer.

![Same percentage, different shooters](artifacts/fig_shots_or_shooter.png)

Trusting a player's number takes volume: about 340 shots before half of his SMOKE is
repeatable skill, and about 790 before 70% is, a volume only 28% of qualified
players reach in a full season. That is why every value here ships with an interval.

## Validation summary

| Test | Result |
|---|---|
| Stability, year over year | 0.48, above effective field goal percentage (0.43) |
| Reliability, within season | 0.54 alternating shots, 0.63 first versus second half (split-half, Spearman-Brown corrected) |
| Predicts next-season shot-making | 0.40, versus 0.31 for efficiency; adds information beyond efficiency (p < 0.001) |
| Convergent validity | true shooting 0.62 down to usage 0.19 |
| Confound checks | opponent quality and venue explain 0.6% of variance; rankings correlate 0.996 after controls |
| Archetype fairness | no playing style penalized overall (p = 0.17; interior finishers p = 0.41); Interior Finishers are the best-measured group |

Every published value ships with a confidence interval. Only 53 of
266 qualified players separate statistically from league average within a single
season, and the leaderboard says which ones.

![The leaderboard, with uncertainty shown](artifacts/fig3_leaderboard.png)

Full detail, method, and limitations for all eight tests: [VALIDATION.md](VALIDATION.md).
Technical writeup: [METHODS.md](METHODS.md).
Common objections, answered with the supporting figure: [FAQ.md](FAQ.md).

## Reproducing the results

1. Create a virtual environment and install dependencies:

   ```
   python -m venv .venv
   .venv/Scripts/python.exe -m pip install -r requirements.txt
   ```

2. Place the source datasets. They are free and public but not redistributed here; see
   [data/README.md](data/README.md) for sources and expected paths.

3. Build the model outputs, then the validation outputs, in order (later steps read
   what earlier ones write):

   ```
   python -m src.models.build_model_outputs
   python -m src.pulls.build_tracking_season
   # downloads the 2015-16 archive (~3.6 GB) and takes about two hours
   python -m src.validate.reliability
   python -m src.validate.stability
   python -m src.validate.convergent
   python -m src.validate.predictive
   python -m src.validate.confounds
   python -m src.validate.archetype_fairness
   python -m src.report.figures
   ```

   Outputs land in `data/v2/validation/`, where this repository already ships the
   reference outputs from our run. After re-running, `git diff data/v2/validation`
   shows exactly what, if anything, your run changed. The figure suite is rebuilt
   into `artifacts/`.

4. Run the tests:

   ```
   python -m pytest
   ```

## What SMOKE sees that the box score does not

Ranking players by SMOKE instead of raw field goal percentage moves some of them a long
way. Every large move is explainable in a sentence, which is the property that makes the
metric usable rather than merely defensible.

![The same players, ranked by raw FG% and then by SMOKE](artifacts/movers_chart.png)

## Dashboard

```
python -m streamlit run dashboard/app.py
```

Four pages: Player, Leaderboard, Movers, Methods. See [dashboard/README.md](dashboard/README.md).

The dashboard reads the validation outputs, which ship with the repository, so it runs
on a fresh clone without re-running the pipeline.

## Limitations

SMOKE measures shot-making only. It says nothing about defense, playmaking, or rebounding, and it is one input for evaluation rather than a verdict on a player. Public per-shot difficulty data covers 2014-15 (904 of 1,230 games, Oct 28 to Mar 4, 281 shooters) and part of 2015-16 (631 games, about half a season); every load-bearing claim rests on the former. The model's expectations are cross-fitted: each shot is scored by a model that never saw its game. Most individual players are not statistically separable from average on a single season of data, which is why confidence intervals and shrinkage ship with every number.

## Credit

The paper is by **Cole Campbell**, **Marc Rajesh**, and **Calder Wyllie**. SMOKE grew out of a
DAT 490 capstone at Arizona State University built by the three of them with **Germain Meza**,
whose data engineering built the shot pipeline this work extends; within the capstone Rajesh led
the modeling and Wyllie the exploratory analysis and visualization. The post-capstone research,
validation, and writing were led by Campbell.

## License

Code and derived outputs in this repository are MIT licensed (see [LICENSE](LICENSE)). The
license covers this repository's contents only, not the underlying NBA data, which is not
redistributed here and is obtained from the public sources listed in
[data/README.md](data/README.md). See [NOTICE](NOTICE) for the full scope statement.

## Disclaimer

SMOKE is an independent analysis of publicly available NBA data. It is not affiliated with
or endorsed by the National Basketball Association.
