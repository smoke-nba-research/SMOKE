# SMOKE

**Shots Made Over Known Expectation.** Shot-making, separated from shot selection.

SMOKE is a player's actual field goal percentage minus the percentage a shot-difficulty
model expected, given where and how they shot. Positive means a player makes more shots
than the difficulty of their attempts predicts. The construction is deliberately simple.
The contribution is that every claim about the metric is tested and reproducible.

## The premise

A field goal percentage rewards two different skills at once: taking shots that are easy
to make, and making shots that are hard. They are close to independent in the data, so a
player's shot diet tells you almost nothing about whether he beats it.

![Shot selection and shot-making are separate skills](artifacts/fig2_thesis.png)

## Validation summary

| Test | Result |
|---|---|
| Stability, year over year | 0.51, above effective field goal percentage (0.47) |
| Reliability, within season | 0.57 (split-half, Spearman-Brown corrected) |
| Predicts next-season shot-making | 0.51, versus 0.28 for efficiency |
| Convergent validity | Predicted ordering holds: true shooting 0.54 down to usage 0.19 |
| Confound checks | Opponent quality and venue explain 2.2% of variance; rankings correlate 0.99 after controls |
| Archetype fairness | No playing style penalized; interior big men are the best-measured group |

Every published value ships with a confidence interval. Only 49 of 266 qualified players
separate statistically from league average within a single season, and the leaderboard
says which ones.

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

3. Run the validation scripts in order (later tests read what the first one writes):

   ```
   python -m src.validate.reliability
   python -m src.validate.stability
   python -m src.validate.convergent
   python -m src.validate.predictive
   python -m src.validate.confounds
   python -m src.validate.archetype_fairness
   ```

Outputs land in `data/v2/validation/` and should match the figures in `VALIDATION.md`.

To rebuild the figure suite in `artifacts/` from those outputs:

```
python -m src.report.figures
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

The dashboard reads the validation outputs, so run the scripts above before starting it.
A fresh clone does not ship those files, because they are generated rather than authored.

## Limitations

SMOKE measures shot-making only. It says nothing about defense, playmaking, or rebounding,
and it is one input for evaluation rather than a verdict on a player. Public per-shot
difficulty data exists for one full season (2014-15) and part of a second (2015-16); every
load-bearing claim rests on the former. Most individual players are not statistically
separable from average on a single season of data, which is why confidence intervals and
shrinkage ship with every number.

## Credit

SMOKE grew out of a DAT 490 capstone at Arizona State University built with
**Marc Rajesh** (modeling), **Calder Wyllie** (exploratory analysis and visualization),
and **Germain Meza** (data engineering). The work here extends that foundation.

## License

Code and derived outputs in this repository are MIT licensed (see LICENSE). The license
covers this repository's contents only, not the underlying NBA data, which is obtained
from the public sources listed in `data/README.md`.

## Disclaimer

SMOKE is an independent analysis of publicly available NBA data. It is not affiliated with
or endorsed by the National Basketball Association.
