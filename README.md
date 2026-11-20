# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/FandeadAlter/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/FandeadAlter/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for make.fun's bounty "Forecast how many GTA VI players earn each trophy in week one". The program reads one case — every read the house made of Exophase's Grand Theft Auto VI PlayStation trophy page before the case's checkpoint — and forecasts each of the game's 55 trophies' earn rate at the case's target as five quantiles (5%, 25%, 50%, 75%, 95%). The final case is cut at the deadline, 2026-11-21 11:30 UTC, and its target is 2026-11-26 00:00 UTC, a week after launch.

## The model

Three ideas, per trophy, all in logit space against log-time (u = ln(1 + hours since the release, 2026-11-19T00:00Z)).

**Where the rate is heading.** The history's distinct rate steps over the last 24 hours are fitted with a weighted least-squares line of logit(rate) on u, and the slope is shrunk toward zero by its own signal-to-noise ratio. A flat history — launch night, before Exophase's first refresh — yields no trend at all, so the forecast falls back to the last rate. That shrinkage is what keeps the practice cases honest: their histories are perfectly flat, so nothing is claimed about a move that is not in the data.

**How far it can go.** The fitted slope is not extended as a straight line: a line in logit space runs away over a 4.5-day horizon. Instead the forecast approaches an asymptote exponentially, with the rate constant chosen so the approach starts with exactly the fitted slope. A rising rate approaches a ceiling set by the trophy's tier — platinum 4%, gold 30%, silver 25%, bronze 55% for the first week — never below where it already stands; a falling rate decays toward a floor at a quarter of its current rate. Story trophies rise fast then flatten as most active players earn them; grindy and late trophies rise slowly and saturate low. The ceiling is a cap, not a target: a trophy whose slope is small barely moves toward it.

**How sure it is.** The spread (a logit standard deviation) combines four terms: a random-walk term that grows with the square root of the horizon, a trend term proportional to how far the median moves (a big move is an uncertain move), the residual scatter of the fit, and a base term for the refresh jumps Exophase makes on its own schedule. Quantiles are the median shifted by the normal z-scores. The spread is deliberately wider than the house baseline's on a short horizon, because an Exophase refresh can move every rate at once — the practice cases straddle exactly such a refresh — and narrower than a pure random walk on the long horizon, because the trend model earns its keep.

Rates at or near 0% and 100% are clamped to [0.01, 99.99] before the logit, and every quantile is clamped back into [0, 100] and kept non-decreasing, so the trophy already at 100% stays there and no row can come out inverted.

## Scoring

The score is the mean pinball loss on the logit scale over every trophy and quantile: 2 × max(τ × d, (τ − 1) × d), where d = logit(y) − logit(q), y the trophy's earn rate in the answering read and q the quantile at level τ. A forecast whose five quantiles are equal scores exactly its mean absolute logit error. The logit scale makes a miss from 1% to 2% cost about what a miss from 33% to 50% does, so rare trophies count as much as common ones. A prize needs a loss strictly below the best house entry (the persistence baseline, the fitted-curve baseline, and the house model forecast).

On the practice cases (flat launch-night histories, 90-minute horizons) the program's median is persistence — the honest answer when the data shows no move — with a band about three times wider than the house baseline's, so it copes with the refresh the practice straddles. On a synthetic final-like case with two and a half days of rising rates and a growing tracked population, it scored a loss of 0.28 against the persistence baseline's 0.53, because its medians extrapolate the growth instead of standing still.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `*.in` file in the input directory and writes the matching `*.out` to the output directory. It is deterministic: two runs on the same case write the same bytes. The program also reads `$MAKEFUN_INPUT_DIR` and `$MAKEFUN_OUTPUT_DIR` when they are set, else `/input` and `/output`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=me6ajgxn&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
