"""gta6-trophy-rarity-forecast-01 entry in Python 3.13 (standard library only).

Forecasts, for each of GTA VI's 55 PlayStation trophies, its earn rate on
Exophase at the case's target, as five quantiles (5%, 25%, 50%, 75%, 95%),
scored by a pinball loss on the logit scale.

The model, per trophy, in three parts.

1. Where the rate is heading.  Work in logit space, L = logit(rate), against
   log-time u = ln(1 + hours since the release, 2026-11-19T00:00Z).  The
   history's distinct rate steps over the last 24 hours are fitted with a
   weighted least-squares line of L on u; the slope is shrunk toward zero by
   its own signal-to-noise ratio, so a flat history (launch night, no
   refresh yet) yields no trend at all and the forecast falls back to the
   last rate.  That shrinkage is what keeps the practice cases honest: their
   histories are perfectly flat, so nothing is claimed about a move.

2. How far it can go.  The fitted slope is not extended as a straight line:
   a line in logit space runs away over a 4.5-day horizon.  Instead the
   forecast approaches an asymptote exponentially, with the rate constant
   chosen so the approach starts with exactly the fitted slope.  A rising
   rate approaches a ceiling set by the trophy's tier (platinum 4%, gold
   30%, silver 25%, bronze 55% for the first week), never below its current
   rate; a falling rate decays toward a floor (a quarter of its current
   rate).  Story trophies rise fast then flatten as most active players earn
   them; grindy and late trophies rise slowly and saturate low.  The
   ceiling is a cap, not a target: a trophy whose slope is small barely
   moves toward it.

3. How sure it is.  The spread (a logit standard deviation) combines four
   terms in quadrature: a random-walk term that grows with the square root
   of the horizon, a trend term proportional to how far the median moves
   (a big move is an uncertain move), the residual scatter of the fit, and
   a base term for the refresh jumps Exophase makes on its own schedule.
   Quantiles are the median shifted by the normal z-scores.  The spread is
   deliberately wider than the house baseline's on a short horizon, because
   an Exophase refresh can move every rate at once and the practice shows
   exactly that, and narrower than a pure random walk on the long horizon,
   because the trend model earns its keep.

Rates at or near 0% and 100% are clamped to [0.01, 99.99] before the
logit, and every quantile is clamped back into [0, 100] and kept
non-decreasing, so the trophy already at 100% stays there.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so a module shipped beside
main.py imports only after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there): JSON
    {"schema": "makefun/gta6-trophies-case/v1", "id", "at",
    "horizon_minutes", "target", "levels", "game", "trophies", "history",
    "notes"}, where `trophies` lists the case's trophies {id, name,
    description, tier, points, secret} and `history` every read before
    `at`, oldest first, each {slot, at, players_tracked, trophies_earned,
    rates}, `rates` in the order of `trophies` (null where that read
    lacked the trophy);
  - write <id>.out to the output directory: {"trophies": [{"id": 1, "q":
    [q05, q25, q50, q75, q95]}, ...]}, one row for each of the case's
    trophies, each id once, no other key, each q five finite numbers from
    0 to 100 (percent), non-decreasing; at most 65,536 bytes. Anything
    else is an invalid case, scored as if every quantile equalled the
    trophy's rate in the last read, and an entry with an invalid case
    takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# The game's release, the origin of log-time.  Every read before it counts
# as time zero: the first recorder started 10 minutes before midnight.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The ceiling a rising rate approaches, in percent, by tier: what share of
# tracked players could plausibly hold a trophy of that tier a week after
# launch.  A trophy already above its tier's ceiling gets a ceiling above
# where it stands, never below.
CEILING_PCT = {"platinum": 4.0, "gold": 30.0, "silver": 25.0, "bronze": 55.0}
# The floor a falling rate decays toward, as a fraction of its current rate:
# new players keep joining and diluting it, but its earners keep growing,
# so it does not fall to nothing.
FLOOR_FRAC = 0.25

# The window the trend is fitted over, in hours before the checkpoint.
TREND_WINDOW_HOURS = 24.0

# The spread's pieces, in logit units.  Random-walk diffusion per
# square-root day; the base term is the jump an Exophase refresh can make
# on its own schedule; the trend term scales with how far the median moves.
SIGMA_RW = 0.32
SIGMA_BASE = 0.35
SIGMA_TREND = 0.6
SIGMA_FIT_MIN = 0.08


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(when: datetime) -> float:
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def log_time(when: datetime) -> float:
    """u = ln(1 + hours since release): the clock the trend is fitted on."""
    return math.log1p(hours_since_release(when))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.  A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def fit_trend(points):
    """Weighted least squares of logit rate on log-time.

    points is a list of (u, L, weight).  Returns (slope, residual_sd,
    slope_se).  A single point, or points at one time, carry no slope:
    slope 0 and an infinite standard error, so the caller shrinks the
    slope to zero.
    """
    if len(points) < 2:
        return 0.0, 0.0, float("inf")
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return 0.0, 0.0, float("inf")
    um = sum(w * u for u, _, w in points) / sw
    lm = sum(w * l for _, l, w in points) / sw
    sxx = sum(w * (u - um) ** 2 for u, _, w in points)
    sxy = sum(w * (u - um) * (l - lm) for u, l, w in points)
    if sxx <= 0:
        return 0.0, 0.0, float("inf")
    slope = sxy / sxx
    intercept = lm - slope * um
    n = len(points)
    if n > 2:
        # The residual scale is the plain mean of squared residuals, so it
        # does not depend on the recency weights.
        rss_plain = sum((l - (intercept + slope * u)) ** 2 for u, l, _ in points)
        resid_sd = math.sqrt(rss_plain / (n - 2))
    else:
        resid_sd = 0.0
    slope_se = resid_sd / math.sqrt(sxx) if resid_sd > 0 else 0.0
    return slope, resid_sd, slope_se


def forecast_trophy(trophy, series, u_target, horizon_days):
    """series is a list of (when, rate) for this trophy, oldest first,
    every read that carried it.  Returns the five quantiles in percent."""
    # Collapse consecutive reads with the same rate into one step: Exophase
    # refreshes every few hours, so most reads repeat the last refresh and
    # would otherwise overweight the flat stretches.
    steps = []
    for when, rate in series:
        if rate is None:
            continue
        if steps and steps[-1][1] == rate:
            continue
        steps.append((when, rate))
    if not steps:
        raise ValueError("trophy with no rate in the history")

    last_when, last_rate = steps[-1]
    u_last = log_time(last_when)
    l_last = logit(last_rate)

    # The trend window: the last 24 hours of reads, or all of them when the
    # history is shorter than that.
    window_start = last_when.timestamp() - TREND_WINDOW_HOURS * 3600.0
    window = [(w, r) for (w, r) in steps if w.timestamp() >= window_start]
    if len(window) < 2:
        window = steps
    points = []
    for when, rate in window:
        # Recent steps count more: a step's weight halves every 12 hours.
        age_h = (last_when - when).total_seconds() / 3600.0
        points.append((log_time(when), logit(rate), 0.5 ** (age_h / 12.0)))
    slope, resid_sd, slope_se = fit_trend(points)

    # Shrink the slope toward zero by its signal-to-noise ratio.  A flat
    # history gives slope 0; a noisy one gives a slope the data cannot
    # support, so most of it is discarded.
    if slope == 0.0:
        slope_shrunk = 0.0
    elif slope_se == 0.0:
        slope_shrunk = slope
    elif math.isinf(slope_se):
        slope_shrunk = 0.0
    else:
        slope_shrunk = slope * (slope * slope) / (slope * slope + slope_se * slope_se)

    # The asymptote the forecast approaches, and the rate constant that
    # makes the approach start with the fitted slope.
    tier = trophy.get("tier", "bronze")
    du = u_target - u_last
    if slope_shrunk > 0:
        ceil_pct = CEILING_PCT.get(tier, 55.0)
        l_ceil = max(logit(ceil_pct), l_last + 0.5)
        gap = max(0.2, l_ceil - l_last)
        k = slope_shrunk / gap
        l_med = l_ceil - (l_ceil - l_last) * math.exp(-k * du)
    elif slope_shrunk < 0:
        l_floor = min(logit(max(0.05, last_rate * FLOOR_FRAC)), l_last - 0.5)
        gap = max(0.2, l_last - l_floor)
        k = -slope_shrunk / gap
        l_med = l_floor + (l_last - l_floor) * math.exp(-k * du)
    else:
        l_med = l_last

    # The spread: random walk over the horizon, the trend's own
    # uncertainty, the fit's residual scatter, and the refresh jump.
    s_rw = SIGMA_RW * math.sqrt(horizon_days)
    s_trend = SIGMA_TREND * abs(l_med - l_last)
    s_fit = max(resid_sd, SIGMA_FIT_MIN)
    spread = math.sqrt(s_rw * s_rw + s_trend * s_trend + s_fit * s_fit + SIGMA_BASE * SIGMA_BASE)

    out = []
    prev = None
    for z in Z:
        q = inv_logit(l_med + z * spread)
        q = min(100.0, max(0.0, q))
        if prev is not None and q < prev:
            q = prev
        out.append(round4(q))
        prev = q
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    u_target = log_time(target)
    horizon_days = case["horizon_minutes"] / 1440.0
    history = case["history"]
    trophies = case["trophies"]

    rows = []
    for i, trophy in enumerate(trophies):
        series = []
        for read in history:
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            series.append((parse_time(read["at"]), rate))
        rows.append({"id": trophy["id"], "q": forecast_trophy(trophy, series, u_target, horizon_days)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
