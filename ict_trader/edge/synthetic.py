"""Candidate tables with a KNOWN answer, to test the test.

A validator that has never been shown an edge cannot be trusted to find one, and one that has never been shown pure noise cannot be trusted
to decline. `table(edge=...)` builds a history of candidates whose features look like the engine's and whose outcomes follow the model's own
assumption exactly:

    furthest excursion (R) = exp(delta(x) - L),   L ~ Logistic(0, 1)    =>    P(reaches T R) = sigmoid(delta(x) - ln T)

With delta = 0 that is 1 / (1 + T): a random walk. With `edge` > 0, the features lift delta in the directions ICT teaches.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from . import features
from .candidates import Candidate

# the planted lift per unit of `edge`, on the log-odds of reaching a target: a handful of ICT's questions, each pointing the right way
LIFT = {"bias_with": 0.9, "bias_against": -0.9, "pd_depth": 0.8, "pd_wrong": -0.6, "disp": 0.8, "obstacles": -0.8, "raid_external": 0.5, "in_window": 0.5}


def _row(rnd: random.Random) -> list[float]:
    v = dict.fromkeys(features.NAMES, 0.0)
    r = rnd.random()
    v["bias_with"], v["bias_against"] = (1.0, 0.0) if r < 0.4 else (0.0, 1.0) if r < 0.7 else (0.0, 0.0)
    r = rnd.random()
    v["wbias_with"], v["wbias_against"] = (1.0, 0.0) if r < 0.4 else (0.0, 1.0) if r < 0.7 else (0.0, 0.0)
    pd = rnd.random()
    if rnd.random() < 0.5:
        v["pd_depth"] = pd
    else:
        v["pd_wrong"] = pd
    v["pd20_right"] = rnd.choice((-1.0, 0.0, 1.0))
    v["disp"], v["gap"] = rnd.random(), rnd.random()
    v["obstacles"] = rnd.choice((0.0, 1 / 3, 2 / 3, 1.0))
    v["seek_destroy"] = 1.0 if rnd.random() < 0.15 else 0.0
    r = rnd.random()
    v["raid_external"], v["raid_equal"], v["raid_session"] = (1.0, 0.0, 0.0) if r < 0.3 else (0.0, 1.0, 0.0) if r < 0.5 else (0.0, 0.0, 1.0) if r < 0.8 else (0.0, 0.0, 0.0)
    r = rnd.random()
    v["poi_array"], v["poi_pool"] = (1.0, 0.0) if r < 0.3 else (0.0, 1.0) if r < 0.5 else (0.0, 0.0)
    v["htf_trend"] = rnd.choice((-1.0, -0.5, 0.0, 0.5, 1.0))
    v["room_over"] = rnd.choice((0.0, 0.0, 0.0, 0.5, 1.0))
    v["in_window"] = 1.0 if rnd.random() < 0.45 else 0.0
    return [v[n] for n in features.NAMES]


def table(n: int = 1500, edge: float = 0.0, months: int = 12, seed: int = 1, cost_r: float = 0.05, start: datetime | None = None) -> list[Candidate]:
    """`n` candidates spread over `months` months, with the planted `edge` (0 = pure noise; 1 is a strong, 0.4 a modest, edge)."""
    rnd = random.Random(seed)
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    span = months * 30 * 24 * 60
    idx = {nm: i for i, nm in enumerate(features.NAMES)}
    out: list[Candidate] = []
    for _ in range(n):
        x = _row(rnd)
        delta = edge * sum(w * x[idx[nm]] for nm, w in LIFT.items())
        # centre the planted lift so that an average candidate sits on the random-walk baseline and only the contrast carries the edge
        delta -= edge * _CENTRE
        u = min(max(rnd.random(), 1e-12), 1 - 1e-12)
        mfe = math.exp(delta - math.log(u / (1 - u)))
        t = start + timedelta(minutes=rnd.randrange(span))
        direction = rnd.choice((-1, 1))
        out.append(Candidate("SYN", t, direction, 1.0, 1.0 - direction * 0.001, 0.001, 2.0, cost_r, mfe, mfe < 3.0, x))
    out.sort(key=lambda c: c.time)
    return out


def _centre() -> float:
    rnd = random.Random(0)
    idx = {nm: i for i, nm in enumerate(features.NAMES)}
    rows = [_row(rnd) for _ in range(4000)]
    return sum(sum(w * x[idx[nm]] for nm, w in LIFT.items()) for x in rows) / len(rows)


_CENTRE = _centre()
