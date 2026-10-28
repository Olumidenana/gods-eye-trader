"""Every candidate the engine can find, with what price did next.

The engine runs with its judgement gates OFF (the "nonarrative" control: any sweep, shift and gap, under the stop and target sanity rules) so
the model sees the setups ICT would take AND the ones he would not, and can learn which questions separate them. For each order that filled,
the outcome is measured from the candles after the fill: how many R price ran in the trade's favour before it touched the stop (the
furthest excursion). A bar that touches the stop is a stop: if a bar could have done either, the stop is assumed first (conservative).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, time
from typing import Sequence

from ..sessions import Bar, Window, to_ny
from . import features

TARGETS = (1.0, 1.5, 2.0, 3.0)   # the grid of targets (in R) the model estimates a chance for


@dataclass
class Candidate:
    market: str
    time: datetime
    direction: int
    entry: float
    stop: float
    risk: float          # price distance, entry to stop
    planned_rr: float    # what the engine aimed at (not used by the model: the target is chosen by the model)
    cost_r: float        # spread and slippage, in R
    mfe_r: float         # furthest excursion in the trade's favour before the stop, in R
    stopped: bool
    x: list[float]

    @property
    def day(self) -> str:
        return self.time.strftime("%Y-%m-%d")

    @property
    def month(self) -> str:
        return self.time.strftime("%Y-%m")

    def hit(self, t: float) -> int:
        """1 if a target of t R was reached before the stop."""
        return 1 if self.mfe_r >= t else 0

    def result(self, t: float) -> float:
        """The result in R of taking this candidate with a target of t R (stop = -1R), after costs."""
        return (t if self.hit(t) else -1.0) - self.cost_r


def outcome(bars: Sequence[Bar], i0: int, direction: int, entry: float, risk: float, horizon: int) -> tuple[float, bool]:
    """(furthest excursion in R, stopped) for a trade filled in bar i0, looking at the bars after it."""
    stop = entry - direction * risk
    best = 0.0
    for b in bars[i0 + 1:i0 + 1 + horizon]:
        if (b.low <= stop) if direction > 0 else (b.high >= stop):
            return best, True
        best = max(best, ((b.high - entry) if direction > 0 else (entry - b.low)) / risk)
    return best, False


def extract(bars: list[Bar], market_key: str = "TEST", cls: str = "forex", tf: int = 5, spread_pips: float | None = None,
            slippage_pips: float = 0.2, horizon_days: float = 1.0) -> list[Candidate]:
    """Run the engine over `bars` with its gates off and measure every order that fills."""
    from .. import scanner
    from ..models import make_strategy

    m = scanner.Market(market_key, market_key, "", "edge", "", 0, "", cls)
    cfg = scanner.auto_config(m, bars, None)
    cfg = scanner.apply_style(cfg, tf)
    inst = cfg.instrument if spread_pips is None else replace(cfg.instrument, spread_pips=spread_pips)
    st = replace(cfg.strategy, model="nonarrative", max_signals_per_day=20, min_rr=1.0)
    # every hour is a candidate: whether the order was inside a killzone is one of the features, not a filter
    st = replace(st, killzones=(Window("any_hour", time(0, 0), time(23, 59)),), flatten_time=None)
    if cls == "synthetic":   # no time theory on a computer-generated market
        st = replace(st, reference_ranges=(), day_rollover=time(0, 0))
    strat = make_strategy(st, inst)
    step = {b.time: i for i, b in enumerate(bars)}
    horizon = int(horizon_days * 1440 / tf)
    pip = inst.pip_size or 1e-4
    out: list[Candidate] = []
    for b in bars:
        sig = strat.on_bar(b)
        if sig is None:
            continue
        i0 = step.get(sig.time - _tf(tf))
        adv = (sig.meta or {}).get("advice") or {}
        c = adv.get("c")
        risk = abs(sig.entry_ref - sig.stop)
        if i0 is None or not c or risk <= 0:
            continue
        mfe, stopped = outcome(bars, i0, sig.direction, sig.entry_ref, risk, horizon)
        cost = ((inst.spread_pips or 0.0) + slippage_pips) * pip / risk
        out.append(Candidate(market_key, sig.time, sig.direction, sig.entry_ref, sig.stop, risk, sig.rr, cost, mfe, stopped, features.vector(c, to_ny(sig.time).hour, cls != "synthetic")))
    return out


def _tf(minutes: int):
    from datetime import timedelta
    return timedelta(minutes=minutes)
