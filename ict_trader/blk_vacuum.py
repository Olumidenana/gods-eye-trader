"""Vacuum Block: a true gap between two candles (a session open, the weekend, news such as NFP) that price comes back to.

Geometry measured on the lesson frames (ICT Monthly Mentorship, December 2016, re-uploaded Sept 2022: shPGUz9pU-A at 01:25, 07:45
and 13:25; where it differs from the older wording in rules/ict-models.json the frames win). The slide is a bodies-only schematic of
the bullish block; the bearish one is its exact mirror, so the rules below are written for the gap up and the signed direction d
(LONG = +1, SHORT = -1) mirrors them.

  gap     the slide's candles are two down candles A and B, a small up candle C, then D, which opens far above C's close with nothing
          traded in between: low[D] is above high[C] (a gap down: high[D] below low[C]). Their colours are the schematic's, not a
          condition: any candle pair that leaves a true gap counts. It is at least max(4 ticks, half the median candle range) wide.
          The slide draws no wicks, so C's close is its high and D's open is its low (pixel-measured: c_C is the top of C's body,
          o_D the bottom of D's body) and the frames' zone [c_C, o_D] is the true gap [high[C], low[D]]. That is the edge pair used
          here, and the lesson's own words back it ("a vacuum block contains no candle bodies or wicks within the gap window",
          research/ict-videos/20220909-shPGUz9pU-A.md): with wicks it is the only band where no price traded (a band that a wick
          overlaps is a volume imbalance, the vimb model). Near edge = low[D], far edge = high[C]; CE = their midpoint. low[D] and
          not open[D] is the near edge because a candle always closes at or above its low, so the limit never sits above the market.
  entry   price comes back into the gap from above: a limit at the near edge, rested the moment D closes (the spec's reading; see
          the doubt below). It fills on the first candle that enters the gap, which in both return frames comes before the
          drawn bounce, and the stop below carries the rest of the way down.
  stop    just beyond the far edge (the engine's stop buffer): the frames void the block when price closes below c_C. The 13:25
          frame's body ends exactly on c_C (pixel row 431 of 432) and the next candles rally, so a stop one tick under the edge
          survives the frame; the 07:45 body ends 57% of the way down and survives too.
  stand   "enter only with an immediate bounce, otherwise stand aside" is the lifecycle's: a resting order is cancelled when a body
  aside   closes through the CE, when price reaches the target without coming back, or when the gap is not revisited in VC_EXPIRY
          candles. The limit fills on the first touch, so the bounce itself can only be judged after the fill (by the stop at the
          far edge); judging it first would use the candle that has not closed yet. A candle that closes through the CE has
          already traded through the limit, so the CE cancel only bites when the engine could not fill (the day's signal cap is
          used up, or another order fired on the same candle). The report's "react bars = 3" has no constant for the same reason.

Doubts kept from the frames (the report grades the reaction level and the stop low confidence): at 07:45 the bounce line sits at
y 362, 59% of the way down the gap, but it is also the top of candle A (y 364), so it may be A's high rather than a gap level; at
13:25 the line is c_C, the far edge. The frames never show where the limit rests, so the near edge is the spec's choice.

Gaps are rare on continuously quoted series (a 24/7 synthetic index opens each candle at the last close), which is fine: the model
stays quiet there and works on any series that does have them (indices at the open, FX after the weekend, jump markets). Price
only: no clock, no session. The target is the engine's nearest opposing liquidity, so a gap above everything (nothing left to
aim at) is rejected as TP-01. The daily bias is respected like the other pattern models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .opening import FirstPresentedStrategy
from .sessions import Level
from .strategy import LONG, SHORT

VC_MIN_TICKS = 4         # the gap is at least this many ticks (the pip size) wide ...
VC_MIN_RANGE = 0.5       # ... and at least this fraction of the median candle range
VC_EXPIRY = 20           # candles the order may wait for price to come back into the gap
VC_WARMUP = 20           # candles of history needed before the first gap is read (a full median-range window)
_EPS = 1e-6              # float slack on the size comparison, as a fraction of a tick (0.04 must count as 4 ticks of 0.01)


@dataclass
class VacuumStrategy(FirstPresentedStrategy):
    widen_tight_stops = True
    _seen: set = field(default_factory=set)        # indexes of the candles whose gap was already read (pruned to the last VC_EXPIRY)

    def _expiry_bars(self) -> int:
        """The life of this model's limit: the live order list and the analysis page count the candles left from it (the base's
        90 minutes is 18 candles on 5m, 6 on 15m and 1 on 1h, none of them this model's 20)."""
        return VC_EXPIRY

    def _detect(self, idx: int, close_time: datetime, pip: float) -> None:
        if idx < VC_WARMUP or idx in self._seen:
            return
        cfg, b = self.config, self._bars
        bar, prev = b[idx], b[idx - 1]
        if bar.low > prev.high:                              # a gap up: price comes back down into it, a buy
            d, near, far = LONG, bar.low, prev.high
        elif bar.high < prev.low:                            # a gap down: price comes back up into it, a sell
            d, near, far = SHORT, bar.high, prev.low
        else:
            return                                           # some price traded between the two candles: not a vacuum
        size, slack = (near - far) * d, _EPS * pip
        if size < VC_MIN_TICKS * pip - slack:
            return                                           # cheap exit: too small whatever the volatility is
        mr = self._median_range()                            # only computed when there is a gap to measure
        if size < max(VC_MIN_TICKS * pip, VC_MIN_RANGE * mr) - slack:
            return
        self._seen = {i for i in self._seen if i > idx - VC_EXPIRY}
        self._seen.add(idx)
        bias, why = self.daily_bias()
        if bias and bias != d:
            self._log(close_time, "RL-05", f"vacuum block ignored: against the daily draw ({why})")
            return
        f = self._fmt
        lo, hi = min(near, far), max(near, far)
        ce = (lo + hi) / 2
        entry = near                                         # the edge price meets first on the way back
        stop = far - d * cfg.stop_buffer_pips * pip          # just beyond the other edge
        up = d == LONG
        pre = (f"VC-01 true gap {'up' if up else 'down'}: the gap candle's {'low' if up else 'high'} {f(near)} is {'above' if up else 'below'} "
               f"the {'high' if up else 'low'} {f(far)} of the candle before it, so no price traded between them",
               f"VC-02 vacuum block {f(lo)} to {f(hi)}, {size / pip:.1f} ticks and {size / mr if mr else float('inf'):.1f}x the median candle range "
               f"(at least {VC_MIN_TICKS} ticks and {VC_MIN_RANGE}x), consequent encroachment {f(ce)}",
               f"VC-03 price came back {'down' if up else 'up'} to the gap's {'upper' if up else 'lower'} edge {f(entry)}: {'buy' if up else 'sell'} limit there, "
               f"stop {f(stop)} beyond the {'lower' if up else 'upper'} edge; stand aside if a body closes through {f(ce)} or it is not revisited in {VC_EXPIRY} candles",
               f"RL-05 not against the daily draw ({why})")
        lvl = Level(f"VC {f(entry)}", entry, "low" if up else "high")
        self._stage(d, lvl, entry, stop, (lo, hi), idx, close_time, "vacuum", "vacuum block entry", pre=pre, ce_cancel=True,
                    expiry=VC_EXPIRY, extreme=far, sweep_time=prev.time, swing_time=prev.time)
