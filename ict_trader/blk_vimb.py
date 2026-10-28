"""Volume Imbalance (VI): two adjacent candles whose bodies do not overlap while their wicks do.

Geometry measured on the lesson frames, where the frames win over the older wording in rules/ict-models.json. Written for a bullish
VI; a bearish VI is the exact mirror (prices negated, highs and lows swapped), so one set of rules covers both directions.
  eTMS36H7HA0 at 0:01   the orange box runs from the top of candle 1's body (an up candle's close) to candle 2's open (an up candle),
                        candle 2's lower wick reaches back into candle 1's wick; a later wick tags the box's lower edge, price reverses.
  -Td-D-vKJDg at 14:40  two lines 3 ticks apart: candle 1's body top and candle 2's open; candle 1's upper wick ends right at the
                        upper line (wick contact can be razor thin); a retest wick goes one tick under the lower line, bodies stay out.
  krdRxohMI3g at 15:00  NQ daily box from the small up candle's close to the next big up candle's open.
  KHxQjsITOLo at 2:00   only shows a daily VI used as a draw (the target of a long): that use is not coded (the engine picks the target).
All the measured pairs are same-colour. No frame shows where inside the box the limit rests (the 1m long at 2:00 enters about a point
above its small box), so the CE default is the written dossier rule and "edge" is the alternative.

  imbalance  candle 2 opens above candle 1's body (bodies apart) while its low still reaches candle 1's high (wicks touching),
             and it closes up: the displacement that made it. With two up candles that is "open 2 above close 1". The zone runs
             from candle 1's body top (its close) to candle 2's open; its midpoint is the consequent encroachment (CE).
             It is not a fair value gap (three candles, wick to wick: no gap between the bodies of two neighbours) and not a
             vacuum (no wick contact: a true gap). It is at least max(2 ticks, 5% of the median candle range) tall.
  entry      a limit as the VI forms, at the CE (VI_ENTRY = "ce") or at the zone's near edge (VI_ENTRY = "edge"), the buy for a
             bullish VI and the sell for a bearish one: only in the direction of the move that made it.
  stop       beyond the far edge by one tick plus the engine's stop buffer. The target is the nearest opposing liquidity (the
             engine's usual one). The order rests VI_EXPIRY candles.
  void       a body closing through the far edge (below the zone of a bullish VI): the zone failed and behaves as an inversion.
             A wick through it is tolerated. The base order lifecycle cannot say this (a candle that closes through the far edge
             has already traded through the limit, so it would fill), so on_bar drops the order first: that candle does not fire.

The liquidity void of the same lesson (a run of one-sided candles with a hole back to the origin, HTQgH11W37o) is NOT coded: it is
only an illustration in the frames, with no stop, target or partial-fill rule.

Price only: no clock, so it also runs on 24/7 synthetic indices. It respects the daily bias like the other pattern models. A tick is
the instrument's pip size, as in the other block models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .opening import FirstPresentedStrategy
from .sessions import Bar, Level
from .strategy import LONG, SHORT, Signal

VI_MIN_TICKS = 2       # the zone is at least this many ticks (the pip size) tall ...
VI_MIN_RANGE = 0.05    # ... and at least this fraction of the median candle range
VI_ENTRY = "ce"        # where the limit rests: "ce" = the zone's midpoint, "edge" = its near edge (the side price comes back to)
VI_STOP_TICKS = 1      # the stop sits this many ticks beyond the far edge, on top of the engine's stop_buffer_pips
VI_EXPIRY = 20         # candles a resting order may wait for the return
VI_WARMUP = 10         # candles of history needed before the first imbalance is read (so the median range means something)
_SEEN_KEEP = 8         # (colour, index of candle 1) pairs remembered: only the pair that just closed is ever asked about again


def _view(b: Bar, m: int) -> tuple[float, float, float, float]:
    """(open, high, low, close) as a bullish imbalance sees the candle: as it is for m = +1; for m = -1 every price is negated,
    so the low becomes the high and a down candle looks like an up candle. A real price is m times its view."""
    return (b.open, b.high, b.low, b.close) if m > 0 else (-b.open, -b.low, -b.high, -b.close)


@dataclass
class VolumeImbalanceStrategy(FirstPresentedStrategy):
    widen_tight_stops = True
    _seen: set = field(default_factory=set)       # (colour, index of candle 1) of the imbalances already read

    def on_bar(self, bar: Bar) -> Signal | None:
        self._drop_void(bar)
        return super().on_bar(bar)

    def _expiry_bars(self) -> int:
        """The life of this model's limit: the live order list counts the candles left from it (the base's 90 minutes is not ours)."""
        return VI_EXPIRY

    def _drop_void(self, bar: Bar) -> None:
        """Cancel the resting orders whose zone this candle failed: its close is through the far edge. Done before the base
        lifecycle runs, because that candle has already traded through the limit and the base would fill it."""
        if not self._setups:
            return
        eps = self.instrument.pip_size * 1e-3
        keep = []
        for s in self._setups:
            far = s.fvg[0] if s.direction == LONG else s.fvg[1]                 # the zone is stored as (low, high)
            if (bar.close - far) * s.direction < -eps:
                self._log(bar.time + timedelta(minutes=self.config.timeframe_minutes), "EN-04",
                          f"vimb volume imbalance limit at {self._fmt(s.limit)} cancelled: a body closed {'below' if s.direction == LONG else 'above'} "
                          f"the far edge {self._fmt(far)} (the zone failed and becomes an inversion)")
            else:
                keep.append(s)
        if len(keep) != len(self._setups):
            self._setups = keep

    def _detect(self, idx: int, close_time: datetime, pip: float) -> None:
        if idx < VI_WARMUP:
            return
        cfg, b = self.config, self._bars
        one, two = b[idx - 1], b[idx]
        if two.close == two.open:
            return                                                  # no displacement: no direction
        m = 1 if two.close > two.open else -1                       # +1: a bullish VI (buy), -1: the mirror
        o1, h1, _, c1 = _view(one, m)
        o2, _, l2, _ = _view(two, m)
        top1 = max(o1, c1)                                          # candle 1's body top (its close when it is an up candle)
        gap, eps = o2 - top1, pip * 1e-3
        if gap <= 0 or l2 > h1 + eps:
            return                                                  # bodies overlap, or no wick contact (a vacuum)
        mr = self._median_range()
        if gap < max(VI_MIN_TICKS * pip, VI_MIN_RANGE * mr) - eps or (m, idx - 1) in self._seen:
            return
        d = LONG if m > 0 else SHORT
        bias, why = self.daily_bias()
        if bias and bias != d:
            return
        self._seen.add((m, idx - 1))
        if len(self._seen) > _SEEN_KEEP:
            self._seen = {k for k in self._seen if k[1] >= idx - 2}

        lo, hi = sorted((m * top1, m * o2))                         # the zone in real prices
        far, near = (lo, hi) if d == LONG else (hi, lo)             # far edge: where the zone fails; near edge: where price comes back to
        ce = (lo + hi) / 2
        entry = ce if VI_ENTRY == "ce" else near
        stop = far - d * (VI_STOP_TICKS + cfg.stop_buffer_pips) * pip
        f, up = self._fmt, d == LONG
        pre = (f"VI-01 candle 2 opened at {f(two.open)}, {'above' if up else 'below'} candle 1's body at {f(m * top1)} (bodies apart), while its "
               f"{'low' if up else 'high'} {f(two.low if up else two.high)} still reaches candle 1's {'high' if up else 'low'} "
               f"{f(one.high if up else one.low)} (wicks touching: not a vacuum) and it closed {'up' if up else 'down'}",
               f"VI-02 {'bullish' if up else 'bearish'} volume imbalance {f(lo)} to {f(hi)}, {gap / pip:.1f} ticks"
               f"{f' = {gap / mr:.2f} of the median range' if mr else ''}, consequent encroachment {f(ce)}",
               f"VI-03 {'buy' if up else 'sell'} limit at the {'CE' if VI_ENTRY == 'ce' else 'near edge'} {f(entry)}, stop {f(stop)} beyond the far edge "
               f"{f(far)}; void if a body closes {'below' if up else 'above'} {f(far)} (it becomes an inversion)",
               f"RL-05 not against the daily draw ({why})")
        lvl = Level(f"VI {f(ce)}", ce, "low" if up else "high")
        self._stage(d, lvl, entry, stop, (lo, hi), idx, close_time, "vimb", "volume imbalance", pre=pre, ce_cancel=False,
                    expiry=VI_EXPIRY, extreme=far, sweep_time=one.time, swing_time=one.time)
