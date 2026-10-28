"""Suspension Block: three candles closing the same way with a volume imbalance either side of the middle one.

Geometry measured on the lesson frames (the NQ daily blocks in 486N8ow3U2A and zyKAPNk_prA, the intraday blocks in
-Td-D-vKJDg; where it differs from the older wording in rules/ict-models.json the frames win). Written for three up
candles; three down candles are the exact mirror (prices negated), so one set of rules covers both.

  candles   A, B, C close up. Lower imbalance: B opens above A's close while B's low still reaches A's high (bodies apart,
            wicks touching). Upper imbalance: C opens above B's close while C's low still reaches B's high. A gap with no
            wick contact is a vacuum, not an imbalance. Each imbalance is at least max(2 ticks, 5% of the median range).
  box       from A's close to C's open: the far edges of the two imbalances, NOT B's high and low (B's wicks overrun both).
            The consequent encroachment (CE) is its midpoint.
  side      the colour of the candles says nothing about direction: the same up-up-up shape is a sell block at a session high
            and a buy block after a low. Location decides: a box whose CE is in premium of the last SB_RANGE candles' range
            is a sell block, in discount a buy block (on the equilibrium: no trade). Down-down-down works the same way.
  entry     a limit at the CE, the first touch of the far half (the upper half for a sell, the lower for a buy). When price
            is already on the wrong side of that limit (a sell box that formed under three up candles: price sits above it)
            the box waits armed until a candle closes back through the limit, then the order rests for the return.
  stop      beyond the box edge by one tick (or the stop buffer when that is larger). The target is the nearest opposing
            liquidity, the engine's usual one.
  void      a body closing outside the box on the stop side (above the top of a sell box, below the bottom of a buy box).
            A wick through the edge is tolerated; a close only just beyond it (within the one tick) too. A resting order is
            cancelled by the base lifecycle at that same moment, because a close beyond edge + tick has traded through the
            stop. An armed box is voided here, but only once price has come back into the box (a close at or inside the
            edge): while the run that built the box is still going, its candles close out beyond the edge just as candle C
            itself does, and that is not a failed retest (-Td-D-vKJDg@14m40: an orange sell block on candles A, B, C, then
            two more up candles closing above its top, and the lesson still trades the later return into it).
            A newer box that shares a candle replaces the older one on the same side, armed or resting (a run of four
            candles makes two overlapping boxes: the newest, at the end of the run, is the one the detector follows).

Price only: no clock, so it also runs on 24/7 synthetic indices (the only calendar it knows is the trading day the engine
clears its resting orders on). It respects the daily bias like the other pattern models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .opening import FirstPresentedStrategy
from .sessions import Bar, Level, trading_day
from .strategy import LONG, SHORT

SB_VI_TICKS = 2          # each volume imbalance is at least this many ticks (the pip size) wide ...
SB_VI_RANGE = 0.05       # ... and at least this fraction of the median candle range
SB_RANGE = 60            # candles of the dealing range whose midpoint splits premium (sell) from discount (buy)
SB_WARMUP = 20           # candles of history needed before the first box is read (a full median-range window)
SB_OVERSHOOT_TICKS = 1   # a close may sit this far beyond the box edge; the stop sits that far (or the stop buffer) beyond it
SB_ENTRY_FRAC = 0.0      # limit position from the CE (0) towards the box edge on the stop side (1)
SB_ARM_BARS = 30         # candles an armed box waits for price to leave the entry before it is dropped
SB_EXPIRY = 30           # candles a resting order may wait for the return
SB_MAX_ARMED = 4         # armed boxes kept at once (the oldest is dropped)
_EPS = 1e-9              # float slack on price comparisons


def _hi(b: Bar, m: int) -> float:
    """The candle's high as seen by a block of colour m (+1 up: the high; -1 down: the negated low)."""
    return b.high if m > 0 else -b.low


def _lo(b: Bar, m: int) -> float:
    return b.low if m > 0 else -b.high


@dataclass(slots=True)
class _Block:
    m: int           # colour of the three candles: +1 up, -1 down
    d: int           # the trade: SHORT for a box in premium, LONG in discount
    a: int           # index of candle A
    born: int        # index of candle C, the bar the box completed on
    day: object      # the trading day it formed on
    lo: float        # the box: A's close and C's open, sorted
    hi: float
    ce: float
    vis: tuple       # the two imbalances, each (low, high)
    span: tuple      # the dealing range: (low, high, candles)
    entry: float
    stop: float
    why: str         # the daily-draw text for the rules trail
    left: bool = False   # price had to leave the entry first (the box was armed)
    tested: bool = False  # an armed box: price has since closed back at or inside the stop-side edge (a close out there now voids it)


@dataclass
class SuspensionStrategy(FirstPresentedStrategy):
    widen_tight_stops = True
    _seen: set = field(default_factory=set)       # (colour, index of A) of the boxes already used
    _armed: list = field(default_factory=list)    # boxes waiting for price to leave their entry so a limit can rest

    # ------------------------------------------------------------------ detection
    def _detect(self, idx: int, close_time: datetime, pip: float) -> None:
        if self._armed:
            self._work_armed(idx, close_time, pip)
        if idx < SB_WARMUP:
            return
        x = self._find(idx, pip)
        if x is None:
            return
        if (x.entry - self._bars[idx].close) * -x.d > 0:       # price is already past the entry: the limit can rest now
            if not self._place(x, idx, close_time):
                return
        else:
            f = self._fmt
            for o in self._armed:                                    # a newer box sharing a candle replaces the older armed one
                if o.d == x.d and abs(o.a - x.a) <= 2:
                    self._log(close_time, "SB-03", f"suspension block {f(o.lo)} to {f(o.hi)} dropped: replaced by the newer box {f(x.lo)} to {f(x.hi)}")
            self._armed = [o for o in self._armed if not (o.d == x.d and abs(o.a - x.a) <= 2)]
            self._armed.append(x)
            del self._armed[:-SB_MAX_ARMED]
            self._log(close_time, "SB-03", f"suspension block {f(x.lo)} to {f(x.hi)} in {'premium' if x.d == SHORT else 'discount'}: "
                                           f"waiting for price to leave {f(x.entry)} before the {'sell' if x.d == SHORT else 'buy'} limit rests")
        self._seen = {k for k in self._seen if k[1] >= idx - 2 * SB_RANGE}
        self._seen.add((x.m, x.a))

    def _find(self, idx: int, pip: float) -> _Block | None:
        """The box that the candle which just closed completes, or None."""
        b = self._bars
        a, k, c = b[idx - 2], b[idx - 1], b[idx]
        if c.close > c.open and k.close > k.open and a.close > a.open:
            m = 1
        elif c.close < c.open and k.close < k.open and a.close < a.open:
            m = -1
        else:
            return None
        g1, g2 = m * (k.open - a.close), m * (c.open - k.close)          # the two imbalances: bodies apart ...
        if g1 <= 0 or g2 <= 0 or _lo(k, m) > _hi(a, m) or _lo(c, m) > _hi(k, m):     # ... wicks touching
            return None
        need = max(SB_VI_TICKS * pip, SB_VI_RANGE * self._median_range()) - _EPS
        if g1 < need or g2 < need or (m, idx - 2) in self._seen:
            return None
        lo, hi = (a.close, c.open) if m > 0 else (c.open, a.close)
        ce = (lo + hi) / 2
        first = max(0, idx - SB_RANGE + 1)
        top = max(b[i].high for i in range(first, idx + 1))
        bot = min(b[i].low for i in range(first, idx + 1))
        eq = (top + bot) / 2
        if abs(ce - eq) <= _EPS:
            return None                                              # neither premium nor discount: no side (float noise counts as equal)
        d = SHORT if ce > eq else LONG
        bias, why = self.daily_bias()
        if bias and bias != d:
            return None
        edge = hi if d == SHORT else lo
        entry = ce + SB_ENTRY_FRAC * (edge - ce)
        stop = edge - d * self._buffer(pip)
        return _Block(m, d, idx - 2, idx, trading_day(c.time, self.config.day_rollover), lo, hi, ce,
                      (tuple(sorted((a.close, k.open))), tuple(sorted((k.close, c.open)))), (bot, top, idx + 1 - first), entry, stop, why)

    def _buffer(self, pip: float) -> float:
        """How far beyond the box edge a close or the stop may sit: one tick, or the engine's stop buffer when that is wider."""
        return max(SB_OVERSHOOT_TICKS, self.config.stop_buffer_pips) * pip

    # ------------------------------------------------------------------ armed boxes
    def _work_armed(self, idx: int, close_time: datetime, pip: float) -> None:
        """Drop the armed boxes this bar outlived or voided (a close beyond the edge counts only after price came back into the box);
        rest the order of the ones whose entry price has been left behind."""
        bar, buf, keep, f = self._bars[idx], self._buffer(pip), [], self._fmt
        day = trading_day(bar.time, self.config.day_rollover)
        for x in self._armed:
            edge = x.hi if x.d == SHORT else x.lo
            if day != x.day or idx - x.born > SB_ARM_BARS:
                continue                                             # the engine clears resting orders at the day roll; so do we
            if (bar.close - edge) * -x.d > buf + _EPS:               # a close out beyond the stop-side edge ...
                if x.tested:                                         # ... after price had come back into the box: the retest failed
                    self._log(close_time, "SB-03", f"suspension block {f(x.lo)} to {f(x.hi)} void: price came back into it, then a body closed "
                                                   f"{'above' if x.d == SHORT else 'below'} {f(edge)}")
                else:
                    keep.append(x)                                   # ... while the run that built the box is still going: not a retest yet
                continue
            x.tested = True                                          # a close at or inside the edge: price is back at the box
            if (x.entry - bar.close) * -x.d > 0:                     # a close back through the entry: the limit now rests for the return
                x.left = True
                self._place(x, idx, close_time)
            else:
                keep.append(x)
        self._armed = keep

    # ------------------------------------------------------------------ order
    def _place(self, x: _Block, idx: int, close_time: datetime) -> bool:
        """Rest the limit for a box (the engine checks the stop and the reward first). True when it was staged."""
        cfg, b, f = self.config, self._bars, self._fmt
        d, edge = x.d, (x.hi if x.d == SHORT else x.lo)
        lvl = Level(f"SB {f(x.ce)}", x.ce, "high" if d == SHORT else "low")
        if not self._stage(d, lvl, x.entry, x.stop, (x.lo, x.hi), idx, close_time, "suspension", "suspension block entry", pre=self._trail(x),
                           ce_cancel=False, expiry=SB_EXPIRY, extreme=edge, sweep_time=b[x.a].time, swing_time=b[x.a].time):
            return False
        s = self._setups[-1]
        s.sb_a = x.a                                                 # which box it is (see the replacement below)
        s.fvg_time = b[x.a].time + timedelta(minutes=cfg.timeframe_minutes)    # the chart draws the box from candle A
        for e in reversed(self.events):
            if e["type"] == "staged":
                e["fvg_time"] = s.fvg_time
                break
        older = {id(o) for o in self._setups[:-1] if o.direction == d and abs(getattr(o, "sb_a", -10 ** 9) - x.a) <= 2}
        if older:                                                    # a newer box sharing a candle replaces the resting order
            for o in self._setups[:-1]:
                if id(o) in older:
                    self._log(close_time, "EN-04", f"suspension block limit at {f(o.limit)} cancelled: replaced by the newer box {f(x.lo)} to {f(x.hi)}")
            self._setups = [o for o in self._setups if id(o) not in older]
        return True

    def _trail(self, x: _Block) -> tuple:
        f, up, sell = self._fmt, x.m > 0, x.d == SHORT
        (a0, a1), (b0, b1), (bot, top, n) = x.vis[0], x.vis[1], x.span
        return (f"SB-01 three {'up' if up else 'down'} candles in a row with a volume imbalance either side of the middle one: "
                f"{f(a0)} to {f(a1)} and {f(b0)} to {f(b1)} (bodies apart, wicks touching)",
                f"SB-02 suspension block {f(x.lo)} to {f(x.hi)}, the far edges of the two imbalances, consequent encroachment {f(x.ce)}",
                f"SB-03 the block sits in {'premium' if sell else 'discount'} of the last {n} candles' range {f(bot)} to {f(top)} "
                f"(equilibrium {f((bot + top) / 2)}): a {'sell' if sell else 'buy'} block, limit at {f(x.entry)} in its {'upper' if sell else 'lower'} half, "
                f"stop {f(x.stop)} beyond the {'top' if sell else 'bottom'}" + (", placed once price had left the block and came back" if x.left else ""),
                f"RL-05 not against the daily draw ({x.why})")
