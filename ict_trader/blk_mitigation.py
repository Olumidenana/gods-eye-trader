"""Mitigation Block: a failure swing, then the break of its low; the last candle at that low is where price returns to.

ICT teaches it as the cousin of the breaker that did NOT take the old high (20220907_FOUzW0QmsfI, 20240306_0OIplRTKYOA;
the geometry below was measured on those frames, where it differs from the older wording in rules/ict-models.json).
Written for a sell; a buy is the exact mirror (highs and lows swapped, up candles for down candles).

  swing     a swing high H0 (the last one before A), the next swing low A, then a rally to a top B that stays BELOW H0 (a lower
            high: the failure swing). If B took H0 it is a higher high and the block would be a breaker, so it is skipped.
  break     the first candle that CLOSES below A. A wick through A that closes back above is not a break; only a close is.
  block     the last down-close candle before the first up candle of the A-to-B rally (in the lesson chart it is A's own
            candle). The zone runs from its OPEN (the top; the upper wick is left out) down to its low (the lower wick is in).
  entry     a limit at the open of the block (MB_ENTRY_FRAC moves it down the zone), the stop beyond the zone top by 0.4 of the
            zone height. The target is the -1 standard deviation projection (the block's close minus the rally's body height
            above it) when that pays enough, otherwise the engine's nearest liquidity.
  void      a close above the block's open, or any trade above H0, before the fill: the zone was run, it is no longer a block.
            Each new lower swing makes a new block (the stepladder), and the newest replaces the older resting order.

Price only: no clock, so it also runs on 24/7 synthetic indices. It respects the daily bias like the other pattern models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .opening import FirstPresentedStrategy
from .sessions import Bar, Level
from .strategy import LONG, SHORT, Signal

MB_SWING = 2           # a swing high/low is the extreme of this many candles on each side
MB_MAX_H0_A = 30       # the old high may sit at most this many candles before the swing low A
MB_MAX_A_B = 30        # the rally from A may take at most this many candles to reach its top B
MB_MAX_B_BREAK = 20    # the break of A must come within this many candles of B
MB_ENTRY_FRAC = 0.0    # limit position along the zone from the block's open (0) towards its far wick (1)
MB_STOP_FRAC = 0.4     # the stop sits this fraction of the zone height beyond the zone top (at least stop_buffer_pips)
MB_SD = 1.0            # preferred target: this many rally body heights beyond the block's close
MB_EXPIRY = 60         # candles a resting order may wait


# The geometry is written for a sell and mirrored for a buy by negating prices. These read a candle "as if selling":
# its high for a sell, its low (negated) for a buy, and so on.
def _top(b: Bar, d: int) -> float:
    return b.high if d == SHORT else -b.low


def _bot(b: Bar, d: int) -> float:
    return b.low if d == SHORT else -b.high


def _opn(b: Bar, d: int) -> float:
    return b.open if d == SHORT else -b.open


def _cls(b: Bar, d: int) -> float:
    return b.close if d == SHORT else -b.close


@dataclass
class MitigationStrategy(FirstPresentedStrategy):
    widen_tight_stops = True
    _seen: set = field(default_factory=set)       # (direction, index of A) pairs already used

    def on_bar(self, bar: Bar) -> Signal | None:
        self._drop_void()
        return super().on_bar(bar)

    def _void_reason(self, s, bar: Bar) -> str | None:
        """Why this resting order is no longer a mitigation block after `bar` closed, or None."""
        guard = getattr(s, "mb_guard", None)
        if guard is None:
            return None
        old, edge = guard                                    # the old high (low) never retaken, the block's open
        m = -s.direction                                     # +1 for a sell, -1 for a buy
        if ((bar.high if m > 0 else bar.low) - old) * m > 0:
            return f"price traded {'above' if m > 0 else 'below'} the old {'high' if m > 0 else 'low'} {self._fmt(old)} before the fill: a breaker, not a mitigation block"
        if (bar.close - edge) * m > 0:
            return f"a body closed {'above' if m > 0 else 'below'} the block's open {self._fmt(edge)}: the zone is violated"
        return None

    def _drop_void(self) -> None:
        """Drop the orders the last closed bar voided. Run before the next bar is processed, so only closed bars are used.

        The base lifecycle fills a limit the moment it is touched, and any bar that closes beyond the block's open (or runs the
        old high) has first traded through the limit, so this only bites an order that could not fill (the day's signal cap, or
        another order filled on the same bar) and would otherwise rest on and fill later against a zone price already ran.
        """
        if not self._setups or not self._bars:
            return
        last = self._bars[-1]
        keep = []
        for s in self._setups:
            why = self._void_reason(s, last)
            if why:
                self._log(last.time + timedelta(minutes=self.config.timeframe_minutes), "EN-04", f"mitigation block limit cancelled: {why}")
            else:
                keep.append(s)
        if len(keep) != len(self._setups):
            self._setups = keep

    def _detect(self, idx: int, close_time: datetime, pip: float) -> None:
        if idx < 6 * MB_SWING:
            return
        for d in (SHORT, LONG):
            found = self._find(d, idx)
            if found is None:
                continue
            bias, why = self.daily_bias()
            if bias and bias != d:
                continue
            self._seen.add((d, found[0]))
            self._stage_block(d, found, idx, close_time, pip, why)
            return

    # ------------------------------------------------------------------ geometry
    def _find(self, d: int, idx: int):
        """The failure swing whose break is the candle that just closed: indices (A, B, H0, block), or None.

        A is a swing low (lower than the MB_SWING candles each side) that the trigger closes below for the first time: no
        candle since has CLOSED below it (a wick through it that closed back above does not break it, only a close does).
        Walks back from the trigger keeping the lowest close seen since the candidate; once that is at or under the trigger's
        close no older low can still qualify, so the walk stops there.
        """
        b, n = self._bars, MB_SWING
        t = b[idx]
        ct = _cls(t, d)
        mc = min(_cls(b[x], d) for x in range(idx - n, idx))   # the lowest close after the first candidate A (idx - n - 1)
        for a in range(idx - n - 1, max(n, idx - MB_MAX_A_B - MB_MAX_B_BREAK) - 1, -1):
            if mc <= ct:
                break                                          # a close at or under the trigger's lies after every older low
            la = _bot(b[a], d)
            if ct < la <= mc and (d, a) not in self._seen and all(_bot(b[a - i], d) > la and _bot(b[a + i], d) > la for i in range(1, n + 1)):
                hit = self._shape(d, a, idx, _top(t, d))
                if hit is not None:
                    return hit
            mc = min(mc, _cls(b[a], d))
        return None

    def _shape(self, d: int, a: int, idx: int, ht: float):
        """The failed rally from the swing low A that the trigger candle `idx` broke: (A, B, H0, block) indices or None."""
        b, n = self._bars, MB_SWING
        bi, hb = a + 1, _top(b[a + 1], d)                      # B: the top of the rally since A (the first of equal tops)
        for x in range(a + 2, idx):
            v = _top(b[x], d)
            if v > hb:
                bi, hb = x, v
        if bi - a > MB_MAX_A_B or idx - bi > MB_MAX_B_BREAK or ht >= hb:
            return None                                        # too slow, or the trigger candle itself took B
        if any(_top(b[x], d) >= hb for x in range(bi - n, min(bi + n, idx) + 1) if x != bi):
            return None                                        # B is not a swing top (an equal or higher high beside it)
        h0 = next((x for x in range(a - 1, max(n, a - MB_MAX_H0_A) - 1, -1)       # H0: the last swing top before A (the nearest one)
                   if all(_top(b[y], d) < _top(b[x], d) for y in range(x - n, x + n + 1) if y != x)), None)
        if h0 is None or hb >= _top(b[h0], d):
            return None                                        # no old high, or B retook it: a higher high, so a breaker, not a mitigation block
        j = next((x for x in range(a, bi + 1) if _cls(b[x], d) > _opn(b[x], d)), None)     # first up-close candle of the rally
        if j is None:
            return None
        k = next((x for x in range(j - 1, max(h0 + 1, a - n) - 1, -1) if _cls(b[x], d) < _opn(b[x], d)), None)   # last down-close before it
        if k is None or _opn(b[k], d) >= max(_opn(b[bi], d), _cls(b[bi], d)):
            return None                                        # no block candle, or its open is not below the rally's body top
        return a, bi, h0, k

    # ------------------------------------------------------------------ order
    def _stage_block(self, d: int, found: tuple, idx: int, close_time: datetime, pip: float, why: str) -> None:
        cfg, b, f = self.config, self._bars, self._fmt
        a, bi, h0, k = found
        m = -d                                                 # back from "as if selling" prices: +1 for a sell, -1 for a buy
        top, bot = _opn(b[k], d), _bot(b[k], d)
        height = top - bot
        entry = m * (top - MB_ENTRY_FRAC * height)
        stop = m * (top + max(MB_STOP_FRAC * height, cfg.stop_buffer_pips * pip))
        zone = (min(m * top, m * bot), max(m * top, m * bot))
        old, failed, broken = m * _top(b[h0], d), m * _top(b[bi], d), m * _bot(b[a], d)
        ck = _cls(b[k], d)
        sd = ck - MB_SD * (max(_opn(b[bi], d), _cls(b[bi], d)) - ck)       # the rally's body height measured down from the block's close
        tgt = Level(f"{'-' if d == SHORT else '+'}{MB_SD:g} SD {f(m * sd)}", m * sd, "low" if d == SHORT else "high") \
            if sd < _bot(b[idx], d) else None                  # a projection price has already traded through is no target
        own, opp = ("high", "low") if d == SHORT else ("low", "high")
        pre = (f"MB-01 failure swing: a {own} at {f(old)}, a {opp} at {f(broken)}, then a {'lower' if d == SHORT else 'higher'} {own} at {f(failed)} that never retook {f(old)}",
               f"MB-02 structure broke: a candle closed {'below' if d == SHORT else 'above'} {f(broken)}, the {opp} the failed {'rally' if d == SHORT else 'decline'} started from",
               f"MB-03 mitigation block: the last {'down' if d == SHORT else 'up'} candle before that {'rally' if d == SHORT else 'decline'}, zone [{f(zone[0])}, {f(zone[1])}] (open to {opp} wick), limit at {f(entry)}",
               f"RL-05 not against the daily draw ({why})")
        lvl = Level(f"{'SWH' if d == SHORT else 'SWL'} {f(failed)}", failed, own)
        if not self._stage(d, lvl, entry, stop, zone, idx, close_time, "mitigation", "block entry", pre=pre, ce_cancel=False,
                           expiry=MB_EXPIRY, extreme=failed, sweep_time=b[bi].time, swing_time=b[a].time, target=tgt):
            return
        s = self._setups[-1]
        s.mb_guard = (old, m * top)                            # what voids it before the fill (see _void_reason)
        s.mss_level = broken                                   # the swing low whose break confirmed the block
        s.fvg_time = b[k].time + timedelta(minutes=cfg.timeframe_minutes)    # the chart draws the zone from the block candle
        for e in reversed(self.events):
            if e["type"] == "staged":
                e["fvg_time"] = s.fvg_time
                break
        older = [x for x in self._setups[:-1] if x.direction == d]
        if older:                                              # stepladder: the newest block replaces the resting one on this side
            for x in older:
                self._log(close_time, "EN-04", f"mitigation block limit at {f(x.limit)} cancelled: replaced by the block at the new swing")
            self._setups = [x for x in self._setups[:-1] if x.direction != d] + [s]
