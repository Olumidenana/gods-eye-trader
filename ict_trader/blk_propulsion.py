"""Propulsion Block: a down candle that trades into a bullish order block and takes over as the support.

The geometry was measured on ICT's lesson frames (20220908_glu98jAH8vE, 20241017_2QD_2AG7Syo, 08d62cZDXUk, 20260310_CodF_0llnuk), where it
differs from the older wording in rules/ict-models.json. Written for a buy; a sell is the exact mirror (highs and lows swapped, up
candles for down candles).

  A      a bullish order block: the last down-close candle before a displacement (an up candle with a big body) that closes above the
         closest swing high over A. It is found with the engine's own swing helper; no clock and no volume are used.
  B      a LATER down-close candle, at most PB_MAX_AB candles after A, whose low reaches into A (B.low <= A.high). B alone is the block.
         Its zone runs from its OPEN (the top; an upper wick is left out) down to its mean threshold MT = (open + close) / 2.
  valid  once a later candle's high trades above B's open (it has "proved itself"); it stays valid until it is voided. The order is
         staged on the first candle, within PB_VALID_BARS of B, that closes above the entry, so the limit rests on the right side of
         the market. That is normally the validating candle itself; if that one wicks over B's open but closes back inside the zone,
         the block is still valid and the order waits for the first later candle that closes above the entry. Each block is looked at
         once: staged, rejected, or skipped against the daily bias.
  entry  a limit at B's open (PB_ENTRY_DEPTH moves it down the zone towards MT), the stop one tick under MT, the target the engine's
         nearest external liquidity.
  void   a BODY close below MT less the one-tick tolerance. Before the order is staged the detector checks it (a close under MT after
         B kills the block, however high price then runs); afterwards the order lifecycle already holds the rule, because the stop
         sits at that level: a bar that closes under it has traded down through the limit first (the order fills, then the trade
         stops out) or, if the order could not fill, has touched the stop and the order is cancelled. A wick through MT, or a close
         one tick under it, does not void the block: the lesson's NQ example is exactly that case and price ran. The older rule's
         "or any trade-through of MT" is therefore dropped.

Price only: no clock, so it also runs on 24/7 synthetic indices. It respects the daily bias like the other pattern models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import NamedTuple

from .opening import FirstPresentedStrategy
from .sessions import Level
from .strategy import LONG, SHORT

PB_MAX_AB = 10         # the propulsion candle B comes at most this many candles after the order block A
PB_DISP_BARS = 3       # the displacement that makes A an order block starts within this many candles after A
PB_SWING_BACK = 30     # candles before A searched for the swing high that the displacement takes
PB_VALID_BARS = 10     # the order is staged within this many candles of B (immediate response; bounds the look-back)
PB_TOL_TICKS = 1       # ticks of tolerance under MT: the stop sits there, and a close within it does not void the block
PB_ENTRY_DEPTH = 0.0   # limit position down the zone: 0 = B's open (the top), 1 = MT
PB_EXPIRY = 12         # candles a resting order may wait for the retest (not in the lessons: a staging choice)


# The geometry is written for a buy and mirrored for a sell by negating prices: these read a candle "as if buying".
def _hi(b, d: int) -> float:
    return b.high if d == LONG else -b.low


def _lo(b, d: int) -> float:
    return b.low if d == LONG else -b.high


def _op(b, d: int) -> float:
    return b.open if d == LONG else -b.open


def _cl(b, d: int) -> float:
    return b.close if d == LONG else -b.close


class _Hit(NamedTuple):
    a: int          # the order block
    k: int          # its displacement candle
    swing: float    # the swing high (low for a sell) the displacement took, as a real price
    swing_i: int
    b: int          # the propulsion candle
    top: float      # B's open, "as if buying" (negated for a sell)
    mt: float       # B's mean threshold, "as if buying"
    vhigh: float    # the highest high (lowest low for a sell) after B up to the staging candle, as a real price: what validated B


@dataclass
class PropulsionStrategy(FirstPresentedStrategy):
    widen_tight_stops = True
    _seen: set = field(default_factory=set)       # (direction, index of A) pairs already staged: one propulsion block per order block
    _tried: set = field(default_factory=set)      # (direction, index of B) pairs already looked at: a block is staged, rejected or skipped once

    # ------------------------------------------------------------------ detection
    def _detect(self, idx: int, close_time: datetime, pip: float) -> None:
        if idx < 6:
            return
        for d in (LONG, SHORT):
            hit = self._find(d, idx, pip)
            if hit is None:
                continue
            bias, why = self.daily_bias()
            if bias and bias != d:
                continue
            if self._stage_block(d, hit, idx, close_time, pip, why):
                self._seen.add((d, hit.a))
                return

    def _find(self, d: int, idx: int, pip: float) -> _Hit | None:
        """The propulsion candle that is ready to be staged at the close of candle `idx`, with its order block, or None.

        Candidates B are the down-close candles of the last PB_VALID_BARS. B is ready when, counting the candles after it up to and
        including `idx`: some high traded above B's open (it has proved itself), no candle closed under MT less the tolerance (it is
        not void), and `idx` closes above the entry (the limit rests below the market). Only then is an order block A looked for behind
        B, and each B is looked at once (`_tried`), so a block that was staged, rejected or skipped is not met again on the next candle.
        """
        b, t = self._bars, self._bars[idx]
        tol, eps = PB_TOL_TICKS * pip, pip * 1e-3
        ct = _cl(t, d)
        run_hi, run_cl = _hi(t, d), ct                  # over the candles after B, up to idx included: the highest high / lowest close
        mb = None
        for bi in range(idx - 1, max(1, idx - PB_VALID_BARS) - 1, -1):
            B = b[bi]
            op, cl, hb = (B.open, B.close, B.high) if d == LONG else (-B.open, -B.close, -B.low)      # _op, _cl, _hi inlined: the hot loop
            if cl < op and run_hi > op + eps and (d, bi) not in self._tried:
                mt = (op + cl) / 2
                if run_cl >= mt - tol - eps and ct > op - PB_ENTRY_DEPTH * (op - mt) + eps:
                    if mb is None:
                        mb = self._median_body()
                        if not mb:
                            return None
                    self._tried.add((d, bi))
                    a = self._order_block(d, bi, mb)
                    if a is not None and (d, a[0]) not in self._seen:
                        return _Hit(a[0], a[1], a[2], a[3], bi, op, mt, d * run_hi)
            if hb > run_hi:
                run_hi = hb
            if cl < run_cl:
                run_cl = cl
        return None

    def _order_block(self, d: int, bi: int, mb: float):
        """The nearest order block behind the propulsion candle `bi`: (a, k, swing, swing_i), or None."""
        b, B = self._bars, self._bars[bi]
        for a in range(bi - 1, max(1, bi - PB_MAX_AB) - 1, -1):
            A = b[a]
            if _cl(A, d) >= _op(A, d) or _lo(B, d) > _hi(A, d):
                continue                                 # not a down-close candle, or B's low never reached into it
            sw = self._swing_over(d, a)
            if sw is None:
                continue
            for k in range(a + 1, min(a + PB_DISP_BARS, bi - 1) + 1):
                K = b[k]
                if _cl(K, d) < _op(K, d):
                    break                                # another down candle came first: A is not the last one before the move
                if _cl(K, d) - _op(K, d) >= self.config.displacement_body_mult * mb and _cl(K, d) > d * sw[0]:
                    return a, k, sw[0], sw[1]
        return None

    def _swing_over(self, d: int, a: int):
        """The closest swing high (low for a sell) behind candle `a` that lies above its high: (price, index), or None."""
        before, after, hi_a = a - 1, max(1, a - PB_SWING_BACK), _hi(self._bars[a], d)
        while before >= after:
            sw = self._last_swing(d, before, after)
            if sw is None:
                return None
            if d * sw[0] > hi_a:
                return sw
            before = sw[1] - 1
        return None

    # ------------------------------------------------------------------ order
    def _stage_block(self, d: int, h: _Hit, idx: int, close_time: datetime, pip: float, why: str) -> bool:
        cfg, b, f = self.config, self._bars, self._fmt
        A, B = b[h.a], b[h.b]
        tol = PB_TOL_TICKS * pip
        entry = d * (h.top - PB_ENTRY_DEPTH * (h.top - h.mt))
        stop = d * (h.mt - tol)
        mt, top = d * h.mt, d * h.top
        zone = (min(mt, top), max(mt, top))
        up = d == LONG
        pre = (f"PB-01 {'bullish' if up else 'bearish'} order block: the last {'down' if up else 'up'} candle [{f(A.low)}, {f(A.high)}] before a displacement "
               f"that took the swing {'high' if up else 'low'} at {f(h.swing)}",
               f"PB-02 propulsion candle: a later {'down' if up else 'up'} candle (open {f(B.open)}, close {f(B.close)}) traded {'down' if up else 'up'} into the block, "
               f"its {'low' if up else 'high'} {f(B.low if up else B.high)} reaching {f(A.high if up else A.low)}",
               f"PB-03 validated: a later candle's {'high' if up else 'low'} {f(h.vhigh)} took its open {f(B.open)}; zone [{f(zone[0])}, {f(zone[1])}], "
               f"limit at {f(entry)}, stop {f(stop)} {'under' if up else 'over'} the mean threshold {f(mt)}; a body close {'below' if up else 'above'} {f(stop)} voids the block",
               f"RL-05 not against the daily draw ({why})")
        lvl = Level(f"PB {f(B.open)}", B.open, "low" if up else "high")
        if not self._stage(d, lvl, entry, stop, zone, idx, close_time, "propulsion", "propulsion block", pre=pre, ce_cancel=False,
                           expiry=PB_EXPIRY, extreme=B.low if up else B.high, sweep_time=B.time, swing_time=b[h.swing_i].time):
            return False
        s = self._setups[-1]
        s.fvg_time = B.time + timedelta(minutes=cfg.timeframe_minutes)    # the chart draws the zone from the propulsion candle
        for e in reversed(self.events):
            if e["type"] == "staged":
                e["fvg_time"] = s.fvg_time
                break
        return True
