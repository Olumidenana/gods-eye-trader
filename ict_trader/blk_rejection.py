"""Rejection Block: the body-to-wick zone of a swing high (or low) that printed with long wicks.

ICT's definition (December 2016 Monthly Mentorship, then the 2022 and 2025 re-teaches): a price high forms with
long wicks on the high(s) of the candles, and price reaches up above their bodies to run the buy-side liquidity
before it declines. The zone is the composite upper wick, "viewed like a candle all in and of itself": from the
highest body edge of the cluster up to its highest high. It is not an order block (no displacement or gap is
needed) and not Turtle Soup (no new extreme is needed). The geometry below was measured on the lesson frames
(see the geometry report, section "2. REJECTION BLOCK"); stops and targets are not drawn there, so those are
fixed here and written in each signal's reasons.

  bearish  a cluster of candles around a swing high (RB_SIDE each side; the swing holds the highest high of
           RB_SWING candles), with a long upper wick (wick/body and wick/range both large; two such candles, or
           one stronger one). Zone = [highest body edge, highest high]. The idea is confirmed by the first close
           below the cluster's lowest low (within RB_CONFIRM_BARS candles); a close above the wick high before
           that kills it. Then a limit at the body edge, the stop just beyond the wick extreme (the pierce of the
           zone is the trade), the target the nearest opposing liquidity.
  bullish  the mirror: a swing low with long lower wicks, zone = [lowest low, lowest body edge]; a limit at the body edge, the
           stop just below the wick low, confirmed by a close above the cluster's highest high.

Price only (no clock), so it runs on 24/7 synthetic indices; only closed candles are read, never the future.
The maths is written once for the bearish case on mirrored prices: with s = +1 the code reads highs, with s = -1
every price is negated so a low reads as a high (see _out, _in, _top, _bot).

Cost: a swing is tested once, on the candle that completes its cluster, and then waits as a small tuple; each
later candle only compares its close with two numbers per waiting swing (a handful at most), so the work per
candle is constant and nothing is rescanned. What the model remembers is bounded too: the waiting swings die
after RB_CONFIRM_BARS candles and the judged ones are forgotten once they are too old to matter (see _forget).

One idea at a time: a swing that holds the same extreme as one already staged (equal highs inside the swing
window), or any swing while an order of the same side still rests, is the same block seen again and is not staged
twice. Garbage prices (NaN, infinity) never reach an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .opening import FirstPresentedStrategy
from .sessions import Level
from .strategy import LONG, SHORT

RB_SWING = 20            # the swing candle holds the highest high (lowest low) of this many candles up to and including itself
RB_SIDE = 3              # candles on each side of the swing that make the cluster (none of the ones after may go beyond it)
RB_WICK_BODY = 0.75      # a long wick is at least this multiple of its own candle's body...
RB_WICK_RANGE = 0.40     # ...and at least this fraction of that candle's whole range
RB_STRONG_BODY = 1.0     # one candle whose wick reaches this multiple of its body...
RB_STRONG_RANGE = 0.50   # ...and this fraction of its range is enough on its own (the range test already implies the body one)
RB_MIN_WICKS = 2         # otherwise this many long-wick candles are needed in the cluster
RB_MIN_ZONE = 0.15       # the zone must be at least this fraction of the median candle range (an ATR(14)-like scale)
RB_CONFIRM_BARS = 10     # the confirming close must come within this many candles after the cluster's last candle
RB_BUFFER_PIPS = 2       # stop buffer beyond the wick extreme, in pips (1-2 ticks); a wider configured stop_buffer_pips wins
RB_EXPIRY = 5            # candles the limit order may rest unfilled
RB_KEEP = 128            # judged swings remembered before the ones too old to matter are forgotten (bounds the memory)


def _out(x, s: int) -> float:
    """The wick-side extreme (mirrored): the high for s = +1, minus the low for s = -1."""
    return x.high if s > 0 else -x.low


def _in(x, s: int) -> float:
    """The far-side extreme (mirrored): the low for s = +1, minus the high for s = -1."""
    return x.low if s > 0 else -x.high


def _top(x, s: int) -> float:
    """The body edge on the wick side (mirrored)."""
    return max(x.open, x.close) if s > 0 else -min(x.open, x.close)


def _bot(x, s: int) -> float:
    """The body edge on the far side (mirrored)."""
    return min(x.open, x.close) if s > 0 else -max(x.open, x.close)


@dataclass
class RejectionStrategy(FirstPresentedStrategy):
    widen_tight_stops = True
    _seen: set = field(default_factory=set)             # (direction, swing candle) pairs already judged: a cluster is judged once
    _wait: list = field(default_factory=list)           # swings whose cluster is complete and that wait for the confirming close: (direction, swing, wick, far)
    _blocks: dict = field(default_factory=dict)         # (direction, wick extreme) -> the swing candle of the block staged there: an equal high is the same block

    def _expiry_bars(self) -> int:
        """The life of this model's limit: the live order list counts the candles left from it (the base's 90 minutes is not ours)."""
        return RB_EXPIRY

    def _swing(self, p: int, s: int):
        """The cluster's far extreme (mirrored) if candle p is a swing completed by the latest candle, else None."""
        b = self._bars
        wick = _out(b[p], s)
        for j in range(p + 1, p + RB_SIDE + 1):                # nothing higher in the candles just after it...
            if _out(b[j], s) > wick:
                return None
        for j in range(p - 1, p - RB_SWING, -1):               # ...nor in the RB_SWING candles up to it: a swing high
            if _out(b[j], s) > wick:
                return None
        return min(_in(b[j], s) for j in range(p - RB_SIDE, p + RB_SIDE + 1))

    def _geometry(self, p: int, s: int):
        """(highest body edge, number of long-wick candles) when the cluster around swing p has the wicks of a rejection block, else None.

        A wick counts when it is long for its own candle (wick/body and wick/range) and pokes above the highest body
        edge of the cluster: a wick buried beneath another candle's body is not part of the zone. Two such candles
        are needed, or one that is long enough on its own."""
        b = self._bars
        body = max(_top(b[j], s) for j in range(p - RB_SIDE, p + RB_SIDE + 1))
        n_long, strong = 0, False
        for j in range(p - RB_SIDE, p + RB_SIDE + 1):
            x = b[j]
            wick, rng, bd = _out(x, s) - _top(x, s), _out(x, s) - _in(x, s), _top(x, s) - _bot(x, s)
            if rng <= 0 or _out(x, s) <= body:
                continue
            if wick >= RB_WICK_BODY * bd and wick >= RB_WICK_RANGE * rng:
                n_long += 1
                strong = strong or (wick >= RB_STRONG_BODY * bd and wick >= RB_STRONG_RANGE * rng)
        return (body, n_long) if strong or n_long >= RB_MIN_WICKS else None

    def _detect(self, idx: int, close_time: datetime, pip: float) -> None:
        b = self._bars
        hits = []
        if self._wait:                                          # swings already waiting: the latest close decides each one
            close, keep = b[idx].close, []
            for w in self._wait:
                d, p, wick, far = w
                c = close * -d                                  # mirrored close (SHORT reads prices as they are)
                if c > wick or idx - p > RB_SIDE + RB_CONFIRM_BARS:
                    continue                                    # accepted beyond the wick, or the confirmation came too late: dead
                if c < far:
                    hits.append(w)                              # the first close beyond the cluster's far side: confirmed
                else:
                    keep.append(w)
            self._wait = keep
        p = idx - RB_SIDE                                       # the swing whose right-hand candles have all just closed
        if p >= RB_SWING - 1:
            for d in (SHORT, LONG):
                far = self._swing(p, -d)
                if far is not None:
                    self._wait.append((d, p, _out(b[p], -d), far))
        for d, p, wick, far in reversed(hits):                  # the most recent cluster first
            if (d, p) in self._seen:
                continue
            self._seen.add((d, p))
            self._try(d, p, wick, far, idx, close_time, pip)    # (a second block of the same side is turned away inside)
        if len(self._seen) > RB_KEEP:
            self._forget(idx)

    def _forget(self, idx: int) -> None:
        """Drop the judged swings that can no longer matter: a swing is judged once (it waits at most RB_SIDE + RB_CONFIRM_BARS candles
        after its swing candle) and only blocks an equal high that still shares a RB_SWING window with it."""
        cut = idx - (RB_SWING + RB_SIDE + RB_CONFIRM_BARS + 1)
        self._seen = {k for k in self._seen if k[1] >= cut}
        self._blocks = {k: v for k, v in self._blocks.items() if v >= cut}

    def _try(self, d: int, p: int, wick: float, far: float, idx: int, close_time: datetime, pip: float) -> bool:
        """The cluster around swing p was confirmed on candle idx: check its wicks, size and the daily draw, then rest the limit."""
        s, b = -d, self._bars
        if any(x.direction == d for x in self._setups):
            return False                                        # an order of this side already rests: a nearby swing is the same idea, not a second trade
        prev = self._blocks.get((d, wick))
        if prev is not None and p - prev < RB_SWING:
            return False                                        # an equal high still inside the swing window: the block was already staged
        geo = self._geometry(p, s)
        if geo is None:
            return False
        body, n_long = geo
        mr = self._median_range()
        if not all(map(math.isfinite, (body, wick, far, mr))):
            return False                                        # a missing or broken price: no order is ever built on it
        if not mr or wick - body < RB_MIN_ZONE * mr:
            return False                                        # a sliver of a zone: no room for a wick to speak of
        bias, why = self.daily_bias()
        if bias and bias != d:
            return False
        entry, ext, low_c = s * body, s * wick, s * far         # back to real prices
        stop = ext + s * max(RB_BUFFER_PIPS, self.config.stop_buffer_pips) * pip
        zone = (min(entry, ext), max(entry, ext))
        lo_i = min(range(p - RB_SIDE, p + RB_SIDE + 1), key=lambda j: _in(b[j], s))    # the candle that printed the cluster's far extreme
        side, edge = ("upper", "high") if d == SHORT else ("lower", "low")
        pre = (f"RJ-01 rejection block: {n_long} candle{'s' if n_long > 1 else ''} with long {side} wicks around the {edge} at {self._fmt(ext)}, "
               f"zone {self._fmt(zone[0])} to {self._fmt(zone[1])} (the body edge to the wick {edge})",
               f"RJ-02 rejection confirmed: a close {'below' if d == SHORT else 'above'} the cluster's {'lowest low' if d == SHORT else 'highest high'} {self._fmt(low_c)}, "
               f"no close {'above' if d == SHORT else 'below'} the wick extreme {self._fmt(ext)}",
               f"RJ-03 not against the daily draw ({why})")
        if not self._stage(d, Level(f"RB {self._fmt(ext)}", ext, edge), entry, stop, zone, idx, close_time, "rejection", "rejection block",
                           pre=pre, ce_cancel=False, expiry=RB_EXPIRY, extreme=ext, sweep_time=b[p].time, swing_time=b[lo_i].time):
            return False
        self._blocks[(d, wick)] = p
        s_ = self._setups[-1]
        s_.mss_level = low_c                                    # the shift the chart and the story show is the close through the cluster's far extreme, not the entry
        s_.fvg_time = b[p - RB_SIDE].time + timedelta(minutes=self.config.timeframe_minutes)    # the chart draws the zone from the start of the cluster
        for e in reversed(self.events):
            if e["type"] == "staged":
                e["fvg_time"] = s_.fvg_time
                break
        return True
