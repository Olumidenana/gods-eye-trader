"""The higher timeframe first: ICT's top-down read.

ICT does not look for an entry on the chart he executes on. The higher timeframe says where price is allowed to turn: the 4-hour and
1-hour fair value gaps and order blocks (premium and discount arrays) that have not been traded through. The lower timeframe then
does what it is for: it waits for the raid and the shift inside or at that array and gives the precise entry. An entry on a 5-minute
gap that sits in the middle of nowhere on the higher timeframes is a coin flip; the same gap inside a 1-hour array is a trade.

This module builds the higher-timeframe candles from the execution candles the strategy already sees (so it needs no second feed and
the backtest uses only candles that had CLOSED by the entry: no look-ahead), finds the arrays on them, and answers one question:
does this entry zone sit at an unmitigated higher-timeframe array of the right direction?

Candles are bucketed on New York time: 4-hour candles open at 01, 05, 09, 13, 17 and 21 NY (the forex day opens at 17:00), 1-hour
candles on the hour. A synthetic index has no 17:00 open, so its buckets start at New York midnight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .pd_arrays import detect_breaker_blocks, detect_rejection_blocks, detect_volume_imbalances
from .sessions import Bar, to_ny

LONG, SHORT = 1, -1
TF_NAMES = {60: "1h", 240: "4h", 1440: "daily", 10080: "weekly"}
# The PD-array matrix, as extracted (rules/ict-models.json): read from equilibrium outward, premium for a sell, discount for a buy:
# mitigation block, breaker, liquidity void, fair value gap, order block, rejection block, then the old high / low. The further from
# equilibrium an array sits, the stronger the place to turn, so when an entry sits in several the outermost one is named first.
# (The mitigation block is not read on the higher timeframes: it is a lower-timeframe block, tested in the Research lab.)
RANK = {"BRK": 1, "VI": 2, "FVG": 3, "OB": 4, "RB": 5}
KIND_WORDS = {"FVG": "fair value gap", "OB": "order block", "BRK": "breaker", "VI": "liquidity void", "RB": "rejection block"}


@dataclass(frozen=True)
class Array:
    tf: int
    kind: str          # "FVG", "OB", "BRK" (breaker), "VI" (liquidity void) or "RB" (rejection block)
    dir: int           # +1 bullish (support for a buy), -1 bearish (resistance for a sell)
    lo: float
    hi: float
    time: datetime

    def text(self, fmt: Any = lambda v: f"{v:g}") -> str:
        return f"{TF_NAMES.get(self.tf, str(self.tf) + 'm')} {'bullish' if self.dir > 0 else 'bearish'} {KIND_WORDS.get(self.kind, self.kind)} [{fmt(self.lo)}, {fmt(self.hi)}]"

    def to_json(self, fmt: Any = lambda v: v) -> dict[str, Any]:
        return {"tf": self.tf, "kind": self.kind, "dir": "bull" if self.dir > 0 else "bear", "lo": fmt(self.lo), "hi": fmt(self.hi), "time": int(self.time.timestamp())}


def ladder(tf: int) -> dict[str, Any]:
    """Which timeframe the bias comes from, for the timeframe you execute on. ICT reads top-down in steps: the chart you enter on
    is a rung or two below the chart that gives the bias. Trade the 1-minute and the bias is the 1-hour (confirmed by the 4-hour);
    trade the 5-minute and it is the 4-hour (confirmed by the daily); 15-minute and up, the daily (confirmed by the weekly)."""
    if tf <= 3:
        return {"tf": 60, "up": 240, "name": "1-hour", "up_name": "4-hour"}
    if tf <= 5:
        return {"tf": 240, "up": 1440, "name": "4-hour", "up_name": "daily"}
    return {"tf": 1440, "up": 10080, "name": "daily", "up_name": "weekly"}


def candle_bias(h: list[Bar], fmt: Any = lambda v: f"{v:g}") -> tuple[int, str]:
    """The bias from the last two CLOSED candles of a timeframe, read the way the daily bias is: a close beyond the prior candle's
    range, or a run on it that closed back inside (the run was the raid, the move back is the draw)."""
    if len(h) < 2:
        return 0, "not enough closed candles yet"
    a, b = h[-2], h[-1]
    if b.close > a.high:
        return LONG, f"the last candle closed {fmt(b.close)} above the prior candle's high {fmt(a.high)}"
    if b.close < a.low:
        return SHORT, f"the last candle closed {fmt(b.close)} below the prior candle's low {fmt(a.low)}"
    if b.high > a.high and b.low >= a.low:
        return SHORT, f"the last candle ran the prior high {fmt(a.high)} and closed back inside"
    if b.low < a.low and b.high <= a.high:
        return LONG, f"the last candle ran the prior low {fmt(a.low)} and closed back inside"
    return 0, "inside or two-sided: no clear draw"


def trend_of(highs: list[float], lows: list[float], left: int = 2) -> int:
    """The trend over MANY candles from their swing points, not from one candle: +1 higher highs and higher lows, -1 lower highs and
    lower lows, 0 two-sided or too few swings to say. A swing is a high (low) above (below) the `left` candles either side."""
    hi, lo = [], []
    for i in range(left, len(highs) - left):
        if all(highs[i] > highs[j] for j in range(i - left, i)) and all(highs[i] >= highs[j] for j in range(i + 1, i + left + 1)):
            hi.append(highs[i])
        if all(lows[i] < lows[j] for j in range(i - left, i)) and all(lows[i] <= lows[j] for j in range(i + 1, i + left + 1)):
            lo.append(lows[i])
    if len(hi) < 2 or len(lo) < 2:
        return 0
    up, dn = hi[-1] > hi[-2] and lo[-1] > lo[-2], hi[-1] < hi[-2] and lo[-1] < lo[-2]
    return 1 if up else -1 if dn else 0


class HtfArrays:
    """Higher-timeframe candles built from execution candles, and the arrays on them."""

    def __init__(self, tfs: tuple[int, ...], clock: bool = True, keep: int = 160) -> None:
        self.tfs = tuple(sorted(tfs))
        self.anchor = 17 * 60 if clock else 0
        self.keep = keep
        self._cur: dict[int, list] = {}
        self._start: dict[int, int] = {}
        self._done: dict[int, list[Bar]] = {t: [] for t in self.tfs}
        self._cache: dict[int, list[Array]] = {}
        self.fixed: dict[int, list[Bar]] = {}   # whole-day and whole-week candles handed in by the strategy (daily, weekly)

    @property
    def all_tfs(self) -> tuple[int, ...]:
        """Every timeframe that has arrays: the intraday ones built from the feed, then the daily and weekly ones."""
        return self.tfs + tuple(sorted(self.fixed))

    def set_fixed(self, tf: int, candles: list[Bar]) -> None:
        """Hand in COMPLETED candles of a timeframe the feed cannot build (daily, weekly: the strategy keeps the days)."""
        self.fixed[tf] = candles[-self.keep:]
        self._cache.pop(tf, None)

    def _bucket(self, t: datetime, tf: int) -> int:
        ny = to_ny(t)
        return (ny.toordinal() * 1440 + ny.hour * 60 + ny.minute - self.anchor) // tf

    def feed(self, bar: Bar) -> None:
        """Add an execution candle. A higher-timeframe candle counts only once the next one has begun."""
        for tf in self.tfs:
            b, cur = self._bucket(bar.time, tf), self._cur.get(tf)
            if cur is None or cur[0] != b:
                if cur is not None and cur[0] != self._start[tf]:   # the first bucket is a partial candle: not used
                    self._done[tf].append(Bar(cur[5], cur[1], cur[2], cur[3], cur[4]))
                    del self._done[tf][:-self.keep]
                    self._cache.pop(tf, None)
                self._start.setdefault(tf, b)
                self._cur[tf] = [b, bar.open, bar.high, bar.low, bar.close, bar.time]
            else:
                cur[2], cur[3], cur[4] = max(cur[2], bar.high), min(cur[3], bar.low), bar.close

    def arrays(self, tf: int) -> list[Array]:
        """The unmitigated PD arrays on this timeframe's closed candles: fair value gaps and order blocks (until a candle closes through
        them), and the breakers, liquidity voids and rejection blocks that price has not traded back into."""
        if tf in self._cache:
            return self._cache[tf]
        h, out = self._done.get(tf) or self.fixed.get(tf, []), []
        for i in range(2, len(h)):
            a, c = h[i - 2], h[i]
            if c.low > a.high:
                d, lo, hi = LONG, a.high, c.low
            elif c.high < a.low:
                d, lo, hi = SHORT, c.high, a.low
            else:
                continue
            later = h[i + 1:]
            if not any((x.close < lo) if d == LONG else (x.close > hi) for x in later):   # closed through the gap: it is inverted, not a support
                out.append(Array(tf, "FVG", d, lo, hi, c.time))
            # the order block: the last opposite candle before the displacement (a down-close before a rally, an up-close before a drop)
            for j in range(i - 2, max(-1, i - 6), -1):
                if (h[j].close < h[j].open) == (d == LONG) and h[j].close != h[j].open:
                    blo, bhi = h[j].low, h[j].high
                    if not any((x.close < blo) if d == LONG else (x.close > bhi) for x in h[j + 1:]):
                        out.append(Array(tf, "OB", d, blo, bhi, h[j].time))
                    break
        for kind, fn in (("BRK", detect_breaker_blocks), ("VI", detect_volume_imbalances), ("RB", detect_rejection_blocks)):
            for a in fn(h, lookback=len(h)):
                later = h[a.bar_index + 1:]
                if a.mitigated or a.broken or a.top <= a.bottom or any(x.low <= a.midpoint <= x.high for x in later):
                    continue    # (the detectors of the breaker and the void do not track a touch: a candle through the midpoint ends it)
                out.append(Array(tf, kind, a.direction, a.bottom, a.top, a.time))
        self._cache[tf] = out
        return out

    def trend(self, tf: int) -> int:
        """The structure trend (+1 / -1 / 0) over this timeframe's last 60 closed candles."""
        h = self._done.get(tf, [])[-60:]
        return trend_of([x.high for x in h], [x.low for x in h])

    def at(self, d: int, lo: float, hi: float) -> Array | None:
        """The higher-timeframe array of direction `d` that the zone [lo, hi] sits in or touches: the outermost on the matrix (RANK),
        and of those the higher timeframe."""
        best = None
        for tf in self.all_tfs:
            for a in self.arrays(tf):
                tol = 0.1 * (a.hi - a.lo)
                if a.dir == d and lo <= a.hi + tol and hi >= a.lo - tol and (best is None or (RANK.get(a.kind, 0), a.tf) > (RANK.get(best.kind, 0), best.tf)):
                    best = a
        return best

    def nested(self, d: int, lo: float, hi: float) -> list[str]:
        """The timeframes that have an array of direction `d` at this zone: the more of them stack, the stronger the place (the nesting test)."""
        return [TF_NAMES.get(tf, str(tf) + "m") for tf in self.all_tfs if any(
            a.dir == d and lo <= a.hi + 0.1 * (a.hi - a.lo) and hi >= a.lo - 0.1 * (a.hi - a.lo) for a in self.arrays(tf))]
