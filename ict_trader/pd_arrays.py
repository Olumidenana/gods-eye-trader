"""Institutional Premium & Discount (PD) Array Engine.

Implements algorithmic ICT PD Arrays derived from the institutional core curriculum:
  1. Order Blocks (OB) & Mean Threshold (MT = 50% of candle body)
  2. Breaker Blocks (failed highs/lows inverted after structural displacement)
  3. Fair Value Gaps (FVG) & Consequent Encroachment (CE = 50% of gap)
  4. Inversion FVGs (IFVG: blown-through gaps flipping support/resistance)
  5. Volume Imbalances (body-to-body opening gaps)
  6. Rejection Blocks (extended wick extremes where orders were absorbed)

PD Arrays are classified by:
  - Premium (Bearish arrays: sell side liquidity, bearish OB, bearish FVG, bearish Breaker)
  - Discount (Bullish arrays: buy side liquidity, bullish OB, bullish FVG, bullish Breaker)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Any

from .sessions import Bar
from .strategy import LONG, SHORT


@dataclass(frozen=True)
class PDArray:
    kind: str                   # 'order_block', 'breaker', 'fvg', 'ifvg', 'volume_imbalance', 'rejection_block'
    direction: int              # LONG (+1 = discount/bullish) or SHORT (-1 = premium/bearish)
    top: float                  # Upper bound of the array
    bottom: float               # Lower bound of the array
    midpoint: float             # Mean Threshold (OB) or Consequent Encroachment (FVG / wick)
    bar_index: int              # Index in the source bar series where array formed
    time: datetime              # Timestamp of the bar
    mitigated: bool = False     # Whether price has touched/tested the array
    broken: bool = False        # Whether price has closed through the array
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_premium(self) -> bool:
        return self.direction == SHORT

    @property
    def is_discount(self) -> bool:
        return self.direction == LONG

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def tested_by(self, bar: Bar) -> bool:
        """True if bar enters or trades through this PD array."""
        return not (bar.high < self.bottom or bar.low > self.top)

    def respects_midpoint(self, bar: Bar) -> bool:
        """Checks ICT invalidation rule: candle bodies must not close beyond Mean Threshold / CE."""
        if self.direction == LONG:
            return bar.close >= self.midpoint
        return bar.close <= self.midpoint


def calculate_median_body(bars: list[Bar], lookback: int = 20) -> float:
    """Calculates median candle body size over recent lookback bars."""
    if not bars:
        return 0.0
    recent = bars[-lookback:]
    bodies = [abs(b.close - b.open) for b in recent]
    return median(bodies) if bodies else 0.0


# ---------------------------------------------------------------- Order Blocks
def detect_order_blocks(
    bars: list[Bar],
    min_displacement_mult: float = 1.2,
    lookback: int = 50,
) -> list[PDArray]:
    """Detects institutional Order Blocks:

    - Bullish OB: The lowest down-close candle (or cluster) prior to an upward
      displacement move that breaks recent swing structure.
      Mean Threshold = 50% of candle body (or full range).
    - Bearish OB: The highest up-close candle (or cluster) prior to a downward
      displacement move that breaks recent swing structure.
      Mean Threshold = 50% of candle body (or full range).
    """
    if len(bars) < 5:
        return []

    obs: list[PDArray] = []
    med_body = calculate_median_body(bars, lookback=min(len(bars), lookback))
    start_idx = max(2, len(bars) - lookback)

    for i in range(start_idx, len(bars) - 1):
        prev = bars[i - 1]
        curr = bars[i]
        nxt = bars[i + 1]

        disp_body = abs(nxt.close - nxt.open)
        is_displacement = disp_body >= (med_body * min_displacement_mult) if med_body > 0 else True

        # Bullish OB: down candle (curr.close < curr.open) followed by strong up candle
        if curr.close < curr.open and nxt.close > nxt.open and is_displacement:
            # Check displacement breaks previous candle high
            if nxt.close > curr.high:
                body_top = curr.open
                body_bottom = curr.close
                mt = body_bottom + (body_top - body_bottom) * 0.5
                obs.append(PDArray(
                    kind="order_block",
                    direction=LONG,
                    top=curr.high,
                    bottom=curr.low,
                    midpoint=mt,
                    bar_index=i,
                    time=curr.time,
                    details={
                        "body_top": body_top,
                        "body_bottom": body_bottom,
                        "mean_threshold": mt,
                        "displacement_bar": i + 1,
                    },
                ))

        # Bearish OB: up candle (curr.close > curr.open) followed by strong down candle
        elif curr.close > curr.open and nxt.close < nxt.open and is_displacement:
            # Check displacement breaks previous candle low
            if nxt.close < curr.low:
                body_top = curr.close
                body_bottom = curr.open
                mt = body_bottom + (body_top - body_bottom) * 0.5
                obs.append(PDArray(
                    kind="order_block",
                    direction=SHORT,
                    top=curr.high,
                    bottom=curr.low,
                    midpoint=mt,
                    bar_index=i,
                    time=curr.time,
                    details={
                        "body_top": body_top,
                        "body_bottom": body_bottom,
                        "mean_threshold": mt,
                        "displacement_bar": i + 1,
                    },
                ))

    return obs


# ---------------------------------------------------------------- Breaker Blocks
def detect_breaker_blocks(bars: list[Bar], lookback: int = 60) -> list[PDArray]:
    """Detects institutional Breaker Blocks:

    - Bearish Breaker: Low (A) -> High (B) -> Higher High / Sweep (C) ->
      Displacement leg breaking below Low (A).
      The up-candle(s) at Low (A) that failed now invert into resistance (Bearish Breaker).
    - Bullish Breaker: High (A) -> Low (B) -> Lower Low / Sweep (C) ->
      Displacement leg breaking above High (A).
      The down-candle(s) at High (A) that failed now invert into support (Bullish Breaker).
    """
    if len(bars) < 7:
        return []

    breakers: list[PDArray] = []
    start = max(1, len(bars) - lookback)

    # Find 3-candle swing highs and swing lows
    swing_highs: list[tuple[int, float]] = []  # (index, price)
    swing_lows: list[tuple[int, float]] = []   # (index, price)

    for i in range(start, len(bars) - 1):
        if bars[i].high > bars[i - 1].high and bars[i].high > bars[i + 1].high:
            swing_highs.append((i, bars[i].high))
        if bars[i].low < bars[i - 1].low and bars[i].low < bars[i + 1].low:
            swing_lows.append((i, bars[i].low))

    # Bearish Breakers: Swing Low A -> Swing High B -> Swing High C (higher than B) -> Break below A
    for a_idx, a_low in swing_lows:
        # Candidate highs after A
        highs_after_a = [h for h in swing_highs if h[0] > a_idx]
        if len(highs_after_a) < 2:
            continue
        for idx_b in range(len(highs_after_a) - 1):
            b_idx, b_high = highs_after_a[idx_b]
            c_idx, c_high = highs_after_a[idx_b + 1]
            if c_high > b_high:  # Liquidity purge / higher high
                # Check if subsequent candles displace and close below A's low
                for d_idx in range(c_idx + 1, len(bars)):
                    if bars[d_idx].close < a_low:
                        ref_bar = bars[a_idx]
                        mt = (ref_bar.high + ref_bar.low) * 0.5
                        breakers.append(PDArray(
                            kind="breaker",
                            direction=SHORT,
                            top=ref_bar.high,
                            bottom=ref_bar.low,
                            midpoint=mt,
                            bar_index=a_idx,
                            time=ref_bar.time,
                            details={"swing_a": a_low, "swing_b": b_high, "swing_c": c_high, "break_bar": d_idx},
                        ))
                        break

    # Bullish Breakers: Swing High A -> Swing Low B -> Swing Low C (lower than B) -> Break above A
    for a_idx, a_high in swing_highs:
        lows_after_a = [lw for lw in swing_lows if lw[0] > a_idx]
        if len(lows_after_a) < 2:
            continue
        for idx_b in range(len(lows_after_a) - 1):
            b_idx, b_low = lows_after_a[idx_b]
            c_idx, c_low = lows_after_a[idx_b + 1]
            if c_low < b_low:  # Liquidity purge / lower low
                for d_idx in range(c_idx + 1, len(bars)):
                    if bars[d_idx].close > a_high:
                        ref_bar = bars[a_idx]
                        mt = (ref_bar.high + ref_bar.low) * 0.5
                        breakers.append(PDArray(
                            kind="breaker",
                            direction=LONG,
                            top=ref_bar.high,
                            bottom=ref_bar.low,
                            midpoint=mt,
                            bar_index=a_idx,
                            time=ref_bar.time,
                            details={"swing_a": a_high, "swing_b": b_low, "swing_c": c_low, "break_bar": d_idx},
                        ))
                        break

    return breakers


# ---------------------------------------------------------------- Fair Value Gaps & IFVGs
def detect_fvgs(bars: list[Bar], lookback: int = 50) -> list[PDArray]:
    """Detects 3-candle Fair Value Gaps (FVG) and Consequent Encroachment (CE = 50%)."""
    if len(bars) < 3:
        return []

    fvgs: list[PDArray] = []
    start = max(2, len(bars) - lookback)

    for i in range(start, len(bars)):
        c1, c2, c3 = bars[i - 2], bars[i - 1], bars[i]

        # Bullish FVG: Bar 1 high is below Bar 3 low
        if c1.high < c3.low:
            gap_bottom = c1.high
            gap_top = c3.low
            ce = gap_bottom + (gap_top - gap_bottom) * 0.5
            fvgs.append(PDArray(
                kind="fvg",
                direction=LONG,
                top=gap_top,
                bottom=gap_bottom,
                midpoint=ce,
                bar_index=i - 1,
                time=c2.time,
                details={"bar1": i - 2, "bar2": i - 1, "bar3": i, "consequent_encroachment": ce},
            ))

        # Bearish FVG: Bar 1 low is above Bar 3 high
        elif c1.low > c3.high:
            gap_top = c1.low
            gap_bottom = c3.high
            ce = gap_bottom + (gap_top - gap_bottom) * 0.5
            fvgs.append(PDArray(
                kind="fvg",
                direction=SHORT,
                top=gap_top,
                bottom=gap_bottom,
                midpoint=ce,
                bar_index=i - 1,
                time=c2.time,
                details={"bar1": i - 2, "bar2": i - 1, "bar3": i, "consequent_encroachment": ce},
            ))

    return fvgs


def detect_inversion_fvgs(bars: list[Bar], fvgs: list[PDArray]) -> list[PDArray]:
    """Detects Inversion Fair Value Gaps (IFVG):

    When an active FVG is completely breached with full body closes, its polarity flips:
    - Bullish FVG closed below bottom -> becomes Bearish IFVG (resistance).
    - Bearish FVG closed above top -> becomes Bullish IFVG (support).
    """
    if not fvgs or not bars:
        return []

    ifvgs: list[PDArray] = []
    for fvg in fvgs:
        idx = fvg.bar_index
        # Check bars following the FVG formation
        for k in range(idx + 2, len(bars)):
            bar = bars[k]
            if fvg.direction == LONG and bar.close < fvg.bottom:
                # Bullish FVG blown through to downside -> flips to Bearish IFVG
                ifvgs.append(PDArray(
                    kind="ifvg",
                    direction=SHORT,
                    top=fvg.top,
                    bottom=fvg.bottom,
                    midpoint=fvg.midpoint,
                    bar_index=k,
                    time=bar.time,
                    details={"original_fvg_time": fvg.time, "inversion_bar": k},
                ))
                break
            elif fvg.direction == SHORT and bar.close > fvg.top:
                # Bearish FVG blown through to upside -> flips to Bullish IFVG
                ifvgs.append(PDArray(
                    kind="ifvg",
                    direction=LONG,
                    top=fvg.top,
                    bottom=fvg.bottom,
                    midpoint=fvg.midpoint,
                    bar_index=k,
                    time=bar.time,
                    details={"original_fvg_time": fvg.time, "inversion_bar": k},
                ))
                break

    return ifvgs


# ---------------------------------------------------------------- Volume Imbalances
def detect_volume_imbalances(bars: list[Bar], lookback: int = 50) -> list[PDArray]:
    """Detects Volume Imbalances: where candle bodies do not meet (open != previous close),

    leaving a real gap in trading volume even if wicks overlap.
    """
    if len(bars) < 2:
        return []

    vis: list[PDArray] = []
    start = max(1, len(bars) - lookback)

    for i in range(start, len(bars)):
        prev = bars[i - 1]
        curr = bars[i]

        # Gap up: prev close < curr open
        if curr.open > prev.close and curr.open > prev.open:
            gap_bottom = max(prev.open, prev.close)
            gap_top = curr.open
            if gap_top > gap_bottom:
                ce = (gap_bottom + gap_top) * 0.5
                vis.append(PDArray(
                    kind="volume_imbalance",
                    direction=LONG,
                    top=gap_top,
                    bottom=gap_bottom,
                    midpoint=ce,
                    bar_index=i,
                    time=curr.time,
                ))

        # Gap down: prev close > curr open
        elif curr.open < prev.close and curr.open < prev.open:
            gap_top = min(prev.open, prev.close)
            gap_bottom = curr.open
            if gap_top > gap_bottom:
                ce = (gap_bottom + gap_top) * 0.5
                vis.append(PDArray(
                    kind="volume_imbalance",
                    direction=SHORT,
                    top=gap_top,
                    bottom=gap_bottom,
                    midpoint=ce,
                    bar_index=i,
                    time=curr.time,
                ))

    return vis


# ---------------------------------------------------------------- Rejection Blocks
def detect_rejection_blocks(bars: list[Bar], lookback: int = 60, wick_share: float = 0.4) -> list[PDArray]:
    """Detects Rejection Blocks: the wick of the candle at a swing extreme, where price was turned away.

    - Bearish (premium): a swing high whose upper wick is at least `wick_share` of its range; the block runs from the top of the
      body to the high.
    - Bullish (discount): the mirror at a swing low, from the low to the bottom of the body.
    A candle smaller than a typical one is noise, not a rejection. `broken` once a candle closes beyond the extreme; `mitigated`
    once price has traded back into the wick's midpoint.
    """
    if len(bars) < 6:
        return []
    med = sorted(x.high - x.low for x in bars[-max(6, lookback):])[min(len(bars), max(6, lookback)) // 2]
    out: list[PDArray] = []
    for i in range(max(2, len(bars) - lookback), len(bars) - 2):
        c, rng = bars[i], bars[i].high - bars[i].low
        if rng <= 0 or rng < 0.8 * med:
            continue
        around = bars[i - 2:i] + bars[i + 1:i + 3]
        later = bars[i + 1:]
        top_body, bot_body = max(c.open, c.close), min(c.open, c.close)
        if c.high > max(x.high for x in around) and (c.high - top_body) >= wick_share * rng:
            out.append(PDArray(kind="rejection_block", direction=SHORT, top=c.high, bottom=top_body, midpoint=(c.high + top_body) / 2,
                               bar_index=i, time=c.time, mitigated=any(x.high >= (c.high + top_body) / 2 for x in later),
                               broken=any(x.close > c.high for x in later)))
        elif c.low < min(x.low for x in around) and (bot_body - c.low) >= wick_share * rng:
            out.append(PDArray(kind="rejection_block", direction=LONG, top=bot_body, bottom=c.low, midpoint=(c.low + bot_body) / 2,
                               bar_index=i, time=c.time, mitigated=any(x.low <= (c.low + bot_body) / 2 for x in later),
                               broken=any(x.close < c.low for x in later)))
    return out


def detect_bprs(bars: list[Bar], fvgs: list[PDArray] | None = None, lookback: int = 60) -> list[PDArray]:
    """Detects Balanced Price Ranges (BPR, Rule IV-02):

    An aggressive price move in one direction creating an FVG, followed by an
    opposing displacement move creating an overlapping FVG in the opposite direction.
    The overlap zone [max(f1.bottom, f2.bottom), min(f1.top, f2.top)] is the BPR.
    """
    if fvgs is None:
        fvgs = detect_fvgs(bars, lookback=lookback)
    if len(fvgs) < 2:
        return []
    bprs: list[PDArray] = []
    for i, f1 in enumerate(fvgs):
        for f2 in fvgs[i + 1:]:
            if f2.direction == f1.direction or f2.bar_index - f1.bar_index > 24:
                continue
            bot, top = max(f1.bottom, f2.bottom), min(f1.top, f2.top)
            if top <= bot:
                continue
            mid = (bot + top) * 0.5
            later = bars[f2.bar_index + 2:] if bars else []
            bprs.append(PDArray(
                kind="bpr",
                direction=f2.direction,
                top=top,
                bottom=bot,
                midpoint=mid,
                bar_index=f2.bar_index,
                time=f2.time,
                mitigated=any((x.low <= mid if f2.direction == LONG else x.high >= mid) for x in later),
                broken=any((x.close < bot if f2.direction == LONG else x.close > top) for x in later),
                details={"first_fvg_bar": f1.bar_index, "second_fvg_bar": f2.bar_index},
            ))
            break
    return bprs


# ---------------------------------------------------------------- Unified Scanner
def scan_pd_arrays(bars: list[Bar], lookback: int = 60) -> list[PDArray]:
    """Runs all institutional PD Array detectors across the given bar series."""
    obs = detect_order_blocks(bars, lookback=lookback)
    breakers = detect_breaker_blocks(bars, lookback=lookback)
    fvgs = detect_fvgs(bars, lookback=lookback)
    ifvgs = detect_inversion_fvgs(bars, fvgs)
    bprs = detect_bprs(bars, fvgs, lookback=lookback)
    vis = detect_volume_imbalances(bars, lookback=lookback)

    all_arrays = obs + breakers + fvgs + ifvgs + bprs + vis + detect_rejection_blocks(bars, lookback=lookback)
    return sorted(all_arrays, key=lambda a: a.time)
