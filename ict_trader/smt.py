"""Smart Money Technique (SMT) Intermarket Divergence Engine.

Detects institutional divergence across correlated and inversely-correlated asset pairs:
  - Correlated Equities: ES (S&P 500) vs NQ (Nasdaq 100) vs YM (Dow Jones)
  - Correlated FX: EUR/USD vs GBP/USD
  - Inverse Macro: EUR/USD vs USDX (DXY), ES vs USDX (DXY)

Core ICT Principles:
  1. Bullish SMT:
     - Correlated: One asset runs sell-side liquidity to a Lower Low, while the sister
       asset fails to confirm by holding a Higher Low. Strong institutional accumulation.
     - Inverse: Target asset makes a Lower Low, while DXY fails to make a Higher High.
  2. Bearish SMT:
     - Correlated: One asset runs buy-side liquidity to a Higher High, while the sister
       asset fails to confirm by holding a Lower High. Heavy institutional distribution.
     - Inverse: Target asset makes a Higher High, while DXY fails to make a Lower Low.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .sessions import Bar
from .strategy import LONG, SHORT

CORRELATED_PAIRS = {
    ("US500", "NAS100"),
    ("NAS100", "US500"),
    ("EURUSD", "GBPUSD"),
    ("GBPUSD", "EURUSD"),
    ("US500", "US30"),
}

INVERSE_PAIRS = {
    ("EURUSD", "DXY"),
    ("GBPUSD", "DXY"),
    ("US500", "DXY"),
    ("NAS100", "DXY"),
}


@dataclass(frozen=True)
class SMTDivergence:
    asset_a: str
    asset_b: str
    direction: int            # LONG (bullish accumulation) or SHORT (bearish distribution)
    time: datetime
    a_highs: tuple[float, float] | None = None  # (prev_high, curr_high)
    a_lows: tuple[float, float] | None = None   # (prev_low, curr_low)
    b_highs: tuple[float, float] | None = None  # (prev_high, curr_high)
    b_lows: tuple[float, float] | None = None   # (prev_low, curr_low)
    is_inverse: bool = False
    description: str = ""


def _find_recent_swings(bars: list[Bar], lookback: int = 40) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Returns ((index, high), (index, low)) for 3-bar swing points in recent bars."""
    if len(bars) < 3:
        return [], []
    start = max(1, len(bars) - lookback)
    highs, lows = [], []
    for i in range(start, len(bars) - 1):
        if bars[i].high > bars[i - 1].high and bars[i].high > bars[i + 1].high:
            highs.append((i, bars[i].high))
        if bars[i].low < bars[i - 1].low and bars[i].low < bars[i + 1].low:
            lows.append((i, bars[i].low))
    return highs, lows


def detect_smt_divergence(
    bars_a: list[Bar],
    bars_b: list[Bar],
    symbol_a: str,
    symbol_b: str,
    is_inverse: bool = False,
    lookback: int = 40,
) -> SMTDivergence | None:
    """Evaluates the two most recent swing points of Asset A against Asset B.

    Returns SMTDivergence if a crack in correlation is confirmed.
    """
    if len(bars_a) < 5 or len(bars_b) < 5:
        return None

    highs_a, lows_a = _find_recent_swings(bars_a, lookback=lookback)
    highs_b, lows_b = _find_recent_swings(bars_b, lookback=lookback)

    # 1. Bearish SMT (Check Highs)
    if len(highs_a) >= 2 and len(highs_b) >= 2:
        a_prev_idx, a_prev_h = highs_a[-2]
        a_curr_idx, a_curr_h = highs_a[-1]

        b_prev_idx, b_prev_h = highs_b[-2]
        b_curr_idx, b_curr_h = highs_b[-1]

        t = bars_a[a_curr_idx].time

        if not is_inverse:
            # Correlated: A makes Higher High, B makes Lower High
            if a_curr_h > a_prev_h and b_curr_h < b_prev_h:
                return SMTDivergence(
                    asset_a=symbol_a,
                    asset_b=symbol_b,
                    direction=SHORT,
                    time=t,
                    a_highs=(a_prev_h, a_curr_h),
                    b_highs=(b_prev_h, b_curr_h),
                    is_inverse=False,
                    description=f"Bearish SMT: {symbol_a} printed Higher High ({a_curr_h:.2f} > {a_prev_h:.2f}) while {symbol_b} failed with Lower High ({b_curr_h:.2f} < {b_prev_h:.2f})",
                )
            # Or B makes Higher High, A makes Lower High
            elif a_curr_h < a_prev_h and b_curr_h > b_prev_h:
                return SMTDivergence(
                    asset_a=symbol_a,
                    asset_b=symbol_b,
                    direction=SHORT,
                    time=t,
                    a_highs=(a_prev_h, a_curr_h),
                    b_highs=(b_prev_h, b_curr_h),
                    is_inverse=False,
                    description=f"Bearish SMT: {symbol_b} printed Higher High ({b_curr_h:.2f} > {b_prev_h:.2f}) while {symbol_a} showed weakness with Lower High ({a_curr_h:.2f} < {a_prev_h:.2f})",
                )
        else:
            # Inversely Correlated (e.g. Asset A vs DXY):
            # If A makes Higher High, DXY should make Lower Low.
            # If DXY makes a Higher Low (fails to make lower low), that's Bearish SMT for Asset A!
            if len(lows_b) >= 2:
                b_prev_l_idx, b_prev_l = lows_b[-2]
                b_curr_l_idx, b_curr_l = lows_b[-1]
                if a_curr_h > a_prev_h and b_curr_l > b_prev_l:
                    return SMTDivergence(
                        asset_a=symbol_a,
                        asset_b=symbol_b,
                        direction=SHORT,
                        time=t,
                        a_highs=(a_prev_h, a_curr_h),
                        b_lows=(b_prev_l, b_curr_l),
                        is_inverse=True,
                        description=f"Bearish Inverse SMT: {symbol_a} made Higher High but {symbol_b} failed to make Lower Low (Higher Low {b_curr_l:.2f} > {b_prev_l:.2f})",
                    )

    # 2. Bullish SMT (Check Lows)
    if len(lows_a) >= 2 and len(lows_b) >= 2:
        a_prev_idx, a_prev_l = lows_a[-2]
        a_curr_idx, a_curr_l = lows_a[-1]

        b_prev_idx, b_prev_l = lows_b[-2]
        b_curr_idx, b_curr_l = lows_b[-1]

        t = bars_a[a_curr_idx].time

        if not is_inverse:
            # Correlated: A makes Lower Low, B makes Higher Low
            if a_curr_l < a_prev_l and b_curr_l > b_prev_l:
                return SMTDivergence(
                    asset_a=symbol_a,
                    asset_b=symbol_b,
                    direction=LONG,
                    time=t,
                    a_lows=(a_prev_l, a_curr_l),
                    b_lows=(b_prev_l, b_curr_l),
                    is_inverse=False,
                    description=f"Bullish SMT: {symbol_a} purged to Lower Low ({a_curr_l:.2f} < {a_prev_l:.2f}) while {symbol_b} accumulated with Higher Low ({b_curr_l:.2f} > {b_prev_l:.2f})",
                )
            # Or B makes Lower Low, A makes Higher Low
            elif a_curr_l > a_prev_l and b_curr_l < b_prev_l:
                return SMTDivergence(
                    asset_a=symbol_a,
                    asset_b=symbol_b,
                    direction=LONG,
                    time=t,
                    a_lows=(a_prev_l, a_curr_l),
                    b_lows=(b_prev_l, b_curr_l),
                    is_inverse=False,
                    description=f"Bullish SMT: {symbol_b} purged to Lower Low ({b_curr_l:.2f} < {b_prev_l:.2f}) while {symbol_a} held strong with Higher Low ({a_curr_l:.2f} > {a_prev_l:.2f})",
                )
        else:
            # Inversely Correlated (e.g. Asset A vs DXY):
            # If A makes Lower Low, DXY should make Higher High.
            # If DXY makes a Lower High (fails to make higher high), that's Bullish SMT for Asset A!
            if len(highs_b) >= 2:
                b_prev_h_idx, b_prev_h = highs_b[-2]
                b_curr_h_idx, b_curr_h = highs_b[-1]
                if a_curr_l < a_prev_l and b_curr_h < b_prev_h:
                    return SMTDivergence(
                        asset_a=symbol_a,
                        asset_b=symbol_b,
                        direction=LONG,
                        time=t,
                        a_lows=(a_prev_l, a_curr_l),
                        b_highs=(b_prev_h, b_curr_h),
                        is_inverse=True,
                        description=f"Bullish Inverse SMT: {symbol_a} made Lower Low but {symbol_b} failed to make Higher High (Lower High {b_curr_h:.2f} < {b_prev_h:.2f})",
                    )

    return None
