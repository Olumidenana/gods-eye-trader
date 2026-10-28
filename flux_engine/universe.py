"""Market Universe & Configuration for God's Eye Flux (Zero Synthetics, Zero Stock Indices).

Strictly institutional, session-driven markets with real order flow:
- Gold & Silver
- Forex Majors
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class TimeWindow:
    name: str
    start: time
    end: time

    def contains(self, t: time) -> bool:
        if self.start <= self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end


@dataclass(frozen=True)
class MarketSpec:
    key: str
    label: str
    asset_class: str          # "forex" | "metal"
    deriv_symbol: str         # Deriv symbol (frxXAUUSD, etc.)
    yahoo_symbol: str         # Yahoo Finance ticker fallback
    pip_size: float
    typical_spread_pips: float
    decimals: int
    smt_partner: str          # Correlated triad/dyad partner key for SMT divergence
    region: str = "us"        # "us" | "eu" | "asia" | "global"


# Strict Institutional Universe — Zero Synthetics, Zero Stock Indices (Forex Majors & Gold/Silver only)
UNIVERSE: tuple[MarketSpec, ...] = (
    # --- Precious Metals ---
    MarketSpec("XAUUSD", "Gold (XAU/USD)", "metal", "frxXAUUSD", "GC=F", 0.1, 2.0, 2, "XAGUSD", "global"),
    MarketSpec("XAGUSD", "Silver (XAG/USD)", "metal", "frxXAGUSD", "SI=F", 0.01, 2.0, 3, "XAUUSD", "global"),

    # --- Forex Majors ---
    MarketSpec("EURUSD", "EUR/USD", "forex", "frxEURUSD", "EURUSD=X", 0.0001, 0.7, 5, "GBPUSD", "global"),
    MarketSpec("GBPUSD", "GBP/USD", "forex", "frxGBPUSD", "GBPUSD=X", 0.0001, 0.9, 5, "EURUSD", "global"),
    MarketSpec("USDJPY", "USD/JPY", "forex", "frxUSDJPY", "JPY=X", 0.01, 0.8, 3, "EURUSD", "global"),
    MarketSpec("AUDUSD", "AUD/USD", "forex", "frxAUDUSD", "AUDUSD=X", 0.0001, 0.9, 5, "NZDUSD", "global"),
    MarketSpec("NZDUSD", "NZD/USD", "forex", "frxNZDUSD", "NZDUSD=X", 0.0001, 1.1, 5, "AUDUSD", "global"),
    MarketSpec("USDCAD", "USD/CAD", "forex", "frxUSDCAD", "CAD=X", 0.0001, 1.0, 5, "EURUSD", "global"),
)

BY_KEY: dict[str, MarketSpec] = {m.key: m for m in UNIVERSE}
BY_DERIV: dict[str, MarketSpec] = {m.deriv_symbol: m for m in UNIVERSE if m.deriv_symbol}
ASSET_CLASSES: list[str] = ["metal", "forex"]


# Standard Institutional Sessions & Killzones (NY EST Time)
ICT_KILLZONES: tuple[TimeWindow, ...] = (
    TimeWindow("ASIAN_RANGE", time(20, 0), time(0, 0)),
    TimeWindow("LONDON_OPEN", time(2, 0), time(5, 0)),
    TimeWindow("NY_AM", time(8, 30), time(11, 0)),
    TimeWindow("NY_PM", time(13, 30), time(16, 0)),
)

MACRO_WINDOWS: tuple[TimeWindow, ...] = (
    TimeWindow("MACRO_0850_0910", time(8, 50), time(9, 10)),
    TimeWindow("MACRO_0950_1010", time(9, 50), time(10, 10)),
    TimeWindow("MACRO_1050_1110", time(10, 50), time(11, 10)),
    TimeWindow("MACRO_1150_1210", time(11, 50), time(12, 10)),
    TimeWindow("MACRO_1310_1340", time(13, 10), time(13, 40)),
    TimeWindow("MACRO_1515_1545", time(15, 15), time(15, 45)),
)


def killzones_for(spec: MarketSpec) -> tuple[TimeWindow, ...]:
    return ICT_KILLZONES


def is_macro_window(t: time) -> tuple[bool, str]:
    for w in MACRO_WINDOWS:
        if w.contains(t):
            return True, w.name
    return False, ""
