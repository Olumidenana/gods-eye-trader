"""Advanced Institutional Sessions, Power of 3 (AMD), and Algorithmic Macro Windows.

Derives the institutional time-delivery principles:
  1. Power of Three (AMD):
     - Accumulation: Asian session consolidation (19:00 - 00:00 / 00:00 - 02:00 NY)
     - Manipulation: London Judas Swing (02:00 - 05:00 NY) engineers High/Low of Day
     - Distribution: New York Session (07:00 - 11:00 AM NY) delivers true expansion
  2. ICT Algorithmic 20-Minute Macro Windows:
     - London Open Macro:     02:33 - 03:00 NY
     - NY AM Open Macro:       08:50 - 09:10 NY (sweeps pre-market liquidity)
     - NY AM Mid Macro:        09:50 - 10:10 NY (key session turning point)
     - London Close Macro:     10:30 - 11:00 NY (daily range extension or retracement)
     - NY AM Post-Close Macro: 10:50 - 11:10 NY (pre-lunch distribution)
     - NY PM Re-price Macro:   13:50 - 14:10 NY (post-lunch continuation)
     - NY Final Settlement:    15:15 - 15:45 NY (closing run into equity settlement)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

from .sessions import to_ny


@dataclass(frozen=True)
class MacroWindow:
    name: str
    start: time
    end: time
    description: str

    def contains(self, t: time) -> bool:
        return self.start <= t <= self.end


# Official ICT 20-minute algorithmic macro windows (all times New York local)
MACRO_WINDOWS: tuple[MacroWindow, ...] = (
    MacroWindow("london_open_macro", time(2, 33), time(3, 0), "London Open liquidity sweep & initial expansion"),
    MacroWindow("ny_am_open_macro", time(8, 50), time(9, 10), "Pre-9:30 cash open sweep & algorithmic injection"),
    MacroWindow("ny_am_mid_macro", time(9, 50), time(10, 10), "Silver bullet / AM turning point & trend resumption"),
    MacroWindow("london_close_macro", time(10, 30), time(11, 0), "London close liquidity injection & reversal window"),
    MacroWindow("ny_am_close_macro", time(10, 50), time(11, 10), "AM session completion & distribution into lunch"),
    MacroWindow("ny_pm_reprice_macro", time(13, 50), time(14, 10), "Afternoon trend continuation & liquidity purge"),
    MacroWindow("ny_settlement_macro", time(15, 15), time(15, 45), "Final hour algorithmic settlement & ramp"),
)


def in_macro(dt: datetime) -> MacroWindow | None:
    """Checks if the given UTC or timezone-aware datetime falls inside an ICT macro window."""
    ny_dt = to_ny(dt)
    t = ny_dt.time()
    for w in MACRO_WINDOWS:
        if w.contains(t):
            return w
    return None


def get_amd_phase(dt: datetime) -> str:
    """Classifies current time into the Power of 3 (AMD) phase for the trading day.

    - Accumulation: 19:00 - 02:00 NY (Asia & pre-London)
    - Manipulation: 02:00 - 05:00 NY (London Judas swing)
    - Distribution: 07:00 - 11:30 NY (NY AM session expansion)
    - Re-distribution / Consolidation: 11:30 - 13:30 NY (Lunch pause)
    - PM Distribution: 13:30 - 16:00 NY (NY PM session)
    """
    ny_dt = to_ny(dt)
    h = ny_dt.hour
    m = ny_dt.minute

    # Asian Accumulation
    if h >= 19 or h < 2:
        return "accumulation"
    # London Judas Manipulation
    if 2 <= h < 5:
        return "manipulation"
    # NY AM Distribution
    if 7 <= h < 11 or (h == 11 and m <= 30):
        return "distribution"
    # NY Lunch Re-consolidation
    if (h == 11 and m > 30) or (h == 12) or (h == 13 and m < 30):
        return "lunch_consolidation"
    # NY PM Distribution
    if (h == 13 and m >= 30) or (14 <= h < 16):
        return "pm_distribution"

    return "inter_session"
