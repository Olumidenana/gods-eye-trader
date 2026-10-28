"""ICT strict: only the setups ICT himself would take.

The engine finds every sweep, shift and gap on every market. Strict keeps the subset that
matches how ICT trades the 2022 model:
- markets he applies it to (index futures, the major currency pairs, gold, silver, crude);
  never computer-generated synthetic indices, which have no institutional order flow;
- 5- and 15-minute execution inside the killzones;
- no Mondays, and no Friday afternoons;
- grade A or better at 2R or more.

Each signal carries `strict` (True/False) and `strict_why` (the reasons it isn't strict), so
the track record can be judged on strict trades alone and the bots can trade only them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .sessions import to_ny

STRICT_MARKETS = frozenset({"EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD",
                            "XAUUSD", "XAGUSD", "US500", "NAS100", "US30", "WTI"})
STRICT_TIMEFRAMES = (5, 15)
MIN_GRADE = "A"
MIN_RR = 2.0
GRADES = ("A+", "A", "B", "C")


def strict_why(sig: dict[str, Any]) -> list[str]:
    """Why a signal falls outside ICT strict; an empty list means it qualifies."""
    why = []
    if sig.get("market") not in STRICT_MARKETS:
        why.append("not a market ICT trades")
    if int(sig.get("timeframe_minutes") or 5) not in STRICT_TIMEFRAMES:
        why.append("not a 5m or 15m killzone trade")
    t = sig.get("time")
    if t:
        ny = to_ny(datetime.fromisoformat(t))
        if ny.weekday() == 0:
            why.append("Monday")
        elif ny.weekday() == 4 and ny.hour >= 12:
            why.append("Friday afternoon")
        elif ny.weekday() >= 5:
            why.append("weekend")
    g = (sig.get("quality") or {}).get("grade", "C")
    if g not in GRADES or GRADES.index(g) > GRADES.index(MIN_GRADE):
        why.append(f"grade {g} (needs {MIN_GRADE} or better)")
    if float(sig.get("rr") or 0) < MIN_RR:
        why.append(f"{float(sig.get('rr') or 0):.1f}R (needs {MIN_RR:.0f}R)")
    return why


def mark(sig: dict[str, Any]) -> None:
    why = strict_why(sig)
    sig["strict"], sig["strict_why"] = not why, why
