"""Market conditions, read the way a trader reads the tape before choosing a model.

Five labels, each from price (and the clock where the market has one):

  trend   trending / mixed / ranging   how directly price has travelled over the last 48 candles
  state   expansion / retracement / consolidation / mixed   what the market is doing: a directed leg, a pullback inside one, or a range
  vol     high / normal / low          recent candle size against the market's own norm
  draw    with / none / against        the daily draw against the trade (for a market read: up / down / none)
  phase   open / killzone / off / 24h  the first half hour after an ICT open, a killzone, outside both; 24h on markets with no clock
  path    clear / some / busy / seek   how many opposing gaps stand between entry and target; seek = both sides being stopped out
  array   fvg / ote / ob / breaker / ifvg / unicorn   which PD array the entry sits in (stamped on a trade, so the lab shows which arrays pay)
  poi     htf-pool / 1h fvg / 4h ob ...   the higher-timeframe point the entry sits at (top-down, htf.py)

The same labels tag every backtested trade (Research lab, "By condition") and describe the market now
(the live "best fit" on Home), so what the lab learns about when a model works is what the selector uses.
"""

from __future__ import annotations

from statistics import median
from typing import Any

TREND_AT, RANGE_AT = 0.25, 0.12      # efficiency ratio cut-offs
VOL_HIGH, VOL_LOW = 1.4, 0.7         # recent median range / long-run median range
DIMENSIONS = ("trend", "state", "vol", "draw", "phase", "path", "array", "poi")


def efficiency(closes: list[float]) -> float | None:
    if len(closes) < 24:
        return None
    walk = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
    return abs(closes[-1] - closes[0]) / walk if walk else None


def label_trend(closes: list[float]) -> str | None:
    er = efficiency(closes)
    return None if er is None else "trending" if er >= TREND_AT else "ranging" if er <= RANGE_AT else "mixed"


def label_state(closes: list[float]) -> str | None:
    """What the market is doing now, from price alone (so it holds on a synthetic index too): consolidation (it is going nowhere, the
    middle of the range is 50/50), expansion (a directed leg, the last candles moving with it: do not fade it), retracement (a
    directed leg whose last candles are pulling back: the entry is the array it retraces into), or mixed."""
    recent = closes[-49:]
    er = efficiency(recent)
    if er is None:
        return None
    if er <= RANGE_AT:
        return "consolidation"
    if er >= TREND_AT:
        return "expansion" if (recent[-1] - recent[-13]) * (recent[-1] - recent[0]) >= 0 else "retracement"
    return "mixed"


def label_vol(ranges: list[float]) -> str | None:
    if len(ranges) < 100:
        return None
    base = median(ranges)
    recent = median(ranges[-24:])
    if not base:
        return None
    r = recent / base
    return "high" if r >= VOL_HIGH else "low" if r <= VOL_LOW else "normal"


def tag(c: dict[str, Any]) -> str:
    """The rules-trail line the Research lab parses back (CN-00 trend=trending; vol=normal; ...)."""
    return "CN-00 conditions: " + "; ".join(f"{k}={c[k]}" for k in DIMENSIONS if c.get(k))


def parse(reasons) -> dict[str, str]:
    for r in reasons or ():
        if r.startswith("CN-00 conditions: "):
            return dict(p.split("=", 1) for p in r[len("CN-00 conditions: "):].split("; ") if "=" in p)
    return {}
