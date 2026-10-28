"""Plain-text signal feed for the ICT Flux MT5 robot (mql5/ICTFluxMT5.mq5).

MQL5 has no JSON parser, so /api/scan?format=mt5 serves one pipe-separated line per
open signal from the last hour. The robot decides freshness, limits and sizing itself.

    ICTFLUX1|<generated epoch>
    id|cls|mt5 symbol|label|buy/sell|entry|stop|tp1|target|rr|grade|signal epoch|tf minutes|share off at TP1
"""

from __future__ import annotations

from datetime import datetime, timezone

HEADER = "ICTFLUX1"
MAX_AGE_MINUTES = 60


def mt5_symbol(sig: dict) -> str:
    """Deriv MT5 names: frxEURUSD -> EURUSD, cryBTCUSD -> BTCUSD; synthetics and the rest use the label
    (e.g. 'Volatility 75 Index'), which is what Deriv's MT5 servers call them."""
    bs = sig.get("broker_symbol") or ""
    if bs[:3] in ("frx", "cry"):
        return bs[3:]
    return sig.get("label") or bs


def _clean(v) -> str:
    return str(v if v is not None else "").replace("|", "/").replace("\n", " ")


def feed(report: dict, now: datetime | None = None, strict: bool = False) -> str:
    now = now or datetime.now(timezone.utc)
    lines = [f"{HEADER}|{int(now.timestamp())}"]
    for m in report.get("markets", []):
        if not m.get("ok"):
            continue
        for s in m.get("today", {}).get("signals", []):
            t = datetime.fromisoformat(s["time"])
            if (s.get("outcome") or {}).get("status") != "open" or (now - t).total_seconds() > MAX_AGE_MINUTES * 60 or (strict and not s.get("strict")):
                continue
            tp1 = (s.get("setup") or {}).get("tp1") or 0
            grade = (s.get("quality") or {}).get("grade", "")
            row = [s["id"], m.get("cls") or s.get("cls"), mt5_symbol(s), s.get("label"), "buy" if s["direction"] == "long" else "sell",
                   s["entry"], s["stop"], tp1, s["target"], s.get("rr", 0), grade, int(t.timestamp()), s.get("timeframe_minutes", 5),
                   (s.get("management") or {}).get("partial", 0.5)]
            lines.append("|".join(_clean(v) for v in row))
    return "\n".join(lines) + "\n"
