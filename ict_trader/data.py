"""Loading bar data, plus a synthetic generator for demos and tests.

CSV columns: time, open, high, low, close[, volume]. `time` is the bar open
time in UTC: ISO 8601 or MetaTrader's "2026.01.05 13:05" export format. Use
--tz to declare the timezone of naive timestamps (MT5 server time is often
UTC+2/+3; get this wrong and every killzone is shifted).
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .sessions import Bar, to_ny


def _parse_time(value: str, tz: ZoneInfo | timezone) -> datetime:
    value = value.strip().replace("Z", "+00:00")
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
        try:
            ts = datetime.strptime(value, fmt)
            break
        except ValueError:
            continue
    else:
        ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=tz)
    return ts.astimezone(timezone.utc)


def load_bars_csv(path: str | Path, tz: str = "UTC") -> list[Bar]:
    zone = timezone.utc if tz.upper() == "UTC" else ZoneInfo(tz)
    bars: list[Bar] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower().strip("<> "): c for c in reader.fieldnames or []}
        if "time" not in cols and "date" in cols:
            cols["time"] = cols["date"]
        for row in reader:
            g = lambda k: row[cols[k]]  # noqa: E731
            bars.append(Bar(
                _parse_time(g("time"), zone), float(g("open")), float(g("high")), float(g("low")), float(g("close")),
                float(row[cols["volume"]]) if "volume" in cols and row[cols["volume"]] else 0.0,
            ))
    bars.sort(key=lambda b: b.time)
    for prev, cur in zip(bars, bars[1:]):
        if cur.time == prev.time:
            raise ValueError(f"Duplicate bar at {cur.time.isoformat()}")
    return bars


def save_bars_csv(bars: list[Bar], path: str | Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b.time.strftime("%Y-%m-%dT%H:%M:%SZ"), b.open, b.high, b.low, b.close, b.volume])


def synthetic_bars(start: datetime, days: int, timeframe_minutes: int = 5, price: float = 1.1000, seed: int = 7) -> list[Bar]:
    """A random walk with session-dependent volatility, weekends skipped.

    It has no edge by construction, so a strategy that looks profitable on it
    is overfit. Use it to exercise the pipeline, never to judge the strategy.
    """
    rng = random.Random(seed)
    bars: list[Bar] = []
    ts = start
    end = start + timedelta(days=days)
    step = timedelta(minutes=timeframe_minutes)
    while ts < end:
        ny = to_ny(ts)
        # Forex week: Sunday 17:00 to Friday 17:00 New York.
        closed = ny.weekday() == 5 or (ny.weekday() == 4 and ny.hour >= 17) or (ny.weekday() == 6 and ny.hour < 17)
        if not closed:
            hour = ny.hour
            vol = 0.00012 if 2 <= hour < 11 else 0.00005
            o = price
            path = [o]
            for _ in range(4):
                path.append(path[-1] + rng.gauss(0, vol / 2))
            c = path[-1]
            hi = max(path) + abs(rng.gauss(0, vol / 4))
            lo = min(path) - abs(rng.gauss(0, vol / 4))
            bars.append(Bar(ts, round(o, 5), round(hi, 5), round(lo, 5), round(c, 5), float(rng.randint(50, 500))))
            price = c
        ts += step
    return bars

