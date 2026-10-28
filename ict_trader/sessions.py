"""Trading sessions, New York time, and reference levels (PDH/PDL, Asian range).

ICT rules are defined in New York local time (America/New_York, EST/EDT). All
bar timestamps in the engine are timezone-aware UTC datetimes and are converted
to NY time here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from .config import StrategyConfig, Window

try:
    NY = ZoneInfo("America/New_York")
except Exception:
    NY = timezone(timedelta(hours=-4), "EDT")

Side = Literal["high", "low", "mid"]


@dataclass(frozen=True)
class Bar:
    time: datetime  # bar open time (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass(frozen=True)
class Level:
    name: str   # e.g. "PDH", "PDL", "ASIA_H", "ASIA_L"
    price: float
    side: Side


def to_ny(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(NY)


def trading_day(dt: datetime, rollover: time = time(17, 0)) -> date:
    """Forex trading day in NY time: bars at/after rollover belong to the next calendar day."""
    ny = to_ny(dt)
    if rollover == time(0, 0):
        return ny.date()
    return ny.date() + timedelta(days=1) if ny.time() >= rollover else ny.date()


def in_any(windows: tuple[Window, ...], dt: datetime) -> Window | None:
    t = to_ny(dt).time()
    for w in windows:
        if w.contains(t):
            return w
    return None


@dataclass
class _Range:
    high: float = float("-inf")
    low: float = float("inf")
    high_time: datetime | None = None
    low_time: datetime | None = None
    seen: bool = False
    complete: bool = False


@dataclass
class LevelTracker:
    """Builds reference levels bar by bar, never using data from the future.

    A range level only becomes available on the first bar after its window
    ends, and prior-day levels only once the day has rolled over.
    """

    config: StrategyConfig
    day: date | None = None
    _day_high: float = float("-inf")
    _day_low: float = float("inf")
    _day_high_time: datetime | None = None
    _day_low_time: datetime | None = None
    _prior_high: float | None = None
    _prior_low: float | None = None
    _prior_high_time: datetime | None = None
    _prior_low_time: datetime | None = None
    _ranges: dict[str, _Range] = field(default_factory=dict)

    def update(self, bar: Bar) -> bool:
        """Feed a bar. Returns True if this bar started a new trading day."""
        day = trading_day(bar.time, self.config.day_rollover)
        if self.config.level_period == "week":
            day = day - timedelta(days=day.weekday())  # the week's Monday: prior-period = prior week
        new_day = day != self.day
        if new_day:
            if self.day is not None and self._day_high > float("-inf"):
                self._prior_high, self._prior_low = self._day_high, self._day_low
                self._prior_high_time, self._prior_low_time = self._day_high_time, self._day_low_time
            self.day = day
            self._day_high, self._day_low = float("-inf"), float("inf")
            self._day_high_time, self._day_low_time = None, None
            self._ranges = {w.name: _Range() for w in self.config.reference_ranges}

        t = to_ny(bar.time).time()
        for w in self.config.reference_ranges:
            r = self._ranges[w.name]
            if r.complete:
                continue
            if w.contains(t):
                if bar.high >= r.high:
                    r.high = bar.high
                    r.high_time = bar.time
                if bar.low <= r.low:
                    r.low = bar.low
                    r.low_time = bar.time
                r.seen = True
            elif r.seen:
                r.complete = True

        if bar.high >= self._day_high:
            self._day_high = bar.high
            self._day_high_time = bar.time
        if bar.low <= self._day_low:
            self._day_low = bar.low
            self._day_low_time = bar.time
        return new_day

    def levels(self) -> list[Level]:
        """Levels that are fully formed as of the last bar fed to update().

        None of them include that bar: prior-day levels exclude the current
        day, and a range completes only on the first bar after its window.
        """
        out: list[Level] = []
        if self.config.use_prior_day_levels and self._prior_high is not None:
            hi, lo = ("PWH", "PWL") if self.config.level_period == "week" else ("PDH", "PDL")
            out.append(Level(hi, self._prior_high, "high"))
            out.append(Level(lo, self._prior_low, "low"))  # type: ignore[arg-type]
        for name, r in self._ranges.items():
            if r.complete:
                out.append(Level(f"{name.upper()}_H", r.high, "high"))
                out.append(Level(f"{name.upper()}_L", r.low, "low"))
        return out

    def origin_times(self) -> dict[str, datetime]:
        """Exact bar timestamps where each tracked level formed."""
        out: dict[str, datetime] = {}
        hi, lo = ("PWH", "PWL") if self.config.level_period == "week" else ("PDH", "PDL")
        if self._prior_high_time is not None:
            out[hi] = self._prior_high_time
        if self._prior_low_time is not None:
            out[lo] = self._prior_low_time
        if self._day_high_time is not None:
            out["HOD"] = self._day_high_time
        if self._day_low_time is not None:
            out["LOD"] = self._day_low_time
        for name, r in self._ranges.items():
            if r.high_time is not None:
                out[f"{name.upper()}_H"] = r.high_time
            if r.low_time is not None:
                out[f"{name.upper()}_L"] = r.low_time
        return out


def flatten_deadline(entry: datetime, flatten_time: time | None) -> datetime | None:
    """The next NY flatten_time at or after entry (rule EX-1)."""
    if flatten_time is None:
        return None
    ny = to_ny(entry)
    day = ny.date() if ny.time() < flatten_time else ny.date() + timedelta(days=1)
    return datetime.combine(day, flatten_time, tzinfo=NY)


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
