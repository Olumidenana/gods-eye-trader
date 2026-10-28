"""Fundamentals filter: stay out of the market around high-impact news.

The calendar is a CSV with columns: time_utc, currency, impact, title
(time_utc in ISO 8601, e.g. 2026-10-02T12:30:00Z). Export it from your
calendar provider of choice; the system never trades blind if the file is
missing in live mode (see live.py).
"""

from __future__ import annotations

import csv
from bisect import bisect_left
from dataclasses import dataclass
import re
import zlib
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

from .config import InstrumentConfig, NewsConfig


NY = ZoneInfo("America/New_York")
FOMC = re.compile(r"FOMC|Federal Funds Rate|Fed Chair|Powell Speaks", re.I)
NFP = re.compile(r"Non-?Farm (Employment|Payroll)", re.I)


@dataclass(frozen=True)
class NewsEvent:
    time: datetime
    currency: str
    impact: str
    title: str


def parse_utc(value: str) -> datetime:
    value = value.strip().replace("Z", "+00:00")
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def load_calendar_csv(path: str | Path) -> list[NewsEvent]:
    with open(path, newline="") as fh:
        rows = csv.DictReader(fh)
        events = [
            NewsEvent(parse_utc(r["time_utc"]), r["currency"].strip().upper(), r["impact"].strip().lower(), r.get("title", "").strip())
            for r in rows
        ]
    return sorted(events, key=lambda e: e.time)


def symbol_currencies(instrument: InstrumentConfig) -> tuple[str, ...]:
    if instrument.news_currencies:
        return tuple(c.upper() for c in instrument.news_currencies)
    s = instrument.symbol.upper()
    if len(s) >= 6 and s[:6].isalpha():
        return (s[:3], s[3:6])
    raise ValueError(f"Set instrument.news_currencies for non-FX symbol {instrument.symbol!r}")


def market_currencies(key: str, cls: str, region: str = "us") -> tuple[str, ...]:
    """Whose news moves a market (NF-01). Synthetic indices have none."""
    if cls == "synthetic":
        return ()
    if cls == "forex" and len(key) == 6:
        return (key[:3], key[3:])
    if cls == "index":
        return {"UK100": ("GBP",), "DE40": ("EUR",), "FR40": ("EUR",), "EU50": ("EUR",), "JP225": ("JPY",),
                "AU200": ("AUD",), "HK33": ("CNY",)}.get(key, ("USD",))
    return ("USD",)  # metals, energy, grains and crypto trade off US data


class NewsFilter:
    def __init__(self, config: NewsConfig, instrument: InstrumentConfig, events: list[NewsEvent]):
        self.config = config
        currencies = set(symbol_currencies(instrument))
        impacts = set(config.impacts)
        self.events = [e for e in events if e.currency in currencies and e.impact in impacts]
        self._times = [e.time for e in self.events]
        mine = [e for e in events if e.currency in currencies]
        ny = lambda t: t.astimezone(NY)  # noqa: E731
        self.fomc_days = {ny(e.time).date() for e in mine if e.currency == "USD" and e.impact == "high" and FOMC.search(e.title)}
        self.nfp_days = {ny(e.time).date() for e in mine if e.currency == "USD" and NFP.search(e.title)}
        self.holidays = {ny(e.time).date(): e for e in mine if e.impact == "holiday"}

    def no_trade(self, ts: datetime) -> str | None:
        """Why the whole session is off-limits at ts (NF-03/04/05), or None."""
        c = self.config
        t = ts.astimezone(NY)
        d, after_noon = t.date(), t.time() >= time(12, 0)
        if c.holiday_blackout:
            if d in self.holidays:
                h = self.holidays[d]
                return f"NF-05 {h.currency} bank holiday ({h.title}): no trading today"
            if t.weekday() == 4 and after_noon and (d + timedelta(days=3)) in self.holidays:
                return "NF-05 Friday afternoon before a holiday Monday: no trading"
        if c.fomc_blackout and d in self.fomc_days and after_noon:
            return "NF-03 FOMC day: no afternoon trading"
        if c.nfp_week_cutoff:
            for nfp in self.nfp_days:
                monday = nfp - timedelta(days=nfp.weekday())
                if monday <= d <= nfp and (t.weekday() >= 3 or (t.weekday() == 2 and after_noon)):
                    return f"NF-04 NFP week ({nfp:%a %d %b}): trading stops at Wednesday 12:00 NY"
        return None

    def _first_event_after(self, ts: datetime) -> int:
        return bisect_left(self._times, ts)

    def blocking_event(self, ts: datetime) -> NewsEvent | None:
        """The event whose blackout window covers ts, if any (rule NF-1)."""
        if not self.config.enabled:
            return None
        before, after = timedelta(minutes=self.config.minutes_before), timedelta(minutes=self.config.minutes_after)
        i = self._first_event_after(ts - after)
        if i < len(self.events) and self.events[i].time <= ts + before:
            return self.events[i]
        return None

    def should_flatten(self, ts: datetime) -> NewsEvent | None:
        """An upcoming event close enough that open trades must close (NF-2)."""
        if not self.config.enabled or self.config.flatten_minutes_before <= 0:
            return None
        i = self._first_event_after(ts)
        if i < len(self.events) and self.events[i].time - ts <= timedelta(minutes=self.config.flatten_minutes_before):
            return self.events[i]
        return None


# ---------------------------------------------------------------- live calendar
# ForexFactory's free weekly calendar feed (no key). Each item: title, country (currency
# code), date (ISO with offset), impact (High / Medium / Low / Holiday).
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_cache: dict = {"at": 0.0, "events": None}


def parse_ff(items: list[dict]) -> list[NewsEvent]:
    out = []
    for it in items:
        try:
            ts = parse_utc(str(it["date"]))
        except (KeyError, ValueError):
            continue
        out.append(NewsEvent(ts, str(it.get("country", "")).upper(), str(it.get("impact", "")).lower(), str(it.get("title", ""))))
    return sorted(out, key=lambda e: e.time)


def fetch_calendar(timeout: float = 8, max_age_s: float = 1800) -> list[NewsEvent] | None:
    """This week's events, cached for 30 minutes. None if the feed can't be reached."""
    import json
    import time as _time
    import urllib.request

    if _cache["events"] is not None and _time.time() - _cache["at"] < max_age_s:
        return _cache["events"]
    try:
        req = urllib.request.Request(FF_URL, headers={"User-Agent": "ict-flux/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            events = parse_ff(json.loads(r.read()))
    except Exception:
        return _cache["events"]  # keep the last good copy if there is one
    _cache.update(at=_time.time(), events=events)
    return events


# ---------------------------------------------------------------- the calendar in the app
def _event_json(e: NewsEvent, now: datetime) -> dict:
    t = e.time.astimezone(NY)
    return {"id": f"{e.time:%Y%m%dT%H%MZ}-{e.currency}-{zlib.crc32(e.title.encode()) % 10000:04d}", "time": e.time.isoformat(),
            "time_ny": f"{t:%a %d %b %H:%M}", "currency": e.currency, "impact": e.impact, "title": e.title,
            "minutes": int((e.time - now).total_seconds() // 60)}


def digest(events: list[NewsEvent] | None, markets: list[dict], now: datetime, config: NewsConfig | None = None,
           days: int = 7, keep: tuple[str, ...] = ("high", "medium", "holiday")) -> dict:
    """What the app shows of the economic calendar (NF-01..05): the coming events that move markets, which of our
    markets each currency touches, the blackouts running right now and the day-level rules (FOMC, NFP week, holidays).

    `markets` are the scanned markets (dicts with key, label, cls). `events` is None when the feed could not be reached."""
    cfg = config or NewsConfig()
    if events is None:
        return {"ok": False, "source": "ForexFactory", "events": [], "blackout": [], "day_notes": [], "next_high": None,
                "error": "The calendar feed could not be reached just now. Trade as normal but check a calendar before the open."}
    by_ccy: dict[str, list[str]] = {}
    for m in markets:
        for c in market_currencies(m.get("key", ""), m.get("cls", ""), m.get("region", "us")):
            by_ccy.setdefault(c, []).append(m.get("label", m.get("key", "")))
    lo, hi = now - timedelta(hours=2), now + timedelta(days=days)
    rows = [_event_json(e, now) for e in events if e.impact in keep and lo <= e.time <= hi]
    before, after = timedelta(minutes=cfg.minutes_before), timedelta(minutes=cfg.minutes_after)
    blackout = []
    for e in events:
        if e.impact == "high" and e.time - before <= now <= e.time + after and e.currency in by_ccy:
            j = _event_json(e, now)
            j.update(until=(e.time + after).isoformat(), markets=by_ccy[e.currency])
            blackout.append(j)
    notes = []
    for ccy in sorted({e.currency for e in events if e.currency in by_ccy}):
        try:
            nf = NewsFilter(cfg, InstrumentConfig(symbol=ccy + "XXX", news_currencies=(ccy,)), events)
        except Exception:
            continue
        why = nf.no_trade(now)
        if why:
            notes.append({"currency": ccy, "note": why, "markets": by_ccy.get(ccy, [])})
    nxt = next((r for r in rows if r["impact"] == "high" and r["minutes"] >= 0 and r["currency"] in by_ccy), None)
    return {"ok": True, "source": "ForexFactory", "window": {"before": cfg.minutes_before, "after": cfg.minutes_after},
            "events": rows, "blackout": blackout, "day_notes": notes, "next_high": nxt, "markets_by_currency": by_ccy}
