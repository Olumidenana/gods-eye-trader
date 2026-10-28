"""Market data feed.

Primary: Deriv public WebSocket (wss://api.derivws.com/trading/v1/options/ws/public) for every market Deriv
lists: forex, metals (XAU, XAG, XPT, XPD) and the OTC indices. One connection serves every symbol.

Deriv does not list crude oil, natural gas or single stocks, so those come from Yahoo Finance and are labelled
as such everywhere. Nothing is ever labelled "Deriv" unless it came from Deriv.

Deriv keeps about 27 days of 5-minute candles. Every fetch is merged into a disk cache (.cache/bars), so the
research lab's history grows deeper every day the app runs. Later fetches only ask for the candles since the
last cached one.
"""

from __future__ import annotations

import json
import threading
import time as _time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .universe import BY_KEY, MarketSpec

NY = ZoneInfo("America/New_York")
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "bars"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DERIV_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"
DERIV_MAX_PAGES = 8          # 8 x 5000 requested; Deriv returns its own retention (~27 days on 5m)
LIVE_REFRESH_S = 20.0        # a market is refetched at most this often

_RAM: dict[tuple[str, int], tuple[float, list["Bar"], str]] = {}
_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low


def to_ny(dt: datetime) -> datetime:
    return dt.astimezone(NY)


# ---------------------------------------------------------------- disk cache
def _path(key: str, tf: int) -> Path:
    return CACHE_DIR / f"{key}_{tf}m.json"


def _load(key: str, tf: int) -> list[Bar]:
    p = _path(key, tf)
    if not p.exists():
        return []
    try:
        return [Bar(datetime.fromtimestamp(r[0], tz=timezone.utc), r[1], r[2], r[3], r[4], r[5] if len(r) > 5 else 0.0)
                for r in json.loads(p.read_text("utf-8"))]
    except Exception:
        return []


def _save(key: str, tf: int, bars: list[Bar]) -> None:
    try:
        _path(key, tf).write_text(json.dumps([[int(b.time.timestamp()), b.open, b.high, b.low, b.close, b.volume]
                                              for b in bars[-60000:]], separators=(",", ":")), "utf-8")
    except Exception:
        pass


def _merge(a: list[Bar], b: list[Bar]) -> list[Bar]:
    d = {x.time: x for x in a}
    d.update({x.time: x for x in b})
    return sorted(d.values(), key=lambda x: x.time)


def _floor(epoch: int, tf: int) -> datetime:
    s = tf * 60
    return datetime.fromtimestamp((int(epoch) // s) * s, tz=timezone.utc)


# ---------------------------------------------------------------- Deriv
def _deriv_fetch(symbol: str, tf: int, since: datetime | None, days: int) -> list[Bar]:
    """Candles for one Deriv symbol. With `since`, only what came after it."""
    import websocket

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Origin": "https://app.deriv.com",
    }
    ws = websocket.create_connection(DERIV_WS_URL, header=headers, timeout=15)
    try:
        out: list[Bar] = []
        cutoff = since or (datetime.now(timezone.utc) - timedelta(days=days))
        end: Any = "latest"
        for page in range(DERIV_MAX_PAGES):
            rid = page + 1
            ws.send(json.dumps({"ticks_history": symbol, "style": "candles", "granularity": tf * 60,
                                "count": 5000, "end": end, "req_id": rid}))
            deadline = _time.monotonic() + 10
            msg = None
            while _time.monotonic() < deadline:
                m = json.loads(ws.recv())
                if m.get("req_id") == rid:
                    msg = m
                    break
            if msg is None or msg.get("error"):
                if not out:
                    raise RuntimeError(f"Deriv {symbol}: {(msg or {}).get('error', {}).get('message', 'no reply')}")
                break
            candles = msg.get("candles") or []
            if not candles:
                break
            page_bars = [Bar(_floor(c["epoch"], tf), float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"]))
                         for c in candles]
            out = page_bars + out
            if page_bars[0].time <= cutoff or len(candles) < 50:
                break
            end = int(candles[0]["epoch"]) - 1
        return _merge([], out)
    finally:
        try:
            ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------- Yahoo (oil and stocks only)
def _yahoo_fetch(ticker: str, tf: int, days: int) -> list[Bar]:
    interval, cap = {1: ("1m", 7), 5: ("5m", 58), 15: ("15m", 58), 60: ("60m", 700)}.get(tf, ("5m", 58))
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
           f"?interval={interval}&range={min(days, cap)}d&includePrePost=false")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (gods-eye-flux)"})
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.load(r)
    res = ((data.get("chart") or {}).get("result") or [None])[0]
    if not res:
        raise LookupError(f"no data for {ticker}")
    q = res["indicators"]["quote"][0]
    out = []
    for i, ts in enumerate(res.get("timestamp") or []):
        o, h, lo, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, lo, c):
            continue
        out.append(Bar(_floor(ts, tf), float(o), float(h), float(lo), float(c), float((q.get("volume") or [0])[i] or 0)))
    return _merge([], out)


# ---------------------------------------------------------------- public
def source_of(spec: MarketSpec) -> str:
    return "deriv" if spec.deriv_symbol else "yahoo"


def fetch_bars(key: str, tf_minutes: int = 5, days: int = 30, force: bool = False) -> tuple[list[Bar], str]:
    spec = BY_KEY.get(key.upper())
    if spec is None:
        raise LookupError(f"unknown market {key}")
    ck = (spec.key, tf_minutes)
    now = _time.monotonic()
    with _LOCK:
        hit = _RAM.get(ck)
    if hit and not force and now - hit[0] < LIVE_REFRESH_S:
        bars, label = hit[1], hit[2]
    else:
        cached = _load(spec.key, tf_minutes)
        try:
            if spec.deriv_symbol:
                since = cached[-1].time - timedelta(minutes=tf_minutes * 3) if len(cached) > 200 else None
                new = _deriv_fetch(spec.deriv_symbol, tf_minutes, since, days)
                label = f"Deriv {spec.deriv_symbol} (real-time)"
            else:
                new = _yahoo_fetch(spec.yahoo_symbol, tf_minutes, max(days, 58))
                label = f"Yahoo {spec.yahoo_symbol} (not on Deriv)"
            bars = _merge(cached, new)
            _save(spec.key, tf_minutes, bars)
        except Exception as e:
            if not cached:
                raise RuntimeError(f"{spec.key}: feed unavailable ({e})") from e
            bars, label = cached, f"cached copy ({e.__class__.__name__} on refresh)"
        with _LOCK:
            _RAM[ck] = (now, bars, label)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    sliced = [b for b in bars if b.time >= cutoff]
    return (sliced if len(sliced) >= 100 else bars), label


def prefetch(keys, tf_minutes: int = 5, days: int = 12) -> dict[str, tuple[list[Bar], str]]:
    out: dict[str, tuple[list[Bar], str]] = {}

    def one(k):
        try:
            return k, fetch_bars(k, tf_minutes, days)
        except Exception:
            return k, None

    with ThreadPoolExecutor(max_workers=min(12, max(1, len(keys)))) as pool:
        for k, r in pool.map(one, list(keys)):
            if r and len(r[0]) >= 60:
                out[k] = r
    return out


prefetch_universe = prefetch  # old name
