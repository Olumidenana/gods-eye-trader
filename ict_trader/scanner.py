"""Market scanner: runs the strategy on fresh 5-minute data across many markets.

Used by the web app's serverless API (api/scan.py, api/history.py). For each
market it downloads recent 5-minute bars, replays them through the exact
strategy and backtester used everywhere else, and reports:

- live mode: today's signals with their outcome so far and the decision log;
- history mode: a paper track record over the last month.

Data: with OANDA_API_TOKEN set, real-time OANDA candles for the whole universe
below (forex majors, crosses and exotics, metals, energy, grains, global
indices, crypto). Without it, delayed Yahoo Finance data for the core markets.

Markets with a hand-tuned config in examples/ use it. Every other market gets
its thresholds (sweep size, stop buffer, min/max stop) scaled to its own
recent 5-minute range, so one rule set works from EUR/USD to the Nikkei.
High-impact news (ForexFactory's weekly calendar) blocks entries 30 minutes either
side for each market's currencies (NF-01). Treat results as a paper-trading record,
not as what a broker account would have done.
"""

from __future__ import annotations

import json
import math
import time as _time
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

from . import advisor, aplus, autolearn, management, tiers
from .backtest import Backtester
from .config import Config, InstrumentConfig, RiskConfig, StrategyConfig, Window, load_config
from .sessions import Bar, LevelTracker, to_ny
from .analysis import analyze
from .news import NewsFilter, digest as news_digest, fetch_calendar, market_currencies
from .models import CORE_MODELS, LIVE_MODELS, MODEL_INFO, kind_of, make_strategy, model_available
from .strategy import LONG, SHORT, Signal

log = logging.getLogger("ict_trader.scanner")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

CLASS_LABEL = {"forex": "Forex", "metal": "Metals", "energy": "Energy", "agri": "Agriculture",
               "index": "Indices", "crypto": "Crypto", "synthetic": "Synthetics (Volatility, Jump, Crash/Boom…)", "stock": "Stocks"}


@dataclass(frozen=True)
class Market:
    key: str
    label: str
    feed: str  # Yahoo Finance ticker ("" if none)
    feed_label: str
    config_file: str  # hand-tuned config in examples/, or "" to auto-calibrate
    decimals: int  # 0 = derive from price
    oanda: str = ""  # OANDA v20 instrument
    cls: str = "forex"
    region: str = "us"  # for indices: us / eu / asia (sets the killzones)
    core: bool = False  # scanned on the free Yahoo fallback too


def _fx(pair: str, core_file: str = "", core: bool = False) -> Market:
    a, b = pair.split("_")
    return Market(f"{a}{b}", f"{a}/{b}", f"{a}{b}=X", "Yahoo FX quote", core_file, 3 if b == "JPY" else 5,
                  pair, "forex", core=core)


_FX_PAIRS = ["USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
             "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
             "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
             "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
             "NZD_JPY", "NZD_CHF", "NZD_CAD",
             "USD_ZAR", "USD_MXN", "USD_SGD", "USD_NOK", "USD_SEK", "USD_PLN", "EUR_NOK", "EUR_SEK", "EUR_PLN"]

MARKETS: tuple[Market, ...] = (
    Market("XAUUSD", "Gold", "GC=F", "COMEX gold futures (GC, ~10 min delay)", "xauusd.toml", 2, "XAU_USD", "metal", core=True),
    _fx("GBP_USD", "gbpusd.toml", True),
    _fx("EUR_USD", "config.example.toml", True),
    Market("US500", "S&P 500", "ES=F", "CME E-mini S&P (ES, ~10 min delay)", "us500.toml", 2, "SPX500_USD", "index", core=True),
    Market("NAS100", "Nasdaq 100", "NQ=F", "CME E-mini Nasdaq (NQ, ~10 min delay)", "nas100.toml", 2, "NAS100_USD", "index", core=True),
    *(_fx(p) for p in _FX_PAIRS),
    Market("XAGUSD", "Silver", "SI=F", "COMEX silver", "", 0, "XAG_USD", "metal"),
    Market("XPTUSD", "Platinum", "PL=F", "NYMEX platinum", "", 0, "XPT_USD", "metal"),
    Market("XPDUSD", "Palladium", "PA=F", "NYMEX palladium", "", 0, "XPD_USD", "metal"),
    Market("XCUUSD", "Copper", "HG=F", "COMEX copper", "", 0, "XCU_USD", "metal"),
    Market("BRENT", "Brent crude", "BZ=F", "ICE Brent", "", 0, "BCO_USD", "energy"),
    Market("WTI", "WTI crude", "CL=F", "NYMEX WTI", "", 0, "WTICO_USD", "energy"),
    Market("NATGAS", "Natural gas", "NG=F", "NYMEX natural gas", "", 0, "NATGAS_USD", "energy"),
    Market("CORN", "Corn", "ZC=F", "CBOT corn", "", 0, "CORN_USD", "agri"),
    Market("WHEAT", "Wheat", "ZW=F", "CBOT wheat", "", 0, "WHEAT_USD", "agri"),
    Market("SOYBEAN", "Soybeans", "ZS=F", "CBOT soybeans", "", 0, "SOYBN_USD", "agri"),
    Market("SUGAR", "Sugar", "SB=F", "ICE sugar", "", 0, "SUGAR_USD", "agri"),
    Market("US30", "Dow Jones 30", "YM=F", "CBOT Dow", "", 0, "US30_USD", "index"),
    Market("US2000", "Russell 2000", "RTY=F", "CME Russell", "", 0, "US2000_USD", "index"),
    Market("UK100", "FTSE 100", "^FTSE", "FTSE index", "", 0, "UK100_GBP", "index", "eu"),
    Market("DE40", "DAX 40", "^GDAXI", "DAX index", "", 0, "DE30_EUR", "index", "eu"),
    Market("EU50", "Euro Stoxx 50", "^STOXX50E", "Stoxx index", "", 0, "EU50_EUR", "index", "eu"),
    Market("FR40", "CAC 40", "^FCHI", "CAC index", "", 0, "FR40_EUR", "index", "eu"),
    Market("JP225", "Nikkei 225", "NIY=F", "CME Nikkei", "", 0, "JP225_USD", "index", "asia"),
    Market("AU200", "ASX 200", "^AXJO", "ASX index", "", 0, "AU200_AUD", "index", "asia"),
    Market("HK33", "Hang Seng", "^HSI", "HSI index", "", 0, "HK33_HKD", "index", "asia"),
    Market("CN50", "China A50", "", "", "", 0, "CN50_USD", "index", "asia"),
    Market("BTCUSD", "Bitcoin", "BTC-USD", "Yahoo crypto", "", 0, "BTC_USD", "crypto"),
    Market("ETHUSD", "Ether", "ETH-USD", "Yahoo crypto", "", 0, "ETH_USD", "crypto"),
    # Deriv Volatility indices: computer-generated, 24/7, no sessions. Price model only (no
    # killzones); keys are Deriv's own symbols. Live on Deriv only.
    *(Market(sym, name, "", "Deriv synthetic", "", 0, "", "synthetic") for sym, name in (
        ("R_10", "Volatility 10"), ("R_25", "Volatility 25"), ("R_50", "Volatility 50"),
        ("R_75", "Volatility 75"), ("R_100", "Volatility 100"),
        ("1HZ10V", "Volatility 10 (1s)"), ("1HZ25V", "Volatility 25 (1s)"), ("1HZ50V", "Volatility 50 (1s)"),
        ("1HZ75V", "Volatility 75 (1s)"), ("1HZ100V", "Volatility 100 (1s)"))),
)

Fetcher = Callable[[str, str], Any]  # (ticker, range) -> list[Bar] or (list[Bar], spread)

# Execution timeframes the engine can run on, for every market.
TIMEFRAMES = (1, 2, 3, 5, 15, 60, 240)

# Trader styles: which timeframes they use and how the rules adapt.
#   scalp / intraday: killzones, prior-day and Asia levels, flat by 16:00 NY.
#   swing:    1-hour candles, any hour, prior-day levels, held until stop or target.
#   position: 4-hour candles, any hour, prior-WEEK levels (RL-04), held for weeks.
STYLES = {
    "scalp": {"label": "Scalper", "timeframes": (1, 2, 3), "default": 1, "holds": "minutes"},
    "intraday": {"label": "Intraday", "timeframes": (5, 15), "default": 5, "holds": "hours, flat by 16:00 NY"},
    "swing": {"label": "Swing", "timeframes": (60,), "default": 60, "holds": "days"},
    "position": {"label": "Long-term", "timeframes": (240,), "default": 240, "holds": "weeks"},
}


def style_of(tf: int) -> str:
    return next((k for k, v in STYLES.items() if tf in v["timeframes"]), "intraday")


def signal_window_days(tf: int) -> int:
    """How far back 'current' signals go: today for intraday styles, longer for swing/position."""
    return 1 if tf <= 15 else (7 if tf <= 60 else 30)


def apply_style(cfg: Config, tf: int) -> Config:
    s = replace(cfg.strategy, timeframe_minutes=tf)
    if tf >= 60:
        s = replace(s, killzones=(Window("any_hour", time(0, 0), time(23, 59)),), reference_ranges=(),
                    flatten_time=None, level_period="week" if tf >= 240 else "day",
                    max_signals_per_day=2 if tf >= 240 else 1)
    return replace(cfg, strategy=s)
LIVE_BARS = 1000  # 5-minute default; see live_bars()
HISTORY_DAYS = 30


def live_bars(tf: int = 5) -> int:
    """Enough candles for the prior-week, 7-day track record, and Asia levels (about 7.5 days), within one request."""
    return min(5000, max(400, math.ceil(7.5 * 1440 / tf)))


def history_days(tf: int = 5) -> int:
    """Track-record window: shorter on fast timeframes so ~60 markets fit in one function run."""
    return {1: 7, 2: 10, 3: 14, 5: 30, 15: 60, 60: 180, 240: 540}.get(tf, 30)


def oanda_enabled() -> bool:
    return bool(os.environ.get("OANDA_API_TOKEN"))


def allow_delayed() -> bool:
    """Delayed data is off unless ALLOW_DELAYED=1: the site shows live prices only."""
    return os.environ.get("ALLOW_DELAYED", "") == "1"


def data_source() -> str:
    """'oanda' if a token is set, else 'deriv' (free, real-time, no account); DATA_SOURCE overrides."""
    forced = os.environ.get("DATA_SOURCE", "").lower()
    if forced in ("oanda", "deriv", "yahoo"):
        return forced
    return "oanda" if oanda_enabled() else "deriv"


def universe() -> tuple[Market, ...]:
    """Markets to scan for the active data source."""
    src = data_source()
    if src == "oanda":
        return tuple(m for m in MARKETS if m.oanda)
    if src == "deriv":
        return MARKETS
    return tuple(m for m in MARKETS if m.core)


# ---------------------------------------------------------------- Deriv feed
# Deriv's public WebSocket serves real-time market data with no account, app id or
# token (https://developers.deriv.com/docs/options/ws-public/). The legacy endpoint,
# which needs a numeric app id (1089 is Deriv's shared test id), is kept as a fallback.

DERIV_URLS = (
    "wss://api.derivws.com/trading/v1/options/ws/public",
    "wss://ws.derivws.com/websockets/v3?app_id={legacy_app_id}",
)
DERIV_NAMES = {  # display names on Deriv for markets without a frx/cry symbol
    "US500": ["US 500"], "NAS100": ["US Tech 100"], "US30": ["Wall Street 30"],
    "UK100": ["UK 100"], "DE40": ["Germany 40"], "EU50": ["Euro 50"], "FR40": ["France 40"],
    "JP225": ["Japan 225"], "AU200": ["Australia 200"], "HK33": ["Hong Kong 50"],
    "BRENT": ["Oil/USD", "Brent Crude Oil"], "WTI": ["US Oil", "WTI Crude Oil"],
}


def deriv_resolve(active: list[dict[str, Any]], markets: tuple[Market, ...]) -> dict[str, str]:
    """Map our market keys to Deriv symbols using the live active_symbols list."""
    # New API: underlying_symbol / underlying_symbol_name; legacy: symbol / display_name.
    def sym(a: dict[str, Any]) -> str:
        return a.get("underlying_symbol") or a.get("symbol") or ""

    by_symbol = {sym(a): a for a in active}
    by_name = {(a.get("underlying_symbol_name") or a.get("display_name") or "").lower(): sym(a) for a in active}
    out: dict[str, str] = {}
    for m in markets:
        for sym in (m.key if m.cls == "synthetic" else "", f"frx{m.key}", f"cry{m.key}"):
            if sym in by_symbol:
                out[m.key] = sym
                break
        else:
            for name in DERIV_NAMES.get(m.key, []):
                if name.lower() in by_name:
                    out[m.key] = by_name[name.lower()]
                    break
    return out


def _deriv_candles(c: dict[str, Any]) -> Bar:
    t = datetime.fromtimestamp(int(c["epoch"]), tz=timezone.utc)
    return Bar(t, float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"]))


DISCOVER_MAX = 90  # extra Deriv markets added per scan, synthetics first


def deriv_class(a: dict[str, Any]) -> tuple[str, int] | None:
    """Our class for a row of Deriv's active_symbols, and how early it is listed (synthetics first);
    None for a kind of market the engine does not read."""
    mkt = str(a.get("market") or a.get("market_type") or "").lower()
    sub = str(a.get("submarket") or "").lower()
    if "synthetic" in mkt or "derived" in mkt:
        return "synthetic", 0
    if mkt == "forex":
        return "forex", 1
    if mkt == "indices":
        return "index", 2
    if mkt == "commodities":
        return ("metal" if "metal" in sub else "energy" if "energy" in sub else "agri"), 2
    if "crypto" in mkt:
        return "crypto", 2
    if "stock" in mkt or "equit" in mkt:
        return "stock", 3
    return None


def deriv_region(name: str) -> str:
    """Which session set an index follows, from its display name."""
    low = name.lower()
    return "eu" if any(w in low for w in ("germany", "france", "euro", "uk ", "netherlands", "spain", "swiss")) else \
           "asia" if any(w in low for w in ("japan", "australia", "hong kong", "china")) else "us"


def discover_markets(active: list[dict[str, Any]], known: tuple[Market, ...], taken: set[str]) -> list[tuple[Market, str]]:
    """Every market Deriv lists that the fixed list doesn't cover: all synthetic indices
    (Volatility, Jump, Crash/Boom, Step, Range Break, DEX, Drift...) and any forex,
    index, commodity, crypto or stock symbols. Returns (market, deriv symbol)."""
    keys = {m.key for m in known}
    norm = lambda x: "".join(ch for ch in str(x).upper() if ch.isalnum())  # noqa: E731
    names = {norm(m.label) for m in known} | {norm(m.key) for m in known}
    out: list[tuple[int, Market, str]] = []
    for a in active:
        sym = a.get("underlying_symbol") or a.get("symbol") or ""
        if not sym or sym in taken or a.get("is_trading_suspended"):
            continue
        name = a.get("underlying_symbol_name") or a.get("display_name") or sym
        kind = deriv_class(a)
        if kind is None:
            continue
        cls, rank = kind
        key = sym[3:] if sym[:3] in ("frx", "cry") else sym
        if key in keys or norm(name) in names or norm(key) in names:
            continue  # already listed under another spelling (GBPUSD vs GBP/USD)
        out.append((rank, Market(key, name, "", "Deriv", "", 0, "", cls, deriv_region(name)), sym))
        keys.add(key)
        names.add(norm(name))
    out.sort(key=lambda x: x[0])
    return [(m, sym) for _, m, sym in out[:DISCOVER_MAX]]


def fetch_deriv_bulk(markets: tuple[Market, ...], mode: str, connect: Callable[[str], Any] | None = None,
                     deadline_s: float = 40, tf: int = 5, days: int | None = None, discover: bool = False) -> dict[str, Any]:
    """Candles for many markets over one WebSocket connection.

    Returns {market key: (bars, symbol)} or {market key: Exception}. Live mode
    takes the latest LIVE_BARS; history pages back to HISTORY_DAYS.
    """
    import time as _time

    if connect is None:
        import websocket  # websocket-client, see requirements.txt

        def connect(url: str) -> Any:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Origin": "https://app.deriv.com",
            }
            return websocket.create_connection(url, header=headers, timeout=15)

    legacy_id = os.environ.get("DERIV_LEGACY_APP_ID", "1089")
    ws, errors = None, []
    for url in DERIV_URLS:
        try:
            ws = connect(url.format(legacy_app_id=legacy_id))
            break
        except Exception as e:  # try the next endpoint
            errors.append(f"{url.split('?')[0]}: {e}")
    if ws is None:
        raise ConnectionError("; ".join(errors))
    try:
        def call_many(reqs: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
            items = list(reqs.items())
            for idx in range(0, len(items), 8):
                for rid, body in items[idx:idx + 8]:
                    ws.send(json.dumps({**body, "req_id": rid}))
                if idx + 8 < len(items):
                    _time.sleep(0.04)
            got: dict[int, dict[str, Any]] = {}
            end = _time.monotonic() + deadline_s
            while len(got) < len(reqs) and _time.monotonic() < end:
                try:
                    msg = json.loads(ws.recv())
                except Exception:
                    break
                rid = msg.get("req_id")
                if rid in reqs:
                    got[rid] = msg
            return got

        active = call_many({1: {"active_symbols": "brief"}}).get(1, {})
        if "error" in active:
            raise RuntimeError(f"Deriv: {active['error'].get('message')}")
        symbols = deriv_resolve(active.get("active_symbols") or [], markets)
        extra: list[Market] = []
        if discover:
            for m, sym in discover_markets(active.get("active_symbols") or [], markets, set(symbols.values())):
                extra.append(m)
                symbols[m.key] = sym
            markets = markets + tuple(extra)
        result: dict[str, Any] = {m.key: LookupError("not offered on Deriv") for m in markets if m.key not in symbols}
        keys = [m.key for m in markets if m.key in symbols]
        bars: dict[str, list[Bar]] = {k: [] for k in keys}
        ends: dict[str, Any] = {k: "latest" for k in keys}
        days = days or history_days(tf)
        pages = 1 if mode == "live" else math.ceil(days * 1440 / tf / 5000) + 1
        count = live_bars(tf) if mode == "live" else 5000
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        for _ in range(pages):
            todo = [k for k in keys if k not in result]
            if not todo:
                break
            reqs = {100 + i: {"ticks_history": symbols[k], "style": "candles", "granularity": tf * 60,
                              "count": count, "end": ends[k]} for i, k in enumerate(todo)}
            got = call_many(reqs)
            for i, k in enumerate(todo):
                msg = got.get(100 + i)
                if msg is None:
                    result[k] = TimeoutError("no reply from Deriv")
                    continue
                if "error" in msg:
                    err = msg["error"]
                    err_msg = str(err.get("message", ""))
                    err_code = str(err.get("code", ""))
                    if "rate" in err_code.lower() or "too many" in err_msg.lower() or "rate" in err_msg.lower():
                        _time.sleep(1.0)
                        try:
                            ws.send(json.dumps({**reqs[100 + i], "req_id": 900 + i}))
                            retry_end = _time.monotonic() + 5
                            while _time.monotonic() < retry_end:
                                r_msg = json.loads(ws.recv())
                                if r_msg.get("req_id") == 900 + i:
                                    msg = r_msg
                                    break
                        except Exception:
                            pass
                    if "error" in msg:
                        err = msg["error"]
                        result[k] = (LookupError if err.get("code") in ("InvalidSymbol", "InputValidationFailed") else RuntimeError)(
                            f"Deriv: {err.get('message')}")
                        continue
                page = [_deriv_candles(c) for c in msg.get("candles") or []]
                bars[k] = page + bars[k]
                if page:
                    ends[k] = int(page[0].time.timestamp()) - 1
                if mode == "history" and page and page[0].time <= cutoff:
                    result[k] = None  # enough history; mark done
        for k in keys:
            if not isinstance(result.get(k), Exception):
                b = sorted({x.time: x for x in bars[k]}.values(), key=lambda x: x.time)
                if mode == "history":
                    b = [x for x in b if x.time >= cutoff]
                result[k] = (b, symbols[k])
        result["__extra__"] = extra
        if mode in ("live", "history"):
            # Daily candles (about 70) for the higher-timeframe rules: the weekly draw and the 20/40/60-day ranges
            # need more history than the few days of intraday candles the scan loads. Optional: no reply, no seed.
            try:
                ok_keys = [k for k in keys if not isinstance(result.get(k), Exception)]
                dgot = call_many({5000 + i: {"ticks_history": symbols[k], "style": "candles", "granularity": 86400,
                                             "count": 70, "end": "latest"} for i, k in enumerate(ok_keys)})
                result["__daily__"] = {k: [_deriv_candles(c) for c in (dgot.get(5000 + i) or {}).get("candles") or []]
                                       for i, k in enumerate(ok_keys)}
            except Exception as e:
                log.warning("daily candles unavailable: %s", e)
        return result
    finally:
        try:
            ws.close()
        except Exception:
            pass


def fetch_yahoo(ticker: str, range_: str = "60d", interval: str = "5m") -> list[Bar]:
    """5-minute bars from Yahoo's public chart API, oldest first, UTC bar-open times."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
        f"?interval={interval}&range={range_}&includePrePost=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ict-flux scanner)"})
    with urllib.request.urlopen(req, timeout=12) as resp:
        data = json.load(resp)
    result = (data.get("chart") or {}).get("result") or []
    if not result:
        err = (data.get("chart") or {}).get("error") or "empty response"
        raise RuntimeError(f"Yahoo returned no data for {ticker}: {err}")
    r = result[0]
    q = r["indicators"]["quote"][0]
    bars: list[Bar] = []
    for i, ts in enumerate(r.get("timestamp") or []):
        o, h, lo, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, lo, c):
            continue
        t = datetime.fromtimestamp(ts, tz=timezone.utc)
        if t.second:
            continue  # the live, still-forming bar sometimes arrives off-grid
        bars.append(Bar(t, float(o), float(h), float(lo), float(c), float((q.get("volume") or [0])[i] or 0)))
    bars.sort(key=lambda b: b.time)
    dedup: list[Bar] = []
    for b in bars:
        if dedup and dedup[-1].time == b.time:
            dedup[-1] = b
        else:
            dedup.append(b)
    return dedup


def _oanda_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    token = os.environ["OANDA_API_TOKEN"]
    host = "api-fxtrade.oanda.com" if os.environ.get("OANDA_ENV") == "live" else "api-fxpractice.oanda.com"
    req = urllib.request.Request(f"https://{host}{path}?{urllib.parse.urlencode(params)}",
                                 headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:160]
        if e.code in (400, 404):
            raise LookupError(f"not available on this OANDA account ({detail})") from e
        if e.code == 401:
            raise PermissionError("OANDA rejected the API token (check OANDA_API_TOKEN / OANDA_ENV)") from e
        raise


def fetch_oanda(instrument: str, days: int | None = None, count: int = LIVE_BARS,
                granularity: str = "M5") -> tuple[list[Bar], float | None]:
    """Real-time 5-minute candles from OANDA's v20 REST API.

    Returns mid-price bars plus the median bid/ask spread. With `days`, pages
    forward from that many days ago; otherwise fetches the latest `count` bars.
    """
    bars: list[Bar] = []
    spreads: list[float] = []

    def take(candles: list[dict[str, Any]]) -> None:
        for c in candles:
            if not c.get("complete"):
                continue  # the forming candle; the strategy only sees closed bars
            t = datetime.fromisoformat(c["time"][:19]).replace(tzinfo=timezone.utc)
            m = c["mid"]
            bars.append(Bar(t, float(m["o"]), float(m["h"]), float(m["l"]), float(m["c"]), float(c.get("volume", 0))))
            if "ask" in c and "bid" in c:
                spreads.append(float(c["ask"]["c"]) - float(c["bid"]["c"]))

    path = f"/v3/instruments/{instrument}/candles"
    if days is None:
        take(_oanda_get(path, {"granularity": granularity, "price": "MBA", "count": count}).get("candles") or [])
    else:
        cursor = datetime.now(timezone.utc) - timedelta(days=days)
        for _ in range(4):
            candles = _oanda_get(path, {"granularity": granularity, "price": "MBA", "count": 5000,
                                        "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ")}).get("candles") or []
            take(candles)
            if len(candles) < 5000 or not bars:
                break
            cursor = bars[-1].time + timedelta(minutes=5)
    return bars, (median(spreads[-300:]) if spreads else None)


def _closed_bars(bars: list[Bar], now: datetime, tf_minutes: int) -> list[Bar]:
    """Drop the bar that is still forming: the strategy only ever sees closed bars."""
    tf = timedelta(minutes=tf_minutes)
    return [b for b in bars if b.time + tf <= now]


def _decimals(market: Market, price: float) -> int:
    if market.decimals:
        return market.decimals
    if price >= 1000:
        return 1
    if price >= 20:
        return 2
    if price >= 2:
        return 3
    return 4


def _killzones(market: Market) -> tuple[tuple[Window, ...], tuple[Window, ...]]:
    """Execution windows and the session ranges whose highs/lows are liquidity pools.

    Besides PDH/PDL, ICT marks the session highs and lows formed before each killzone as
    the liquidity it runs: the Asian range for London (RL-02), the London range for New
    York (Ep 21, Ep 41: "ICT London Open Killzone" low as the target), and for indices
    the pre-market range before the 08:30 open (Ep 41: the 07:00 high as buyside liquidity).
    """
    w = lambda n, a, b: Window(n, time(*a), time(*b))  # noqa: E731
    asia, london = w("asia", (20, 0), (0, 0)), w("london", (2, 0), (5, 0))
    if market.cls in ("synthetic", "crypto"):
        return (Window("24h", time(0, 0), time(23, 59)),), (asia, london)
    if market.cls == "forex":
        return (w("london", (2, 0), (5, 0)), w("new_york", (7, 0), (10, 0))), (asia, london)  # KZ-04, KZ-05
    if market.cls == "index" and market.region == "us":
        return ((w("ny_am", (8, 30), (11, 0)), w("ny_pm", (13, 30), (15, 0))),  # KZ-06, KZ-09
                (asia, london, w("premarket", (5, 0), (8, 30)), w("lunch", (12, 0), (13, 30))))  # SW-04
    return (w("london", (2, 0), (5, 0)), w("ny_am", (8, 30), (11, 0))), (asia, london)


def auto_config(market: Market, bars: list[Bar], spread: float | None) -> Config:
    """Rules scaled to the market's own volatility (median 5-minute range, last 3 days)."""
    dp = _decimals(market, bars[-1].close)
    pip = 10.0 ** -(dp - 1)
    rng = median([b.high - b.low for b in bars[-864:]]) or pip
    unit = rng / pip  # one typical 5-minute range, in pips
    kz, refs = _killzones(market)
    timed = market.cls != "synthetic"
    strategy = StrategyConfig(
        model="ict2022",
        killzones=kz, reference_ranges=refs,
        min_sweep_pips=round(0.15 * unit, 3), stop_buffer_pips=round(0.25 * unit, 3),
        min_stop_pips=round(1.0 * unit, 3), max_stop_pips=round(8.0 * unit, 3),
        enforce_liquidity_purge_order=timed,
        day_open_rule=timed,
        htf_context=True,
        target_pref="draw",
        mss_max_bars=8,
        trim_stop_at_50_pct=True, move_be_at_75_pct=True,
        move_be_at_1r=True, partial_at_tp1=0.5,
        regime_filter=True, max_obstacles=2, min_score=40, bias_required=True,
    )
    instrument = InstrumentConfig(
        symbol=market.key, pip_size=pip,
        spread_pips=round((spread if spread else 0.1 * rng) / pip, 3),
        contract_size=1, commission_per_lot=0.0, asset_class=market.cls,
    )
    return Config(instrument=instrument, strategy=strategy, risk=RiskConfig(max_lots=1e12))


def _fmt(x: float, dp: int) -> float:
    return round(x, dp)


def _signal_id(market: Market, sig: Signal, tf: int = 5) -> str:
    stamp = sig.time.strftime('%Y%m%dT%H%MZ')
    return f"{market.key}-{stamp}" if tf == 5 else f"{market.key}-{tf}m-{stamp}"


def _best_r(trade: Any, sig: Signal, bars: list[Bar]) -> float | None:
    """The furthest the trade went in its favour, in R, from the fill to the exit (or now)."""
    from bisect import bisect_left, bisect_right
    risk = abs(trade.entry_price - sig.stop)
    if not risk or not bars:
        return None
    i = bisect_left(bars, trade.entry_time, key=lambda b: b.time)
    end = trade.exit_time if trade.exit_time and not trade.exit_reason.startswith("END") else bars[-1].time
    j = bisect_right(bars, end, key=lambda b: b.time)
    span = bars[i:j]
    if not span:
        return None
    best = max(b.high for b in span) - trade.entry_price if sig.direction == LONG else trade.entry_price - min(b.low for b in span)
    return round(max(0.0, best / risk), 2)


def _outcome(trade: Any | None, sig: Signal, bars: list[Bar]) -> dict[str, Any]:
    if trade is None:
        return {"status": "not_taken", "label": "Not taken (risk limits or news blackout)", "r": None}
    out = _outcome_of(trade, sig, bars)
    out["best_r"] = _best_r(trade, sig, bars)
    out["tp1_hit"] = "TP1 partial taken" in trade.exit_reason  # the first leg was banked (a twin trade)
    return out


def _outcome_of(trade: Any, sig: Signal, bars: list[Bar]) -> dict[str, Any]:
    reason = trade.exit_reason
    r = round(trade.r_multiple, 2)
    if reason.startswith("END"):
        # marked at the last close, with any TP1 partial already banked counted in
        return {"status": "open", "label": "Still open · TP1 banked" if "TP1" in reason else "Still open", "r": r}
    if reason.startswith("TP"):
        return {"status": "win", "label": "Target hit", "r": r}
    if reason.startswith("SL"):
        if "TP1" in reason and r > 0.05:
            return {"status": "win", "label": "TP1 partial taken, rest stopped at breakeven", "r": r}
        if r > 0.05:
            return {"status": "win", "label": "Stopped in profit (trailed stop)", "r": r}
        if abs(r) <= 0.05:
            return {"status": "breakeven", "label": "Stopped at breakeven", "r": r}
        return {"status": "loss", "label": "Stop loss hit", "r": r}
    if reason.startswith("EX"):
        return {"status": "win" if r > 0 else "loss", "label": "Closed at 16:00 NY", "r": r}
    return {"status": "win" if r > 0 else "loss", "label": reason, "r": r}


def _setup_json(sig: Signal, dp: int) -> dict[str, Any] | None:
    """Where the sweep, shift, gap and midnight open were, for drawing the setup on a chart."""
    m = sig.meta
    if not m:
        return None
    ts = lambda t: int(t.timestamp()) if t else None  # noqa: E731
    px = lambda v: _fmt(v, dp) if v is not None else None  # noqa: E731
    return {"level_price": px(m.get("level_price")), "sweep_price": px(m.get("sweep_price")), "sweep_time": ts(m.get("sweep_time")),
            "mss_level": px(m.get("mss_level")), "mss_time": ts(m.get("mss_time")),
            "fvg": [px(v) for v in m.get("fvg") or ()], "fvg_time": ts(m.get("fvg_time")),
            "midnight_open": px(m.get("midnight_open")), "target_name": m.get("target_name"), "entry_mode": m.get("entry_mode"),
            "swing_time": ts(m.get("swing_time")), "tp1": px(m.get("tp1")),
            "levels": [{"name": lv["name"], "price": px(lv["price"]), "side": lv["side"], "spent": lv["spent"]} for lv in m.get("levels") or []]}


# SMT divergence (SMT-01/02): correlated markets that should run the same liquidity.
SMT_PAIRS = {"US500": "NAS100", "NAS100": "US500", "US30": "US500", "US2000": "US500", "EURUSD": "GBPUSD", "GBPUSD": "EURUSD",
             "XAUUSD": "XAGUSD", "XAGUSD": "XAUUSD", "AUDUSD": "NZDUSD", "NZDUSD": "AUDUSD", "BTCUSD": "ETHUSD", "ETHUSD": "BTCUSD",
             "DE40": "EU50", "EU50": "DE40", "USDCHF": "USDJPY", "EURJPY": "GBPJPY", "GBPJPY": "EURJPY"}


def smt_check(strat_cfg: StrategyConfig, partner: tuple[str, list[Bar]], level_name: str, sweep_time: datetime,
              direction: int, tf: int) -> dict[str, Any] | None:
    """Did the correlated market also run its own version of the swept level?

    ICT's SMT divergence: when one market takes a low (or high) and its partner does not,
    the move is more likely a sweep than a breakout. Returns None when the partner has no
    such level yet."""
    label, pbars = partner
    tracker = LevelTracker(strat_cfg)
    seen = []
    for b in pbars:
        if b.time > sweep_time:
            break
        tracker.update(b)
        seen.append(b)
    lv = next((x for x in tracker.levels() if x.name == level_name), None)
    if lv is None or not seen:
        return None
    since = sweep_time - timedelta(hours=2)
    window = [b for b in seen if b.time >= since]
    if not window:
        return None
    if direction == LONG:  # sell-side was run: did the partner make the lower low too?
        took = min(b.low for b in window) < lv.price
    else:
        took = max(b.high for b in window) > lv.price
    side = "low" if direction == LONG else "high"
    return {"with": label, "divergence": not took,
            "text": (f"SMT: {label} did not take its {level_name} ({lv.price:g}) — the {side} was only run here"
                     if not took else f"No SMT: {label} also ran its {level_name}")}


GRADES = ("A+", "A", "B", "C")


def grade(rr: float, bias_aligned: bool, smt: dict | None, midnight: bool) -> dict[str, Any]:
    """Quality of a setup from its confluences with anti-ruin and HTF bias enforcement."""
    if not bias_aligned:
        return {"grade": "C", "score": 0, "confluence": ["against HTF bias"]}
    if rr < 1.8:
        return {"grade": "C", "score": 0, "confluence": [f"insufficient R:R ({rr:.1f}R)"]}
    why, score = ["HTF bias aligned"], 1
    if 1.8 <= rr <= 4.5:
        score += 1
        why.append(f"{rr:.1f}R target")
    if smt and smt.get("divergence"):
        score += 1
        why.append("SMT divergence")
    if midnight:
        score += 1
        why.append("midnight open")
    g = "A+" if score >= 4 else "A" if score == 3 else "B" if score == 2 else "C"
    return {"grade": g, "score": score, "confluence": why}


def _judge(js: dict[str, Any], lb: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The judgement matrix for a trade from a model that does not keep its own (Turtle Soup, the institutional engine):
    scored on what the trade itself shows (the draw, the liquidity taken, the reward, the time), the other rows left out. `lb`: the main
    strategy's top-down bias for this timeframe (ladder_bias), so these models are read against the same 1h / 4h / daily story as the rest.
    It describes the market NOW, so it is used for a trade from the last day only."""
    try:
        d = 1 if js["direction"] == "long" else -1
        rs = js.get("reasons") or []
        ny = to_ny(datetime.fromisoformat(js["time"]))
        fresh = lb is not None and (datetime.now(timezone.utc) - datetime.fromisoformat(js["time"])).total_seconds() < 20 * 3600
        a = advisor.assess({
            "dir": d, "bias": lb["p"][0] if fresh else (d if any(r.startswith("RL-05") for r in rs) else 0), "wbias": lb["u"][0] if fresh else 0,
            "bias_name": lb["names"][0] if fresh else "daily", "wbias_name": lb["names"][1] if fresh else "weekly", "level": js.get("level") or "",
            "rr": float(js["rr"]), "target": "the target", "window": js.get("killzone") or None, "timed": js.get("cls") != "synthetic",
            "entry_price": float(js["entry"]) if js.get("entry") else None,
            "target_price": float(js["target"]) if js.get("target") else None,
            "market": js.get("market"),
            "weekday": ny.weekday(), "hour": ny.hour, "midnight_ok": True if any(r.startswith("KZ-01") for r in rs) else None,
            "unknown": ["Premium / discount", "Displacement and gap", "Path to the target"]})
        a["partial"] = True  # judged on fewer rows than a model that keeps the full matrix
        return a
    except (KeyError, TypeError, ValueError):
        return None


def _ck(advice: dict[str, Any] | None) -> str:
    """The A+ checklist as its own alert line (empty when the trade carries none)."""
    t = aplus.text(advice)
    return t + "\n" if t else ""


def _grade_sig(js: dict[str, Any], lb: dict[str, Any] | None = None) -> None:
    if not js.get("advice") and js.get("rr") is not None:
        js["advice"] = _judge(js, lb)
    rs = js.get("reasons") or []
    if js.get("advice") and js.get("smt"):   # a correlated market was read: the checklist gets its SMT box
        js["advice"]["aplus"] = aplus.checklist(js["advice"], js["smt"])
    # the grade is the checklist's (ICT's three required conditions); the old confluence points only stand in for a trade with no judgement at all
    js["quality"] = aplus.quality(js.get("advice")) or grade(js["rr"], any(r.startswith("RL-05") for r in rs), js.get("smt"), any(r.startswith("KZ-01") for r in rs))
    if "market" in js:
        from .strict import mark
        mark(js)


def _model_state(strat: Any, signals: list[dict[str, Any]], st: StrategyConfig, dp: int, last: Bar, lb: dict[str, Any] | None = None) -> dict[str, Any]:
    """One model's live state on one market, for the Desk."""
    px = lambda v: _fmt(v, dp) if v is not None else None  # noqa: E731
    d = lambda v: "long" if v == LONG else "short"  # noqa: E731
    orders, forming, triggers = [], [], []
    for s in getattr(strat, "_setups", []):  # ICT 2022 / Silver Bullet
        if s.stage == "limit":
            risk = abs(s.limit - s.stop) or 1e-12
            bias_ok = hasattr(strat, "daily_bias") and strat.daily_bias()[0] == s.direction
            mo = getattr(strat, "_midnight_open", None)
            tp1_val = getattr(s, "tp1", None)
            fvg_val = getattr(s, "fvg", None)
            exp_fn = getattr(strat, "_expiry_bars", lambda: 10)
            orders.append({"quality": aplus.quality(getattr(s, "advice", None)) or grade(abs(s.target - s.limit) / risk, bias_ok, None, mo is not None and (s.extreme - mo) * s.direction < 0),
                           "direction": d(s.direction), "entry": px(s.limit), "stop": px(s.stop), "tp1": px(tp1_val), "tp2": px(s.target),
                           "tp2_name": getattr(s, "target_name", ""), "rr": round(abs(s.target - s.limit) / risk, 2), "level": s.level.name,
                           "fvg": [px(fvg_val[0]), px(fvg_val[1])] if fvg_val else None, "candles_left": exp_fn() - s.bars,
                           "placed": int(s.mss_time.timestamp()) if getattr(s, "mss_time", None) else None, "window": getattr(s, "window", ""),
                           "advice": getattr(s, "advice", None),
                           "management": management.describe(st.timeframe_minutes, st.model)})
        else:
            found = strat._last_swing(s.direction, len(strat._bars) - 1, max(0, s.sweep_index - st.swing_lookback))
            need_px = found[0] if isinstance(found, (tuple, list)) else found
            forming.append({"direction": d(s.direction), "level": s.level.name, "level_price": px(s.level.price), "extreme": px(s.extreme),
                            "need": px(need_px) if need_px is not None else None, "candles_left": max(0, st.mss_max_bars - s.bars), "window": getattr(s, "window", "")})
    for p in getattr(strat, "_pending", []):  # sweep & reclaim: a close back inside triggers a market order
        triggers.append({"direction": d(p.direction), "level": p.level.name, "level_price": px(p.level.price), "extreme": px(p.extreme),
                         "trigger": f"a candle close back {'below' if p.direction == SHORT else 'above'} {px(p.level.price)}",
                         "candles_left": max(0, st.max_confirm_bars - p.bars_waited)})
    open_trades = [s for s in signals if s["outcome"]["status"] == "open"]
    stage = ("in_trade" if open_trades else "order" if orders else "trigger" if triggers else "forming" if forming else "idle")
    for s in signals:
        if "quality" not in s:
            _grade_sig(s, lb)
        s.pop("reasons", None)
    return {"available": True, "stage": stage, "orders": orders, "forming": forming, "triggers": triggers,
            "signals": signals, "price": px(last.close)}


def _stats(trades: list[Any]) -> dict[str, Any]:
    closed = [t for t in trades if not t.exit_reason.startswith("END")]
    wins = [t for t in closed if t.r_multiple > 0.05]
    losses = [t for t in closed if t.r_multiple < -0.05]
    gross_win = sum(t.r_multiple for t in wins)
    gross_loss = -sum(t.r_multiple for t in losses)
    total_r = sum(t.r_multiple for t in closed)
    n = len(closed)
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": n - len(wins) - len(losses),
        "win_rate": round(len(wins) / n * 100, 1) if n else None,
        "total_r": round(total_r, 2),
        "avg_r": round(total_r / n, 2) if n else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
    }


def _load(market: Market, fetch: Fetcher | None, mode: str,
          pre: Any = None, tf: int = 5) -> tuple[list[Bar], float | None, bool, str]:
    """Bars, spread (price units, if known), whether the feed is real-time, and its label."""
    if isinstance(pre, tuple):
        bars, symbol = pre
        return bars, None, True, f"Deriv {symbol} · real-time"
    if fetch is None and market.oanda and data_source() == "oanda":
        gran = {1: "M1", 2: "M2", 5: "M5", 15: "M15", 60: "H1", 240: "H4"}.get(tf)
        if gran is None:
            raise LookupError(f"OANDA has no {tf}-minute candles")
        if mode == "history":
            bars, spread = fetch_oanda(market.oanda, days=history_days(tf), granularity=gran)
        else:
            bars, spread = fetch_oanda(market.oanda, count=live_bars(tf), granularity=gran)
        return bars, spread, True, f"OANDA {market.oanda} · real-time"
    if fetch is None and not market.feed:
        raise pre if isinstance(pre, Exception) else LookupError("no free data source for this market")
    # Not on the live feed: only fall back to delayed Yahoo data if explicitly allowed.
    if isinstance(pre, Exception) and (not isinstance(pre, LookupError) or not allow_delayed()):
        raise pre
    range_ = "1mo" if mode == "history" else "5d"
    if fetch is not None:
        raw = fetch(market.feed, range_)
    else:
        if tf not in (1, 2, 5, 15, 60):
            raise LookupError(f"no {tf}-minute delayed data for this market")
        yrange = {1: "7d", 60: "1y" if mode == "history" else "1mo"}.get(tf, range_)
        raw = fetch_yahoo(market.feed, "5d" if tf == 1 and mode == "live" else yrange, "60m" if tf == 60 else f"{tf}m")
    bars, spread = raw if isinstance(raw, tuple) else (raw, None)
    return bars, spread, False, market.feed_label + (" (delayed)" if fetch is None and "delay" not in market.feed_label else "")


def scan_market(market: Market, fetch: Fetcher | None = None, now: datetime | None = None,
                mode: str = "live", pre: Any = None, tf: int = 5,
                candles_for: set[str] | None = None, as_of: datetime | None = None, news: list | None = None,
                partner: tuple[str, list[Bar]] | None = None, daily: list[Bar] | None = None, light: bool = False) -> dict[str, Any]:
    """`light`: run only the six core models (when the scan is running out of time).
    `daily`: about 70 daily candles that seed the higher-timeframe record (weekly draw, 20/40/60-day ranges).
    `candles_for`: in history mode, the signal ids that get their chart candles
    (default: the 8 most recent). `as_of`: replay a past day: the engine sees only the
    candles up to that moment (its read, setups and signals as they were then), while
    the trades' outcomes use everything that happened after."""
    now = now or datetime.now(timezone.utc)
    real_now = now
    out: dict[str, Any] = {"key": market.key, "label": market.label, "cls": market.cls, "ok": False}
    try:
        raw, spread, realtime, feed_label = _load(market, fetch, "history" if as_of else mode, pre, tf)
        if len(raw) < 300:
            raise RuntimeError(f"only {len(raw)} bars returned")
        if market.config_file and tf == 5:
            # Hand-tuned 5-minute config; dynamically expand max_stop_pips if real market volatility exceeds static bounds.
            cfg: Config = load_config(EXAMPLES / market.config_file)
            if spread:
                cfg = replace(cfg, instrument=replace(cfg.instrument, spread_pips=spread / cfg.instrument.pip_size))
            if market.config_file != "config.example.toml" and raw:
                rng_pips = median([b.high - b.low for b in raw[-864:]]) / max(cfg.instrument.pip_size, 1e-12)
                dyn_max = round(18.0 * rng_pips, 1)
                if dyn_max > cfg.strategy.max_stop_pips:
                    cfg = replace(cfg, strategy=replace(cfg.strategy, max_stop_pips=dyn_max))
        else:
            cfg = auto_config(market, raw, spread)
        cfg = autolearn.heal_config(cfg, market.key, raw)
        cfg = apply_style(cfg, tf)
        st = replace(cfg.strategy, model=os.environ.get("SIGNAL_MODEL", "ict2022"))
        if market.cls == "synthetic":
            # No time theory on a computer-generated market: price model at any hour.
            st = replace(st, killzones=(Window("any_hour", time(0, 0), time(23, 59)),), reference_ranges=(),
                         flatten_time=None, max_signals_per_day=3, day_rollover=time(0, 0))
        cfg = replace(cfg, strategy=management.apply(st, tf, live=True))  # this style's and model's ICT trade management
        all_bars = _closed_bars(raw, now, tf)
        if as_of:
            now = min(as_of, now)
        bars = _closed_bars(raw, now, tf)
        if len(bars) < 100:
            raise RuntimeError("not enough candles before that day")
    except LookupError as e:
        out.update(error=str(e), unavailable=True)
        return out
    except Exception as e:  # network, parsing or config problems are reported, never raised
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    dp = _decimals(market, bars[-1].close)
    broker_symbol = pre[1] if isinstance(pre, tuple) else None  # e.g. frxEURUSD, for the trading bot
    strategy = make_strategy(cfg.strategy, cfg.instrument)
    if not daily and fetch is None and market.feed and market.cls != "synthetic":
        try:
            raw_d = fetch_yahoo(market.feed, "3mo", "1d")
            daily = raw_d[0] if isinstance(raw_d, tuple) else raw_d
        except Exception:
            daily = None
    if daily and hasattr(strategy, "seed_days"):
        strategy.seed_days(daily, before=bars[0].time.date())
    sig_by_bar: dict[datetime, Signal] = {}
    for b in bars:
        if (s := strategy.on_bar(b)) is not None:
            sig_by_bar[b.time] = s
    signals = list(sig_by_bar.values())
    try:
        autolearn.heal_config(cfg, market.key, bars, getattr(strategy, "decisions", []), signals)
    except Exception:
        pass
    try:   # the top-down bias for this timeframe, so the models with no matrix of their own (Turtle Soup, the institutional engine) are read against it too
        lb_now = strategy.ladder_bias() if hasattr(strategy, "ladder_bias") else None
    except Exception:
        lb_now = None
    # NF-01: no entries 30 minutes either side of high-impact news for this market's currencies
    # Synthetic (volatility) indices have no currency, so no news, FOMC, NFP or holiday rule applies.
    ccy = market_currencies(market.key, market.cls, market.region)
    nf = NewsFilter(cfg.news, replace(cfg.instrument, news_currencies=ccy), news) if news and ccy else None
    result = Backtester(cfg, news=nf).run(all_bars, precomputed=sig_by_bar if not as_of else None)  # outcomes: everything that happened, even after `as_of`
    by_entry = {t.entry_time: t for t in result.trades}
    step = timedelta(minutes=tf)

    def sig_json(s: Signal, by_entry: dict = by_entry, model: str | None = None) -> dict[str, Any]:
        trade = by_entry.get(s.time)
        oc = _outcome(trade, s, all_bars)
        done = trade is not None and oc["status"] != "open"
        return {
            "id": _signal_id(market, s, tf) + (f"-{model}" if model else ""),
            "timeframe_minutes": tf,
            "market": market.key,
            "label": market.label,
            "cls": market.cls,
            "broker_symbol": broker_symbol,
            "decimals": dp,
            "time": s.time.isoformat(),
            "time_ny": to_ny(s.time).strftime("%a %d %b %H:%M"),
            "direction": "long" if s.direction == LONG else "short",
            "entry": _fmt(s.entry_ref, dp),
            "stop": _fmt(s.stop, dp),
            "target": _fmt(s.target, dp),
            "rr": round(s.rr, 2),
            "killzone": s.killzone,
            "entry_type": s.entry_type,
            "level": s.level.name,
            "reasons": list(s.reasons),
            "fill": _fmt(trade.entry_price, dp) if trade else None,
            "exit": _fmt(trade.exit_price, dp) if done else None,
            "exit_time": trade.exit_time.isoformat() if done else None,
            "exit_time_ny": to_ny(trade.exit_time).strftime("%H:%M") if done else None,
            "outcome": oc,
            "setup": _setup_json(s, dp),
            "advice": s.meta.get("advice"),  # the adviser's TAKE / CAUTION / SKIP with its reasons
            "management": management.describe(tf, model or cfg.strategy.model),
            "model": model or cfg.strategy.model,  # which ICT model gave it, shown beside it everywhere
        }

    last = bars[-1]
    out.update({
        "ok": True,
        "feed": feed_label,
        "broker_symbol": broker_symbol,
        "realtime": realtime,
        "tuned": bool(market.config_file) and tf == 5,
        "timeframe_minutes": tf,
        "last_price": _fmt(last.close, dp),
        "last_bar_ny": to_ny(last.time + step).strftime("%H:%M"),
        "stale_minutes": round((real_now - (last.time + step)).total_seconds() / 60) if not as_of else 0,
        "as_of": as_of.isoformat() if as_of else None,
        "since": to_ny(bars[0].time).strftime("%d %b"),
    })
    if mode == "history":
        all_trades = list(result.trades)
        hist_pairs: list[tuple[Any, dict[str, Any], str | None, str]] = [(s, by_entry, None, _signal_id(market, s, tf)) for s in signals]
        # The track record is ONE model per market (its main model): a trade is counted once. Adding every model's trades to it counted one
        # idea several times over (hundreds a day) and made the record say nothing about what a person following the app would have done.
        hist_pairs.sort(key=lambda x: x[0].time)
        out["stats"] = _stats(all_trades)
        recent = []
        index = {b.time: i for i, b in enumerate(bars)}
        want = candles_for if candles_for is not None else {sid for _, _, _, sid in hist_pairs[-8:]}
        for s, by_m, m_name, _ in hist_pairs[::-1]:  # every trade in the window, newest first (the trade review)
            js = sig_json(s, by_m, m_name)
            if js["id"] in want:
                # The real candles around the trade: from 20 bars before the sweep (or 30
                # before the signal) to a few bars after the exit (or the latest bar).
                sweep_t = s.meta.get("sweep_time")
                i0 = index.get(sweep_t, index.get(s.time - step, len(bars) - 1)) - (20 if sweep_t else 30)
                exit_t = datetime.fromisoformat(js["exit_time"]) if js["exit_time"] else bars[-1].time
                i1 = min(len(bars), index.get(exit_t, len(bars) - 1) + 8, max(0, i0) + 160)
                js["candles"] = [[int(b.time.timestamp()), _fmt(b.open, dp), _fmt(b.high, dp), _fmt(b.low, dp), _fmt(b.close, dp)]
                                 for b in bars[max(0, i0):i1]]
            js["signal_bar"] = int((s.time - step).timestamp())
            js["decimals"] = dp
            _grade_sig(js, lb_now)
            recent.append(js)
        out["recent"] = recent
    else:
        # "Current" signals: today (NY) for scalp/intraday; the last week / month for swing / long-term.
        first_day = to_ny(now).date() - timedelta(days=signal_window_days(tf) - 1)
        current = lambda ts: to_ny(ts).date() >= first_day  # noqa: E731
        fmt = "%H:%M" if tf <= 15 else "%a %d %b %H:%M"
        out["staged"] = [{"direction": o["direction"], "limit": _fmt(o["limit"], dp), "stop": _fmt(o["stop"], dp),
                          "target": _fmt(o["target"], dp), "level": o["level"],
                          "candles_left": o.get("bars_left", o.get("bars_waiting", 0)),
                          "fvg": [_fmt(o.get("fvg", (0, 0))[0], dp), _fmt(o.get("fvg", (0, 0))[1], dp)]}
                         for o in getattr(strategy, "staged", [])]
        out["model"] = cfg.strategy.model
        todays = [sig_json(s) for s in signals if current(s.time)]
        index = {b.time: i for i, b in enumerate(bars)}
        for s, js in zip([s for s in signals if current(s.time)], todays):
            # The candles the engine saw, for the marked-up chart of each of today's signals.
            sweep_t = s.meta.get("sweep_time")
            i0 = max(0, index.get(sweep_t, len(bars) - 1) - 40) if sweep_t else max(0, len(bars) - 120)
            js["candles"] = [[int(b.time.timestamp()), _fmt(b.open, dp), _fmt(b.high, dp), _fmt(b.low, dp), _fmt(b.close, dp)]
                             for b in bars[i0:i0 + 200]]
            js["signal_bar"] = int((s.time - step).timestamp())
        if partner:
            pb = (partner[0], _closed_bars(partner[1], now, tf))
            for s, js in zip([s for s in signals if current(s.time)], todays):
                if s.meta.get("sweep_time"):
                    js["smt"] = smt_check(cfg.strategy, pb, s.level.name, s.meta["sweep_time"], s.direction, tf)
        for js in todays:
            _grade_sig(js, lb_now)
        if hasattr(strategy, "events"):
            try:
                out["analysis"] = analyze(strategy, bars, cfg, tf, dp, now, todays, news=nf)
                sw = next((e for e in reversed(strategy.events) if e["type"] == "sweep"), None)
                if partner and sw and current(sw["time"]):
                    out["analysis"]["smt"] = smt_check(cfg.strategy, (partner[0], _closed_bars(partner[1], now, tf)),
                                                       sw["level"], sw["time"], sw["dir"], tf)
                    if out["analysis"]["smt"]:
                        out["analysis"]["narrative"].insert(-1, out["analysis"]["smt"]["text"] + " (SMT-01).")
                if out["analysis"]["events"] or todays:
                    # Today's candles for the marked-up chart of a market where something happened.
                    start = out["analysis"]["day_start"] - 4 * 3600 if tf < 60 else 0
                    chart = [b for b in bars if b.time.timestamp() >= start][-220:]
                    out["chart"] = [[int(b.time.timestamp()), _fmt(b.open, dp), _fmt(b.high, dp), _fmt(b.low, dp), _fmt(b.close, dp)]
                                    for b in chart]
            except Exception as e:  # the read is extra; never let it break the scan
                log.warning("analysis failed for %s: %s", market.key, e)
        # Every model's live state on this market: resting limits, setups forming,
        # market executions about to trigger, and today's trades (the Desk).
        out["models"] = {}
        recent_cut = to_ny(now).date() - timedelta(days=max(7, signal_window_days(tf) - 1))
        recent_win = lambda ts: to_ny(ts).date() >= recent_cut  # noqa: E731
        recent_map: dict[str, tuple[Any, dict[str, Any]]] = {}
        for s in signals:
            if recent_win(s.time):
                js = sig_json(s)
                js["signal_bar"] = int((s.time - step).timestamp())
                js["decimals"] = dp
                _grade_sig(js, lb_now)
                recent_map[js["id"]] = (s, js)
        for name in (CORE_MODELS if light else LIVE_MODELS):
            if not model_available(cfg.strategy, name):
                out["models"][name] = {"available": False}
                continue
            if name == cfg.strategy.model:
                strat_m, sigs_m, by_m = strategy, signals, by_entry
            else:
                cfg_m = replace(cfg, strategy=management.apply(replace(cfg.strategy, model=name), tf, live=True))
                strat_m = make_strategy(cfg_m.strategy, cfg.instrument)
                if daily and hasattr(strat_m, "seed_days"):
                    strat_m.seed_days(daily, before=bars[0].time.date())
                sigs_m_by_bar = {}
                for b in bars:
                    if (s := strat_m.on_bar(b)) is not None:
                        sigs_m_by_bar[b.time] = s
                sigs_m = list(sigs_m_by_bar.values())
                # Run the backtester whenever the model has any signal in the 7-day track-record window.
                by_m = {t.entry_time: t for t in Backtester(cfg_m, news=nf).run(all_bars, precomputed=sigs_m_by_bar if not as_of else None).trades} if any(recent_win(s.time) for s in sigs_m) else {}
                if kind_of(name) == "model":
                    for s in sigs_m:
                        if recent_win(s.time):
                            time_dir = (s.time, s.direction)
                            match = next((item for item in recent_map.values() if (item[0].time, item[0].direction) == time_dir), None)
                            if match:
                                _, m_js = match
                                cf = m_js.setdefault("confluence_models", [m_js.get("model", "ict2022")])
                                if name not in cf:
                                    cf.append(name)
                                continue
                            js = sig_json(s, by_m, name)
                            js["confluence_models"] = [name]
                            if js["id"] not in recent_map:
                                js["signal_bar"] = int((s.time - step).timestamp())
                                js["decimals"] = dp
                                _grade_sig(js, lb_now)
                                recent_map[js["id"]] = (s, js)
            out["models"][name] = _model_state(strat_m, [sig_json(s, by_m, name) for s in sigs_m if current(s.time)],
                                               cfg_m.strategy if name != cfg.strategy.model else cfg.strategy, dp, last, lb_now)
        recent_items = sorted(recent_map.values(), key=lambda x: x[1]["time"], reverse=True)
        for idx_r, (s, js) in enumerate(recent_items):
            if idx_r < 8 and "candles" not in js:
                sweep_t = s.meta.get("sweep_time")
                i0 = index.get(sweep_t, index.get(s.time - step, len(bars) - 1)) - (20 if sweep_t else 30)
                exit_t = datetime.fromisoformat(js["exit_time"]) if js.get("exit_time") else bars[-1].time
                i1 = min(len(bars), index.get(exit_t, len(bars) - 1) + 8, max(0, i0) + 160)
                js["candles"] = [[int(b.time.timestamp()), _fmt(b.open, dp), _fmt(b.high, dp), _fmt(b.low, dp), _fmt(b.close, dp)]
                                 for b in bars[max(0, i0):i1]]
        out["recent"] = [js for _, js in recent_items]
        # Models telling the same story: which directions have live orders, triggers, setups forming or open trades,
        # and how many models stand behind each. One model alone is a hint; several agreeing is a stronger read.
        agree: dict[str, list[str]] = {"long": [], "short": []}
        for name, stt in out["models"].items():
            if not isinstance(stt, dict) or not stt.get("available") or kind_of(name) != "model":
                continue  # only trade models vouch for a direction; a building block is a part, not a second opinion
            dirs = {o["direction"] for k in ("orders", "triggers", "forming") for o in stt.get(k, [])}
            dirs |= {sg["direction"] for sg in stt.get("signals", []) if (sg.get("outcome") or {}).get("status") == "open"}
            for dd in dirs:
                agree[dd].append(name)
        out["agreement"] = {k: v for k, v in agree.items() if v}
        try:  # which model suits this market right now (conditions.py + selector.py)
            from . import selector
            conds = strategy.read_conditions() if hasattr(strategy, "read_conditions") else None
            if conds:
                out["conductor"] = selector.recommend(conds, [k for k, v in out["models"].items() if v.get("available") and kind_of(k) == "model"], key=market.key,
                                                      label=_label)
        except Exception as e:  # the read is extra; never let it break the scan
            log.warning("conductor failed for %s: %s", market.key, e)
        today_all_sigs = [js for s, js in recent_items if current(s.time)] if recent_items else todays
        out["today"] = {
            "signals": today_all_sigs,
            "decisions": [
                {"time_ny": to_ny(d.time).strftime(fmt), "rule": d.rule, "message": d.message}
                for d in result.decisions if current(d.time) and d.time <= now and d.rule not in ("SIGNAL", "ENTRY")
            ][-12:],
        }
        # Bridge to genuine GodsEyeFluxEngine (single-pass institutional state machine)
        try:
            from flux_engine.universe import BY_KEY as GE_BY_KEY
            from flux_engine.engine import GodsEyeFluxEngine
            from flux_engine.feed import Bar as FBar
            if market.key in GE_BY_KEY:
                spec = GE_BY_KEY[market.key]
                ge_eng = GodsEyeFluxEngine(spec=spec, tf_minutes=tf)
                f_bars = [FBar(b.time, b.open, b.high, b.low, b.close, b.volume) for b in bars]
                ge_sigs, ge_trades, ge_radar = ge_eng.run(f_bars)
                
                # If God's Eye has an active stage 3 intercept (resting limit / order):
                if ge_radar.stage_code == 3 and ge_radar.staged_intercept:
                    si = ge_radar.staged_intercept
                    ge_order = {
                        "quality": si.get("grade", "A+"),
                        "direction": si.get("direction", "long"),
                        "entry": _fmt(si.get("entry"), dp),
                        "stop": _fmt(si.get("stop"), dp),
                        "tp1": _fmt(si.get("tp1"), dp),
                        "tp2": _fmt(si.get("tp2"), dp),
                        "tp2_name": si.get("tp2_name", "DOL"),
                        "rr": round(si.get("rr", 3.5), 2),
                        "level": si.get("archetype", "CHOKEPOINT"),
                        "fvg": None,
                        "candles_left": 10,
                        "placed": int(bars[-1].time.timestamp()),
                        "window": ge_radar.active_killzone or "LONDON/NY",
                        "advice": si.get("verdict", "EXECUTE"),
                        "management": "★ God's Eye Flux (+0.78R BE · +1.25R 70% TP1 · +3.5R Macro Runner)"
                    }
                    if "ict2022" in out["models"] and isinstance(out["models"]["ict2022"], dict):
                        out["models"]["ict2022"]["stage"] = "order"
                        out["models"]["ict2022"]["orders"] = [ge_order]
                elif ge_radar.stage_code == 2:
                    pool = ge_radar.nearest_ssl if ge_radar.bias > 0 else ge_radar.nearest_bsl
                    if pool:
                        ge_forming = {
                            "direction": "long" if ge_radar.bias > 0 else "short",
                            "level": pool.name,
                            "level_price": _fmt(pool.price, dp),
                            "extreme": _fmt(pool.price, dp),
                            "need": _fmt(pool.price, dp),
                            "candles_left": 8,
                            "window": ge_radar.active_killzone or "SETUP"
                        }
                        if "ict2022" in out["models"] and isinstance(out["models"]["ict2022"], dict):
                            out["models"]["ict2022"]["stage"] = "forming"
                            out["models"]["ict2022"]["forming"] = [ge_forming]
                
                # Check for active trade
                if ge_radar.recent_trades and ge_radar.recent_trades[0].get("status") == "open":
                    if "ict2022" in out["models"] and isinstance(out["models"]["ict2022"], dict):
                        out["models"]["ict2022"]["stage"] = "in_trade"

                # Enrich analysis with God's Eye institutional readings
                if "analysis" in out and isinstance(out["analysis"], dict):
                    out["analysis"]["smt"] = ge_radar.smt_summary
                    out["analysis"]["regime"] = ge_radar.regime
                    out["analysis"]["bias"] = ge_radar.bias_label
                    if "dealing" in out["analysis"] and isinstance(out["analysis"]["dealing"], dict):
                        out["analysis"]["dealing"]["pct"] = ge_radar.dealing_range_pct
        except Exception as ex_ge:
            log.warning("GodsEyeFluxEngine scan bridge error for %s: %s", market.key, ex_ge)
    return out


NOTES = [
    "Same rules and conservative fill model as the backtester: next-bar fill, spread charged, stop assumed hit first.",
    "News: entries are blocked 30 minutes either side of high-impact events for each market's currencies (ForexFactory calendar, this week; off if the feed is unreachable). Markets without a hand-tuned config scale their thresholds to their own recent 5-minute range.",
]


def _envelope(now: datetime, results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    realtime = any(r.get("realtime") for r in ok)
    return {
        "generated_at": now.isoformat(),
        "generated_ny": to_ny(now).strftime("%a %d %b %H:%M"),
        "source": {"oanda": "OANDA", "deriv": "Deriv"}.get(data_source(), "Yahoo Finance") + (" real-time 5-minute candles" if realtime else " 5-minute bars (delayed)"),
        "provider": data_source(),
        "realtime": realtime,
        "notes": NOTES,
        "scanned": len(ok),
        "universe": len(results),
        "classes": CLASS_LABEL,
        "markets": results,
    }


def _prefetch(markets: tuple[Market, ...], fetch: Fetcher | None, mode: str, tf: int = 5, days: int | None = None,
              discover: bool = False) -> dict[str, Any]:
    """One Deriv connection for every market; on failure, markets fall back to Yahoo."""
    if fetch is not None or data_source() != "deriv":
        return {}
    try:
        return fetch_deriv_bulk(markets, mode, tf=tf, days=days, discover=discover)
    except Exception as e:
        log.warning("Deriv feed failed: %s", e)
        err = RuntimeError(f"Deriv feed unreachable: {e}")
        return {m.key: (LookupError(str(err)) if m.feed and allow_delayed() else err) for m in markets}


def _enrich_with_smt(results: list[dict[str, Any]], pre: dict[str, Any]) -> None:
    """Detects real-time intermarket SMT divergence across all correlated pairs in SMT_PAIRS

    and attaches institutional confirmation tags to today's active signals.
    """
    from .smt import detect_smt_divergence
    correlated: list[tuple[str, str]] = []
    for a, b in SMT_PAIRS.items():
        if (a, b) not in correlated and (b, a) not in correlated:
            correlated.append((a, b))
    by_key = {r["key"]: r for r in results if r.get("ok")}
    for key_a, key_b in correlated:
        val_a = pre.get(key_a)
        val_b = pre.get(key_b)
        if isinstance(val_a, tuple) and isinstance(val_b, tuple):
            bars_a, bars_b = val_a[0], val_b[0]
            if isinstance(bars_a, list) and isinstance(bars_b, list) and len(bars_a) >= 5 and len(bars_b) >= 5:
                smt = detect_smt_divergence(bars_a, bars_b, key_a, key_b)
                if smt:
                    for k in (key_a, key_b):
                        if k in by_key:
                            by_key[k]["smt"] = {
                                "active": True,
                                "direction": "long" if smt.direction == 1 else "short",
                                "description": smt.description,
                            }
                            today = by_key[k].get("today")
                            if isinstance(today, dict):
                                for sig in today.get("signals", []):
                                    if (sig.get("direction") == "long" and smt.direction == 1) or (sig.get("direction") == "short" and smt.direction == -1):
                                        reasons = list(sig.get("reasons", []))
                                        if not any("SMT-" in r for r in reasons):
                                            reasons.append(f"SMT-01 Intermarket Divergence: {smt.description}")
                                            sig["reasons"] = reasons


REPLAY_DAYS = 10  # how far back a past day can be replayed
DISCOVER = os.environ.get("DERIV_DISCOVER", "1") != "0"  # add every Deriv market, not just the fixed list


def _run(markets: tuple[Market, ...], fetch: Fetcher | None, now: datetime, mode: str, tf: int = 5,
         as_of: datetime | None = None, light: bool = False, discover: bool | None = None) -> list[dict[str, Any]]:
    if tf not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
    do_discover = (DISCOVER if mode == "live" else False) if discover is None else (discover and mode == "live")
    if as_of:
        if as_of > now or (now - as_of).days > REPLAY_DAYS:
            raise ValueError(f"pick a day within the last {REPLAY_DAYS} days")
        # enough candles for the levels before that day, plus everything since (outcomes)
        days = min(history_days(tf), (now - as_of).days + (5 if tf < 60 else 30))
        pre = _prefetch(markets, fetch, "history", tf, days=days, discover=False)
    else:
        pre = _prefetch(markets, fetch, mode, tf, discover=do_discover)
    # every other market Deriv lists joins the scan (synthetics first)
    if do_discover:
        markets = markets + tuple(pre.pop("__extra__", None) or ())
    else:
        pre.pop("__extra__", None)
    daily = pre.pop("__daily__", None) or {}
    news = fetch_calendar() if fetch is None and os.environ.get("NEWS_FILTER", "1") != "0" else None
    t0 = _time.monotonic()
    with ThreadPoolExecutor(max_workers=min(16, len(markets))) as pool:
        labels = {m.key: m.label for m in markets}

        def partner(m: Market) -> tuple[str, list[Bar]] | None:
            k = SMT_PAIRS.get(m.key)
            got = pre.get(k) if k else None
            return (labels.get(k, k), got[0]) if isinstance(got, tuple) and mode == "live" else None

        def safe(m: Market) -> dict[str, Any]:
            # one market's problem must never take the whole scan (and the site) down
            try:
                return scan_market(m, fetch, now, mode, pre.get(m.key), tf, as_of=as_of, news=news, partner=partner(m), daily=daily.get(m.key),
                                   light=light or (_time.monotonic() - t0 > SCAN_BUDGET_S))
            except Exception as e:
                log.warning("scan of %s failed: %s", m.key, e)
                return {"key": m.key, "label": m.label, "cls": m.cls, "ok": False, "error": f"{type(e).__name__}: {e}"}

        results = list(pool.map(safe, markets))
    _enrich_with_smt(results, pre)
    return results


def trade_detail(market_key: str, signal_id: str, tf: int = 5, fetch: Fetcher | None = None,
                 now: datetime | None = None, cls: str | None = None) -> dict[str, Any]:
    """One past trade with its real candles, for the trade review chart."""
    if tf not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
    market = next((m for m in universe() if m.key == market_key), None)
    if market is None and cls:  # a market discovered from Deriv's list
        market = Market(market_key, market_key, "", "Deriv", "", 0, "", cls)
    if market is None:
        raise LookupError(f"unknown market {market_key}")
    now = now or datetime.now(timezone.utc)
    pre = _prefetch((market,), fetch, "history", tf)
    daily = pre.pop("__daily__", None) or {}
    res = scan_market(market, fetch, now, "history", pre.get(market.key), tf, candles_for={signal_id}, daily=daily.get(market.key))
    if not res.get("ok"):
        raise LookupError(res.get("error") or "market unavailable")
    sig = next((sg for sg in res["recent"] if sg["id"] == signal_id), None)
    if sig is None:
        raise LookupError("trade not found in the current history window")
    return sig


def scan(fetch: Fetcher | None = None, now: datetime | None = None,
         markets: tuple[Market, ...] | None = None, tf: int = 5, as_of: datetime | None = None,
         calendar: list | None = None, light: bool = False, discover: bool | None = None) -> dict[str, Any]:
    """Live scan: today's signals and decisions for every market. With `as_of`, the same
    view replayed for a past day (the engine's read at that moment, outcomes since)."""
    now = now or datetime.now(timezone.utc)
    if markets is None:
        markets = universe()
    elif discover is None:
        discover = False  # target markets passed: do not auto-append 25 extra discovered markets
    results = _run(markets, fetch, now, "live", tf, as_of=as_of, light=light, discover=discover)
    env = _envelope(now, results)
    env["timeframe_minutes"] = tf
    env["summary"] = {"signals_today": sum(len(r["today"]["signals"]) for r in results if r.get("ok"))}
    env["as_of"] = as_of.isoformat() if as_of else None
    if os.environ.get("NEWS_FILTER", "1") != "0" and not as_of:  # the economic calendar, for the app and the pre-news alert
        try:
            events = calendar if calendar is not None else (fetch_calendar() if fetch is None else None)
            if events is not None or fetch is None:
                env["news"] = news_digest(events, [{"key": r.get("key"), "label": r.get("label"), "cls": r.get("cls")} for r in results if r.get("ok")], now)
        except Exception as e:  # the calendar must never take a scan down
            log.warning("news digest failed: %s", e)
    return env


def history(fetch: Fetcher | None = None, now: datetime | None = None,
            markets: tuple[Market, ...] | None = None, tf: int = 5) -> dict[str, Any]:
    """Track record over the last month for every market."""
    now = now or datetime.now(timezone.utc)
    markets = markets or universe()
    results = _run(markets, fetch, now, "history", tf)
    env = _envelope(now, results)
    env["timeframe_minutes"] = tf
    env["style"] = style_of(tf)
    env["history_days"] = history_days(tf)
    st = [r["stats"] for r in results if r.get("ok")]
    n, w = sum(s["trades"] for s in st), sum(s["wins"] for s in st)
    gw = sum(s["total_r"] for s in st if s["total_r"] > 0)
    # Keep candles only for the most recent closed trades (the replays); drop the rest to stay small.
    closed = sorted((sg for r in results if r.get("ok") for sg in r["recent"] if sg["outcome"]["status"] in ("win", "loss", "breakeven")),
                    key=lambda sg: sg["time"], reverse=True)
    keep = {sg["id"] for sg in closed[:12]}
    for r in results:
        for sg in r.get("recent", []):
            if sg["id"] not in keep:
                sg.pop("candles", None)
    env["summary"] = {"trades": n, "wins": w, "win_rate": round(w / n * 100, 1) if n else None,
                      "total_r": round(sum(s["total_r"] for s in st), 2),
                      "markets_positive": sum(1 for s in st if s["total_r"] > 0), "gross_positive_r": round(gw, 2)}
    return env


# ---------------------------------------------------------------- notifications

SCAN_BUDGET_S = float(os.environ.get("SCAN_BUDGET_S", "25"))  # after this long, the rest of the markets run the six core models only
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
DEFAULT_TOPIC = "ictflux-signals-x7q4m2"


def ntfy_topic() -> str:
    return os.environ.get("NTFY_TOPIC", DEFAULT_TOPIC)


def _http(url: str, data: bytes | None = None, headers: dict[str, str] | None = None, timeout: float = 8) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _log_topic(topic: str) -> str:
    """Nobody subscribes to this one: it only remembers the alerts that tier-wise went to some phones and not the shared topic."""
    return f"{topic}-log"


def already_sent(topic: str, since: str = "24h") -> dict[str, str | None]:
    """Event ids already pushed to the topic (with the grade each was sent with), read back from ntfy's message cache.

    This keeps the scanner stateless: any cron cadence, retries or overlaps
    never notify the same event twice.
    """
    raw = _http(f"{NTFY_SERVER}/{topic},{_log_topic(topic)}/json?poll=1&since={since}").decode("utf-8", "replace")
    ids: dict[str, str | None] = {}
    for line in raw.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        tags = msg.get("tags") or []
        grade = next((t[3:] for t in tags if t.startswith("gr-")), None)
        for tag in tags:
            if tag.startswith("id-"):
                ids[tag[3:]] = grade
    return ids


KILLZONE_ALERTS = (  # (id, name, start, end) in NY time; alerted once in their first 15 minutes
    ("london", "London killzone", time(2, 0), time(5, 0)),
    ("ny", "New York killzone (forex)", time(7, 0), time(10, 0)),
    ("ny_idx", "NY AM killzone (indices)", time(8, 30), time(11, 0)),
    ("ny_pm", "NY PM killzone", time(13, 30), time(15, 0)),
)


def _min_grade() -> int:
    """Events are built for every tier (A+, A, B); each phone and the shared channel then take what their tier allows (tiers.py)."""
    return GRADES.index("B")


# Models whose setups are pushed besides the main one (NOTIFY_MODELS="ote,unicorn" to narrow, "none" to stop).
# Quiet (low-priority) pushes when a setup starts forming; NOTIFY_FORMING=0 turns them off.
NOTIFY_FORMING = os.environ.get("NOTIFY_FORMING", "1") != "0"
NOTIFY_MODELS = tuple(x.strip() for x in os.environ.get("NOTIFY_MODELS", ",".join(m for m in LIVE_MODELS if kind_of(m) == "model")).split(",") if x.strip())


def pending_events(report: dict[str, Any], sent: frozenset[str] | set[str] = frozenset()) -> list[dict[str, Any]]:
    """Everything worth a push: a killzone opening, a setup forming, a limit order set (and
    cancelled), a trade filled, its progress, and its close, each naming the model(s) behind it.
    Trades below B are never alerted. Each event carries its `grade` (None for a heads-up) and `kind`, so the
    tier a phone chose (A+ only, A and up, B and up) decides who gets it. `sent`: event ids already pushed
    (a dict of id -> grade when it comes from ntfy's history, so a cancelled limit follows the grade it was sent with)."""
    events: list[dict[str, Any]] = []
    min_g = _min_grade()
    ok_grade = lambda q: q is None or GRADES.index(q.get("grade", "C")) <= min_g  # noqa: E731
    word = lambda d: "BUY" if d == "long" else "SELL"  # noqa: E731
    arrow = lambda d: "▲" if d == "long" else "▼"  # noqa: E731
    gtag = lambda q: f" · {q['grade']}" if q else ""  # noqa: E731
    ok = [m for m in report.get("markets", []) if m.get("ok")]

    # killzone opening: who has a bias and what is forming
    if report.get("generated_at"):
        now = to_ny(datetime.fromisoformat(report["generated_at"]))
        for kid, name, start, end in KILLZONE_ALERTS:
            st = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
            if now.weekday() < 5 and st <= now < st + timedelta(minutes=15):
                leaning = [m for m in ok if (m.get("analysis") or {}).get("bias") in ("bullish", "bearish")]
                lean = ", ".join(f"{m['label']} {arrow('long' if m['analysis']['bias'] == 'bullish' else 'short')}" for m in leaning[:6])
                forming = sum(1 for m in ok if (m.get("analysis") or {}).get("stage") in ("await_mss", "limit"))
                events.append({"id": f"kz-{kid}-{now:%Y%m%d}", "title": f"🟢 {name} is open ({start:%H:%M}–{end:%H:%M} NY)",
                               "body": (f"Bias: {lean}." if lean else "No clear bias yet.") + (f" {forming} setup(s) forming." if forming else "")
                                       + " Watch for a liquidity sweep, then the shift.",
                               "priority": "default", "emoji": "alarm_clock", "kind": "headsup", "grade": None})

    events += _news_events(report)

    now_utc = datetime.fromisoformat(report["generated_at"]) if report.get("generated_at") else None
    tfm = report.get("timeframe_minutes", 5)
    for m in ok:
        events += _market_events(m, report, now_utc, tfm, ok_grade, sent)
    return events


NEWS_LEAD_MIN = 15  # a heads-up this many minutes before a high-impact release


def _news_events(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One heads-up per release time, shortly before it: which markets pause and what to do about open trades."""
    nw, at = report.get("news") or {}, report.get("generated_at")
    if not nw.get("ok") or not at:
        return []
    now = datetime.fromisoformat(at)
    win = nw.get("window") or {}
    before, after = win.get("before", 30), win.get("after", 30)
    by_ccy = nw.get("markets_by_currency") or {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in nw.get("events", []):
        if e.get("impact") == "high" and e.get("currency") in by_ccy:
            m = (datetime.fromisoformat(e["time"]) - now).total_seconds() / 60
            if 0 < m <= NEWS_LEAD_MIN:
                groups.setdefault(e["time"], []).append(e)
    out = []
    for iso, es in groups.items():
        t = datetime.fromisoformat(iso)
        mins = max(1, round((t - now).total_seconds() / 60))
        ccys = sorted({e["currency"] for e in es})
        touched = sorted({lbl for c in ccys for lbl in by_ccy.get(c, [])})
        names = "; ".join(dict.fromkeys(f"{e['currency']} {e['title']}" for e in es))
        out.append({"id": f"news-{t:%Y%m%dT%H%M}", "kind": "news", "grade": None, "priority": "high", "emoji": "newspaper",
                    "title": f"📰 High-impact news in {mins} min ({to_ny(t):%H:%M} NY): {names}"[:140],
                    "body": (f"New entries pause on {len(touched)} markets ({', '.join(ccys)}) from {to_ny(t - timedelta(minutes=before)):%H:%M} to "
                             f"{to_ny(t + timedelta(minutes=after)):%H:%M} NY. Open trades on them should be closed or covered before the release.\n"
                             "ICT's way: don't trade into the number. Let the first spike run one side's liquidity, then look for the retest of an inversion or first-presented FVG.")})
    return out


def _label(k: str) -> str:
    return MODEL_INFO.get(k, {}).get("label", k)


def _names(models: list[str]) -> str:
    return " + ".join(dict.fromkeys(_label(k) for k in models))


def _lasted(minutes: float) -> str:
    minutes = max(0, int(round(minutes)))
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes} min"


def _dp(v: Any) -> int:
    t = str(v)
    return len(t.split(".")[1]) if "." in t else 0


def _when(iso: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(iso) if iso else None
    except ValueError:
        return None


# Progress steps of an open trade, pushed once each and only the furthest new one per scan.
def _milestones(s: dict[str, Any]) -> list[tuple[float, str, str, str]]:
    rr = float(s.get("rr") or 0)
    try:
        entry = float(s.get("fill") or s["entry"])
        risk = abs(entry - float(s["stop"]))
    except (KeyError, TypeError, ValueError):
        return []
    mg = s.get("management") or {}
    part = int(round((mg.get("partial") or 0) * 100))
    tp1 = (s.get("setup") or {}).get("tp1")
    marks = [(1.0, "1r", "+1R reached", "Your risk is covered by the move; follow the plan, don't take it early.")]
    if tp1 is not None and risk:
        tp1_r = abs(float(tp1) - entry) / risk
        marks.append((tp1_r, "tp1", f"TP1 reached ({tp1})",
                      f"Take {part}% off and move the stop to entry." if part else "Move the stop to entry; hold the rest to the target."))
    if rr:
        if mg.get("trim_50", True):
            marks.append((0.5 * rr, "p50", "Halfway to the target", "Cut the stop's risk by 25% (SL-04)."))
        if mg.get("be_75", True):
            marks.append((0.75 * rr, "p75", "75% of the way to the target", "Move the stop to entry (SL-05): this trade can no longer lose."))
    for r, key in ((2.0, "2r"), (3.0, "3r"), (5.0, "5r")):
        marks.append((r, key, f"+{r:.0f}R reached", "Keep the stop where the plan puts it and let it run."))
    return sorted((x for x in marks if 0 < x[0] < rr or not rr), key=lambda x: x[0])


def _market_events(m: dict[str, Any], report: dict[str, Any], now: datetime | None, tfm: int,
                   ok_grade: Any, sent: frozenset[str] | set[str]) -> list[dict[str, Any]]:
    """One market's pushes, every model together: the same setup in several models is one alert naming them all."""
    word = lambda d: "BUY" if d == "long" else "SELL"  # noqa: E731
    arrow = lambda d: "▲" if d == "long" else "▼"  # noqa: E731
    gtag = lambda q: f" · {q['grade']}" if q else ""  # noqa: E731
    key, label, price = m.get("key", m.get("label", "")), m["label"], m.get("last_price")
    main = m.get("model") or "ict2022"
    states = m.get("models") or {}
    names = [main] + [n for n in states if n != main and n in NOTIFY_MODELS and (states[n] or {}).get("available")]
    recent = lambda t: now is None or t is None or now - t <= timedelta(hours=3)  # noqa: E731
    out: list[dict[str, Any]] = []

    def group(items: list[tuple[str, dict[str, Any]]], kf: Any) -> list[tuple[dict[str, Any], list[str]]]:
        by: dict[Any, tuple[dict[str, Any], list[str]]] = {}
        for name, x in items:
            k = kf(x)
            if k in by:
                by[k][1].append(name)
            else:
                by[k] = (x, [name])
        return list(by.values())

    # limit orders resting in the gap: the trade, before it fills
    orders = [(n, o) for n in names for o in (states.get(n) or {}).get("orders", []) if ok_grade(o.get("quality"))]
    live_placed = {str(o.get("placed")) for st in states.values() for o in (st or {}).get("orders", [])}
    for o, ms in group(orders, lambda o: (o["direction"], o["entry"])):
        first = ms[0]
        mins = (o.get("candles_left") or 0) * tfm
        away = ""
        try:
            dist = abs(float(price) - float(o["entry"]))
            risk = abs(float(o["entry"]) - float(o["stop"])) or 1
            away = f"\nNow {price}: {dist:.{_dp(o['entry'])}f} from the limit ({dist / risk:.1f}× the stop distance). It fills if price comes back to {o['entry']}."
        except (TypeError, ValueError):
            pass
        by = f" (by {to_ny(now + timedelta(minutes=mins)):%H:%M} NY)" if now and mins else ""
        og = (o.get("quality") or {}).get("grade")
        out.append({
            "id": f"{key}-order-{o.get('placed')}" if first == main else f"{key}-{first}-order-{o.get('placed')}",
            "title": f"⏳ {_names(ms)}: {word(o['direction'])} LIMIT {label}{gtag(o.get('quality'))} · {o['rr']}R",
            "body": (tiers.suggestion(og, o.get("advice"), len(set(ms))) + "\n" + _ck(o.get("advice")) +
                     f"{word(o['direction'])} limit {o['entry']} · SL {o['stop']}" + (f" · TP1 {o['tp1']}" if o.get("tp1") is not None else "")
                     + f" · TP2 {o['tp2']}" + away + (f"\nCancels if not filled within {mins} min{by}." if mins else "") + f"\nModel: {_names(ms)}."),
            "priority": "high", "emoji": "hourglass_flowing_sand", "kind": "trade", "grade": og})

    # every trade, once, with all the models that took it
    sigs = [(main, x) for x in m.get("today", {}).get("signals", [])]
    sigs += [(n, x) for n in names[1:] for x in (states.get(n) or {}).get("signals", [])]
    filled_from = {str((x.get("setup") or {}).get("mss_time")) for _, x in sigs}

    # a limit pushed earlier that is gone without a fill: the setup failed before entry
    for sid in sent:
        if not sid.startswith(f"{key}-") or "-order-" not in sid or sid.endswith("-cancelled"):
            continue
        placed = sid.rsplit("-order-", 1)[1]
        try:  # only today's recent limits: older ones may have filled on a day no longer in the report
            at = datetime.fromtimestamp(int(placed), timezone.utc)
        except (ValueError, OverflowError, OSError):
            continue
        if now is None or not timedelta(0) <= now - at <= timedelta(hours=6) or to_ny(at).date() != to_ny(now).date():
            continue
        if placed in live_placed or placed in filled_from or f"{sid}-cancelled" in sent:
            continue
        mid = sid[len(key) + 1:].rsplit("-order-", 1)[0]
        out.append({"id": f"{sid}-cancelled", "title": f"✖️ {_label(mid if mid in MODEL_INFO else main)}: {label} limit cancelled",
                    "body": "The limit was taken off before it filled: price ran to the target or through the stop first, or its time ran out. No trade.",
                    "priority": "default", "emoji": "x", "kind": "cancel",
                    "grade": (sent.get(sid) if hasattr(sent, "get") else None)})

    for s, ms in group(sigs, lambda x: (x["direction"], x.get("time") or x["id"])):
        if not ok_grade(s.get("quality")):
            continue
        oc = s.get("outcome") or {}
        st, sid = oc.get("status"), s["id"]
        t0 = _when(s.get("time"))
        who = _names(ms)
        su = s.get("setup") or {}
        q = s.get("quality")
        g = (q or {}).get("grade")
        conf = ", ".join(q["confluence"]) if q and q.get("confluence") else ""
        ago = f" ({_lasted((now - t0).total_seconds() / 60)} ago)" if now and t0 else ""
        if st in ("open", "win", "loss", "breakeven") and (st == "open" or ms[0] == main) and recent(t0):
            out.append({
                "id": sid,
                "title": f"{arrow(s['direction'])} {who}: {word(s['direction'])} {label}{gtag(q)} · {s['rr']}R",
                "body": ((tiers.suggestion(g, s.get("advice"), len(set(ms)), filled=True) + "\n" + _ck(s.get("advice"))) if st == "open" else "")
                         + (f"{word(s['direction'])} {s['entry']} · SL {s['stop']}" + (f" · TP1 {su['tp1']}" if su.get("tp1") is not None else "")
                         + f" · TP2 {s['target']}\n"
                         + f"{(s.get('killzone') or '').replace('_', ' ')} · {s.get('level', '')} swept" + (f" · {conf}" if conf else "")
                         + (" · SMT ✓" if (s.get("smt") or {}).get("divergence") else "")
                         + (f"\nFilled at {s['time_ny'][-5:]} NY{ago}" + (f", now {oc['r']:+.2f}R." if st == "open" and oc.get("r") is not None else ".") if s.get("time_ny") else "")
                         + (f"\nPlan: {(s.get('management') or {}).get('text')}" if (s.get("management") or {}).get("text") else "")
                         + f"\nModel: {who}."),
                "priority": "high", "kind": "trade", "grade": g,
                "emoji": "chart_with_upwards_trend" if s["direction"] == "long" else "chart_with_downwards_trend"})
        if st == "open":
            best, now_r = oc.get("best_r") if oc.get("best_r") is not None else oc.get("r") or 0, oc.get("r") or 0
            crossed = [x for x in _milestones(s) if best >= x[0]]
            if crossed:
                r_at, mk, what, todo = crossed[-1]
                pct = f", {min(100, best / float(s['rr']) * 100):.0f}% of the way to TP2 {s['target']}" if s.get("rr") else ""
                out.append({"id": f"{sid}-m-{mk}", "title": f"📈 {who}: {label} {word(s['direction'])} · {what}",
                            "body": f"Best so far {best:+.2f}R{pct}; now {now_r:+.2f}R.\nNext: {todo}\nModel: {who}.",
                            "priority": "default", "emoji": "chart_with_upwards_trend", "kind": "trade", "grade": g})
            if now_r <= -0.7:
                out.append({"id": f"{sid}-warn", "title": f"⚠️ {who}: {label} under pressure · {now_r:+.2f}R",
                            "body": f"Price is {abs(now_r) * 100:.0f}% of the way to the stop {s['stop']}. The plan holds to the stop: don't widen it, don't add.\nModel: {who}.",
                            "priority": "default", "emoji": "warning", "kind": "trade", "grade": g})
        if st in ("win", "loss", "breakeven") and recent(_when(s.get("exit_time"))):
            best = oc.get("best_r")
            t1 = _when(s.get("exit_time"))
            took = f" after {_lasted((t1 - t0).total_seconds() / 60)}" if t0 and t1 else ""
            far = (f"\nAt best it reached {best:+.2f}R" + (" before the stop." if st == "loss" else ".")) if best is not None else ""
            sign = "+" if (oc.get("r") or 0) > 0 else ""
            out.append({
                "id": sid + "-closed",
                "title": f"{'✅' if st == 'win' else '⛔' if st == 'loss' else '➖'} {who}: {label} {oc.get('label', '')} · {sign}{oc.get('r')}R",
                "body": f"Exit {s.get('exit')} at {s.get('exit_time_ny')} NY (entry {s.get('fill')}){took}.{far}\nModel: {who}.",
                "priority": "default", "emoji": "white_check_mark" if st == "win" else "no_entry", "kind": "trade", "grade": g})

    # market executions armed (a level was run; a close back inside fires) and setups forming
    trig = [(n, t) for n in names for t in (states.get(n) or {}).get("triggers", [])]
    for t, ms in group(trig, lambda t: (t["direction"], t.get("level"), t.get("level_price"))):
        mins = (t.get("candles_left") or 0) * tfm
        out.append({"id": f"{key}-{ms[0]}-trigger-{t.get('level')}-{report.get('generated_at', '')[:10]}",
                    "title": f"👀 {_names(ms)}: {label} {word(t['direction'])} setup forming",
                    "body": f"{t['level']} {t['level_price']} was run to {t['extreme']}. Market entry on {t['trigger']}"
                            + (f", within the next {mins} min." if mins else ".") + f"\nModel: {_names(ms)}.",
                    "priority": "default", "emoji": "eyes", "kind": "forming", "grade": None})
    forming = [(n, f) for n in names for f in (states.get(n) or {}).get("forming", [])] if NOTIFY_FORMING else []
    for f, ms in group(forming, lambda f: (f["direction"], f.get("level"), f.get("level_price"))):
        mins = (f.get("candles_left") or 0) * tfm
        need = (f"Needs a strong close {'below' if f['direction'] == 'short' else 'above'} {f['need']} for the shift in structure, "
                f"then a limit goes in the gap") if f.get("need") is not None else "Waiting for the shift in structure, then a limit goes in the gap"
        out.append({"id": f"{key}-{ms[0]}-forming-{f.get('level')}-{report.get('generated_at', '')[:10]}",
                    "title": f"🧩 {_names(ms)}: {label} {word(f['direction'])} setup forming",
                    "body": f"{f['level']} {f['level_price']} swept to {f['extreme']}. {need}" + (f" (within {mins} min)." if mins else ".")
                            + f"\nModel: {_names(ms)}.",
                    "priority": "low", "emoji": "jigsaw", "kind": "forming", "grade": None})
    return out


def _send_ntfy(topic: str, ev: dict[str, Any], click: str | None) -> None:
    headers = {
        "Title": ev["title"].encode("utf-8").decode("latin-1", "replace"),
        "Priority": ev["priority"],
        "Tags": f"{ev['emoji']},id-{ev['id']}" + (f",gr-{ev['grade']}" if ev.get("grade") else ""),
    }
    if click:
        headers["Click"] = click
    _http(f"{NTFY_SERVER}/{topic}", data=ev["body"].encode("utf-8"), headers=headers)


def _send_telegram(ev: dict[str, Any]) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    payload = json.dumps({"chat_id": chat, "text": f"<b>{ev['title']}</b>\n{ev['body']}", "parse_mode": "HTML"}).encode()
    _http(f"https://api.telegram.org/bot{token}/sendMessage", data=payload, headers={"Content-Type": "application/json"})


def notify(report: dict[str, Any], click: str | None = None) -> dict[str, Any]:
    """Push every event not pushed before. Returns what was sent."""
    topic = ntfy_topic()
    try:
        sent_before = already_sent(topic)
    except Exception as e:
        # Without the history we cannot de-duplicate; sending blind would spam on every run.
        return {"sent": [], "error": f"could not read ntfy history: {e}"}
    sent, errors, done = [], [], []
    channel, log = tiers.channel_tier(), _log_topic(topic)
    for ev in pending_events(report, sent_before):
        if ev["id"] in sent_before:
            continue
        wanted = tiers.allows(channel, ev)  # the shared topic and Telegram follow NOTIFY_MIN_GRADE (A by default)
        try:
            _send_ntfy(topic if wanted else log, ev, click)  # the rest is written to a silent log topic so it is not sent twice
            done.append(ev)
            if wanted:
                sent.append(ev["id"])
        except Exception as e:
            errors.append(f"{ev['id']}: ntfy {e}")
            continue
        if wanted:
            try:
                _send_telegram(ev)
            except Exception as e:
                errors.append(f"{ev['id']}: telegram {e}")
    out = {"topic": topic, "sent": sent, "errors": errors}
    if done:  # straight to phones that turned notifications on in the app; each keeps what its own tier allows
        from .webpush import broadcast
        out["app_push"] = broadcast(done, click)
    return out
