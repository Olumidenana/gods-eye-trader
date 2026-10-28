"""Research: does the ICT engine have an edge on a market it was never designed for?

Runs the exact strategy and conservative backtester on months of real Deriv candles
for any symbol and any execution timeframe (1, 2, 3, 5, 15 minutes), and reports how
often setups form and whether they paid. Built first for Deriv's Volatility indices,
which ICT never taught: they are synthetic, trade 24/7, and have no sessions, news
or institutional order flow, so whether killzones, sweeps and gaps mean anything
there is an open question this answers with data rather than opinion.

Two time modes are compared:
- "ict":    the standard ICT killzones (London 02:00-05:00, NY 07:00-10:00 New York
            time), one setup per day, flat by 16:00, exactly as on forex.
- "allday": the same price rules at any hour (the market never closes), up to three
            setups per day, no forced flatten. Tests the price model without the
            time model.
"""

from __future__ import annotations

import json
import math
import os
import time as _time
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from statistics import median
from typing import Any, Callable

from . import aplus, management
from .backtest import Backtester
from .config import Window
from .metrics import summarize
from .models import MODEL_INFO, MODELS, is_model, label_of, make_strategy, model_available
from .scanner import DERIV_URLS, MARKETS, Market, _deriv_candles, apply_style, auto_config
from .sessions import Bar, to_ny

FOREX_METALS: dict[str, str] = {
    "frxXAUUSD": "Gold / USD", "frxXAGUSD": "Silver / USD",
    "frxEURUSD": "EUR/USD", "frxGBPUSD": "GBP/USD", "frxUSDJPY": "USD/JPY",
    "frxAUDUSD": "AUD/USD", "frxUSDCAD": "USD/CAD", "frxNZDUSD": "NZD/USD",
}
SYNTHETICS = FOREX_METALS
# Every candle size Deriv serves: 1-15 min use the killzones, 30 min keeps them, 1 h and up run at any hour.
TIMEFRAMES = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 480, 1440)
# How far back each timeframe goes: enough trades to judge, inside the 60 s function limit.
DEFAULT_DAYS = {1: 30, 2: 45, 3: 60, 5: 90, 10: 120, 15: 180, 30: 270, 60: 365, 120: 540, 240: 730, 480: 1095, 1440: 1825}
MODES = ("ict", "allday")

_CANDLE_CACHE: dict[tuple[str, int], tuple[float, list[Bar]]] = {}
_CANDLE_CACHE_TTL = 900.0  # 15 minutes


def fetch_deriv_history(symbol: str, tf_minutes: int, days: int,
                        connect: Callable[[str], Any] | None = None, deadline_s: float = 35) -> list[Bar]:
    """Real Deriv candles for `symbol`, paging back 5,000 at a time over one connection with in-memory caching."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Use cached candles if available and covering the required window
    if connect is None:
        key = (symbol, tf_minutes)
        now_mono = _time.monotonic()
        if key in _CANDLE_CACHE:
            ts, cached_bars = _CANDLE_CACHE[key]
            if now_mono - ts < _CANDLE_CACHE_TTL and cached_bars and cached_bars[0].time <= cutoff:
                return [b for b in cached_bars if b.time >= cutoff]

    if connect is None:
        import websocket

        def connect(url: str) -> Any:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Origin": "https://app.deriv.com",
            }
            return websocket.create_connection(url, header=headers, timeout=15)

    ws, errors = None, []
    bars: dict[datetime, Bar] = {}
    end: Any = "latest"
    stop_at = _time.monotonic() + deadline_s
    for attempt in range(3):
        ws, errors = None, []
        for url in DERIV_URLS:
            try:
                ws = connect(url.format(legacy_app_id=os.environ.get("DERIV_LEGACY_APP_ID", "1089")))
                break
            except Exception as e:
                errors.append(str(e))
                _time.sleep(0.2)
        if ws is None:
            if attempt < 2:
                _time.sleep(0.5)
                continue
            raise ConnectionError("; ".join(errors))
        try:
            for rid in range(1, 40):
                ws.send(json.dumps({"ticks_history": symbol, "style": "candles", "granularity": tf_minutes * 60,
                                    "count": 5000, "end": end, "req_id": rid}))
                while True:
                    msg = json.loads(ws.recv())
                    if msg.get("req_id") == rid or msg.get("req_id") == rid + 500:
                        break
                if "error" in msg:
                    err_text = str(msg["error"].get("message", ""))
                    err_code = str(msg["error"].get("code", "")).lower()
                    if "rate" in err_code or "rate" in err_text.lower() or "too many" in err_text.lower():
                        _time.sleep(1.0)
                        ws.send(json.dumps({"ticks_history": symbol, "style": "candles", "granularity": tf_minutes * 60,
                                            "count": 5000, "end": end, "req_id": rid + 500}))
                        while True:
                            msg = json.loads(ws.recv())
                            if msg.get("req_id") == rid + 500:
                                break
                        if "error" in msg:
                            raise LookupError(f"Deriv: {msg['error'].get('message')}")
                    else:
                        raise LookupError(f"Deriv: {msg['error'].get('message')}")
                page = [_deriv_candles(c) for c in msg.get("candles") or []]
                for b in page:
                    bars[b.time] = b
                if not page or page[0].time <= cutoff or _time.monotonic() > stop_at:
                    break
                end = int(page[0].time.timestamp()) - 1
                _time.sleep(0.05)
            break
        except Exception:
            if attempt == 2:
                if bars:
                    break
                raise
            _time.sleep(0.5)
        finally:
            try:
                ws.close()
            except Exception:
                pass
    sorted_bars = sorted((b for b in bars.values() if b.time >= cutoff), key=lambda b: b.time)
    min_cache = 15 if tf_minutes >= 1440 else (30 if tf_minutes >= 240 else (60 if tf_minutes >= 60 else 100))
    if connect is None and len(sorted_bars) >= min_cache:
        _CANDLE_CACHE[(symbol, tf_minutes)] = (_time.monotonic(), sorted_bars)
    return sorted_bars


def correlated_symbol(sym: str) -> str | None:
    """Returns the primary correlated pair for SMT divergence (e.g. GBP/USD <-> EUR/USD, NAS100 <-> US500)."""
    s = sym.upper().replace("FRX", "").replace("_", "")
    mapping = {
        "GBPUSD": "frxEURUSD" if sym.startswith("frx") else "EURUSD",
        "EURUSD": "frxGBPUSD" if sym.startswith("frx") else "GBPUSD",
        "XAUUSD": "frxXAGUSD" if sym.startswith("frx") else "XAGUSD",
        "XAGUSD": "frxXAUUSD" if sym.startswith("frx") else "XAUUSD",
        "NAS100": "US500",
        "US500": "NAS100",
        "USTEC": "US500",
        "SPX500": "NAS100",
    }
    return mapping.get(s)


def build_config(market: Market, bars: list[Bar], tf_minutes: int, mode: str, model: str = "ict2022",
                 mgmt: str | None = None):
    cfg = apply_style(auto_config(market, bars, None), tf_minutes)  # swing/long-term rules on 1h/4h
    s = management.apply(replace(cfg.strategy, model=model), tf_minutes)
    if mgmt in ("godseye", "titanium"):
        s = replace(s, partial_at_tp1=0.70, move_be_at_75_pct=True, move_be_at_1r=True, trim_stop_at_50_pct=True, trail_structure=True, target_pref="draw", min_rr=3.0, target_rr=3.5)
    elif mgmt == "hold":
        s = replace(s, partial_at_tp1=0.0, move_be_at_75_pct=False, move_be_at_1r=False, trim_stop_at_50_pct=False)
    elif mgmt == "fixed2":
        s = replace(s, partial_at_tp1=0.0, move_be_at_75_pct=False, move_be_at_1r=False, trim_stop_at_50_pct=False, min_rr=2.0, target_rr=2.0, target_mode="rr")
    elif mgmt == "fixed3":
        s = replace(s, partial_at_tp1=0.0, move_be_at_75_pct=False, move_be_at_1r=False, trim_stop_at_50_pct=False, min_rr=3.0, target_rr=3.0, target_mode="rr")
    elif mgmt == "fixed5":
        s = replace(s, partial_at_tp1=0.0, move_be_at_75_pct=False, move_be_at_1r=False, trim_stop_at_50_pct=False, min_rr=5.0, target_rr=5.0, target_mode="rr")
    elif mgmt == "fixed10":
        s = replace(s, partial_at_tp1=0.0, move_be_at_75_pct=False, move_be_at_1r=False, trim_stop_at_50_pct=False, min_rr=10.0, target_rr=10.0, target_mode="rr")
    elif mgmt == "fixed20":
        s = replace(s, partial_at_tp1=0.0, move_be_at_75_pct=False, move_be_at_1r=False, trim_stop_at_50_pct=False, min_rr=20.0, target_rr=20.0, target_mode="rr")
    elif mgmt in ("pd_array", "htf_draw"):
        s = replace(s, target_pref="draw", htf_context=True, htf_arrays=True, partial_at_tp1=0.5, move_be_at_75_pct=True, move_be_at_1r=True, trail_structure=True)
    elif mgmt in ("sd", "standard_dev"):
        s = replace(s, target_pref="sd", sd_targets=True, cbdr_targets=True, min_rr=2.0)
    if mode == "allday" or market.cls in ("synthetic", "crypto"):
        s = replace(s, killzones=(Window("all_day", time(0, 0), time(23, 59)),), flatten_time=None,
                    max_signals_per_day=3)
    if market.cls in ("synthetic", "crypto"):
        # No session, no 17:00 forex close: the "day" is the plain calendar day, so yesterday's high
        # and low mean something on a market that never stops.
        s = replace(s, day_rollover=time(0, 0))
    return replace(cfg, strategy=s)


def _excursion(trades: list, bars: list[Bar]) -> list[float]:
    """For each trade, the furthest it went in its favour (in R) before it ended."""
    from bisect import bisect_left, bisect_right
    out = []
    for t in trades:
        risk = abs(t.entry_price - t.stop) or 1e-12
        span = bars[bisect_left(bars, t.entry_time, key=lambda b: b.time):bisect_right(bars, t.exit_time, key=lambda b: b.time)]
        if not span:
            out.append(0.0)
            continue
        best = max(b.high for b in span) - t.entry_price if t.direction > 0 else t.entry_price - min(b.low for b in span)
        out.append(max(0.0, best / risk))
    return out


def _fixed_exit(trades: list, bars: list[Bar], spread: float, target_r: float, half_at: float | None = None) -> list[float]:
    """The same entries with a different exit: a fixed target at `target_r`, or (with `half_at`) half off
    at that many R with the stop moved to entry and the rest held to the trade's own target. Conservative
    like the backtester: the stop counts first when a candle touches both, gaps fill at the open."""
    from bisect import bisect_left
    out = []
    for t in trades:
        d, e, risk = t.direction, t.entry_price, abs(t.entry_price - t.stop) or 1e-12
        far = abs(t.target - e) / risk
        adj = 0.0 if d > 0 else spread  # exits on the bid for longs, the ask for shorts
        stop, banked, size = t.stop, 0.0, 1.0
        half = half_at is not None and far > half_at  # its own target nearer than the half: it simply runs to it
        first = half_at if half else far if half_at is not None else target_r
        goal = e + d * first * risk
        res = None
        for b in bars[bisect_left(bars, t.entry_time, key=lambda b: b.time):]:
            o, hi, lo = b.open + adj, b.high + adj, b.low + adj
            fav = lambda p: (p - e) * d / risk  # noqa: E731
            if (o - stop) * d <= 0 and b.time > t.entry_time:
                res = banked + size * fav(o)
                break
            if (lo if d > 0 else hi) * d <= stop * d:
                res = banked + size * fav(stop)
                break
            if b.time == t.entry_time:  # the fill candle: only the stop counts (it may have come first)
                continue
            if (hi if d > 0 else lo) * d >= goal * d:
                if not half or size < 1:
                    res = banked + size * fav(goal)
                    break
                banked, size, stop, goal = 0.5 * first, 0.5, e, t.target  # half off, stop to entry, rest to the target
                if (hi if d > 0 else lo) * d >= goal * d:
                    res = banked + size * far
                    break
        out.append(round(res, 3) if res is not None else round(banked + size * ((bars[-1].close - e) * d / risk), 3))
    return out


def _trade_grade(t: Any) -> str:
    """The same A+/A/B/C grade the live signals carry, from the trade's own rules trail. Evaluates SMT
    if a correlated market confirmation was detected."""
    from .scanner import grade
    g = aplus.parse_grade(t.reasons)
    if g:
        return g   # the grade the live signals carry: ICT's checklist
    risk = abs(t.entry_price - t.stop) or 1e-12
    rs = t.reasons or ()
    smt_box = {"divergence": True} if any("SMT-01" in r for r in rs) else None
    return grade(abs(t.target - t.entry_price) / risk, any(r.startswith("RL-05") for r in rs), smt_box,
                 any(r.startswith("KZ-01") for r in rs))["grade"]


def _trade_verdict(t: Any) -> str | None:
    """The adviser's verdict for a trade, from its rules trail (AD-00 advisor <score>/100 <VERDICT>)."""
    for r in t.reasons or ():
        if r.startswith("AD-00 advisor "):
            return r.split("/100 ", 1)[1].split(":", 1)[0]
    return None


def sample_plan(trades: int, trading_days: int, trades_needed: int | None, cal_per_day: float = 1.4, floor: int = 30) -> dict[str, Any]:
    """How much data this test needs before it can say anything, from how often the setup actually trades: the days to reach the
    30 trades that are the least before judging, and (when there is a positive edge to confirm) the days to confirm an edge this
    size (t = 2). `cal_per_day`: calendar days per trading day in this window (about 1.4 for forex, 1 for a 24/7 synthetic index)."""
    per_day = trades / trading_days if trading_days and trades else 0.0
    if per_day <= 0:
        return {"per_day": 0.0, "days_for_30": None, "days_to_confirm": None,
                "text": "It took no trades in this window, so there is nothing to size: widen the window, add markets, or ease the timeframe."}
    d30 = math.ceil(floor / per_day)
    dn = math.ceil(trades_needed / per_day) if trades_needed else None
    wk = lambda d: f"{d} trading days (about {max(1, math.ceil(d * cal_per_day / 7))} weeks)" if d >= 10 else f"{d} trading days"   # noqa: E731
    text = (f"At {per_day:.2f} trades a trading day, it needs about {wk(d30)} of data to reach 30 trades, the least before judging"
            + (f", and about {wk(dn)} to confirm an edge this size beyond luck." if dn else ", and there is no positive edge yet to size a confirmation for."))
    return {"per_day": round(per_day, 3), "days_for_30": d30, "days_to_confirm": dn, "text": text}


TRADE_CHARTS_MAX = 150  # every trade of a normal test; the payload stays around 1 MB


def _trade_charts(cfg: Any, market: Market, bars: list[Bar], trades: list, tf_minutes: int, n: int = TRADE_CHARTS_MAX,
                  daily_bars: list[Bar] | None = None) -> list[dict[str, Any]]:
    """The last `n` trades with the real candles around them and the setup the engine saw (sweep,
    shift, gap, entry, stop, targets), in the same shape as the track record, so the app draws
    each one exactly like a live trade review: market, date and time, how it formed and ended."""
    from .scanner import _setup_json, _decimals, _fmt
    strat = make_strategy(cfg.strategy, cfg.instrument)
    if daily_bars and hasattr(strat, "seed_days"):
        strat.seed_days(daily_bars, before=bars[0].time.date() if bars else None)
    sigs = {s.time: s for b in bars if (s := strat.on_bar(b)) is not None}
    index = {b.time: i for i, b in enumerate(bars)}
    dp = _decimals(market, bars[-1].close)
    step = timedelta(minutes=tf_minutes)
    out = []
    for t in trades[-n:][::-1]:
        sig = sigs.get(t.entry_time)
        i_in = index.get(t.entry_time - step, index.get(t.entry_time, 0))
        i_out = index.get(t.exit_time, i_in)
        lo, hi = max(0, i_in - 48), min(len(bars), max(i_out, i_in) + 8, i_in + 120)
        r = round(t.r_multiple, 2)
        status = "win" if r > 0.05 else "loss" if r < -0.05 else "breakeven"
        out.append({
            "id": f"{market.key}-{t.entry_time:%Y%m%dT%H%M}", "market": market.key, "label": market.label, "cls": market.cls,
            "time": t.entry_time.isoformat(), "time_ny": to_ny(t.entry_time).strftime("%a %d %b %Y %H:%M"),
            "exit_time": t.exit_time.isoformat(), "exit_time_ny": to_ny(t.exit_time).strftime("%a %d %b %H:%M"),
            "direction": "long" if t.direction > 0 else "short", "timeframe_minutes": tf_minutes, "decimals": dp,
            "entry": _fmt(t.entry_price, dp), "fill": _fmt(t.entry_price, dp), "stop": _fmt(t.stop, dp), "target": _fmt(t.target, dp),
            "exit": _fmt(t.exit_price, dp), "rr": round(abs(t.target - t.entry_price) / abs(t.entry_price - t.stop), 2) if t.entry_price != t.stop else None,
            "killzone": sig.killzone if sig else "", "level": sig.level.name if sig else "",
            "setup": _setup_json(sig, dp) if sig else None,
            "outcome": {"status": status, "label": t.exit_reason, "r": r},
            "candles": [[int(b.time.timestamp()), _fmt(b.open, dp), _fmt(b.high, dp), _fmt(b.low, dp), _fmt(b.close, dp)] for b in bars[lo:hi]],
            "signal_bar": int((t.entry_time - step).timestamp()),
        })
    return out


# What happened to every liquidity raid the model saw, from the rules log: the answer to "why so few trades?".
_FUNNEL = (
    ("Raids on a pool inside a window", ("SW-01", "NR-01", "RL-05", "HT-02", "OK-01", "OK-02", "OK-03", "OK-04", "WP-01", "WP-02", "SW-02"), True),
    ("Ignored: no story, or against it", ("NR-01", "RL-05", "HT-02", "OK-01", "OK-02", "OK-03", "OK-04", "WP-01", "WP-02"), False),
    ("Ignored: the other side was purged first", ("SW-02",), False),
    ("Raid, then no shift in structure in time", ("MS-02", "KZ-1"), False),
    ("Shifted, but the plan was rejected", ("FV-01", "FV-02", "PD-01", "DO-01", "MC-01", "RG-01", "RG-02", "HT-01", "KZ-01", "SL-03", "TP-01", "TP-02", "AD-01", "UN-01", "BR-01", "RP-01", "HA-01", "IV-02"), False),
    ("Limit orders placed", ("EN-01",), False),
    ("Cancelled before they filled", ("EN-04",), False),
    ("Trades taken", ("SIGNAL",), False),
)


def _funnel(decisions: list[Any]) -> list[dict[str, Any]]:
    import re
    out = []
    for label, rules, _total in _FUNNEL:
        msgs: dict[str, int] = {}
        n = 0
        for d in decisions:
            if d.rule not in rules:
                continue
            if d.rule == "MS-02" and "cancelled" not in d.message:
                continue                                    # "structure broken but without displacement; waiting" is not an exit
            n += 1
            m = re.sub(r"[-+]?\d+(\.\d+)?", "#", re.sub(r"^\S+ (?=(sweep|setup))", "", d.message))[:110]
            msgs[m] = msgs.get(m, 0) + 1
        top = sorted(msgs.items(), key=lambda kv: -kv[1])[:3]
        out.append({"label": label, "count": n, "reasons": [[m, c] for m, c in top] if label not in ("Trades taken", "Limit orders placed", "Raids on a pool inside a window") else []})
    return out


def research(symbol: str, tf_minutes: int = 5, days: int | None = None, mode: str = "ict",
             fetch: Callable[[str, int, int], list[Bar]] | None = None, label: str | None = None,
             model: str = "ict2022", cls: str | None = None, daily_bars: list[Bar] | None = None,
             mgmt: str | None = None, partner_bars: list[Bar] | None = None) -> dict[str, Any]:
    if not is_model(model):
        raise ValueError(f"model must be one of {MODELS}, or custom:entry=...;stop=...;target=...;filter=...;mgmt=...")
    if tf_minutes not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    days = days or DEFAULT_DAYS[tf_minutes]
    bars = (fetch or fetch_deriv_history)(symbol, tf_minutes, days)
    min_bars = 15 if tf_minutes >= 1440 else (30 if tf_minutes >= 240 else (60 if tf_minutes >= 60 else (100 if tf_minutes >= 15 else 150)))
    if len(bars) < min_bars:
        raise RuntimeError(f"only {len(bars)} candles returned for {symbol} (at least {min_bars} required)")
    tf = timedelta(minutes=tf_minutes)
    now = bars[-1].time + tf
    bars = [b for b in bars if b.time + tf <= now]
    if daily_bars is None and fetch is None and tf_minutes < 1440:
        try:
            daily_bars = fetch_deriv_history(symbol, 1440, days + 70, deadline_s=10)
        except Exception:
            daily_bars = None
    partner_sym = correlated_symbol(symbol)
    if partner_bars is None and fetch is None and partner_sym:
        try:
            partner_bars = fetch_deriv_history(partner_sym, tf_minutes, days, deadline_s=12)
        except Exception:
            partner_bars = None
    clean_key = symbol.replace("frx", "").replace("cry", "").upper()
    matched = next((m for m in MARKETS if m.key.upper() == clean_key or m.key.upper() == symbol.upper()), None)
    known = ("forex", "metal", "index", "energy", "crypto", "synthetic", "agri", "stock")
    if matched:
        cls = cls if cls in known else matched.cls
        decimals = matched.decimals
        region = matched.region
        label = label or matched.label
    else:
        cls = cls if cls in known else ("synthetic" if symbol in SYNTHETICS else "forex")
        decimals = 0
        region = ""
    market = Market(symbol, label or SYNTHETICS.get(symbol, symbol), "", "Deriv", "", decimals, "", cls, region)
    cfg = build_config(market, bars, tf_minutes, mode, model, mgmt=mgmt)
    if not model_available(cfg.strategy, model):
        raise ValueError(f"{model} requires a market with session killzones on a 1-30 minute timeframe")
    
    from flux_engine.universe import BY_KEY as GE_BY_KEY
    from flux_engine.engine import GodsEyeFluxEngine
    from flux_engine.feed import Bar as FBar
    from .backtest import Trade

    is_gods_eye = clean_key in GE_BY_KEY and (mgmt == "gdeye" or model in ("ict2022", "all", "sweep", "godseye"))
    if is_gods_eye:
        spec = GE_BY_KEY[clean_key]
        f_bars = [FBar(b.time, b.open, b.high, b.low, b.close, b.volume) for b in bars]
        f_partner = [FBar(b.time, b.open, b.high, b.low, b.close, b.volume) for b in partner_bars] if partner_bars else None
        ge_eng = GodsEyeFluxEngine(spec=spec, tf_minutes=tf_minutes, mode=mode)
        _, ge_trades, _ = ge_eng.run(f_bars, partner_bars=f_partner)
        raw_trades = [
            Trade(
                symbol=t.signal.symbol,
                direction=1 if t.signal.direction == 1 else -1,
                entry_time=t.entry_time,
                entry_price=t.entry_price,
                stop=t.stop_price,
                target=t.tp2_price,
                lots=1.0,
                risk_amount=100.0,
                exit_time=t.exit_time,
                exit_price=t.exit_price,
                exit_reason=t.exit_reason,
                pnl=round(t.r_multiple * 100.0, 2),
                r_multiple=t.r_multiple,
                reasons=tuple(t.signal.reasons)
            )
            for t in ge_trades if t.status != "open"
        ]
        result_equity = [(b.time, 10000.0) for b in bars]
        result = type("BacktestResultMock", (), {"trades": raw_trades, "equity_curve": result_equity, "decisions": []})()
    else:
        result = Backtester(cfg).run(bars, daily_bars=daily_bars)
        raw_trades = [t for t in result.trades if not t.exit_reason.startswith("END")]

    # If a correlated partner market was fetched, detect SMT divergence and enrich trades
    trades = []
    if partner_bars and len(partner_bars) >= 20 and partner_sym:
        from .smt import detect_smt_divergence
        from bisect import bisect_left
        for t in raw_trades:
            idx_a = bisect_left(bars, t.entry_time, key=lambda b: b.time)
            idx_b = bisect_left(partner_bars, t.entry_time, key=lambda b: b.time)
            slice_a = bars[max(0, idx_a - 40):idx_a + 1]
            slice_b = partner_bars[max(0, idx_b - 40):idx_b + 1]
            smt = detect_smt_divergence(slice_a, slice_b, symbol, partner_sym)
            if smt and ((t.direction > 0 and smt.direction == 1) or (t.direction < 0 and smt.direction == -1)):
                t = replace(t, reasons=t.reasons + (f"SMT-01 Intermarket Divergence with {partner_sym}: {smt.description}",))
            trades.append(t)
    else:
        trades = raw_trades

    stats = summarize(trades, result.equity_curve, cfg.initial_equity)
    span_days = max(1e-9, (bars[-1].time - bars[0].time).total_seconds() / 86400)
    trading_days = len({to_ny(b.time).date() for b in bars})

    by_window: dict[str, list[float]] = {}
    by_month: dict[str, list[float]] = {}
    by_weekday: dict[str, list[float]] = {}
    by_hour: dict[str, list[float]] = {}
    by_date: dict[str, list[float]] = {}
    for t in trades:
        kz = next((r.split(" inside ")[1].split(" ")[0] for r in t.reasons if " inside " in r), "?")
        by_window.setdefault(kz, []).append(t.r_multiple)
        ny = to_ny(t.entry_time)
        by_month.setdefault(ny.strftime("%Y-%m"), []).append(t.r_multiple)
        by_weekday.setdefault(ny.strftime("%a"), []).append(t.r_multiple)
        by_hour.setdefault(ny.strftime("%H:00"), []).append(t.r_multiple)
        by_date.setdefault(ny.date().isoformat(), []).append(t.r_multiple)
    agg = lambda rs: {"trades": len(rs), "total_r": round(sum(rs), 2),  # noqa: E731
                      "win_rate": round(sum(1 for r in rs if r > 0.05) / len(rs) * 100, 1) if rs else None}

    curve = result.equity_curve
    step = max(1, len(curve) // 160)
    exp, t = stats["expectancy_r"], stats["expectancy_t_stat"]
    if stats["trades"] < 30:
        n = stats["trades"]
        lo, hi = stats["expectancy_ci95"]
        so_far = (f" So far {exp:+.2f}R per trade, but luck alone could put the true figure anywhere from {lo:+.2f}R to {hi:+.2f}R."
                  + (f" To tell an edge of {exp:+.2f}R from luck at this spread you would need about {stats['trades_needed']} trades." if stats["trades_needed"] else "")
                  ) if n >= 2 else ""
        verdict = ("not enough setups", f"Only {n} trade{'' if n == 1 else 's'} in {trading_days} days, so no verdict yet. Thirty trades is the least before an "
                                        f"average means anything (confirming a small edge takes 100 or more).{so_far} This model is selective: pool more "
                                        "markets (the table below adds them up), use a longer window or a faster timeframe, or the 'All day' mode.")
    elif stats["statistically_meaningful"]:
        verdict = ("edge", f"Positive expectancy of {exp:+.2f}R per trade, unlikely to be luck (t = {t:.1f}). "
                           "Confirm on a later, unseen period before trusting it.")
    elif exp > 0:
        verdict = ("unproven", f"Slightly positive ({exp:+.2f}R per trade over {stats['trades']} trades), but not yet more than luck could "
                               f"explain (t = {t:.1f}; about 2 or more is convincing). More trades will tell.")
    else:
        verdict = ("no edge", f"Negative expectancy ({exp:+.2f}R per trade). The rules lose money here after costs.")

    # The same setups under different trade management, so the take-profit rules are chosen
    # by what paid on real candles, not by opinion.
    def managed(name: str, partial: float) -> dict[str, Any]:
        c = replace(cfg, strategy=replace(cfg.strategy, partial_at_tp1=partial))
        rs = [t.r_multiple for t in Backtester(c).run(bars).trades if not t.exit_reason.startswith("END")]
        wins, losses = [r for r in rs if r > 0.05], [r for r in rs if r < -0.05]
        return {"name": name, "partial": partial, "trades": len(rs), "total_r": round(sum(rs), 2),
                "win_rate": round(len(wins) / len(rs) * 100, 1) if rs else None,
                "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
                "avg_loss": round(sum(losses) / len(losses), 2) if losses else None}
    management_plans = [managed(f"ICT plan for this style: {management.describe(tf_minutes, model, trail=False)['text']}", cfg.strategy.partial_at_tp1),
                         managed("Hold the whole trade to the final target", 0.0)]

    # The same entries with fixed exits: a win more often, smaller each time.
    spread = Backtester(cfg).spread

    def fixed(name: str, rs: list[float]) -> dict[str, Any]:
        wins, losses = [r for r in rs if r > 0.05], [r for r in rs if r < -0.05]
        return {"name": name, "partial": None, "trades": len(rs), "total_r": round(sum(rs), 2),
                "win_rate": round(len(wins) / len(rs) * 100, 1) if rs else None,
                "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
                "avg_loss": round(sum(losses) / len(losses), 2) if losses else None}
    management_plans += [fixed("Fixed target at 2R", _fixed_exit(trades, bars, spread, 2.0)),
                         fixed("Fixed target at 3R", _fixed_exit(trades, bars, spread, 3.0)),
                         fixed("Fixed target at 5R", _fixed_exit(trades, bars, spread, 5.0)),
                         fixed("Fixed target at 10R", _fixed_exit(trades, bars, spread, 10.0)),
                         fixed("Fixed target at 20R (runner)", _fixed_exit(trades, bars, spread, 20.0)),
                         fixed("Half off at 2R, stop to entry, rest to final target", _fixed_exit(trades, bars, spread, 0, half_at=2.0)),
                         fixed("Half off at 3R, stop to entry, rest to 10R runner", _fixed_exit(trades, bars, spread, 10.0, half_at=3.0))]

    plan = sample_plan(stats["trades"], trading_days, stats.get("trades_needed"), (span_days / trading_days) if trading_days and span_days else 1.4)
    verdict = (verdict[0], verdict[1] + (" " + plan["text"] if plan["text"] else ""))

    # How far the trades went in their favour before they ended (from 0.5R to 20R+).
    best = _excursion(trades, bars)
    losers = [b for b, t in zip(best, trades) if t.r_multiple < -0.05]
    reach = {"trades": len(best),
             "levels": [{"r": r, "n": sum(1 for b in best if b >= r), "pct": round(sum(1 for b in best if b >= r) / len(best) * 100, 1) if best else 0}
                        for r in (0.5, 1, 2, 3, 5, 8, 10, 15, 20)],
             "median_best": round(median(best), 2) if best else None,
             "losers": len(losers),
             "losers_reached_1r": round(sum(1 for b in losers if b >= 1) / len(losers) * 100, 1) if losers else None,
             "losers_reached_2r": round(sum(1 for b in losers if b >= 2) / len(losers) * 100, 1) if losers else None,
             "losers_reached_3r": round(sum(1 for b in losers if b >= 3) / len(losers) * 100, 1) if losers else None}

    # Out of sample: the same rules on the older two thirds of the period and on the recent third.
    # A result that only works on one of them is a fit to that stretch, not an edge.
    cut = bars[0].time + (bars[-1].time - bars[0].time) * (2 / 3)
    older_r = [t.r_multiple for t in trades if t.entry_time < cut]
    recent_r = [t.r_multiple for t in trades if t.entry_time >= cut]

    def half(rs: list[float]) -> dict[str, Any]:
        return {"trades": len(rs), "win_rate": round(sum(1 for r in rs if r > 0.05) / len(rs) * 100, 1) if rs else None,
                "expectancy_r": round(sum(rs) / len(rs), 3) if rs else None, "total_r": round(sum(rs), 2)}
    by_advice: dict[str, list[float]] = {}
    for t in trades:
        v = _trade_verdict(t)
        if v:
            by_advice.setdefault(v, []).append(t.r_multiple)
    # When does the model work? Results by market condition at entry (trend, volatility, the draw, the phase of the day, the path).
    from . import conditions as cn
    by_cond: dict[str, dict[str, list[float]]] = {}
    for t in trades:
        for dim, val in cn.parse(t.reasons).items():
            by_cond.setdefault(dim, {}).setdefault(val, []).append(t.r_multiple)
    split = {"cut": cut.isoformat(), "older": half(older_r), "recent": half(recent_r),
             "holds": (sum(older_r) > 0 and sum(recent_r) > 0) if len(older_r) >= 10 and len(recent_r) >= 5 else None}

    return {
        "split": split,
        "funnel": _funnel(result.decisions),
        "by_condition": {dim: {val: {**agg(rs), "expectancy_r": round(sum(rs) / len(rs), 3)} for val, rs in sorted(vals.items())}
                         for dim, vals in by_cond.items()},
        # does the adviser earn its keep? results by its verdict (TAKE / CAUTION / SKIP)
        "by_advice": {k: {**agg(by_advice[k]), "expectancy_r": round(sum(by_advice[k]) / len(by_advice[k]), 3)} for k in ("TAKE", "CAUTION", "SKIP") if k in by_advice},
        "v_list": [_trade_verdict(t) for t in trades],
        "management": management_plans,
        "reach": reach,
        "symbol": symbol, "label": market.label, "timeframe_minutes": tf_minutes, "mode": mode, "model": model,
        "from": bars[0].time.isoformat(), "to": bars[-1].time.isoformat(), "candles": len(bars),
        "calendar_days": round(span_days, 1), "trading_days": trading_days,
        "requested_days": days, "complete": span_days >= days * 0.9,
        "mgmt": mgmt or "godseye",
        "setups_per_day": round(stats["trades"] / trading_days, 2) if trading_days else 0,
        "stats": {k: (round(v, 3) if isinstance(v, float) and v != float("inf") else (None if v == float("inf") else v))
                  for k, v in stats.items()},
        "verdict": {"code": verdict[0], "text": verdict[1]}, "sample_plan": plan,
        "by_window": {k: agg(v) for k, v in by_window.items()},
        "by_month": {k: agg(v) for k, v in sorted(by_month.items())},
        # when the setups happen: day of the week and NY hour of entry, and every date that had a trade
        "by_weekday": {d: agg(by_weekday[d]) for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun") if d in by_weekday},
        "by_hour": {k: agg(v) for k, v in sorted(by_hour.items())},
        "by_date": {k: agg(v) for k, v in sorted(by_date.items(), reverse=True)},
        "equity": [[int(ts.timestamp()), round(e, 2)] for ts, e in curve[::step]],
        "trades": [{"time": t.entry_time.isoformat(), "dir": "long" if t.direction > 0 else "short",
                    "entry": t.entry_price, "exit": t.exit_price, "r": round(t.r_multiple, 2), "exit_reason": t.exit_reason}
                   for t in trades[-15:]][::-1],
        "trade_charts": _trade_charts(cfg, market, bars, trades, tf_minutes, daily_bars=daily_bars),
        "r_list": [round(t.r_multiple, 3) for t in trades],  # every trade's result, so models can be combined exactly
        "g_list": [_trade_grade(t) for t in trades],  # each trade's quality (A+, A, B, C), so results can be cut by grade
        # did each trade pass ICT's A+ checklist (true / false, null when it carries none): the challenger to the grade above
        "k_list": [aplus.parse(t.reasons) for t in trades],
        # each trade's entry candle and side, so a setup two models both take counts once when they are combined
        "r_keys": [f"{t.entry_time.isoformat()}|{'long' if t.direction > 0 else 'short'}" for t in trades],
        "config": {"killzones": [f"{w.name} {w.start:%H:%M}-{w.end:%H:%M}" for w in cfg.strategy.killzones],
                   "min_stop": round(cfg.strategy.min_stop_pips * cfg.instrument.pip_size, 6),
                   "max_stop": round(cfg.strategy.max_stop_pips * cfg.instrument.pip_size, 6),
                   "spread": round(cfg.instrument.spread_pips * cfg.instrument.pip_size, 6),
                   "typical_range": round(median([b.high - b.low for b in bars[-864:]]), 6),
                   "max_setups_per_day": cfg.strategy.max_signals_per_day,
                   "flatten": cfg.strategy.flatten_time.strftime("%H:%M") if cfg.strategy.flatten_time else None},
    }
