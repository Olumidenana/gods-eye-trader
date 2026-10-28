"""One market, every timeframe: ICT's top-down read.

ICT does not decide on the chart he trades. The daily and weekly draw set the story, the 4-hour and 1-hour structure
say whether price is delivering it, the 15-minute shows the setup taking shape and the 5- and 1-minute find the entry.
So a trade is only as good as the timeframes above it agree with it, and a lower timeframe that leans the other way is
usually the pullback an entry is made from, not a reason to stand aside.

This module runs the engine once per timeframe on the same market and puts the answers side by side: the market
structure on that timeframe's own candles, where price sits in its range, the nearest pools of resting stops and open
gaps either side, and what every model is doing there. Then it says whether the timeframes agree. It adds no rules of
its own: every number below the structure read comes from the same scan the Desk and the alerts use. Any market Deriv
lists can be read (the list comes from Deriv itself, not from the scan), and a synthetic index is read the same way, on price
alone (no sessions, no clock).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from . import conditions, scanner
from .sessions import Bar

log = logging.getLogger(__name__)

TFS = (240, 60, 15, 5, 1)   # top-down: the higher timeframe sets the story, the lower one finds the entry
ROLES = {240: "Sets the story", 60: "Confirms it", 15: "Frames the setup", 5: "Looks for the entry", 1: "Fine-tunes the entry"}
HIGHER = 60                 # from here up a timeframe is "higher" (it has to agree); below it is where the entry is found
KEY_OK = re.compile(r"^[A-Za-z0-9_.:\-]{1,32}$")
DROP = ("chart", "since", "conductor", "staged")   # heavy parts of a scan this page does not draw
KEEP = ("bias", "bias_reason", "stage", "draw", "story", "dealing_range", "levels", "narrative", "need_mss", "plan", "checklist",
        "news", "smt", "daily_bias", "price", "timeframe_minutes", "any_hour")


def valid_key(key: str) -> bool:
    """A market key from outside (a request): only the characters Deriv and our own keys use."""
    return bool(KEY_OK.match(key or ""))


def structure(bars: list[Bar], left: int = 2, window: int = 150) -> dict[str, Any]:
    """The market structure on these candles: are the swing highs and lows stepping up, stepping down, or neither?"""
    b = bars[-window:]
    hi: list[float] = []
    lo: list[float] = []
    for i in range(left, len(b) - left):
        if all(b[i].high > b[j].high for j in range(i - left, i)) and all(b[i].high >= b[j].high for j in range(i + 1, i + left + 1)):
            hi.append(b[i].high)
        if all(b[i].low < b[j].low for j in range(i - left, i)) and all(b[i].low <= b[j].low for j in range(i + 1, i + left + 1)):
            lo.append(b[i].low)
    if len(hi) < 2 or len(lo) < 2:
        return {"trend": "unclear", "text": "Not enough swings yet to read the structure."}
    hh, hl = hi[-1] > hi[-2], lo[-1] > lo[-2]
    trend = "up" if hh and hl else "down" if not hh and not hl else "range"
    text = {(True, True): "Higher highs and higher lows: buyers are in control.",
            (False, False): "Lower highs and lower lows: sellers are in control.",
            (True, False): "A higher high but a lower low: two-sided, no clear trend.",
            (False, True): "A lower high but a higher low: price is compressing."}[(hh, hl)]
    return {"trend": trend, "text": text, "swing_high": hi[-1], "swing_low": lo[-1]}


def near(a: dict[str, Any], price: float) -> dict[str, Any]:
    """The closest pool of resting stops and the closest open gap above and below price."""
    pools = [{"name": lv["label"], "price": lv["price"], "pool": lv["pool"]} for lv in a.get("levels") or [] if lv.get("status") == "intact"]
    pools += [{"name": "Equal highs" if e["side"] == "high" else "Equal lows", "price": e["price"], "pool": e["pool"]} for e in a.get("equal_levels") or []]
    gaps = [{"name": ("Bullish" if g["dir"] == "bull" else "Bearish") + " gap", "price": g["ce"], "top": g["top"], "bottom": g["bottom"]}
            for g in a.get("fvgs") or [] if g.get("state") != "filled"]

    def closest(items: list[dict[str, Any]], above: bool) -> dict[str, Any] | None:
        side = [x for x in items if (x["price"] > price) == above and x["price"] != price]
        x = min(side, key=lambda i: abs(i["price"] - price), default=None)
        return {**x, "away_pct": round(abs(x["price"] - price) / price * 100, 3)} if x else None
    return {"pool_above": closest(pools, True), "pool_below": closest(pools, False), "gap_above": closest(gaps, True), "gap_below": closest(gaps, False)}


ACTIVE_TTL_S = 600
CLASS_ORDER = ("synthetic", "forex", "index", "metal", "energy", "agri", "crypto", "stock")
_active: dict[str, Any] = {"at": 0.0, "rows": []}   # Deriv's market list, kept for a few minutes per warm server


def _open(connect: Callable[[str], Any] | None = None) -> Any:
    """A connection to Deriv's public WebSocket (no account or token), trying the new endpoint and then the legacy one."""
    if connect is None:
        import websocket  # websocket-client, see requirements.txt

        def connect(url: str) -> Any:
            return websocket.create_connection(url, timeout=15)
    errors = []
    for url in scanner.DERIV_URLS:
        try:
            return connect(url.format(legacy_app_id=os.environ.get("DERIV_LEGACY_APP_ID", "1089")))
        except Exception as e:   # try the next endpoint
            errors.append(f"{url.split('?')[0]}: {e}")
    raise ConnectionError("; ".join(errors))


def _call(ws: Any, reqs: dict[int, dict[str, Any]], deadline_s: float = 30) -> dict[int, dict[str, Any]]:
    """Send every request, then collect the replies by req_id."""
    for rid, body in reqs.items():
        ws.send(json.dumps({**body, "req_id": rid}))
    got: dict[int, dict[str, Any]] = {}
    end = _time.monotonic() + deadline_s
    while len(got) < len(reqs) and _time.monotonic() < end:
        msg = json.loads(ws.recv())
        if msg.get("req_id") in reqs:
            got[msg["req_id"]] = msg
    return got


def _sym(a: dict[str, Any]) -> str:
    return a.get("underlying_symbol") or a.get("symbol") or ""


def _rows(ws: Any) -> list[dict[str, Any]]:
    """Every market Deriv lists right now (active_symbols)."""
    if _active["rows"] and _time.monotonic() - _active["at"] < ACTIVE_TTL_S:
        return _active["rows"]
    msg = _call(ws, {1: {"active_symbols": "brief"}}).get(1, {})
    if "error" in msg:
        raise RuntimeError(f"Deriv: {msg['error'].get('message')}")
    rows = msg.get("active_symbols") or []
    if rows:
        _active.update(at=_time.monotonic(), rows=rows)
    return rows


def listing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every Deriv market the engine can read, grouped for a picker: all the synthetic indices first (Volatility, Jump,
    Crash/Boom, Step, Range Break, DEX, Drift...), then forex, indices, commodities, crypto and stocks."""
    out = []
    for a in rows:
        sym, kind = _sym(a), scanner.deriv_class(a)
        if not sym or kind is None or not valid_key(sym) or a.get("is_trading_suspended"):
            continue
        group = a.get("submarket_display_name") or str(a.get("submarket") or "").replace("_", " ").title() or ""
        out.append({"symbol": sym, "label": a.get("underlying_symbol_name") or a.get("display_name") or sym, "cls": kind[0],
                    "group": group, "open": bool(a.get("exchange_is_open", 1))})
    out.sort(key=lambda m: (CLASS_ORDER.index(m["cls"]) if m["cls"] in CLASS_ORDER else 99, m["group"], m["label"]))
    return out


def deriv_list(connect: Callable[[str], Any] | None = None) -> list[dict[str, Any]]:
    """The market picker's list, straight from Deriv: every market it offers, not only the ones the scan covers."""
    ws = _open(connect)
    try:
        return listing(_rows(ws))
    finally:
        try:
            ws.close()
        except Exception:
            pass


def _market(entry: dict[str, Any], rows: list[dict[str, Any]]) -> scanner.Market:
    """Our Market for a Deriv symbol: the fixed-list market when it is one of those (its hand-tuned settings and name), else
    one built from what Deriv says about it."""
    sym = _sym(entry)
    ours = {v: k for k, v in scanner.deriv_resolve(rows, scanner.MARKETS).items()}.get(sym)
    known = next((m for m in scanner.MARKETS if m.key == ours), None)
    if known is not None:
        return known
    kind = scanner.deriv_class(entry)
    if kind is None:
        raise LookupError(f"{sym} is not a kind of market the engine reads")
    name = entry.get("underlying_symbol_name") or entry.get("display_name") or sym
    return scanner.Market(sym[3:] if sym[:3] in ("frx", "cry") else sym, name, "", "Deriv", "", 0, "", kind[0], scanner.deriv_region(name))


def _fetch(symbol: str, tfs: tuple[int, ...], connect: Callable[[str], Any] | None = None) -> tuple[scanner.Market, dict[int, Any], list[Bar]]:
    """The market and its latest candles on every timeframe (plus about 70 daily candles for the weekly draw), over one
    connection. A timeframe Deriv would not answer comes back as the error."""
    ws = _open(connect)
    try:
        rows = _rows(ws)
        entry = next((a for a in rows if _sym(a) == symbol), None)
        if entry is None:
            raise LookupError(f"Deriv does not list {symbol}")
        market = _market(entry, rows)
        reqs = {100 + i: {"ticks_history": symbol, "style": "candles", "granularity": tf * 60, "count": scanner.live_bars(tf), "end": "latest"}
                for i, tf in enumerate(tfs)}
        reqs[5000] = {"ticks_history": symbol, "style": "candles", "granularity": 86400, "count": 70, "end": "latest"}
        got = _call(ws, reqs)
    finally:
        try:
            ws.close()
        except Exception:
            pass
    bars: dict[int, Any] = {}
    for i, tf in enumerate(tfs):
        msg = got.get(100 + i)
        bars[tf] = (TimeoutError("no reply from Deriv") if msg is None else
                    RuntimeError(f"Deriv: {msg['error'].get('message')}") if "error" in msg else
                    [scanner._deriv_candles(c) for c in msg.get("candles") or []])
    return market, bars, [scanner._deriv_candles(c) for c in (got.get(5000) or {}).get("candles") or []]


def _one(market: scanner.Market, symbol: str, tf: int, bars: Any, daily: list[Bar], now: datetime) -> dict[str, Any]:
    row: dict[str, Any] = {"tf": tf, "role": ROLES.get(tf, ""), "ok": False}
    try:
        if isinstance(bars, Exception):
            raise bars
        r = scanner.scan_market(market, None, now, "live", (bars, symbol), tf, daily=daily or None)
        if not r.get("ok"):
            row["error"] = r.get("error") or "no read"
            return row
        a = r.get("analysis") or {}
        price = float(r.get("last_price") or a.get("price") or 0)
        closed = scanner._closed_bars(bars, now, tf)
        row.update(ok=True, price=price, structure=structure(closed), state=conditions.label_state([x.close for x in closed]), near=near(a, price) if price else {},
                   analysis={k: a.get(k) for k in KEEP}, events=(a.get("events") or [])[-4:],
                   models=r.get("models") or {}, today={"signals": (r.get("today") or {}).get("signals") or []},
                   stale_minutes=r.get("stale_minutes"), feed=r.get("feed"))
    except Exception as e:   # one timeframe's problem must never take the page down
        log.warning("market view %s %sm failed: %s", symbol, tf, e)
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Do the timeframes agree? The daily and weekly story sets the side; the higher timeframes' structure has to confirm
    it; a lower timeframe leaning the other way is the pullback an entry is made from."""
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return {"state": "unavailable", "side": None, "text": "No timeframe could be read for this market right now."}
    story = next(((r["analysis"].get("story") or {}) for r in ok if r["analysis"].get("story")), {})
    side = {"bullish": 1, "bearish": -1}.get(story.get("bias"), 0)
    sign = lambda r: {"up": 1, "down": -1}.get((r.get("structure") or {}).get("trend"), 0)   # noqa: E731
    word = lambda s: "bullish" if s > 0 else "bearish"                                      # noqa: E731
    buy = lambda s: "buys" if s > 0 else "sells"                                            # noqa: E731
    chips = [{"tf": r["tf"], "trend": (r.get("structure") or {}).get("trend", "unclear")} for r in ok]
    if not side:
        why = story.get("stand_aside") or "no clear daily or weekly draw"
        return {"state": "none", "side": None, "chips": chips, "text": f"No story to lean on: {why}. ICT stands aside until the daily and weekly draw is clear."}
    higher = [sign(r) for r in ok if r["tf"] >= HIGHER and sign(r)]
    lower = [sign(r) for r in ok if r["tf"] < HIGHER and sign(r)]
    if any(s != side for s in higher):
        return {"state": "conflict", "side": None, "chips": chips,
                "text": f"The daily and weekly story is {word(side)}, but the {tfname([r['tf'] for r in ok if r['tf'] >= HIGHER and sign(r) and sign(r) != side])} "
                        f"structure is going the other way. The higher timeframes do not agree: stand aside until they line up."}
    if lower and all(s == side for s in lower):
        return {"state": "agree", "side": side, "chips": chips,
                "text": f"Every timeframe leans {word(side)}. Look for {buy(side)}: wait for a raid on the {'sell' if side > 0 else 'buy'}-side stops, then the shift, then enter in the gap."}
    if lower:
        return {"state": "pullback", "side": side, "chips": chips,
                "text": f"The higher timeframes lean {word(side)} but the lower ones are pulling back. Do not fight the story: the entry comes when the lower "
                        f"timeframes turn back with it (a raid of the {'sell' if side > 0 else 'buy'}-side stops, then a shift {'up' if side > 0 else 'down'})."}
    return {"state": "higher", "side": side, "chips": chips,
            "text": f"The story is {word(side)} and the higher timeframes confirm it; the lower ones have no clear direction yet. Wait for a raid and a shift before looking for {buy(side)}."}


def tfname(tfs: list[int]) -> str:
    return " and ".join(("4-hour" if t == 240 else "1-hour" if t == 60 else f"{t}-minute") for t in tfs) or "higher timeframe"


def read(symbol: str, now: datetime | None = None, tfs: tuple[int, ...] = TFS, connect: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Every timeframe of one Deriv market, read at once, with whether they agree. `symbol` is Deriv's own (R_75, BOOM500,
    frxEURUSD, cryBTCUSD...), as listed by deriv_list()."""
    now = now or datetime.now(timezone.utc)
    market, bars, daily = _fetch(symbol, tfs, connect)
    with ThreadPoolExecutor(max_workers=len(tfs)) as pool:
        rows = list(pool.map(lambda tf: _one(market, symbol, tf, bars[tf], daily, now), tfs))
    return {"key": symbol, "label": market.label, "cls": market.cls, "generated_at": now.isoformat(), "timeframes": rows, "alignment": alignment(rows)}
