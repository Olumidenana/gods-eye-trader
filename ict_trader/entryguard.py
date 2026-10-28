"""Real-time entries that keep the plan: when to fire a trade, and when not to.

The engine plans in price levels: a limit at the array, a stop beyond the raid, a target at the draw. A bot that waits
for the candle to close and then buys "at market" with the plan's distances is not trading that plan. By then price may
have run away from the entry (it would be chasing, with the stop and the target shifted off their levels), or gone far
through it toward the stop (the idea is failing). This module is the one place that decides, for the server's trader
and the browser bot alike (the JavaScript in web/template.html mirrors it line for line):

  touched   a resting limit order has been reached by the live price, so it is entered NOW, not at the next candle close
  check     whether a trade (a fill, or a touched order) may still be entered at this price: not through the stop, not past
            the first target, not chasing, not failing, and still paying the reward the plan promised
  armed_orders  the resting limits of the main model that are worth arming, from the last scan

All distances are in R (the plan's own risk, entry to stop).

Everything here fails closed: a plan or a price that is missing, not a number, or on the wrong side is a trade that is not
entered (and never an exception that stops the other plans).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

# Deriv's public price feed (no login): the same two endpoints the scanner reads candles from
PUBLIC_FEEDS = ("wss://api.derivws.com/trading/v1/options/ws/public", "wss://ws.derivws.com/websockets/v3?app_id={legacy_app_id}")

CHASE_MAX_R = 0.25     # price may have left the entry by this much in the trade's favour; more is a trade already gone
DEEP_MAX_R = 0.75      # ... or be this far through it toward the stop; more and the idea is failing, not retracing
ORDER_MAX_AGE_MIN = 90  # an order from a scan this old is not trusted without a fresh scan
MIN_RISK_FRACTION = 1e-6  # a stop closer than this share of the price is below a tick: not a plan
TICK_MAX_AGE_S = 180   # the live price must be a tick this recent (a closed market or a stalled feed has no live price)
SNAPSHOT_SLACK_S = 60  # the scan's data may end this long before the latest candle close (its age is rounded to minutes)
FEED_TIMEOUT_S = 12    # one price request may take this long in total


def _num(x: Any) -> float | None:
    """`x` as a finite number, or None (missing, a word, NaN, infinity, a boolean)."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class PublicFeed:
    """The live price of a market from Deriv's public feed, over one short-lived connection (the trading connection is for
    trading: prices come from here)."""

    def __init__(self, connect: Callable[[str], Any] | None = None, clock: Callable[[], float] = time.time,
                 timeout: float = FEED_TIMEOUT_S) -> None:
        self._connect, self._ws, self._rid, self._clock, self._timeout = connect, None, 0, clock, timeout

    def _open(self) -> Any:
        if self._ws is not None:
            return self._ws
        connect = self._connect
        if connect is None:
            import websocket  # websocket-client, see requirements.txt

            connect = lambda u: websocket.create_connection(u, timeout=10)  # noqa: E731
        errors = []
        for url in PUBLIC_FEEDS:
            try:
                self._ws = connect(url.format(legacy_app_id=os.environ.get("DERIV_LEGACY_APP_ID", "1089")))
                return self._ws
            except Exception as e:
                errors.append(str(e))
        raise ConnectionError("; ".join(errors))

    def price(self, symbol: str, max_age_s: float | None = TICK_MAX_AGE_S) -> float:
        """The latest tick of `symbol`. Raises when there is no usable price: an error from Deriv, no answer in time,
        a price that is not a positive number, or a last tick older than `max_age_s` (the market is closed or the feed
        has stalled: that is not where price is now)."""
        try:
            ws = self._open()
            self._rid += 1
            rid = self._rid
            ws.send(json.dumps({"ticks_history": symbol, "style": "ticks", "count": 1, "end": "latest", "req_id": rid}))
            end = time.monotonic() + self._timeout
            while True:
                if time.monotonic() > end:
                    raise TimeoutError(f"Deriv did not answer for {symbol} in {self._timeout:g}s")
                msg = json.loads(ws.recv())
                if not isinstance(msg, dict) or msg.get("req_id") != rid:
                    continue
                if "error" in msg:
                    raise RuntimeError((msg["error"] or {}).get("message") or "Deriv returned an error")
                hist = msg.get("history") or {}
                prices = hist.get("prices") or []
                if not prices:
                    raise RuntimeError("Deriv did not return a price")
                price = _num(prices[-1])
                if price is None or price <= 0:
                    raise RuntimeError(f"Deriv returned a price that is not a positive number: {prices[-1]!r}")
                times = hist.get("times") or []
                tick = _num(times[-1]) if times else None
                if max_age_s and tick is not None and abs(self._clock() - tick) > max_age_s:
                    raise RuntimeError(f"the last tick of {symbol} is {abs(self._clock() - tick):.0f}s old: the market is closed or the feed has stalled")
                return price
        except RuntimeError:
            raise         # Deriv answered, and the answer was no good: the connection is fine
        except BaseException:
            self.close()  # a connection that failed or was left half-read is never reused: the next price opens a new one
            raise

    def close(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        self._ws = None


class Skip(Exception):
    """A trade the plan no longer allows; the message says why, in the words shown to the trader."""


@dataclass(frozen=True)
class Check:
    ok: bool
    reason: str
    move_r: float                 # + : price has moved the trade's way since the entry, - : it is deeper than the entry
    rr_now: float | None          # the reward left to the target for a trade entered at this price, in R of the remaining stop


def _d(plan: dict[str, Any]) -> int | None:
    """+1 for a long plan, -1 for a short one, None for anything else (a plan with no direction is never traded)."""
    return {"long": 1, "short": -1}.get(plan.get("direction")) if isinstance(plan, dict) else None


def touched(plan: dict[str, Any], spot: float) -> bool:
    """A resting limit is reached when price trades down to a buy limit or up to a sell limit."""
    try:
        d, entry, s = _d(plan), _num(plan.get("entry")), _num(spot)
    except AttributeError:
        return False
    if d is None or entry is None or s is None or s <= 0:
        return False
    return (s - entry) * d <= 0


def first_target(plan: dict[str, Any]) -> float | None:
    """The plan's first target (TP1) when it has a sensible one: strictly between the entry and the final target, on the
    side of the trade. A TP1 behind the entry, or beyond the final target, is not a first target and is ignored."""
    try:
        d, entry, target = _d(plan), _num(plan.get("entry")), _num(plan.get("target"))
        tp1 = _num(plan.get("tp1"))
        if tp1 is None:
            setup = plan.get("setup")
            tp1 = _num(setup.get("tp1")) if isinstance(setup, dict) else None
    except AttributeError:
        return None
    if d is None or entry is None or target is None or tp1 is None:
        return None
    return tp1 if (tp1 - entry) * d > 0 and (target - tp1) * d > 0 else None


def check(plan: dict[str, Any], spot: float, min_rr: float = 1.5) -> Check:
    """May this plan still be entered at `spot`? `plan` needs direction, entry, stop, target (and tp1, here or in `setup`,
    when it has one). Anything missing, not a number or on the wrong side is a refusal, never an exception."""
    if not isinstance(plan, dict):
        return Check(False, "the plan is not a plan", 0.0, None)
    d, s = _d(plan), _num(spot)
    entry, stop, target = _num(plan.get("entry")), _num(plan.get("stop")), _num(plan.get("target"))
    if d is None:
        return Check(False, "the plan has no direction (long or short)", 0.0, None)
    if entry is None or stop is None or target is None:
        return Check(False, "the plan is missing its entry, stop or target, or one is not a number", 0.0, None)
    if s is None or s <= 0:
        return Check(False, f"the live price {spot!r} is not a positive number", 0.0, None)
    min_rr = _num(min_rr)
    min_rr = 1.5 if min_rr is None else min_rr
    risk = abs(entry - stop)
    if risk <= 0:
        return Check(False, "the plan has no stop distance", 0.0, None)
    if risk < abs(entry) * MIN_RISK_FRACTION:
        return Check(False, f"the stop is {risk:g} from the entry: closer than a tick", 0.0, None)
    move = (s - entry) * d / risk
    if (entry - stop) * d <= 0:
        return Check(False, f"the stop {stop:g} is on the wrong side of the entry {entry:g} for a {plan['direction']}", move, None)
    if (target - entry) * d <= 0:
        return Check(False, f"the target {target:g} is on the wrong side of the entry {entry:g} for a {plan['direction']}", move, None)
    if (s - stop) * d <= 0:
        return Check(False, f"price {s:g} is already through the stop {stop:g}", move, None)
    if (target - s) * d <= 0:
        return Check(False, f"price {s:g} is already at the target {target:g}", move, None)
    tp1 = first_target(plan)
    if tp1 is not None and (tp1 - s) * d <= 0:
        return Check(False, f"price {s:g} is already past the first target {tp1:g}", move, None)
    rr_now = (target - s) * d / ((s - stop) * d)
    if move > CHASE_MAX_R:
        return Check(False, f"price has already run {move:.2f}R from the entry {entry:g}: not chasing it", move, rr_now)
    if move < -DEEP_MAX_R:
        return Check(False, f"price is {-move:.2f}R through the entry {entry:g} toward the stop: the idea is failing, not retracing", move, rr_now)
    if rr_now < min_rr:
        return Check(False, f"only {rr_now:.1f}R is left to the target from {s:g}, under the {min_rr:g}R the plan needs", move, rr_now)
    return Check(True, "price is still at the plan's entry", move, rr_now)


# ---------------------------------------------------------------- which resting limits may be armed
def armed_id(market: Any, placed: Any) -> str:
    """The id of a resting limit as a plan. The fill that comes of it carries the same `market` and `setup.mss_time`."""
    return f"armed:{market}:{placed}"


def trade_key(plan: dict[str, Any]) -> str:
    """One key for one trade, whichever way it was found: a resting limit and the fill it becomes are the same trade."""
    setup = plan.get("setup")
    mss = setup.get("mss_time") if isinstance(setup, dict) else None
    if plan.get("armed") or mss is None or plan.get("market") is None:
        return str(plan.get("id"))
    return armed_id(plan["market"], mss)


def last_close_before(now: datetime, timeframe_minutes: int) -> datetime:
    """The latest candle close of this timeframe strictly before `now` (at the very instant of a close, the candle that is
    closing is not counted yet)."""
    step = max(1, int(timeframe_minutes)) * 60
    ts = now.timestamp()
    close = math.floor(ts / step) * step
    if close >= ts:
        close -= step
    return datetime.fromtimestamp(close, timezone.utc)


def snapshot_is_current(generated_at: datetime, now: datetime, timeframe_minutes: int, stale_minutes: float = 0) -> bool:
    """Whether a scan has seen the latest candle close of the plan's timeframe.

    The engine cancels a resting limit when a candle CLOSES through its rules or its candles run out, and a scan only
    learns that after the close. In the minute after a close, before the new scan is saved, the saved one is older than
    that close: the order it still lists may already be dead. So orders are armed only from a scan generated after the
    latest close (and whose market data reached it: `stale_minutes` is how old the market's last candle was at the scan)."""
    latest = last_close_before(now, timeframe_minutes)
    data_through = generated_at - timedelta(minutes=max(0.0, stale_minutes))
    return generated_at >= latest and data_through >= latest - timedelta(seconds=SNAPSHOT_SLACK_S)


def _when(text: Any) -> datetime | None:
    try:
        at = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def armed_orders(report: dict[str, Any], now: datetime | None = None) -> list[dict[str, Any]]:
    """The resting limit orders of each market's main model in the last scan, as plans: the ones a trader would arm.

    An order is dropped when the scan is too old to trust, when the scan has not seen the latest candle close of the
    order's timeframe (`snapshot_is_current`: the order may have been cancelled by that very candle), when the order's
    candles have run out, or when it cannot be identified or is malformed. One bad market or order never hides the others."""
    out: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return out
    at = _when(report.get("generated_at"))
    if at is None:
        return out
    now = now or at
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if now - at > timedelta(minutes=ORDER_MAX_AGE_MIN) or at - now > timedelta(minutes=2):
        return out
    seen: set[str] = set()
    for m in report.get("markets") or []:
        try:
            if not isinstance(m, dict) or not m.get("ok") or not m.get("broker_symbol"):
                continue
            main = m.get("model") or "ict2022"
            tf = int(m.get("timeframe_minutes") or report.get("timeframe_minutes") or 5)
            stale = _num(m.get("stale_minutes")) or 0.0
            if tf <= 0 or not snapshot_is_current(at, now, tf, stale):
                continue
            base = last_close_before(at + timedelta(seconds=1), tf)   # the candle close this scan's orders were counted from
            key = m.get("key") or m.get("label")
            orders = ((m.get("models") or {}).get(main) or {}).get("orders") or []
        except (AttributeError, TypeError, ValueError):
            continue
        for o in orders:
            try:
                left = int(o.get("candles_left") or 0)
                placed = o.get("placed")
                if left <= 0 or now > base + timedelta(minutes=left * tf) or placed is None or key is None:
                    continue
                oid = armed_id(key, placed)
                direction = o["direction"]
                entry, stop, target = _num(o["entry"]), _num(o["stop"]), _num(o["tp2"])
                if oid in seen or direction not in ("long", "short") or entry is None or stop is None or target is None:
                    continue
                seen.add(oid)
                plan = {"id": oid, "armed": True, "market": key, "label": m["label"], "cls": m.get("cls"),
                        "broker_symbol": m["broker_symbol"], "direction": direction, "entry": entry, "stop": stop,
                        "target": target, "tp1": _num(o.get("tp1")), "rr": _num(o.get("rr")) or 0.0,
                        "quality": o.get("quality"), "advice": o.get("advice"), "timeframe_minutes": tf, "placed": placed,
                        "management": o.get("management") or m.get("management"), "setup": {"tp1": o.get("tp1"), "mss_time": placed}}
                try:  # the strict-only users' filter applies to a resting limit too, judged at the moment it would be entered
                    from .strict import mark
                    probe = {**plan, "time": now.isoformat()}
                    mark(probe)
                    plan["strict"], plan["strict_why"] = probe["strict"], probe["strict_why"]
                except Exception:
                    plan["strict"], plan["strict_why"] = False, ["could not be judged"]
                out.append(plan)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
    return out


def verdict_ok(plan: dict[str, Any], need: str = "TAKE") -> bool:
    """The matrix says pull the trigger. `need` TAKE (default) or CAUTION (also accepts a size-down call)."""
    advice = plan.get("advice") if isinstance(plan, dict) else None
    v = advice.get("verdict") if isinstance(advice, dict) else None
    return v == "TAKE" or (need == "CAUTION" and v == "CAUTION")
