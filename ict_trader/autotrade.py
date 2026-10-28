"""Always-on auto-trading: places the engine's signals on users' own Deriv accounts.

Nothing here runs unless the operator enables it (AUTOTRADE_ENABLED=1) and a user has
explicitly enrolled: read and accepted the risk terms, chosen an account, and given an
API token. Each run (after the minute scan) does, for every enrolled user:

  1. connect to their Deriv account (API token -> one-time password -> WebSocket)
  2. read the balance and open positions; pause for the day at the daily loss limit
  3. place each fresh signal in their chosen markets as a multiplier contract, sized so
     hitting the engine's stop loses their chosen risk, with stop-loss and take-profit
     attached (Deriv holds them, so nothing has to stay connected)
  4. log every placement or skip, with the reason

Tokens are stored encrypted (Fernet, key in AUTOTRADE_KEY) and are never logged or
returned. A user manages their enrollment with a random manage key that only they hold
(stored hashed). Real-money accounts need a second, explicit confirmation.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from . import entryguard
from typing import Any, Callable, Protocol

from .sessions import to_ny

API_BASE = "https://api.derivws.com/trading/v1/options"
TERMS_VERSION = "2026-10-07"
TERMS = [
    "Trading leveraged products such as multipliers, CFDs and synthetic indices carries a high risk of losing money. "
    "Most retail accounts lose money. You can lose all the money in your account.",
    "ICT Flux is software that follows fixed rules. It is not financial advice, and it is not a licensed investment "
    "adviser, broker or portfolio manager. Past results, backtests and demo results do not guarantee future results.",
    "By turning this on you authorise ICT Flux to place, and attach stop-loss and take-profit to, trades on the Deriv "
    "account you choose, automatically and without asking you each time, using the settings you set.",
    "Automation can fail: missed or late signals, internet or server outages, Deriv rejecting or changing orders, "
    "slippage and price gaps. You are responsible for monitoring your account and for every trade on it.",
    "Your Deriv API token is stored encrypted and used only to trade for you. Give it only the Trade and Account "
    "management permissions. You can stop at any time here, and revoke the token on Deriv to cut access immediately.",
    "Start on a demo account. Use a real-money account only with money you can afford to lose, and only where "
    "trading these products is legal for you.",
]
CLASSES = ("forex", "metal", "index", "crypto", "energy", "agri", "synthetic", "stock", "strict")  # "strict": ICT strict setups only
LIMITS = {"risk_pct": (0.1, 2.0), "max_trades_day": (1, 10), "max_open": (1, 10), "day_loss_pct": (0.5, 10.0), "min_rr": (1.5, 10.0)}
GRADES = ("A+", "A", "B", "C")
FRESH_MINUTES = 6  # only signals this recent are placed (the browser bot uses the same window)


# ---------------------------------------------------------------- secrets
def _fernet():
    from cryptography.fernet import Fernet  # requirements.txt

    key = os.environ.get("AUTOTRADE_KEY", "")
    if not key:
        raise RuntimeError("AUTOTRADE_KEY is not set")
    return Fernet(key.encode())


def encrypt(token: str) -> str:
    return _fernet().encrypt(token.encode()).decode()


def decrypt(blob: str) -> str:
    return _fernet().decrypt(blob.encode()).decode()


def hash_key(k: str) -> str:
    return hashlib.sha256(k.encode()).hexdigest()


# ---------------------------------------------------------------- storage
class Store(Protocol):
    def insert_user(self, row: dict) -> dict: ...
    def get_user(self, user_id: str) -> dict | None: ...
    def update_user(self, user_id: str, fields: dict) -> None: ...
    def delete_user(self, user_id: str) -> None: ...
    def enabled_users(self) -> list[dict]: ...
    def add_log(self, row: dict) -> None: ...
    def logs(self, user_id: str, limit: int = 50, kind: str | None = None) -> list[dict]: ...
    def logged_signal(self, user_id: str, signal_id: str) -> bool: ...
    def count_claims(self, user_id: str, signal_id: str) -> int: ...


class SupabaseStore:
    """Supabase (Postgres) over its REST API with the service key; see db/autotrade.sql."""

    def __init__(self, url: str | None = None, key: str | None = None) -> None:
        self.url = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = key or os.environ.get("SUPABASE_SERVICE_KEY", "")
        if not self.url or not self.key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    def _req(self, method: str, path: str, body: Any = None, prefer: str | None = None) -> Any:
        headers = {"apikey": self.key, "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        if prefer:
            headers["Prefer"] = prefer
        req = urllib.request.Request(f"{self.url}/rest/v1/{path}", method=method, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(_db_error(e.code, e.read())) from None
        return json.loads(raw) if raw else None

    def insert_user(self, row: dict) -> dict:
        return self._req("POST", "autotrade_users", row, prefer="return=representation")[0]

    def get_user(self, user_id: str) -> dict | None:
        rows = self._req("GET", f"autotrade_users?id=eq.{user_id}&select=*")
        return rows[0] if rows else None

    def update_user(self, user_id: str, fields: dict) -> None:
        self._req("PATCH", f"autotrade_users?id=eq.{user_id}", fields)

    def delete_user(self, user_id: str) -> None:
        self._req("DELETE", f"autotrade_users?id=eq.{user_id}")

    def enabled_users(self) -> list[dict]:
        return self._req("GET", "autotrade_users?enabled=eq.true&select=*")

    def add_log(self, row: dict) -> None:
        self._req("POST", "autotrade_log", row)

    def logs(self, user_id: str, limit: int = 50, kind: str | None = None) -> list[dict]:
        """The user's log, newest first. The 'claimed' rows (see claim()) are bookkeeping, not activity: left out unless asked for."""
        from urllib.parse import quote
        which = f"&kind=eq.{quote(kind)}" if kind else "&kind=neq.claimed"
        return self._req("GET", f"autotrade_log?user_id=eq.{user_id}{which}&select=*&order=created_at.desc&limit={int(limit)}")

    def logged_signal(self, user_id: str, signal_id: str) -> bool:
        from urllib.parse import quote
        return bool(self._req("GET", f"autotrade_log?user_id=eq.{user_id}&signal_id=eq.{quote(signal_id, safe='')}&select=id&limit=1"))

    def count_claims(self, user_id: str, signal_id: str) -> int:
        from urllib.parse import quote
        return len(self._req("GET", f"autotrade_log?user_id=eq.{user_id}&signal_id=eq.{quote(signal_id, safe='')}&kind=eq.claimed&select=id&limit=5") or [])


def _db_error(code: int, raw: bytes) -> str:
    """Supabase's own reason, plus the usual fix, instead of a bare 'HTTP Error 400'."""
    try:
        d = json.loads(raw)
        msg = " ".join(str(d.get(k)) for k in ("code", "message", "hint") if d.get(k))
    except Exception:
        msg = raw[:200].decode("utf-8", "replace")
    fix = ""
    if code in (401, 403):
        fix = " Check SUPABASE_SERVICE_KEY in Vercel (the service_role key), then redeploy."
    elif code in (400, 404) and ("PGRST2" in msg or "column" in msg or "relation" in msg or "table" in msg):
        fix = " Run db/autotrade.sql again in the Supabase SQL editor."
    return f"Database error {code}: {msg}.{fix}"


class MemoryStore:
    """In-memory store for tests and local runs."""

    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.log: list[dict] = []

    def insert_user(self, row: dict) -> dict:
        row = {**row, "id": row.get("id") or secrets.token_hex(8)}
        self.users[row["id"]] = row
        return row

    def get_user(self, user_id: str) -> dict | None:
        return self.users.get(user_id)

    def update_user(self, user_id: str, fields: dict) -> None:
        self.users[user_id].update(fields)

    def delete_user(self, user_id: str) -> None:
        self.users.pop(user_id, None)
        self.log = [x for x in self.log if x["user_id"] != user_id]

    def enabled_users(self) -> list[dict]:
        return [u for u in self.users.values() if u.get("enabled")]

    def add_log(self, row: dict) -> None:
        self.log.append({**row, "created_at": row.get("created_at") or datetime.now(timezone.utc).isoformat()})

    def logs(self, user_id: str, limit: int = 50, kind: str | None = None) -> list[dict]:
        return [x for x in reversed(self.log) if x["user_id"] == user_id and (x.get("kind") == kind if kind else x.get("kind") != "claimed")][:limit]

    def logged_signal(self, user_id: str, signal_id: str) -> bool:
        return any(x["user_id"] == user_id and x.get("signal_id") == signal_id for x in self.log)

    def count_claims(self, user_id: str, signal_id: str) -> int:
        return sum(1 for x in self.log if x["user_id"] == user_id and x.get("signal_id") == signal_id and x.get("kind") == "claimed")


def store_from_env() -> Store:
    return SupabaseStore()


# ---------------------------------------------------------------- Deriv
def _http(url: str, method: str = "GET", token: str = "", body: Any = None) -> Any:
    headers = {"Authorization": f"Bearer {token}", "Deriv-App-ID": os.environ.get("DERIV_APP_ID", "34BbtJiEX2xD1Vqy1Kjd6")}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, method=method, headers=headers, data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", {}).get("message")
        except Exception:
            msg = None
        raise PermissionError(msg or f"Deriv returned {e.code}") from None


def _is_demo(a: dict, acct: str) -> bool:
    """Same rule as the browser bot: Deriv marks demo accounts in account_type/type, and their ids
    start with VR (older accounts) or DOT (demo options trading; real ones start with ROT)."""
    kind = str(a.get("account_type") or a.get("type") or "").lower()
    return "demo" in kind or "virtual" in kind or bool(a.get("is_virtual")) or acct.upper().startswith(("VR", "DOT"))


def deriv_accounts(token: str) -> list[dict]:
    """The accounts a token can trade: [{id, type, currency, balance}]."""
    data = _http(f"{API_BASE}/accounts", token=token)
    rows = data.get("data", data) if isinstance(data, dict) else data
    rows = rows if isinstance(rows, list) else rows.get("accounts", []) if isinstance(rows, dict) else []
    out = []
    for a in rows:
        acct = a.get("account_id") or a.get("id") or a.get("loginid")
        kind = "demo" if _is_demo(a, str(acct)) else "real"
        out.append({"id": str(acct), "type": kind, "currency": a.get("currency") or "USD", "balance": a.get("balance")})
    return out


class Trader(Protocol):
    currency: str
    def balance(self) -> float: ...
    def open_positions(self) -> int: ...
    def place(self, signal: dict, risk: float, spot: float | None = None) -> dict: ...
    def close(self) -> None: ...


class DerivTrader:
    """One short-lived trading connection (same order logic as the browser bot)."""

    def __init__(self, token: str, account_id: str, account_type: str, currency: str = "USD", connect: Callable | None = None) -> None:
        otp = _http(f"{API_BASE}/accounts/{account_id}/otp", method="POST", token=token)
        d = otp.get("data", otp)
        url = d.get("url") or d.get("websocket_url") or d.get("ws_url")
        if not url and (d.get("otp") or d.get("token")):
            url = f"{API_BASE}/ws/{'real' if account_type == 'real' else 'demo'}?otp={d.get('otp') or d.get('token')}".replace("https://", "wss://")
        if not url:
            raise PermissionError("Deriv did not return a trading connection")
        if connect is None:
            import websocket

            connect = lambda u: websocket.create_connection(u, timeout=15)  # noqa: E731
        self.ws = connect(url)
        self.currency = currency
        self._rid = 0

    def _send(self, req: dict) -> dict:
        self._rid += 1
        rid = self._rid
        self.ws.send(json.dumps({**req, "req_id": rid}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("req_id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"].get("message") or msg["error"].get("code"))
                return msg

    def _send_sym(self, req: dict, symbol: str) -> dict:
        try:
            return self._send({**req, "underlying_symbol": symbol})
        except RuntimeError as e:
            if any(w in str(e).lower() for w in ("symbol", "propert", "validation")):
                return self._send({**req, "symbol": symbol})
            raise

    def balance(self) -> float:
        b = self._send({"balance": 1}).get("balance", {})
        self.currency = b.get("currency") or self.currency
        return float(b.get("balance", 0))

    def open_positions(self) -> int:
        return len(self._send({"portfolio": 1}).get("portfolio", {}).get("contracts", []))

    def spot(self, symbol: str) -> float:
        """The latest price from Deriv's public feed (the trading connection is for trading), for entering the plan where it
        stands, not where the candle that made it closed."""
        if getattr(self, "_feed", None) is None:
            self._feed = entryguard.PublicFeed()
        return self._feed.price(symbol)

    def place(self, signal: dict, risk: float, spot: float | None = None) -> dict:
        sym = signal["broker_symbol"]
        if signal.get("direction") not in ("long", "short"):
            raise RuntimeError(f"signal has no direction (long or short): {signal.get('direction')!r}")
        up = signal["direction"] == "long"
        ctype = "MULTUP" if up else "MULTDOWN"
        try:
            cf = self._send({"contracts_for": sym, "currency": self.currency})
        except RuntimeError:
            cf = self._send_sym({"contracts_for": 1, "currency": self.currency}, sym)
        avail = next((c for c in cf.get("contracts_for", {}).get("available", []) if c.get("contract_type") == ctype), None)
        if not avail:
            raise RuntimeError(f"Deriv doesn't offer multipliers on {signal['label']}")
        p = sizing(signal, risk, [float(m) for m in avail.get("multiplier_range") or [10, 20, 30, 50, 100]],
                   float(avail.get("min_stake") or 1), spot)
        prop = self._send_sym({"proposal": 1, "amount": p["stake"], "basis": "stake", "contract_type": ctype, "currency": self.currency,
                               "multiplier": p["multiplier"], "limit_order": {"stop_loss": p["stop_loss"], "take_profit": p["take_profit"]}}, sym)
        pid = prop.get("proposal", {}).get("id")
        if not pid:
            raise RuntimeError("Deriv did not return a price for this trade")
        b = self._send({"buy": pid, "price": p["stake"]}).get("buy", {})
        return {**p, "contract_id": b.get("contract_id"), "price": b.get("buy_price")}

    def open_contracts(self) -> list[dict]:
        """Open contracts with their live profit and stop/target amounts."""
        out = []
        for c in self._send({"portfolio": 1}).get("portfolio", {}).get("contracts", []):
            poc = self._send({"proposal_open_contract": 1, "contract_id": c["contract_id"]}).get("proposal_open_contract", {})
            if poc and not poc.get("is_sold"):
                lo = poc.get("limit_order") or {}
                out.append({"contract_id": str(poc.get("contract_id")), "profit": float(poc.get("profit") or 0),
                            "stop_loss": abs(float((lo.get("stop_loss") or {}).get("order_amount") or 0)),
                            "take_profit": float((lo.get("take_profit") or {}).get("order_amount") or 0)})
        return out

    def update_stop(self, contract_id: str, stop_loss: float, take_profit: float) -> None:
        self._send({"contract_update": 1, "contract_id": int(contract_id),
                    "limit_order": {"stop_loss": stop_loss, "take_profit": take_profit}})

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass
        if getattr(self, "_feed", None) is not None:
            self._feed.close()


def stop_step(profit: float, take_profit: float, stop_now: float, stop0: float, trim: bool, be: bool = True) -> float | None:
    """ICT stop management on a multiplier contract, in money (SL-04, SL-05): halfway to the
    take-profit the stop's risk is cut by 25%; at 75% it goes to (near) entry. Deriv needs a
    positive stop, so "entry" is a stop of 5% of the original risk (at least 0.10). Returns the
    new stop amount, or None when nothing changes. Stops only ever tighten."""
    if take_profit <= 0 or stop0 <= 0:
        return None
    progress = profit / take_profit
    want = None
    if be and progress >= 0.75:
        want = max(0.10, round(0.05 * stop0, 2))
    elif trim and progress >= 0.5:
        want = round(0.75 * stop0, 2)
    return want if want is not None and want < stop_now - 0.005 else None


_PLACED_SL = __import__("re").compile(r"SL ([0-9.]+) /")
_PLACED_LEG = __import__("re").compile(r" (TP1|TP2) · ")   # which leg of a plan a "placed" log row is


def manage_stops(store: "Store", u: dict, t: Any, summary: dict) -> None:
    """Move the stops of this user's open contracts that the bot placed, per their ICT plan."""
    placed = {x.get("contract_id"): x for x in _logs(store, u["id"], 400, "placed") if x.get("contract_id")}
    if not placed:
        return
    legs: dict[str, dict[str, str]] = {}   # signal id -> {"TP1": contract id, "TP2": contract id}: the two legs of one plan
    for cid, x in placed.items():
        lg = _PLACED_LEG.search(x.get("message", ""))
        if lg and x.get("signal_id"):
            legs.setdefault(x["signal_id"], {})[lg.group(1)] = cid
    contracts = t.open_contracts()
    open_ids = {c["contract_id"] for c in contracts}
    for c in contracts:
        row = placed.get(c["contract_id"])
        m = _PLACED_SL.search((row or {}).get("message", ""))
        if not row or not m:
            continue  # not ours, or placed before stop management existed
        new = stop_step(c["profit"], c["take_profit"], c["stop_loss"], float(m.group(1)), "trim 25%" in row["message"])
        why = "near entry, 75% of the way" if new is not None and new < 0.5 * float(m.group(1)) else "risk cut 25%, halfway"
        # TP-04: once the TP1 leg is banked, the runner's stop goes to entry whatever its own progress to the target is
        sib = legs.get(row.get("signal_id"), {})
        if sib.get("TP2") == c["contract_id"] and sib.get("TP1") and sib["TP1"] not in open_ids:
            be = max(0.10, round(0.05 * float(m.group(1)), 2))
            if be < c["stop_loss"] - 0.005 and (new is None or be < new):
                new, why = be, "near entry, TP1 banked"
        if new is None:
            continue
        try:
            t.update_stop(c["contract_id"], new, c["take_profit"])
            store.add_log({"user_id": u["id"], "kind": "managed", "message": f"{row.get('market') or 'Trade'}: stop moved to {new} ({why})",
                           "signal_id": row.get("signal_id"), "market": row.get("market"), "contract_id": c["contract_id"]})
            summary["managed"] = summary.get("managed", 0) + 1
        except Exception as e:
            msg = f"Could not move the stop on {row.get('market') or 'a trade'}: {e}"
            if not _said_lately(store, u["id"], "error", msg, datetime.now(timezone.utc)):   # said once, not at every run
                store.add_log({"user_id": u["id"], "kind": "error", "message": msg, "contract_id": c["contract_id"]})


MIN_STOP_FRACTION = 0.9   # Deriv's stop-loss may fire at this share of the plan's stop distance at the least (see sizing)


def _finite(x: Any) -> float | None:
    return entryguard._num(x)


def sizing(signal: dict, risk: float, multipliers: list[float], min_stake: float, spot: float | None = None) -> dict:
    """Stake and multiplier so a move to the engine's stop LEVEL loses `risk` (Deriv caps the loss at the stake, so the
    largest multiplier that keeps stake >= risk is used), and the take-profit sits at the plan's target LEVEL.

    The trade is entered at `spot` (the live price), not at the plan's entry: the distances are measured from where the
    trade really starts, so the stop stays beyond the raid and the target stays at the draw however far price has
    moved since the plan was made. Without a live price the plan's entry stands in for it.

    Money safety. The loss is capped by the stop-loss amount, which is never more than `risk` (nor the stake), so the
    contract cannot lose more than intended. The danger is the other way: a minimum stake (or no multiplier small enough)
    forces a stake larger than the risk needs, and the stop-loss amount then fires at a fraction of the plan's stop
    distance: stopped out by noise, inside the raid the plan put its stop beyond. So the multiplier is also lowered until
    the stake reaches the minimum, and when even that cannot keep the stop within MIN_STOP_FRACTION (0.9, or the environment's
    AUTOTRADE_MIN_STOP_FRACTION) of the plan's stop distance, the
    trade is refused (RuntimeError, logged) rather than entered with a different stop. A plan on the wrong side of
    the price (spot through the stop or the target), a non-number or a zero distance is refused the same way."""
    direction = {"long": 1, "short": -1}.get(signal.get("direction"))
    if signal.get("direction") is not None and direction is None:
        raise RuntimeError(f"signal has no direction (long or short): {signal.get('direction')!r}")
    entry = _finite(spot if spot is not None else signal.get("entry"))
    stop, target, risk = _finite(signal.get("stop")), _finite(signal.get("target")), _finite(risk)
    if entry is None or entry <= 0 or stop is None or target is None:
        raise RuntimeError("signal has a missing or non-numeric entry, stop or target")
    if risk is None or risk <= 0:
        raise RuntimeError("the risk per trade is zero or not a number")
    dist = abs(entry - stop)
    if dist <= 0:
        raise RuntimeError("signal has no stop distance")
    if direction is not None:
        if (entry - stop) * direction <= 0:
            raise RuntimeError(f"price {entry:g} is already through the stop {stop:g}")
        if (target - entry) * direction <= 0:
            raise RuntimeError(f"price {entry:g} is already at or past the target {target:g}")
    ms = sorted({m for m in (_finite(x) for x in multipliers) if m is not None and m > 0})
    if not ms:
        raise RuntimeError("Deriv returned no usable multipliers")
    floor = max(0.0, _finite(min_stake) or 0.0)
    ratio = entry / dist
    # the largest multiplier whose stake covers the risk (m <= entry/dist) and reaches the minimum stake
    fit = [m for m in ms if m <= ratio and risk * ratio / m >= floor]
    mult = fit[-1] if fit else ms[0]
    stake = round(max(floor, risk * ratio / mult), 2)
    stop_loss = round(min(risk, stake), 2)
    take_profit = round(stop_loss * abs(target - entry) / dist, 2)
    if stake <= 0 or stop_loss <= 0 or take_profit <= 0:
        raise RuntimeError(f"the trade rounds to nothing (stake {stake}, stop-loss {stop_loss}, take-profit {take_profit}): the risk is too small")
    reaches = stop_loss * entry / (stake * mult)       # how far from the entry the stop-loss amount really fires, in price
    least = _finite(os.environ.get("AUTOTRADE_MIN_STOP_FRACTION")) or MIN_STOP_FRACTION   # 0.9; a lower value accepts a tighter stop than the plan's
    if reaches < least * dist:
        raise RuntimeError(f"refused: with the minimum stake {floor:g} and x{mult:g} the stop-loss of {stop_loss:g} fires {reaches / dist:.0%} of the way "
                           f"to the plan's stop, inside the raid it sits beyond; raise the risk per trade or the balance")
    return {"stake": stake, "multiplier": mult, "stop_loss": stop_loss, "take_profit": take_profit, "stop_reach": round(reaches / dist, 4)}


# ---------------------------------------------------------------- enrollment
@dataclass
class Settings:
    risk_pct: float = 1.0
    max_trades_day: int = 2
    max_open: int = 3
    day_loss_pct: float = 2.0
    classes: tuple[str, ...] = ("forex", "metal", "index")
    min_rr: float = 2.0         # only take setups paying at least this
    min_grade: str = "A"        # strict default: only take A and A+ setups (A+ best)

    @classmethod
    def parse(cls, body: dict) -> "Settings":
        s = cls()
        for k, (lo, hi) in LIMITS.items():
            if k in body:
                v = float(body[k])
                if not lo <= v <= hi:
                    raise ValueError(f"{k} must be between {lo} and {hi}")
                setattr(s, k, int(v) if k in ("max_trades_day", "max_open") else v)
        if "min_grade" in body:
            if body["min_grade"] not in GRADES:
                raise ValueError(f"min_grade must be one of {GRADES}")
            s.min_grade = body["min_grade"]
        if "classes" in body:
            c = tuple(x for x in body["classes"] if x in CLASSES)
            if not c:
                raise ValueError("choose at least one market group")
            s.classes = c
        return s

    def row(self) -> dict:
        return {"risk_pct": self.risk_pct, "max_trades_day": self.max_trades_day, "max_open": self.max_open,
                "day_loss_pct": self.day_loss_pct, "classes": list(self.classes), "min_rr": self.min_rr, "min_grade": self.min_grade}


def enroll(store: Store, body: dict, accounts: Callable[[str], list[dict]] = deriv_accounts,
           now: datetime | None = None) -> dict:
    """Create an enrollment. Requires the terms, a valid token, and (for real money) an
    explicit second confirmation. Returns the id and the manage key (shown once)."""
    now = now or datetime.now(timezone.utc)
    if body.get("accept_terms") is not True or body.get("terms_version") != TERMS_VERSION:
        raise ValueError("you must read and accept the current risk terms")
    token = str(body.get("token", "")).strip()
    if len(token) < 8:
        raise ValueError("a Deriv API token is required")
    acct = next((a for a in accounts(token) if a["id"] == str(body.get("account_id"))), None)
    if acct is None:
        raise ValueError("that account was not found for this token")
    if acct["type"] == "real" and body.get("confirm_real") != "REAL":
        raise ValueError("real-money accounts need the separate confirmation (type REAL)")
    s = Settings.parse(body)
    manage = secrets.token_urlsafe(24)
    row = store.insert_user({
        "token_enc": encrypt(token), "account_id": acct["id"], "account_type": acct["type"], "currency": acct["currency"],
        **s.row(), "enabled": True, "manage_hash": hash_key(manage), "terms_version": TERMS_VERSION,
        "accepted_terms_at": now.isoformat(), "real_confirmed_at": now.isoformat() if acct["type"] == "real" else None,
        "day": None, "day_start_balance": None, "paused_reason": None,
        **({"access_code": body["access_code"]} if body.get("access_code") else {}),
    })
    store.add_log({"user_id": row["id"], "kind": "enrolled", "message": f"Auto-trading on for {acct['type']} account {acct['id']}"})
    return {"id": row["id"], "manage_key": manage, "account_id": acct["id"], "account_type": acct["type"]}


def authorize(store: Store, user_id: str, manage_key: str) -> dict:
    u = store.get_user(user_id) if user_id else None
    if not u or not secrets.compare_digest(u["manage_hash"], hash_key(manage_key or "")):
        raise PermissionError("not found")
    return u


def public(u: dict) -> dict:
    """What a user may see about their own enrollment (never the token)."""
    keys = ("id", "account_id", "account_type", "currency", "risk_pct", "max_trades_day", "max_open", "day_loss_pct",
            "classes", "enabled", "terms_version", "accepted_terms_at", "paused_reason", "day_start_balance", "min_rr", "min_grade")
    return {k: u.get(k) for k in keys}


# ---------------------------------------------------------------- the run
def fresh_signals(report: dict, now: datetime) -> list[dict]:
    """The fills of the last few minutes that are still open. A malformed market or signal is skipped, never fatal."""
    out = []
    for m in (report.get("markets") or []) if isinstance(report, dict) else []:
        try:
            if not m.get("ok"):
                continue
            for s in (m.get("today") or {}).get("signals") or []:
                try:
                    t = datetime.fromisoformat(s["time"])
                    t = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
                    age = (now - t).total_seconds() / 60
                    if s.get("broker_symbol") and s["outcome"]["status"] == "open" and age <= max(FRESH_MINUTES, 1.2 * s.get("timeframe_minutes", 5)):
                        out.append({**s, "cls": m.get("cls")})
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
        except AttributeError:
            continue
    return out


def min_verdict() -> str:
    """Which adviser verdicts may be traded: TAKE ("pull the trigger") only, unless AUTOTRADE_MIN_VERDICT=CAUTION."""
    return "CAUTION" if os.environ.get("AUTOTRADE_MIN_VERDICT", "TAKE").upper() == "CAUTION" else "TAKE"


def claim(store: "Store", user_id: str, plan: dict) -> bool:
    """Reserve a trade for this run BEFORE it is placed. True: it is ours, go ahead; False: do not enter it.

    A plan found twice (the minute run and the 5-minute run overlap; two cron hits race; a resting limit and the fill it
    becomes) must be entered once. Nothing can be atomic over Supabase's REST, so the protection is: write a 'claimed' row
    under the trade's key (entryguard.trade_key) first, treat any row under that key as taken (logged_signal), then read the
    claims back and go ahead only if ours is the only one. Two runs that claim at the same moment each see two, and both
    stand down: a missed trade, never a double one. A run that dies after placing and before logging leaves its claim, so
    the next run does not place it again either. If the claim cannot be written the trade is not entered.

    With a unique partial index on the claim rows the database itself refuses the second claim (see the note in db/autotrade.sql
    the owner can add), which makes this atomic: exactly one run wins instead of both standing down."""
    key = entryguard.trade_key(plan)
    try:
        store.add_log({"user_id": user_id, "kind": "claimed", "message": f"Reserved {plan.get('label') or key} so it is entered once",
                       "signal_id": key, "market": plan.get("label")})
    except RuntimeError as e:
        if "23505" in str(e) or "duplicate key" in str(e).lower():
            return False        # a unique index on (user_id, signal_id) where kind = 'claimed' (db/autotrade.sql) made the claim atomic: someone else holds it
        raise
    counter = getattr(store, "count_claims", None)
    return counter is None or counter(user_id, key) == 1


def _logs(store: "Store", user_id: str, limit: int, kind: str | None = None) -> list[dict]:
    """The user's recent log rows of one kind, newest first (a store written before `kind` existed is filtered here)."""
    try:
        return store.logs(user_id, limit, kind=kind) if kind else store.logs(user_id, limit)
    except TypeError:
        return [x for x in store.logs(user_id, limit) if not kind or x.get("kind") == kind]


def _stamp(iso: Any) -> datetime | None:
    try:
        at = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def _said_lately(store: "Store", user_id: str, kind: str, message: str, now: datetime, minutes: float = 30) -> bool:
    """Whether the same line was logged in the last `minutes`: a problem that lasts is said once, not every minute."""
    try:
        for x in _logs(store, user_id, 30):
            if x.get("kind") == kind and x.get("message") == message:
                at = _stamp(x.get("created_at"))
                if at is None or now - at <= timedelta(minutes=minutes):
                    return True
    except Exception:
        return False
    return False


class _Prices:
    """One live price per market per few seconds for the whole run (every user shares it), and a failure is remembered
    too: a market with no price costs one request, not one per user."""

    def __init__(self, source: Any, ttl: float = 5.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._source = source.price if hasattr(source, "price") else source
        self._ttl, self._clock, self._seen = ttl, clock, {}

    def __call__(self, symbol: str) -> float:
        now, hit = self._clock(), self._seen.get(symbol)
        if hit is None or now - hit[0] > self._ttl:
            try:
                hit = (now, self._source(symbol), None)
            except Exception as e:
                hit = (now, None, e)
            self._seen[symbol] = hit
        if hit[2] is not None:
            raise hit[2]
        return hit[1]

    def fresh(self, symbol: str) -> float:
        """The price right now, whatever was seen a moment ago (for the last look before an order goes out)."""
        self._seen.pop(symbol, None)
        return self(symbol)


def _wants(u: dict, need: str) -> tuple[Callable[[dict], bool], float]:
    """This user's filter on a plan (market group, strict-only, reward, grade, verdict) and their minimum reward."""
    min_rr = entryguard._num(u.get("min_rr"))
    min_rr = 1.5 if min_rr is None else min_rr
    name = u.get("min_grade") or "C"
    min_g = GRADES.index(name if name in GRADES else "C")
    classes = u.get("classes") or []
    strict_only = "strict" in classes

    def wants(s: dict) -> bool:
        try:
            q = s.get("quality") or {}
            grade = q.get("grade", "C")
            if q.get("anti_ruin") is False:
                return False  # Vetoed by anti-ruin risk governor
            return (s.get("cls") in classes and (not strict_only or bool(s.get("strict"))) and (entryguard._num(s.get("rr")) or 0.0) >= min_rr
                    and entryguard.verdict_ok(s, need) and GRADES.index(grade if grade in GRADES else "C") <= min_g)
        except Exception:
            return False
    return wants, min_rr


def run(store: Store, report: dict, trader: Callable[[dict, str], Trader] | None = None, now: datetime | None = None,
        codes: Any = None, only_armed: bool = False, prices: Any = None, budget_s: float | None = None) -> dict:
    """Place trades for every enrolled user. Returns a summary (no secrets).

    Two kinds of plan, both entered at the live price and only while the plan still holds (entryguard.py):
      fills   a limit the scan saw filled in the last few minutes (the candle that filled it has closed);
      armed   a resting limit from the last scan that the live price has just reached: entered now, not at the close.
    Only plans the judgement matrix calls PULL THE TRIGGER (TAKE) are traded. `only_armed`: the minute runs between
    candle closes look at resting limits only (and connect to a user's account only when one has been touched).

    `prices`: where live prices come from (an object with .price(symbol), or a function). By default Deriv's public feed,
    read once per market for all users. A trader supplied by the caller that has a `spot` method is used for prices when
    `prices` is not given (older fake traders without `spot` enter fills at the plan's entry, as before).
    `budget_s`: stop starting new work after this many seconds (the minute run must end before the next one starts).

    One user's or one plan's failure never stops the others. Nothing is entered twice (see `claim`)."""
    now = now or datetime.now(timezone.utc)
    deadline = time.monotonic() + budget_s if budget_s else None
    signals = [] if only_armed else fresh_signals(report, now)
    orders = entryguard.armed_orders(report, now)
    summary = {"users": 0, "placed": 0, "skipped": 0, "errors": 0}
    if only_armed and not orders:
        return summary          # between candle closes nothing can happen without a resting limit: no database, no connection
    own_feed = None
    if trader is None:
        def trader(u: dict, token: str) -> Trader:
            return DerivTrader(token, u["account_id"], u["account_type"], u.get("currency") or "USD")
        if prices is None:
            own_feed = prices = entryguard.PublicFeed()
    shared = _Prices(prices) if prices is not None else None
    ctx = {"store": store, "signals": signals, "orders": orders, "need": min_verdict(), "summary": summary, "trader": trader, "codes": codes,
           "now": now, "day": to_ny(now).date().isoformat(), "only_armed": only_armed, "shared": shared, "deadline": deadline}
    try:
        for u in store.enabled_users():
            if deadline is not None and time.monotonic() > deadline:
                summary["deferred"] = summary.get("deferred", 0) + 1     # out of time: the next run picks these up
                continue
            summary["users"] += 1
            try:
                _run_user(ctx, u)
            except Exception as e:  # this user only: the others still run
                summary["errors"] += 1
                msg = f"Run failed: {e}"
                try:
                    if not _said_lately(store, u["id"], "error", msg, now):
                        store.add_log({"user_id": u["id"], "kind": "error", "message": msg})
                except Exception:
                    pass
    finally:
        if own_feed is not None:
            own_feed.close()
    return summary


def _run_user(ctx: dict, u: dict) -> None:
    store, now, day, summary, only_armed = ctx["store"], ctx["now"], ctx["day"], ctx["summary"], ctx["only_armed"]
    uid, deadline = u["id"], ctx["deadline"]
    wants, min_rr = _wants(u, ctx["need"])

    def log(kind: str, msg: str, s: dict | None = None, **x: Any) -> None:
        store.add_log({"user_id": uid, "kind": kind, "message": msg, "signal_id": s.get("id") if s else None,
                       "market": s.get("label") if s else None, **x})

    def say(kind: str, msg: str, s: dict | None = None, **x: Any) -> None:
        """Log without ever raising or repeating: for what is already a problem, and for what happens every minute."""
        try:
            if not _said_lately(store, uid, kind, msg, now):
                log(kind, msg, s, **x)
        except Exception:
            pass

    def taken(plan: dict) -> bool:
        key = entryguard.trade_key(plan)
        return store.logged_signal(uid, plan["id"]) or (key != plan["id"] and store.logged_signal(uid, key))

    # a fill whose resting order was already entered at the touch is the same trade: never entered twice
    seen: set[str] = set()
    todo, armed = [], []
    for s in ctx["signals"]:
        if wants(s) and s.get("id") not in seen and entryguard.trade_key(s) not in seen and not taken(s):
            seen.update((s["id"], entryguard.trade_key(s)))
            todo.append(s)
    for o in ctx["orders"]:
        if wants(o) and o["id"] not in seen and not taken(o):
            seen.add(o["id"])
            armed.append(o)
    memo: dict[str, bool] = {}

    def housekeeping() -> bool:  # stop management / the day's first balance: a reason to connect with nothing to enter
        if only_armed:
            return False
        if "h" not in memo:
            memo["h"] = u.get("day") != day or _recent_placements(store, u, now)
        return memo["h"]
    if not todo and not armed and not housekeeping():
        return                  # nothing new and nothing of ours open: don't even connect
    if only_armed and not armed:
        return
    why = _access_problem(u, ctx["codes"], now)
    if why:  # invite-only: an ended or switched-off code stops the bot until it is renewed
        if u.get("paused_reason") != why:
            store.update_user(uid, {"paused_reason": why})
            store.add_log({"user_id": uid, "kind": "skipped", "message": why})
        summary["skipped"] += len(todo)
        return
    if str(u.get("paused_reason") or "").startswith("Access code"):
        store.update_user(uid, {"paused_reason": None})  # renewed

    def decide(price_of: Callable[[str], float] | None) -> list[tuple[dict, float | None]]:
        """Which plans are worth entering NOW, and at what price. A fill is checked against the live price; a resting limit
        waits (and is looked at again next minute, nothing logged) until the price touches it. A plan that fails the
        check is logged once, under its own id, and is not looked at again. No price: a resting limit cannot be judged
        and a fill is not entered blind (only a trader that has no way to read a price at all enters at the plan's entry)."""
        ready: list[tuple[dict, float | None]] = []
        for plan in todo + armed:
            try:
                sp = price_of(plan["broker_symbol"]) if price_of is not None else None
                if plan.get("armed"):
                    if sp is None or not entryguard.touched(plan, sp):
                        continue
                if sp is not None:
                    chk = entryguard.check(plan, sp, min_rr)
                    if not chk.ok:
                        log("skipped", f"Not entered: {chk.reason}", plan)
                        summary["skipped"] += 1
                        continue
                ready.append((plan, sp))
            except Exception as e:  # no price, a plan that cannot be read: this plan waits, the others go on
                say("error", f"No live price for {plan.get('label')}: {e}" if price_of is not None else f"Could not judge {plan.get('label')}: {e}")
        return ready

    ready, early = None, False
    if ctx["shared"] is not None:   # prices come from the public feed: decide before opening this user's account
        ready, early = decide(ctx["shared"]), True
        if not ready and (only_armed or not housekeeping()):
            return
    try:
        t = ctx["trader"](u, decrypt(u["token_enc"]))
    except Exception as e:
        summary["errors"] += 1
        say("error", f"Could not connect to Deriv: {e}")
        if isinstance(e, PermissionError):
            store.update_user(uid, {"enabled": False, "paused_reason": "Deriv rejected the token; reconnect to resume"})
        return
    try:
        if not only_armed:
            try:
                manage_stops(store, u, t, summary)  # SL-04 / SL-05 on contracts already open
            except Exception as e:
                say("error", f"Stop management failed: {e}")
        bal = t.balance()
        if not (entryguard._num(bal) and bal > 0):
            say("error", f"Deriv reported no usable balance ({bal!r}); nothing entered")
            summary["errors"] += 1
            return
        if u.get("day") != day:
            store.update_user(uid, {"day": day, "day_start_balance": bal, "paused_reason": None})
            u = {**u, "day": day, "day_start_balance": bal, "paused_reason": None}
        start = float(u.get("day_start_balance") or bal)
        if bal <= start * (1 - float(u["day_loss_pct"]) / 100):
            for s, _ in (ready if ready is not None else [(x, None) for x in todo]):
                log("skipped", f"Daily loss limit ({u['day_loss_pct']}%) reached; paused until tomorrow", s)
                summary["skipped"] += 1
            if u.get("paused_reason") != "Daily loss limit reached":
                store.update_user(uid, {"paused_reason": "Daily loss limit reached"})
            return
        if ready is None:       # prices come from the trader's own connection (a caller-supplied trader)
            ready = decide(getattr(t, "spot", None))
        placed_today = len({x.get("signal_id") for x in _logs(store, uid, 400, "placed") if _ny_day(x.get("created_at")) == day})
        open_n = t.open_positions()
        for s, sp in ready:
            if deadline is not None and time.monotonic() > deadline:
                summary["deferred"] = summary.get("deferred", 0) + 1
                continue
            try:
                if placed_today >= int(u["max_trades_day"]):
                    log("skipped", f"Daily trade limit ({u['max_trades_day']}) reached", s)
                    summary["skipped"] += 1
                    continue
                if open_n >= int(u["max_open"]):
                    log("skipped", f"Max open positions ({u['max_open']}) reached", s)
                    summary["skipped"] += 1
                    continue
                if early and sp is not None:   # connecting and the checks took seconds: one last look at the price before the order goes out
                    try:
                        sp = ctx["shared"].fresh(s["broker_symbol"])
                    except Exception as e:     # no price now: it waits for the next run, it is not given up
                        say("error", f"No live price for {s.get('label')}: {e}")
                        continue
                    chk = entryguard.check(s, sp, min_rr)
                    if not chk.ok:
                        log("skipped", f"Not entered: {chk.reason}", s)
                        summary["skipped"] += 1
                        continue
                if not claim(store, uid, s):
                    summary["skipped"] += 1     # another run holds this trade (or has entered it): never twice
                    continue
                legs_placed = _enter(t, s, sp, bal, u, log, say, summary)
                if legs_placed:
                    placed_today += 1
                    open_n += legs_placed
                    summary["placed"] += 1
            except Exception as e:  # this plan only
                summary["errors"] += 1
                say("error", f"{s.get('label')}: not entered: {e}", s)
    except Exception as e:
        summary["errors"] += 1
        say("error", f"Run failed: {e}")
    finally:
        t.close()


def effective_risk_pct(u: dict, bal: float, s: dict | None = None) -> float:
    """The risk percentage for this trade: base `risk_pct`, halved on a second-try re-entry (RE-02)
    or when Ep 41 dynamic risk step-down (`dynamic_risk`, RK-02/RK-03) is active after losses."""
    pct = float(u["risk_pct"])
    if s is not None:
        is_reentry = (bool(s.get("reentry")) or bool((s.get("advice") or {}).get("reentry"))
                      or any(str(r).startswith("RE-02") for r in (s.get("reasons") or ())))
        if is_reentry:
            pct *= 0.5
    if u.get("dynamic_risk"):
        losses = int(u.get("consecutive_losses") or 0)
        start = float(u.get("day_start_balance") or bal)
        if losses >= 2:
            pct *= 0.25
        elif losses == 1 or bal < start:
            pct *= 0.5
    return pct


def _enter(t: Any, s: dict, sp: float | None, bal: float, u: dict, log: Callable, say: Callable, summary: dict) -> int:
    """Place one plan (its TP1 leg and its runner). Returns how many contracts were placed."""
    risk = round(bal * effective_risk_pct(u, bal, s) / 100, 2)
    # TP-04: Deriv can't close part of a contract, so the partial is a second one,
    # sized by the share the ICT plan takes off at TP1.
    tp1 = entryguard.first_target(s)
    part = entryguard._num((s.get("management") or {}).get("partial", 0.5))  # this style's and model's ICT plan
    part = 0.5 if part is None else part
    legs = ([("TP1", {**s, "target": tp1}, risk * part), ("TP2", s, risk * (1 - part))]
            if tp1 is not None and 0 < part < 1 else [("TP", s, risk)])
    placed = 0
    unlogged = []
    for name, leg, leg_risk in legs:
        try:
            r = t.place(leg, round(leg_risk, 2), spot=sp) if sp is not None else t.place(leg, round(leg_risk, 2))
        except Exception as e:
            log("error", f"Order failed ({name}): {e}", s)
            summary["errors"] += 1
            continue
        placed += 1
        try:
            log("placed", f"{'Buy' if s['direction'] == 'long' else 'Sell'} {s.get('label')} {name} · "
                          f"{'at the touch of the resting limit' if s.get('armed') else 'on the fill'}"
                          + (f" ({sp:g}, plan entry {float(s['entry']):g})" if sp is not None else "") + " · "
                          f"stake {r.get('stake')} x{r.get('multiplier')} · SL {r.get('stop_loss')} / TP {r.get('take_profit')} {t.currency} · stop plan: "
                          f"{'trim 25% halfway, entry at 75%' if (s.get('management') or {}).get('trim_50') else 'entry at 75%'}",
                s, contract_id=str(r.get("contract_id") or ""))
        except Exception:
            unlogged.append(str(r.get("contract_id") or "?"))
    if unlogged:  # the orders stand; only their log row failed (the claim already stops a second entry)
        say("error", f"{s.get('label')}: placed contract(s) {', '.join(unlogged)} but could not write the log row", s)
    return placed


def _access_problem(u: dict, codes: Any, now: datetime) -> str | None:
    """Why this user may not trade under invite-only access, or None. Enrollments made before
    access codes existed (no code stored) belong to the owner and keep working."""
    from ict_trader import access
    if not access.required() or not u.get("access_code"):
        return None
    if u["access_code"].startswith("u:"):  # enrolled by a signed-in account: its trial or Pro must still run
        try:
            from ict_trader import accounts
            st = accounts.Store()
            prof = st.profile(u["access_code"][2:]) or {}
            if not accounts.plan_of(prof, prof.get("email") or "", now)["active"]:
                return "Your ICT Flux trial or Pro has ended. Renew in the app to keep auto-trading."
        except Exception:
            return None  # the accounts table is unreachable: don't stop trading over that
        return None
    try:
        access.check_code(u["access_code"], codes, now)
    except PermissionError as e:
        return f"Access code: {e}"
    except Exception:
        return None  # the codes table is unreachable: don't stop trading over that
    return None


def _recent_placements(store: "Store", u: dict, now: datetime) -> bool:
    """Whether the bot placed anything for this user in the last two days (which may still be open)."""
    cutoff = now - timedelta(days=2)
    for x in _logs(store, u["id"], 3, "placed"):
        at = _stamp(x.get("created_at"))
        if at is not None and at >= cutoff:
            return True
    return False


def _ny_day(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        return to_ny(datetime.fromisoformat(str(iso).replace("Z", "+00:00"))).date().isoformat()
    except ValueError:
        return None


def enabled() -> bool:
    return os.environ.get("AUTOTRADE_ENABLED", "") == "1"
