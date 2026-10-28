"""Invite-only access: the app opens only for people the owner has given an access code.

The owner adds codes in Supabase (table access_codes, see db/access.sql), each with an
optional end date, so access can be sold for a period and switched off at any time.
A valid code is exchanged for a short-lived signed pass (no database hit per request),
and the data endpoints refuse requests without one. On by default; ACCESS_REQUIRED=0 switches it off.

OWNER_CODE (env) always works, so the owner can never lock themselves out.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

PASS_TTL = 12 * 3600          # a pass is re-checked against the codes table twice a day
INTERNAL = "_int"             # server-to-server parameter that skips the check (shared edge cache)
CODE_RE = re.compile(r"[A-Za-z0-9-]{3,40}")


def required() -> bool:
    """Is the data locked? Only when explicitly turned on with ACCESS_REQUIRED=1/true, or when an invite list is configured.
    Open by default so users and traders can use the live scanner and view chart markups."""
    if bool(os.environ.get("ALLOWED_EMAILS", "").strip()):
        return True
    return os.environ.get("ACCESS_REQUIRED", "").strip().lower() in ("1", "true", "yes", "on")


def contact() -> str:
    return os.environ.get("ACCESS_CONTACT", "")


def _secret() -> bytes:
    s = os.environ.get("ACCESS_SECRET") or os.environ.get("AUTOTRADE_KEY") or os.environ.get("CRON_SECRET") or ""
    if not s:
        raise RuntimeError("Set ACCESS_SECRET (or AUTOTRADE_KEY) in Vercel")
    return hashlib.sha256(b"ictflux-access-v2:" + s.encode()).digest()


def internal_key() -> str:
    return hmac.new(_secret(), b"internal", hashlib.sha256).hexdigest()[:32]


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(body: str) -> str:
    return _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())


def _parse_time(v: Any) -> datetime | None:
    if not v:
        return None
    t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def issue(code: str, name: str, plan: str, expires_at: datetime | None, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    exp = now + PASS_TTL
    if expires_at:
        exp = min(exp, expires_at.timestamp())
    body = _b64(json.dumps({"c": code, "n": name, "p": plan, "x": int(exp)}).encode())
    return {"token": f"{body}.{_sign(body)}", "name": name, "plan": plan,
            "expires_at": expires_at.isoformat() if expires_at else None, "pass_expires": int(exp)}


def verify(token: str, now: float | None = None) -> dict:
    """The pass's contents, or PermissionError."""
    try:
        body, sig = (token or "").split(".")
    except ValueError:
        raise PermissionError("Enter your access code") from None
    if not hmac.compare_digest(_sign(body), sig):
        raise PermissionError("Enter your access code")
    p = json.loads(_unb64(body))
    if p["x"] < (now if now is not None else time.time()):
        raise PermissionError("Your access pass expired; enter your code again")
    return p


class CodeStore:
    """The access_codes table over Supabase's REST API (same credentials as auto-trading)."""

    def __init__(self) -> None:
        from ict_trader.autotrade import SupabaseStore
        self.db = SupabaseStore()

    def get(self, code: str) -> dict | None:
        rows = self.db._req("GET", f"access_codes?code=ilike.{code}&select=*&limit=1")
        return rows[0] if rows else None

    def touch(self, code: str, now: datetime) -> None:
        self.db._req("PATCH", f"access_codes?code=ilike.{code}", {"last_seen_at": now.isoformat()})


def is_owner(code: str) -> bool:
    owner = os.environ.get("OWNER_CODE", "").strip()
    return bool(owner) and hmac.compare_digest((code or "").strip().upper().encode(), owner.upper().encode())


def check_code(code: str, store: Any = None, now: datetime | None = None) -> dict:
    """{code, name, plan, expires_at} for a usable code, or PermissionError saying why not."""
    code = (code or "").strip()
    now = now or datetime.now(timezone.utc)
    if is_owner(code):
        return {"code": code, "name": "Owner", "plan": "pro", "expires_at": None}
    if not CODE_RE.fullmatch(code):
        raise PermissionError("That access code is not valid")
    row = (store or CodeStore()).get(code)
    if not row or not row.get("enabled", True):
        raise PermissionError("That access code is not valid")
    exp = _parse_time(row.get("expires_at"))
    if exp and exp <= now:
        raise PermissionError(f"That access code ended on {exp.date().isoformat()}. Contact us to renew.")
    return {"code": row["code"], "name": row.get("name") or "", "plan": row.get("plan") or "pro", "expires_at": exp}


def redeem(code: str, store: Any = None, now: datetime | None = None) -> dict:
    """Exchange a code for a pass."""
    now = now or datetime.now(timezone.utc)
    if is_owner(code):
        return issue(code.strip(), "Owner", "pro", None, now.timestamp())
    store = store or CodeStore()
    c = check_code(code, store, now)
    try:
        store.touch(c["code"], now)
    except Exception:
        pass  # last_seen_at is only informational
    return issue(c["code"], c["name"], c["plan"], c["expires_at"], now.timestamp())


# ------------------------------------------------------------------ request gate
def _deny(handler, msg: str) -> None:
    data = json.dumps({"error": msg, "access": "required", "contact": contact()}).encode()
    handler.send_response(401)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Access-Required", "1")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def holder(handler) -> dict | None:
    """Who is calling: the pass contents, or None. Accepts the X-Access pass or a raw ?code= (MT5 robot)."""
    q = parse_qs(urlparse(handler.path).query)
    tok = handler.headers.get("X-Access") or q.get("access", [""])[0]
    if tok:
        return verify(tok)
    code = q.get("code", [""])[0]
    if code:
        c = check_code(code)
        return {"c": c["code"], "n": c["name"], "p": c["plan"]}
    raise PermissionError("Enter your access code")


def own_host(handler) -> str:
    """The host to call back into: ours only. A caller-supplied Host / X-Forwarded-Host that is not one of ours
    must never receive the internal key."""
    mine = {h.strip().lower() for h in os.environ.get("APP_HOSTS", "ictflux.app,ict-trader.vercel.app").split(",") if h.strip()}
    for extra in (os.environ.get("VERCEL_URL"), os.environ.get("VERCEL_BRANCH_URL"), os.environ.get("VERCEL_PROJECT_PRODUCTION_URL")):
        if extra:
            mine.add(extra.strip().lower())
    for cand in (handler.headers.get("x-forwarded-host"), handler.headers.get("host")):
        c = (cand or "").split(",")[0].strip().lower()
        if c in mine:
            return c
    return sorted(mine)[0]


def gate(handler, proxy: bool = True) -> bool:
    """True when the handler should go on and answer. Otherwise the response was sent here.

    With proxy=True a valid request is answered from the shared, edge-cached internal URL,
    so everyone shares one scan instead of each pass causing its own."""
    if not required():
        return True
    q = parse_qs(urlparse(handler.path).query)
    if hmac.compare_digest(q.get(INTERNAL, [""])[0].encode(), internal_key().encode()):
        return True
    try:
        holder(handler)
    except PermissionError as e:
        _deny(handler, str(e))
        return False
    except Exception as e:  # codes table unreachable: fail closed, but say why
        _deny(handler, f"Access check failed: {e}")
        return False
    if not proxy:
        return True
    u = urlparse(handler.path)
    params = [(k, v) for k, vs in parse_qs(u.query).items() if k not in ("access", "code", INTERNAL) for v in vs]
    params.append((INTERNAL, internal_key()))
    host = own_host(handler)
    req = urllib.request.Request(f"https://{host}{u.path}?{urlencode(params)}", headers={"User-Agent": "ictflux-gate"})
    try:
        with urllib.request.urlopen(req, timeout=59) as r:
            status, raw, ctype = r.status, r.read(), r.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as e:
        status, raw, ctype = e.code, e.read(), e.headers.get("Content-Type", "application/json")
    except Exception as e:
        status, raw, ctype = 502, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json"
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Cache-Control", "private, no-store")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)
    return False
