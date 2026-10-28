"""A small per-caller rate limit for the serverless functions.

Each function instance keeps its own count, so this is a best-effort brake on floods and on guessing
access codes, not a hard global limit: the hard limit is Vercel's Firewall rate-limit rules (see
docs/SECURITY.md). It costs nothing and needs no database.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque

_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_MAX_KEYS = 5000


def client_ip(handler) -> str:
    """The caller's address as Vercel reports it (the proxy's own header, not one the caller can set)."""
    h = handler.headers
    fwd = h.get("x-vercel-forwarded-for") or h.get("x-real-ip") or h.get("x-forwarded-for") or ""
    return fwd.split(",")[0].strip() or "unknown"


def allow(bucket: str, who: str, limit: int, window_s: float, now: float | None = None) -> bool:
    now = time.monotonic() if now is None else now
    if len(_hits) > _MAX_KEYS:  # never grow without bound
        _hits.clear()
    q = _hits[(bucket, who)]
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= limit:
        return False
    q.append(now)
    return True


def guard(handler, bucket: str, limit: int, window_s: float = 60) -> bool:
    """True to carry on. Otherwise a 429 was sent and the handler must return."""
    if allow(bucket, client_ip(handler), limit, window_s):
        return True
    data = json.dumps({"error": "Too many requests. Wait a minute and try again."}).encode()
    handler.send_response(429)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Retry-After", str(int(window_s)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
    return False
