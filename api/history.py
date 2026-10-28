"""Vercel serverless function: GET /api/history

Paper track record over the last month for the main markets (see ict_trader/scanner.py). It is the heaviest call the app makes, so a request
NEVER has to wait for it when a saved copy exists, and only ONE rebuild runs at a time:

  1. a copy under 30 minutes old is served;
  2. else any copy under a day old is served (a rebuild is attempted by the next request that finds the lock free);
  3. else, if another request is already rebuilding, this one answers at once with {"warming": true} and the page asks again in a moment;
  4. else this request takes the lock, rebuilds, saves, and answers.

`force=1` asks for a rebuild, but never twice within 5 minutes and never while one is running, so the refresh button (or anyone calling the
endpoint in a loop) cannot make the server rebuild the record over and over.
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import snapshot  # noqa: E402
from ict_trader.scanner import history, universe  # noqa: E402

HIST_FRESH_S = 30 * 60
HIST_STALE_S = 86400       # a saved record may stand in while a fresh one is built, for a day at most
FORCE_MIN_AGE_S = 5 * 60   # force=1 does nothing when the saved record is younger than this
LOCK_S = 150               # a rebuild that has not finished in this long is presumed dead
TOP_KEYS = {"XAUUSD", "GBPUSD", "EURUSD", "US500", "NAS100", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "XAGUSD", "WTI", "US30"}
_BUILDING: dict[int, float] = {}   # this instance's own lock, for when the shared store is not reachable


def _usable(saved: str | None) -> str | None:
    if saved is None:
        return None
    try:
        parsed = json.loads(saved)
    except Exception:
        return None
    return None if isinstance(parsed, dict) and parsed.get("error") else saved


def _lock_held(tf: int) -> bool:
    if time.time() - _BUILDING.get(tf, 0) < LOCK_S:
        return True
    return snapshot.load(f"history-{tf}-lock", LOCK_S) is not None


def _take_lock(tf: int) -> None:
    _BUILDING[tf] = time.time()
    snapshot.save(f"history-{tf}-lock", {"at": _BUILDING[tf]})


def serve(tf: int, force: bool) -> tuple[int, bytes, str]:
    """(status, body, cache-control) for one request. Kept apart from the HTTP handler so it can be tested."""
    key = f"history-{tf}-discipline-v5"   # v5: Titanium Target Delivery (+1.0R BE lock, +1.5R 50% partials, 20/40/60d IPDA)
    fresh = _usable(snapshot.load(key, HIST_FRESH_S))
    recent = _usable(snapshot.load(key, FORCE_MIN_AGE_S)) if force else None
    if fresh is not None and (not force or recent is not None):
        return 200, fresh.encode("utf-8"), "public, s-maxage=60, stale-while-revalidate=120"
    stale = _usable(snapshot.load(key, HIST_STALE_S))
    if _lock_held(tf):
        if stale is not None:
            return 200, stale.encode("utf-8"), "public, s-maxage=30"
        return 200, json.dumps({"ok": True, "warming": True, "markets": [], "message": "Building the track record: one moment."}).encode("utf-8"), "no-store"
    _take_lock(tf)
    try:
        target = tuple(m for m in universe() if m.key in TOP_KEYS) or universe()[:16]
        body = history(tf=tf, markets=target)
    except Exception as e:   # a failed rebuild must not blank the page: serve what is saved, or say so
        if stale is not None:
            return 200, stale.encode("utf-8"), "public, s-maxage=30"
        return 502, json.dumps({"error": f"track record unavailable: {type(e).__name__}"}).encode("utf-8"), "no-store"
    finally:
        _BUILDING.pop(tf, None)
    if isinstance(body, dict) and not body.get("error"):
        snapshot.save(key, body)
    return 200, json.dumps(body).encode("utf-8"), "public, s-maxage=60, stale-while-revalidate=120"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            q = parse_qs(urlparse(self.path).query)
            status, data, cache = serve(int(q.get("tf", ["5"])[0]), q.get("force", ["0"])[0] == "1")
        except ValueError as e:
            status, data, cache = 400, json.dumps({"error": str(e)}).encode("utf-8"), "no-store"
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
