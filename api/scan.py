"""Vercel serverless function: GET /api/scan

Runs the strategy on fresh 5-minute data for every market and returns JSON
(see ict_trader/scanner.py). With ?notify=1 it also pushes any new signal or
trade result to the ntfy topic (and Telegram, if configured). A scheduler
calls that every few minutes; see .github/workflows/scan.yml.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from time import monotonic
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import access, ratelimit, snapshot  # noqa: E402

# The live 5-minute scan saved by the cron is served as is while it is this fresh (a new one
# lands a minute after every 5-minute candle closes, so this only lapses if the cron stops).
FRESH_S = 7 * 60
STALE_S = 24 * 3600  # Up to 24 hours: serving a cached scan avoids cold-start timeouts and browser AbortErrors
SCAN_TOP_KEYS = {"XAUUSD", "GBPUSD", "EURUSD", "US500", "NAS100", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "XAGUSD", "WTI", "US30"}

_SCAN_CACHE: dict[str, tuple[float, str]] = {}


def _get_cached_scan(key: str, max_age: float) -> str | None:
    now_ts = monotonic()
    if key in _SCAN_CACHE:
        ts, body = _SCAN_CACHE[key]
        if (now_ts - ts) <= max_age:
            return body
    res = snapshot.load(key, max_age)
    if res is not None:
        _SCAN_CACHE[key] = (now_ts, res)
    return res


def _set_cached_scan(key: str, report: dict) -> None:
    body = json.dumps(report, separators=(",", ":"))
    _SCAN_CACHE[key] = (monotonic(), body)
    snapshot.save(key, report)

# Time the always-on bot may take. The minute run (resting limits only) must be over before the next minute's starts, so it
# stops starting new work after MINUTE_BUDGET_S; the 5-minute run shares the function's 120 s with the scan (TOTAL_BUDGET_S
# counted from the start of the request).
MINUTE_BUDGET_S = 40
TOTAL_BUDGET_S = 100


def scan_due(now=None) -> bool:
    """True on the minute after a 5-minute candle closes (:01, :06, ... past the hour). SCAN_EVERY=1
    in the environment restores a scan on every call."""
    from datetime import datetime, timezone
    if os.environ.get("SCAN_EVERY") == "1":
        return True
    now = now or datetime.now(timezone.utc)
    return now.minute % 5 == 1


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        started = monotonic()
        q = parse_qs(urlparse(self.path).query)
        tf_str = q.get("tf", ["5"])[0]
        want_notify = q.get("notify", ["0"])[0] == "1"
        secret = os.environ.get("CRON_SECRET")
        if want_notify:
            # The scheduler's call: it must prove it holds CRON_SECRET. With no secret set, nobody may
            # trigger notifications (an unset secret used to leave this open to the whole internet).
            import hmac
            auth = self.headers.get("Authorization", "")
            given = auth[7:] if auth.startswith("Bearer ") else q.get("key", [""])[0]
            if not secret or not hmac.compare_digest(given.encode(), secret.encode()):
                return self._send(401, {"error": "notify requires the CRON_SECRET"}, cache=False)
        elif not ratelimit.guard(self, "scan", 240):
            return
        if want_notify and q.get("force", ["0"])[0] != "1" and not scan_due():
            # The cron may call every minute, but 5-minute candles only close every 5 minutes: the
            # full scan runs on the minute after each close, and the other calls return at once.
            out = {"skipped": "between candle closes; the next scan runs a minute after the next 5-minute close"}
            try:  # but a resting limit is reached in real time, not at the close: look at the live price every minute
                from ict_trader import autotrade
                if autotrade.enabled():
                    saved = snapshot.load("scan-5", FRESH_S)  # only a scan this fresh: an older one may list orders the engine has since cancelled
                    if saved is not None:
                        out["autotrade"] = autotrade.run(autotrade.store_from_env(), json.loads(saved), only_armed=True, budget_s=MINUTE_BUDGET_S)
            except Exception as e:
                out["autotrade"] = {"error": f"{type(e).__name__}: {e}"}
            return self._send(200, out, cache=False)
        if not want_notify and not access.gate(self):
            return

        day = q.get("date", [""])[0]
        mt5 = q.get("format", [""])[0] == "mt5"
        force = q.get("force", ["0"])[0] == "1"
        if not want_notify and not day and not force:
            saved = _get_cached_scan(f"scan-{tf_str}", FRESH_S)  # instant: fresh copy
            if saved is not None:
                if mt5:
                    return self._feed(json.loads(saved), q)
                return self._send_raw(saved.encode("utf-8"))
            stale = _get_cached_scan(f"scan-{tf_str}", STALE_S)  # instant: recent copy (never stall visitors)
            if stale is not None:
                if mt5:
                    return self._feed(json.loads(stale), q)
                return self._send_raw(stale.encode("utf-8"))

        from ict_trader.scanner import notify, ntfy_topic, scan, universe  # the engine: only when a scan must run
        try:
            tf = int(tf_str)
            as_of = None
            if day and not want_notify:
                # a past day: the engine's read as it was at that day's 17:00 NY close
                # (or now, for today), with every outcome since
                from datetime import datetime, time, timezone
                from ict_trader.sessions import NY
                d = datetime.strptime(day, "%Y-%m-%d").date()
                as_of = min(datetime.combine(d, time(17, 0), tzinfo=NY).astimezone(timezone.utc), datetime.now(timezone.utc))
            target_markets = None
            if not want_notify and not day:
                u = universe()
                target_markets = tuple(m for m in u if m.key in SCAN_TOP_KEYS) or u[:14]
            report = scan(tf=tf, as_of=as_of, markets=target_markets, light=not want_notify)
        except ValueError as e:
            return self._send(400, {"error": str(e)}, cache=False)
        report["alerts"] = {"ntfy_topic": ntfy_topic(), "telegram": bool(os.environ.get("TELEGRAM_BOT_TOKEN"))}
        if as_of is None:
            _set_cached_scan(f"scan-{tf}", report)  # every reader gets this one instead of running the engine again
        if mt5 and not want_notify:
            return self._feed(report, q)
        if want_notify:
            from ict_trader import accounts
            report["notified"] = notify(report, click=f"{accounts.app_url(self.headers.get('host'))}/#signals")
            if tf == 5:
                from ict_trader import autotrade
                if autotrade.enabled():  # always-on auto-trading rides the same minute run
                    try:
                        report["autotrade"] = autotrade.run(autotrade.store_from_env(), report, budget_s=max(10.0, TOTAL_BUDGET_S - (monotonic() - started)))
                    except Exception as e:
                        report["autotrade"] = {"error": f"{type(e).__name__}: {e}"}
            # the scheduler only needs the outcome; the full report is too large for cron-job.org
            brief = {k: report.get(k) for k in ("generated_at", "generated_ny", "summary", "notified", "autotrade")}
            return self._send(200, brief, cache=False)
        self._send(200, report, cache=not want_notify, long_cache=bool(day) and report.get("as_of") is not None
                   and report["as_of"] < report["generated_at"])

    def _feed(self, report, q):
        """Plain-text feed for the MT5 robot."""
        from ict_trader.mt5feed import feed
        data = feed(report, strict=q.get("strict", ["0"])[0] == "1").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "public, s-maxage=60, stale-while-revalidate=600")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_raw(self, data):
        """The saved scan, as stored. Shared caches keep it a minute and may serve it a little stale
        while they fetch the next, so the app never waits on the server."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "public, s-maxage=60, stale-while-revalidate=120")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send(self, status, body, cache=True, long_cache=False):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store" if not cache else "public, s-maxage=600, stale-while-revalidate=3600" if long_cache
                         else "public, s-maxage=60, stale-while-revalidate=120")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
