"""Vercel serverless function: GET /api/research

  /api/research?symbol=R_75&tf=5&mode=ict[&days=90]

Backtests the ICT engine on real Deriv candles for one symbol, timeframe and time
mode (see ict_trader/research.py). Results are cached at the edge for 6 hours.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import access, ratelimit  # noqa: E402
from ict_trader.research import DEFAULT_DAYS, MODES, SYNTHETICS, TIMEFRAMES, research  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not ratelimit.guard(self, "research", 120):  # allow batch backtests across timeframes & models
            return
        if not access.gate(self):
            return
        q = parse_qs(urlparse(self.path).query)
        g = lambda k, d=None: q.get(k, [d])[0]  # noqa: E731
        if g("list"):
            return self._send(200, {"synthetics": SYNTHETICS, "timeframes": TIMEFRAMES, "modes": MODES, "default_days": DEFAULT_DAYS}, 86400)
        try:
            tf = int(g("tf", "5"))
            days = int(g("days")) if g("days") else None
            if days is not None:
                days = max(1, min(days, 1825 if tf >= 240 else 365))
            symbol = g("symbol", "frxXAUUSD")
            if not symbol.replace("_", "").isalnum() or len(symbol) > 20:
                raise ValueError("bad symbol")
            label = (g("label") or "")[:40] or None
            mgmt = g("mgmt") or "godseye"
            body = research(symbol, tf, days, g("mode", "ict"), label=label, model=g("model", "ict2022"), cls=g("cls"), mgmt=mgmt)
            self._send(200, body, 21600)
        except (ValueError, LookupError) as e:
            self._send(400, {"error": str(e)}, 0)
        except Exception as e:
            self._send(502, {"error": f"{type(e).__name__}: {e}"}, 0)

    def _send(self, status, body, cache):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", f"public, s-maxage={cache}, stale-while-revalidate=3600" if cache else "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
