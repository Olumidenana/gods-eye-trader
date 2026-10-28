"""Vercel serverless function: GET /api/market?key=R_75

One Deriv market read on every timeframe at once (see ict_trader/mtf.py): the structure on each, the pools of resting stops
and open gaps either side of price, what every model is doing, and whether the timeframes agree. `key` is Deriv's own symbol
(R_75, BOOM500, frxEURUSD, cryBTCUSD...) as listed by /api/markets. The Market view page uses it.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import access, mtf, ratelimit  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not ratelimit.guard(self, "market", 40) or not access.gate(self):
            return
        key = parse_qs(urlparse(self.path).query).get("key", [""])[0]
        if not mtf.valid_key(key):
            return self._send(400, {"error": "unknown market"}, cache=False)
        try:
            self._send(200, mtf.read(key))
        except LookupError as e:
            self._send(404, {"error": str(e)}, cache=False)
        except Exception as e:   # the page shows the message instead of a blank screen
            self._send(502, {"error": f"{type(e).__name__}: {e}"}, cache=False)

    def _send(self, status, body, cache=True):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "public, s-maxage=30, stale-while-revalidate=60" if cache else "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
