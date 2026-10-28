"""Vercel serverless function: GET /api/markets

Every market Deriv offers that the engine can read (ict_trader/mtf.py): all the synthetic indices (Volatility, Jump,
Crash/Boom, Step, Range Break, DEX, Drift...), forex, indices, commodities, crypto and stocks, straight from Deriv's own
list, so the Market view's picker is never limited to what the last scan happened to cover.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import access, mtf, ratelimit  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not ratelimit.guard(self, "markets", 30) or not access.gate(self):
            return
        try:
            self._send(200, {"markets": mtf.deriv_list()})
        except Exception as e:   # the page falls back to the markets of the last scan
            self._send(502, {"error": f"{type(e).__name__}: {e}"}, cache=False)

    def _send(self, status, body, cache=True):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "public, s-maxage=1800, stale-while-revalidate=86400" if cache else "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
