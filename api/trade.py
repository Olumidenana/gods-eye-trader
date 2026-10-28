"""Vercel serverless function: GET /api/trade?market=EURUSD&tf=5&id=...

One past trade from the track record with the real candles around it, so the
trade review can draw how it played out: sweep, shift, gap, fill, stop/target.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import access  # noqa: E402
from ict_trader.scanner import trade_detail  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not access.gate(self):
            return
        q = parse_qs(urlparse(self.path).query)
        try:
            status, body = 200, trade_detail(q.get("market", [""])[0], q.get("id", [""])[0], int(q.get("tf", ["5"])[0]),
                                       cls=q.get("cls", [None])[0])
        except ValueError as e:
            status, body = 400, {"error": str(e)}
        except LookupError as e:
            status, body = 404, {"error": str(e)}
        except Exception as e:  # network problems are reported, never raised
            status, body = 502, {"error": f"{type(e).__name__}: {e}"}
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "public, s-maxage=1800, stale-while-revalidate=3600" if status == 200 else "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
