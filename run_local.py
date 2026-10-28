"""Local & Production Server for God's Eye Flux."""

from __future__ import annotations

import json
import mimetypes
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ict_trader.scanner import scan, universe
from ict_trader.research import research

SCAN_TOP_KEYS = {"XAUUSD", "GBPUSD", "EURUSD", "US500", "NAS100", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "XAGUSD", "WTI", "US30"}


class FluxRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        pass

    def _send_json(self, status: int, data: dict) -> None:
        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)

        if parsed.path == "/api/scan":
            tf = int(q.get("tf", ["5"])[0])
            try:
                u = universe()
                target_markets = tuple(m for m in u if m.key in SCAN_TOP_KEYS) or u[:12]
                res = scan(tf=tf, markets=target_markets, light=True)
                self._send_json(200, res)
            except Exception as e:
                self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            return

        if parsed.path == "/api/research":
            tf = int(q.get("tf", ["5"])[0])
            days = int(q.get("days", ["45"])[0])
            mode = q.get("mode", ["ict"])[0]
            symbol = q.get("symbol", ["XAUUSD"])[0].upper()
            try:
                res = research(symbol=symbol, tf=tf, days=days, mode=mode)
                self._send_json(200, res)
            except Exception as e:
                self._send_json(400, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            return

        return super().do_GET()


def main() -> None:
    port = int(os.environ.get("PORT", "8090"))
    server = ThreadingHTTPServer(("127.0.0.1", port), FluxRequestHandler)
    print(f"God's Eye Flux Live Server running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
