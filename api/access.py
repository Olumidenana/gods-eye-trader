"""Vercel serverless function: /api/access — invite-only access (see ict_trader/access.py).

  GET  ?op=status            {required, contact}
  POST ?op=redeem  {code}    a signed pass for a valid code, or 401 saying why not
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import access, ratelimit  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _op(self):
        return parse_qs(urlparse(self.path).query).get("op", [""])[0]

    def do_GET(self):
        if self._op() == "status":
            return self._send(200, {"required": access.required(), "contact": access.contact()})
        self._send(400, {"error": "unknown op"})

    def do_POST(self):
        if self._op() != "redeem":
            return self._send(400, {"error": "unknown op"})
        if not ratelimit.guard(self, "redeem", 8):  # guessing codes
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            body = {}
        try:
            self._send(200, access.redeem(str(body.get("code", ""))))
        except PermissionError as e:
            self._send(401, {"error": str(e), "contact": access.contact()})
        except Exception as e:
            self._send(500, {"error": f"Access check failed: {e}"})
