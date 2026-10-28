"""Vercel serverless function: /api/deriv — a thin relay to Deriv's account APIs.

The browser cannot call Deriv's REST endpoints directly (CORS), so this relays
four whitelisted operations. It never stores or logs tokens: the user's access
token lives only in their own browser and is passed through per request.

  GET  /api/deriv?op=config                    -> public OAuth settings (client id)
  POST /api/deriv?op=token     {code, code_verifier, redirect_uri}
  GET  /api/deriv?op=accounts  (X-Deriv-Token header)
  POST /api/deriv?op=create_demo (X-Deriv-Token)
  POST /api/deriv?op=otp&account=ID (X-Deriv-Token) -> authenticated WebSocket URL

Docs: https://developers.deriv.com/docs/intro/oauth/ and
https://developers.deriv.com/docs/options/websocket/
"""

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import access  # noqa: E402

AUTH_BASE = "https://auth.deriv.com/oauth2"
API_BASE = "https://api.derivws.com/trading/v1/options"
# The app id is public (it appears in every OAuth redirect); override in Vercel if needed.
APP_ID = os.environ.get("DERIV_APP_ID", "34BbtJiEX2xD1Vqy1Kjd6")


def _call(url, method="GET", headers=None, body=None, form=False):
    data = None
    headers = dict(headers or {})
    if body is not None:
        if form:
            data = urllib.parse.urlencode(body).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # network failure
        return 502, json.dumps({"error": {"message": f"Could not reach Deriv: {e}"}}).encode()


class handler(BaseHTTPRequestHandler):
    def _send(self, status, raw):
        if isinstance(raw, (dict, list)):
            raw = json.dumps(raw).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def _auth_headers(self):
        token = self.headers.get("X-Deriv-Token", "")
        if not token:
            return None
        return {"Authorization": f"Bearer {token}", "Deriv-App-ID": APP_ID}

    def do_GET(self):
        if not access.gate(self, proxy=False):
            return
        q = parse_qs(urlparse(self.path).query)
        op = q.get("op", [""])[0]
        if op == "config":
            return self._send(200, {"client_id": APP_ID, "authorize_url": f"{AUTH_BASE}/auth",
                                    "scope": os.environ.get("DERIV_OAUTH_SCOPE", "trade")})
        if op == "accounts":
            h = self._auth_headers()
            if not h:
                return self._send(401, {"error": {"message": "Not connected to Deriv"}})
            status, raw = _call(f"{API_BASE}/accounts", headers=h)
            return self._send(status, raw)
        self._send(400, {"error": {"message": "unknown op"}})

    def do_POST(self):
        if not access.gate(self, proxy=False):
            return
        q = parse_qs(urlparse(self.path).query)
        op = q.get("op", [""])[0]
        body = self._body()
        if op == "token":
            form = {
                "grant_type": "authorization_code",
                "client_id": APP_ID,
                "code": str(body.get("code", "")),
                "code_verifier": str(body.get("code_verifier", "")),
                "redirect_uri": str(body.get("redirect_uri", "")),
            }
            secret = os.environ.get("DERIV_CLIENT_SECRET")
            if secret:
                form["client_secret"] = secret
            status, raw = _call(f"{AUTH_BASE}/token", "POST", body=form, form=True)
            return self._send(status, raw)
        h = self._auth_headers()
        if not h:
            return self._send(401, {"error": {"message": "Not connected to Deriv"}})
        if op == "create_demo":
            status, raw = _call(f"{API_BASE}/accounts", "POST", headers=h,
                                body={"account_type": "demo", "currency": "USD"})
            return self._send(status, raw)
        if op == "otp":
            account = q.get("account", [""])[0]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", account):
                return self._send(400, {"error": {"message": "bad account id"}})
            status, raw = _call(f"{API_BASE}/accounts/{account}/otp", "POST", headers=h)
            return self._send(status, raw)
        self._send(400, {"error": {"message": "unknown op"}})
