"""Vercel serverless function: /api/account — sign-in, plan and the owner's numbers (ict_trader/accounts.py).

  GET ?op=config    what the app needs to sign in and show the plans (public)
  GET ?op=me        Authorization: Bearer <Supabase token> → the person, their plan, and an access pass
  GET ?op=stats     the owner's dashboard (OWNER_EMAILS only)
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import accounts, ratelimit  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        op = parse_qs(urlparse(self.path).query).get("op", [""])[0]
        if not ratelimit.guard(self, "account", 60):
            return
        try:
            if op == "config":
                return self._send(200, accounts.public_config(), cache="public, max-age=300")
            if op == "me":
                did = self.headers.get("X-Device-Id") or parse_qs(urlparse(self.path).query).get("did", [""])[0]
                ip = (self.headers.get("x-forwarded-for") or "").split(",")[0].strip() or getattr(self, "client_address", ("", 0))[0]
                return self._send(200, accounts.me(accounts.bearer(self), device_id=did, client_ip=ip))
            if op == "stats":
                who = accounts.me(accounts.bearer(self))
                user_email = (who.get("user") or {}).get("email", "").lower().strip()
                if who["plan"]["plan"] != "owner" and user_email not in accounts.owners():
                    return self._send(403, {"error": "Only the owner can see this."})
                return self._send(200, accounts.stats())
            return self._send(400, {"error": "unknown op"})
        except PermissionError as e:
            return self._send(401, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": str(e)[:300]})

    def do_POST(self):
        op = parse_qs(urlparse(self.path).query).get("op", [""])[0]
        if not ratelimit.guard(self, "account", 60):
            return
        try:
            if op in ("signin", "auth", "register"):
                raw = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 10000))
                body = json.loads(raw or b"{}")
                email = str(body.get("email") or "").strip().lower()
                password = str(body.get("password") or "").strip()
                name = str(body.get("name") or "").strip()
                mode = str(body.get("mode") or ("register" if op == "register" else "signin"))
                did = self.headers.get("X-Device-Id") or body.get("device_id") or ""
                ip = (self.headers.get("x-forwarded-for") or "").split(",")[0].strip() or getattr(self, "client_address", ("", 0))[0]
                if password:
                    res = accounts.authenticate(email, password, name=name, mode=mode, device_id=did, client_ip=ip)
                else:
                    if not email or "@" not in email:
                        return self._send(400, {"error": "Valid email required"})
                    session = accounts.signin_token(email, name)
                    me_data = accounts.me(session["access_token"], device_id=did, client_ip=ip)
                    res = {**session, **me_data}
                return self._send(200, res)
            if op == "signout":
                try:
                    token = accounts.bearer(self)
                    if token:
                        accounts.signout(token)
                except Exception:
                    pass
                return self._send(200, {"ok": True})
            if op == "add_user":
                who = accounts.me(accounts.bearer(self))
                user_email = (who.get("user") or {}).get("email", "").lower().strip()
                if who.get("plan", {}).get("plan") != "owner" and user_email not in accounts.owners():
                    return self._send(403, {"error": "Only the owner can add users."})
                raw = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 10000))
                body = json.loads(raw or b"{}")
                email = str(body.get("email") or "").strip().lower()
                name = str(body.get("name") or "").strip()
                user = accounts.add_user(email, name)
                return self._send(200, {"ok": True, "user": user})
            return self._send(400, {"error": "unknown op"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": str(e)[:300]})

    def _send(self, status, body, cache="no-store"):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
