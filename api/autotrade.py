"""Vercel serverless function: /api/autotrade — always-on auto-trading (opt-in).

  GET  ?op=terms                         the risk terms users must accept
  POST ?op=accounts  {token}             the token's Deriv accounts (nothing stored)
  POST ?op=enroll    {token, account_id, accept_terms, terms_version, confirm_real?, settings...}
  GET  ?op=status    (X-Autotrade-Id, X-Autotrade-Key)   settings + recent activity
  POST ?op=update    (same headers) {settings..., enabled?}
  POST ?op=delete    (same headers)  removes the enrollment, its token and its log
  GET  ?op=run       (CRON_SECRET)   one run; normally triggered by /api/scan?notify=1

Off unless AUTOTRADE_ENABLED=1, with SUPABASE_URL, SUPABASE_SERVICE_KEY and
AUTOTRADE_KEY set (see db/autotrade.sql and README).
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import access  # noqa: E402
from ict_trader import autotrade as at  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            return {}

    def _op(self):
        return parse_qs(urlparse(self.path).query).get("op", [""])[0]

    def _guard(self):
        if not at.enabled():
            self._send(503, {"error": "Always-on auto-trading is not switched on for this site yet."})
            return False
        return True

    def _user(self, store):
        return at.authorize(store, self.headers.get("X-Autotrade-Id", ""), self.headers.get("X-Autotrade-Key", ""))

    def do_GET(self):
        op = self._op()
        if op == "terms":
            return self._send(200, {"version": at.TERMS_VERSION, "terms": at.TERMS, "available": at.enabled()})
        if not self._guard():
            return
        try:
            store = at.store_from_env()
            if op == "status":
                u = self._user(store)
                return self._send(200, {"enrollment": at.public(u), "log": [
                    {k: x.get(k) for k in ("created_at", "kind", "message", "market", "contract_id")} for x in store.logs(u["id"], 60)]})
            if op == "run":
                secret = os.environ.get("CRON_SECRET", "")
                q = parse_qs(urlparse(self.path).query)
                if not secret or (self.headers.get("Authorization") != f"Bearer {secret}" and q.get("key", [""])[0] != secret):
                    return self._send(401, {"error": "run requires the CRON_SECRET"})
                from ict_trader.scanner import scan
                return self._send(200, at.run(store, scan(tf=5)))
        except PermissionError:
            return self._send(404, {"error": "No auto-trading found for this device. Set it up again."})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        self._send(400, {"error": "unknown op"})

    def do_POST(self):
        op, body = self._op(), self._body()
        if not self._guard():
            return
        if op in ("accounts", "enroll") and not access.gate(self, proxy=False):
            return
        try:
            if op == "accounts":
                return self._send(200, {"accounts": at.deriv_accounts(str(body.get("token", "")).strip())})
            store = at.store_from_env()
            if op == "enroll":
                if access.required():
                    body["access_code"] = access.holder(self)["c"]
                return self._send(200, at.enroll(store, body))
            u = self._user(store)
            if op == "update":
                fields = at.Settings.parse({**at.public(u), **body}).row()
                if "enabled" in body:
                    fields["enabled"] = bool(body["enabled"])
                    if fields["enabled"]:
                        fields["paused_reason"] = None
                store.update_user(u["id"], fields)
                store.add_log({"user_id": u["id"], "kind": "updated", "message": "Settings updated" + ("" if fields.get("enabled", True) else "; auto-trading paused")})
                return self._send(200, {"enrollment": at.public({**u, **fields})})
            if op == "delete":
                store.delete_user(u["id"])
                return self._send(200, {"deleted": True})
        except PermissionError as e:
            return self._send(404 if str(e) == "not found" else 401, {"error": "No auto-trading found for this device." if str(e) == "not found" else str(e)})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        self._send(400, {"error": "unknown op"})
