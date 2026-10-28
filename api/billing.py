"""Vercel serverless function: /api/billing — Pro through Paystack (ict_trader/accounts.py).

  POST ?op=checkout  {"period": "month"|"year"}, signed in → {"url"}: the Paystack payment page
  GET  ?op=verify&reference=…, signed in → credits the payment once Paystack confirms it
  POST ?op=webhook   Paystack's server calls this on every successful charge (signature checked)
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import accounts, ratelimit  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def _q(self):
        return parse_qs(urlparse(self.path).query)

    def do_GET(self):
        q = self._q()
        if q.get("op", [""])[0] != "verify":
            return self._send(400, {"error": "unknown op"})
        try:
            token = accounts.bearer(self)
            who = accounts.me(token)
            tx = accounts.verify_reference(q.get("reference", [""])[0])
            res = accounts.apply_payment(tx, expect_user=who["user"]["id"])
            return self._send(200, {**res, "me": accounts.me(token)})
        except PermissionError as e:
            return self._send(401, {"error": str(e)})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": str(e)[:300]})

    def do_POST(self):
        if not ratelimit.guard(self, "billing", 40):
            return
        q = self._q()
        op = q.get("op", [""])[0]
        raw = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 200000))
        try:
            if op == "webhook":
                if not accounts.webhook_ok(raw, self.headers.get("x-paystack-signature", "")):
                    return self._send(401, {"error": "bad signature"})
                ev = json.loads(raw or b"{}")
                if ev.get("event") == "charge.success":
                    try:
                        accounts.apply_payment(ev.get("data") or {})
                    except ValueError:
                        pass  # not ours (another product on the same Paystack account) or not the full price
                return self._send(200, {"ok": True})
            if op == "checkout":
                who = accounts.me(accounts.bearer(self))
                body = json.loads(raw or b"{}")
                user = dict(who.get("user", {}))
                if not user.get("email") and body.get("email"):
                    user["email"] = body.get("email")
                return self._send(200, accounts.checkout(user, str(body.get("period", "month")), f"{accounts.app_url(self.headers.get('host'))}/?paid=1"))  # Paystack adds &reference=… to this
            return self._send(400, {"error": "unknown op"})
        except PermissionError as e:
            return self._send(401, {"error": str(e)})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": str(e)[:300]})

    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
