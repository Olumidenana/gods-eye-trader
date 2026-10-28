"""Vercel serverless function: /api/push

GET  ?op=key          the server's public key, for the app to subscribe with
POST ?op=subscribe    {"subscription": <PushSubscription JSON>, "tier": "aplus|a|b"} keeps this phone for alerts
POST ?op=tier         {"endpoint": "...", "tier": "aplus|a|b"} changes which trades this phone hears about
POST ?op=unsubscribe  {"endpoint": "..."} stops alerts to this phone
POST ?op=test         {"subscription": ...} sends this phone one test notification
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ict_trader import access, webpush  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        op = parse_qs(urlparse(self.path).query).get("op", [""])[0]
        if op != "key":
            return self._send(400, {"error": "unknown op"})
        key = webpush.public_key()
        return self._send(200, {"key": key} if key else {"key": None, "error": "Push is not set up: CRON_SECRET is missing in Vercel."})

    def do_POST(self):
        if not access.gate(self, proxy=False):   # a POST has a body: the gate's shared-GET proxy would replay it without one ("unknown op")
            return
        op = parse_qs(urlparse(self.path).query).get("op", [""])[0]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(n, 20000)) or b"{}")
            if op == "subscribe":
                webpush.subscribe(webpush.valid(body.get("subscription")), body.get("tier"))
                return self._send(200, {"ok": True})
            if op == "tier":
                endpoint = body.get("endpoint")
                if not isinstance(endpoint, str) or not endpoint.startswith("https://") or len(endpoint) > 1000:
                    raise ValueError("endpoint missing")
                webpush.set_tier(endpoint, body.get("tier"))
                return self._send(200, {"ok": True})
            if op == "unsubscribe":
                endpoint = body.get("endpoint")
                if not isinstance(endpoint, str):
                    raise ValueError("endpoint missing")
                webpush.unsubscribe(endpoint)
                return self._send(200, {"ok": True})
            if op == "test":
                code = webpush.send(webpush.valid(body.get("subscription")), {
                    "title": "✅ ICT Flux notifications are on",
                    "body": "Trade alerts will arrive here: limits placed, fills, progress and closes for the trades your setting allows, each naming its model.",
                    "tag": "test", "url": "/#home"})
                return self._send(200 if 200 <= code < 300 else 502, {"status": code})
            return self._send(400, {"error": "unknown op"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            msg = str(e)
            if "push_subscriptions" in msg or "404" in msg:
                msg += " — run db/snapshots.sql in the Supabase SQL editor once."
            return self._send(500, {"error": msg[:300]})

    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
