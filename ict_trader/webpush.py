"""Notifications straight to the installed app (Web Push), next to ntfy and Telegram.

A phone that taps "Turn on notifications" in the app hands the server a push subscription
(an endpoint at Google, Apple or Mozilla plus two keys). Each alert is encrypted for that
phone (RFC 8291, aes128gcm) and posted to its endpoint, signed with the server's VAPID key
(RFC 8292). Nothing else needs installing: `cryptography` does the maths.

The VAPID key is derived from CRON_SECRET (or set VAPID_PRIVATE_KEY, the 32-byte key in
base64url), so there is no extra secret to manage. Changing CRON_SECRET changes the key, and
phones then need to turn notifications on again. Subscriptions live in Supabase
(db/snapshots.sql, table push_subscriptions). Each phone also keeps a tier (A+ only, A and up, B and up;
see tiers.py) and only receives the alerts its tier allows.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from . import tiers
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

P256_ORDER = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
TABLE = "push_subscriptions"


def b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def unb64u(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _hkdf(secret: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(secret)


def vapid_key() -> ec.EllipticCurvePrivateKey | None:
    """The server's signing key, or None when push is not set up."""
    raw = os.environ.get("VAPID_PRIVATE_KEY")
    if raw:
        return ec.derive_private_key(int.from_bytes(unb64u(raw), "big"), ec.SECP256R1())
    secret = os.environ.get("CRON_SECRET")
    if not secret:
        return None
    n = int.from_bytes(_hkdf(secret.encode(), b"ictflux-vapid", b"vapid p-256 key", 32), "big")
    return ec.derive_private_key(n % (P256_ORDER - 1) + 1, ec.SECP256R1())


def public_key() -> str | None:
    """The key the app subscribes with (uncompressed P-256 point, base64url)."""
    k = vapid_key()
    return b64u(k.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)) if k else None


def encrypt(payload: bytes, p256dh: str, auth: str, salt: bytes | None = None,
            server_key: ec.EllipticCurvePrivateKey | None = None) -> bytes:
    """RFC 8291 message encryption with the aes128gcm content coding (one record)."""
    ua_public, auth_secret = unb64u(p256dh), unb64u(auth)
    as_key = server_key or ec.generate_private_key(ec.SECP256R1())
    as_public = as_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    shared = as_key.exchange(ec.ECDH(), ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ua_public))
    ikm = _hkdf(shared, auth_secret, b"WebPush: info\x00" + ua_public + as_public, 32)
    salt = salt or os.urandom(16)
    cek = _hkdf(ikm, salt, b"Content-Encoding: aes128gcm\x00", 16)
    nonce = _hkdf(ikm, salt, b"Content-Encoding: nonce\x00", 12)
    body = AESGCM(cek).encrypt(nonce, payload + b"\x02", None)  # \x02: the last (only) record
    return salt + struct.pack(">I", 4096) + bytes([len(as_public)]) + as_public + body


def vapid_header(endpoint: str, key: ec.EllipticCurvePrivateKey, now: float | None = None) -> str:
    """`Authorization: vapid t=<JWT>, k=<public key>` for this push service (RFC 8292)."""
    u = urlparse(endpoint)
    claims = {"aud": f"{u.scheme}://{u.netloc}", "exp": int((now or time.time()) + 12 * 3600),
              "sub": os.environ.get("VAPID_SUBJECT", "https://ictflux.app")}
    signing = b64u(json.dumps({"typ": "JWT", "alg": "ES256"}, separators=(",", ":")).encode()) + "." + \
        b64u(json.dumps(claims, separators=(",", ":")).encode())
    r, s = decode_dss_signature(key.sign(signing.encode(), ec.ECDSA(hashes.SHA256())))
    jwt = signing + "." + b64u(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
    pub = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return f"vapid t={jwt}, k={b64u(pub)}"


def send(sub: dict[str, Any], message: dict[str, Any], key: ec.EllipticCurvePrivateKey | None = None,
         urgency: str = "high", ttl: int = 3600) -> int:
    """Post one message to one subscription. Returns the HTTP status (404/410: the phone unsubscribed)."""
    key = key or vapid_key()
    if key is None:
        raise RuntimeError("push is not set up (CRON_SECRET or VAPID_PRIVATE_KEY)")
    body = encrypt(json.dumps(message, separators=(",", ":")).encode()[:3800], sub["p256dh"], sub["auth"])
    req = urllib.request.Request(sub["endpoint"], data=body, method="POST", headers={
        "Authorization": vapid_header(sub["endpoint"], key), "Content-Encoding": "aes128gcm",
        "Content-Type": "application/octet-stream", "TTL": str(ttl), "Urgency": urgency})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


# ---- subscriptions, kept in Supabase with the service key ----
def _db(method: str, path: str, body: Any = None, prefer: str | None = None) -> Any:
    url, skey = os.environ.get("SUPABASE_URL", "").rstrip("/"), os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not skey:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    headers = {"apikey": skey, "Authorization": f"Bearer {skey}", "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(f"{url}/rest/v1/{path}", method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=8) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def valid(sub: Any) -> dict[str, str]:
    """A subscription from the browser, checked: an https push endpoint and its two keys."""
    if not isinstance(sub, dict):
        raise ValueError("subscription missing")
    endpoint, keys = sub.get("endpoint"), sub.get("keys") or {}
    if not isinstance(endpoint, str) or not endpoint.startswith("https://") or len(endpoint) > 1000:
        raise ValueError("not a push endpoint")
    p256dh, auth = keys.get("p256dh"), keys.get("auth")
    try:
        if len(unb64u(p256dh)) != 65 or len(unb64u(auth)) != 16:
            raise ValueError
    except Exception:
        raise ValueError("bad subscription keys") from None
    return {"endpoint": endpoint, "p256dh": p256dh, "auth": auth}


def _no_column(e: Exception) -> bool:
    """The tier column is missing: db/snapshots.sql has not been re-run since tiers were added."""
    return isinstance(e, urllib.error.HTTPError) and e.code in (400, 404)


def subscribe(sub: dict[str, str], tier: str | None = None) -> None:
    row = dict(sub)
    if tier:
        row["tier"] = tiers.clean(tier)
    try:
        _db("POST", f"{TABLE}?on_conflict=endpoint", row, prefer="resolution=merge-duplicates,return=minimal")
    except Exception as e:
        if not (tier and _no_column(e)):
            raise
        row.pop("tier")  # the phone still gets alerts, on the default tier, until the column is added
        _db("POST", f"{TABLE}?on_conflict=endpoint", row, prefer="resolution=merge-duplicates,return=minimal")


def set_tier(endpoint: str, tier: str) -> None:
    from urllib.parse import quote
    _db("PATCH", f"{TABLE}?endpoint=eq.{quote(endpoint, safe='')}", {"tier": tiers.clean(tier)}, prefer="return=minimal")


def unsubscribe(endpoint: str) -> None:
    from urllib.parse import quote
    _db("DELETE", f"{TABLE}?endpoint=eq.{quote(endpoint, safe='')}")


def subscriptions() -> list[dict[str, str]]:
    try:
        return _db("GET", f"{TABLE}?select=endpoint,p256dh,auth,tier") or []
    except Exception as e:
        if not _no_column(e):
            raise
        return _db("GET", f"{TABLE}?select=endpoint,p256dh,auth") or []


def broadcast(events: list[dict[str, Any]], click: str | None = None) -> dict[str, Any]:
    """Push each event to the subscribed phones whose tier allows it; forget phones that unsubscribed. Never raises."""
    if not events:
        return {"sent": 0}
    try:
        key = vapid_key()
        if key is None:
            return {"sent": 0, "off": "no CRON_SECRET"}
        subs = subscriptions()
    except Exception as e:  # no table yet, or Supabase unreachable: ntfy still went out
        return {"sent": 0, "error": str(e)[:200]}
    sent, gone, errors = 0, [], []
    for sub in subs:
        for ev in events:
            if not tiers.allows(sub.get("tier"), ev):
                continue
            msg = {"title": ev["title"], "body": ev["body"], "tag": ev["id"], "url": click or "/",
                   "quiet": ev.get("priority") == "low"}
            try:
                code = send(sub, msg, key, urgency="normal" if ev.get("priority") == "low" else "high")
            except Exception as e:
                errors.append(str(e)[:120])
                continue
            if code in (404, 410):
                gone.append(sub["endpoint"])
                break
            if 200 <= code < 300:
                sent += 1
            else:
                errors.append(f"HTTP {code}")
    for endpoint in gone:
        try:
            unsubscribe(endpoint)
        except Exception:
            pass
    return {"sent": sent, "phones": len(subs) - len(gone), "removed": len(gone), "errors": errors[:5]}
