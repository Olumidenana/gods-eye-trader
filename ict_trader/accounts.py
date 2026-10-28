"""Accounts, the free trial, Pro through Paystack, and usage numbers for the owner.

Sign-in is Supabase Auth (Google, or a magic link by email), done in the browser; the app
sends the Supabase access token here and gets back who the person is, their plan, and, while
they are on the trial or Pro, the same signed access pass invite codes give (ict_trader/access.py),
so every protected endpoint works unchanged.

  plan   owner  OWNER_EMAILS (comma-separated) — always Pro, sees the owner dashboard
         pro    paid: pro_until in the future (each Paystack payment adds a month or a year)
         trial  the first TRIAL_DAYS (7) days after sign-up
         free   after that: sessions, the live chart and the learning pages only

Tables (db/accounts.sql): profiles, payments, usage_days. Prices in naira: PRO_MONTH_NGN (5000)
and PRO_YEAR_NGN (50000). Paystack: PAYSTACK_SECRET_KEY; its webhook is /api/billing?op=webhook.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

from . import access

PERIOD_DAYS = {"month": 30, "year": 365, "bot_month": 30, "bot_year": 365}


def trial_days() -> int:
    return int(os.environ.get("TRIAL_DAYS", "7"))


def prices() -> dict[str, int]:
    return {
        "month": int(os.environ.get("PRO_MONTH_NGN", "12500")),
        "year": int(os.environ.get("PRO_YEAR_NGN", "125000")),
        "bot_month": int(os.environ.get("BOT_MONTH_NGN", "25000")),
        "bot_year": int(os.environ.get("BOT_YEAR_NGN", "250000")),
    }


DEFAULT_OWNER_EMAILS = {"olumidenana@gmail.com", "aolumide747@gmail.com"}


def owners() -> set[str]:
    raw = f"{os.environ.get('OWNER_EMAILS', '')},{os.environ.get('OWNER_EMAIL', '')}"
    env = {e.strip().lower() for e in raw.split(",") if e.strip()}
    return env | DEFAULT_OWNER_EMAILS


def app_url(host: str | None = None) -> str:
    """Where Paystack sends people back to: APP_URL (default https://ictflux.app), or the request's own host only
    when it is one of ours (APP_HOSTS, comma-separated, default ictflux.app and ict-trader.vercel.app)."""
    mine = {h.strip().lower() for h in os.environ.get("APP_HOSTS", "ictflux.app,ict-trader.vercel.app").split(",") if h.strip()}
    h = (host or "").split(":")[0].lower()
    return f"https://{h}" if h in mine else os.environ.get("APP_URL", "https://ictflux.app").rstrip("/")


def allowed_emails() -> set[str]:
    """ALLOWED_EMAILS (comma-separated): when set, the app is invite-only and only these emails (and the
    owner) get a plan or a pass. Everyone else can sign in but gets nothing."""
    return {e.strip().lower() for e in os.environ.get("ALLOWED_EMAILS", "").split(",") if e.strip()}


def invited(email: str) -> bool:
    allow = allowed_emails()
    return not allow or email.lower() in allow or email.lower() in owners()


def _t(v: Any) -> datetime | None:
    if not v:
        return None
    d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def public_config() -> dict[str, Any]:
    """What the browser needs to sign in and to show the plans (the anon key is public by design)."""
    has_supabase = bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_ANON_KEY"))
    return {"supabase_url": os.environ.get("SUPABASE_URL", "").rstrip("/"), "anon_key": os.environ.get("SUPABASE_ANON_KEY", ""),
            "accounts": True, "has_supabase": has_supabase, "google_client_id": os.environ.get("GOOGLE_CLIENT_ID", "").strip(),
            "payments": bool(os.environ.get("PAYSTACK_SECRET_KEY")), "prices": prices(), "trial_days": trial_days(),
            "required": access.required(), "invite_only": bool(allowed_emails()), "currency": "NGN"}


# ------------------------------------------------------------------ Supabase & Local Store
_DATA_DIR = os.environ.get("DATA_DIR") or ("/tmp" if os.path.exists("/tmp") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch"))


def _load_local_json(filename: str, default: Any) -> Any:
    try:
        p = os.path.join(_DATA_DIR, filename)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_local_json(filename: str, data: Any) -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        p = os.path.join(_DATA_DIR, filename)
        tmp = f"{p}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, p)
    except Exception:
        pass


def dedupe_profiles(profs: list[dict]) -> list[dict]:
    """Deduplicate user profiles strictly by normalized email (or user_id if no email).
    Merges duplicate entries by keeping earliest created_at, latest last_seen_at, active plans, and highest visits."""
    by_email: dict[str, dict] = {}
    no_email: list[dict] = []

    for p in profs:
        raw_email = (p.get("email") or "").strip().lower()
        if not raw_email or "@" not in raw_email:
            no_email.append(p)
            continue

        if raw_email not in by_email:
            by_email[raw_email] = dict(p)
        else:
            existing = by_email[raw_email]
            ex_name = existing.get("name") or ""
            new_name = p.get("name") or ""
            if not ex_name or ex_name.lower() in ("trader", raw_email.split("@")[0]):
                if new_name and new_name.lower() not in ("trader", raw_email.split("@")[0]):
                    existing["name"] = new_name

            ex_created = str(existing.get("created_at") or "")
            new_created = str(p.get("created_at") or "")
            if new_created and (not ex_created or new_created < ex_created):
                existing["created_at"] = new_created

            ex_seen = str(existing.get("last_seen_at") or "")
            new_seen = str(p.get("last_seen_at") or "")
            if new_seen and (not ex_seen or new_seen > ex_seen):
                existing["last_seen_at"] = new_seen

            existing["visits"] = max(existing.get("visits") or 0, p.get("visits") or 0)

            if p.get("pro_until") and (not existing.get("pro_until") or str(p["pro_until"]) > str(existing.get("pro_until") or "")):
                existing["pro_until"] = p["pro_until"]
            if p.get("plan") in ("pro", "owner"):
                existing["plan"] = p["plan"]
            elif existing.get("plan") not in ("pro", "owner") and p.get("plan") == "trial":
                existing["plan"] = "trial"

            if p.get("user_id") and "-" in str(p.get("user_id")) and "-" not in str(existing.get("user_id", "")):
                existing["user_id"] = p["user_id"]

    merged = list(by_email.values())
    seen_ids = {p.get("user_id") for p in merged if p.get("user_id")}
    for p in no_email:
        if p.get("user_id") and p["user_id"] not in seen_ids:
            merged.append(p)
            seen_ids.add(p["user_id"])

    return sorted(merged, key=lambda p: str(p.get("created_at") or ""), reverse=True)


class LocalStore:
    """Fallback store when Supabase is not configured in Vercel, persisting to disk so signups survive restarts."""
    _profiles: dict[str, dict] = {}
    _payments: dict[str, dict] = {}
    _passwords: dict[str, str] = {}
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        if not cls._loaded:
            cls._loaded = True
            p = _load_local_json("local_profiles.json", {})
            if isinstance(p, dict):
                cls._profiles.update(p)
            pw = _load_local_json("local_passwords.json", {})
            if isinstance(pw, dict):
                cls._passwords.update(pw)
            pay = _load_local_json("local_payments.json", {})
            if isinstance(pay, dict):
                cls._payments.update(pay)

    def set_password(self, email: str, password: str) -> None:
        self._ensure_loaded()
        salt = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
        h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
        self._passwords[email.lower()] = f"{salt}:{h}"
        _save_local_json("local_passwords.json", self._passwords)

    def verify_password(self, email: str, password: str) -> bool:
        self._ensure_loaded()
        stored = self._passwords.get(email.lower())
        if not stored:
            return False
        try:
            salt, h = stored.split(":", 1)
            expected = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
            return hmac.compare_digest(h, expected)
        except Exception:
            return False

    def has_password(self, email: str) -> bool:
        self._ensure_loaded()
        return email.lower() in self._passwords

    def user_from_token(self, token: str) -> dict:
        self._ensure_loaded()
        if not token:
            raise PermissionError("Please sign in")
        try:
            p = access.verify(token)
            uid = p.get("c", "").replace("u:", "") or "guest"
            email = p.get("e") or f"{uid}@user.local"
            name = p.get("n") or email.split("@")[0]
            return {"id": uid, "email": email, "email_confirmed_at": "yes", "user_metadata": {"name": name}}
        except Exception:
            pass
        if token.startswith("dev:"):
            email = token.split("dev:", 1)[1].strip().lower()
            uid = hashlib.md5(email.encode()).hexdigest()[:16]
            return {"id": uid, "email": email, "email_confirmed_at": "yes", "user_metadata": {"name": email.split("@")[0]}}
        raise PermissionError("Your sign-in has expired. Please sign in again.")

    def profile(self, user_id: str) -> dict | None:
        self._ensure_loaded()
        return self._profiles.get(user_id)

    def create_profile(self, row: dict) -> dict:
        self._ensure_loaded()
        self._profiles[row["user_id"]] = dict(row)
        _save_local_json("local_profiles.json", self._profiles)
        return self._profiles[row["user_id"]]

    def update_profile(self, user_id: str, fields: dict) -> None:
        self._ensure_loaded()
        if user_id in self._profiles:
            self._profiles[user_id].update(fields)
            _save_local_json("local_profiles.json", self._profiles)

    def count_visit(self, user_id: str, day: str) -> None:
        pass

    def add_payment(self, row: dict) -> bool:
        self._ensure_loaded()
        ref = row.get("reference")
        if ref in self._payments:
            return False
        self._payments[ref] = dict(row)
        uid = row.get("user_id")
        if uid and uid in self._profiles:
            days = PERIOD_DAYS.get(row.get("period", "month"), 30)
            now = datetime.now(timezone.utc)
            cur = _t(self._profiles[uid].get("pro_until"))
            base = max(cur, now) if cur else now
            self._profiles[uid]["pro_until"] = (base + timedelta(days=days)).isoformat()
            _save_local_json("local_profiles.json", self._profiles)
        _save_local_json("local_payments.json", self._payments)
        return True

    def delete_payment(self, reference: str) -> None:
        self._ensure_loaded()
        self._payments.pop(reference, None)
        _save_local_json("local_payments.json", self._payments)

    def payments(self, user_id: str) -> list[dict]:
        self._ensure_loaded()
        return [p for p in self._payments.values() if p.get("user_id") == user_id]

    def all_profiles(self) -> list[dict]:
        self._ensure_loaded()
        return dedupe_profiles(list(self._profiles.values()))

    def all_payments(self) -> list[dict]:
        self._ensure_loaded()
        return list(self._payments.values())

    def usage(self, since: str) -> list[dict]:
        return []


def default_store() -> Any:
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
        try:
            return Store()
        except Exception:
            return LocalStore()
    return LocalStore()


def signin_token(email: str, name: str = "", now: float | None = None) -> dict[str, Any]:
    email = email.strip().lower()
    name = name.strip() or email.split("@")[0]
    uid = hashlib.md5(email.encode()).hexdigest()[:16]
    now_f = now if now is not None else time.time()
    exp = now_f + 30 * 86400  # 30 days session
    body = access._b64(json.dumps({"c": f"u:{uid}", "e": email, "n": name, "p": "pro", "x": int(exp)}).encode())
    token = f"{body}.{access._sign(body)}"
    return {"access_token": token, "user": {"id": uid, "email": email, "name": name}, "expires_in": 30 * 86400}


def authenticate(email: str, password: str, name: str = "", mode: str = "signin",
                 device_id: str | None = None, client_ip: str | None = None) -> dict[str, Any]:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("Valid email address required")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")
    
    store = default_store()
    name = name.strip() or email.split("@")[0]

    if isinstance(store, LocalStore):
        if store.has_password(email):
            if mode == "register":
                if not store.verify_password(email, password):
                    raise ValueError("An account already exists with this email. Please sign in.")
            else:
                if not store.verify_password(email, password):
                    raise ValueError("Incorrect password. Please try again.")
        else:
            store.set_password(email, password)
    
    session = signin_token(email, name)
    me_data = me(session["access_token"], store=store, device_id=device_id, client_ip=client_ip)
    return {**session, **me_data}


def add_user(email: str, name: str = "", store: Any | None = None, now: datetime | None = None) -> dict[str, Any]:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("Valid email address required")
    name = name.strip() or email.split("@")[0]
    uid = hashlib.md5(email.encode()).hexdigest()[:16]
    now = now or datetime.now(timezone.utc)
    trial_end = now + timedelta(days=trial_days())
    store = store or default_store()
    row = {
        "user_id": uid,
        "email": email,
        "name": name,
        "created_at": now.isoformat(),
        "trial_ends_at": trial_end.isoformat(),
        "last_seen_at": now.isoformat(),
        "visits": 1,
    }
    try:
        store.create_profile(row)
    except Exception:
        pass
    LocalStore().create_profile(row)
    return row


def _http(method: str, url: str, body: Any = None, headers: dict[str, str] | None = None, timeout: float = 10) -> tuple[int, Any]:
    default_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
    }
    merged_headers = {**default_headers, **(headers or {})}
    req = urllib.request.Request(url, method=method, headers=merged_headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw) if raw else None
        except ValueError:
            return e.code, {"message": raw[:200].decode("utf-8", "replace")}


class Store:
    """profiles, payments and usage_days over Supabase REST with the service key."""

    def __init__(self) -> None:
        self.url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        if not self.url or not self.key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in Vercel")

    def _req(self, method: str, path: str, body: Any = None, prefer: str | None = None) -> Any:
        h = {"apikey": self.key, "Authorization": f"Bearer {self.key}"}
        if prefer:
            h["Prefer"] = prefer
        code, data = _http(method, f"{self.url}/rest/v1/{path}", body, h)
        if code >= 300:
            msg = (data or {}).get("message") if isinstance(data, dict) else data
            hint = " Run db/accounts.sql in the Supabase SQL editor." if code in (404,) or "does not exist" in str(msg) else ""
            raise RuntimeError(f"Database {code}: {msg}.{hint}")
        return data

    def user_from_token(self, token: str) -> dict:
        """The signed-in Supabase user for an access token, or PermissionError."""
        code, data = _http("GET", f"{self.url}/auth/v1/user", headers={"apikey": self.key, "Authorization": f"Bearer {token}"})
        if code != 200 or not isinstance(data, dict) or not data.get("id"):
            raise PermissionError("Your sign-in has expired. Please sign in again.")
        return data

    def profile(self, user_id: str) -> dict | None:
        rows = self._req("GET", f"profiles?user_id=eq.{user_id}&select=*")
        return rows[0] if rows else None

    def create_profile(self, row: dict) -> dict:
        rows = self._req("POST", "profiles?on_conflict=user_id", row, prefer="resolution=ignore-duplicates,return=representation")
        return rows[0] if rows else self.profile(row["user_id"])  # type: ignore[return-value]

    def update_profile(self, user_id: str, fields: dict) -> None:
        self._req("PATCH", f"profiles?user_id=eq.{user_id}", fields)

    def count_visit(self, user_id: str, day: str) -> None:
        self._req("POST", "rpc/count_visit", {"uid": user_id, "d": day})

    def add_payment(self, row: dict) -> bool:
        """True if this payment is new (a reference is only ever applied once)."""
        rows = self._req("POST", "payments?on_conflict=reference", row, prefer="resolution=ignore-duplicates,return=representation")
        return bool(rows)

    def delete_payment(self, reference: str) -> None:
        self._req("DELETE", f"payments?reference=eq.{quote(reference, safe='')}")

    def payments(self, user_id: str) -> list[dict]:
        return self._req("GET", f"payments?user_id=eq.{user_id}&select=reference,amount_kobo,period,paid_at&order=paid_at.desc&limit=24") or []

    # the owner dashboard
    def all_profiles(self) -> list[dict]:
        by_id: dict[str, dict] = {}
        # 1. Fetch from profiles table if it exists
        try:
            rows = self._req("GET", "profiles?select=user_id,email,name,created_at,trial_ends_at,pro_until,last_seen_at,visits&order=created_at.desc&limit=5000") or []
            for r in rows:
                if r.get("user_id"):
                    by_id[r["user_id"]] = r
        except Exception:
            pass  # if db/accounts.sql hasn't been run yet in Supabase, we still fetch auth users below

        # 2. Always fetch all registered users from Supabase Auth admin API (so no signup is ever missed)
        try:
            code, auth_data = _http("GET", f"{self.url}/auth/v1/admin/users?per_page=1000",
                                    headers={"apikey": self.key, "Authorization": f"Bearer {self.key}"})
            if code == 200 and auth_data:
                users_list = auth_data.get("users") if isinstance(auth_data, dict) else (auth_data if isinstance(auth_data, list) else [])
                for u in users_list or []:
                    uid = u.get("id")
                    if not uid:
                        continue
                    email = (u.get("email") or "").strip().lower()
                    meta = u.get("user_metadata") or {}
                    name = meta.get("full_name") or meta.get("name") or (email.split("@")[0] if "@" in email else "Trader")
                    created = u.get("created_at")
                    last_seen = u.get("last_sign_in_at")
                    if uid in by_id:
                        if not by_id[uid].get("email"):
                            by_id[uid]["email"] = email
                        if not by_id[uid].get("name"):
                            by_id[uid]["name"] = name
                        if not by_id[uid].get("last_seen_at"):
                            by_id[uid]["last_seen_at"] = last_seen
                    else:
                        by_id[uid] = {
                            "user_id": uid,
                            "email": email,
                            "name": name,
                            "created_at": created,
                            "last_seen_at": last_seen,
                            "trial_ends_at": None,
                            "pro_until": None,
                            "visits": 1 if last_seen else 0
                        }
        except Exception:
            pass

        return dedupe_profiles(list(by_id.values()))

    def all_payments(self) -> list[dict]:
        try:
            return self._req("GET", "payments?select=reference,email,amount_kobo,period,paid_at&order=paid_at.desc&limit=5000") or []
        except Exception:
            return []

    def usage(self, since: str) -> list[dict]:
        try:
            return self._req("GET", f"usage_days?day=gte.{since}&select=user_id,day,hits&limit=50000") or []
        except Exception:
            return []


# ------------------------------------------------------------------ plans
def plan_of(profile: dict, email: str, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    pro_until, trial_end = _t(profile.get("pro_until")), _t(profile.get("trial_ends_at"))
    if email.lower() in owners():
        return {"plan": "owner", "active": True, "until": None, "label": "Owner"}
    if pro_until and pro_until > now:
        return {"plan": "pro", "active": True, "until": pro_until.isoformat(), "label": "Pro",
                "days_left": max(0, math.ceil((pro_until - now).total_seconds() / 86400))}
    if trial_end and trial_end > now:
        return {"plan": "trial", "active": True, "until": trial_end.isoformat(), "label": "Free trial",
                "days_left": max(0, math.ceil((trial_end - now).total_seconds() / 86400))}
    # If trial_ends_at is not explicitly set, compute 7 days from created_at
    created = _t(profile.get("created_at"))
    if not trial_end and created and (created + timedelta(days=trial_days())) > now:
        t_end = created + timedelta(days=trial_days())
        return {"plan": "trial", "active": True, "until": t_end.isoformat(), "label": "Free trial",
                "days_left": max(0, math.ceil((t_end - now).total_seconds() / 86400))}
    return {"plan": "free", "active": False, "until": None, "label": "Free",
            "ended": (pro_until or trial_end).isoformat() if (pro_until or trial_end) else None}


_TRIAL_DEVICES: dict[str, str] = {}  # device_id -> user_id


def me(token: str, store: Store | None = None, now: datetime | None = None,
       device_id: str | None = None, client_ip: str | None = None) -> dict[str, Any]:
    """Who is signed in, their plan, and an access pass while the trial or Pro lasts. Creates the
    profile (starting the trial) on first sign-in and counts the day's visit."""
    now = now or datetime.now(timezone.utc)
    store = store or default_store()
    try:
        user = store.user_from_token(token)
    except Exception:
        try:
            ls = LocalStore()
            user = ls.user_from_token(token)
            store = ls
        except Exception:
            raise PermissionError("Your sign-in has expired. Please sign in again.") from None
    uid, raw_email = user["id"], (user.get("email") or "").lower().strip()
    is_confirmed = bool(user.get("email_confirmed_at") or user.get("confirmed_at") or (user.get("app_metadata") or {}).get("provider") == "google")
    effective_email = raw_email if is_confirmed else ""
    meta = user.get("user_metadata") or {}
    name = meta.get("full_name") or meta.get("name") or (raw_email.split("@")[0] if "@" in raw_email else "Trader")
    if not invited(effective_email):  # invite-only: no profile, no trial, no pass
        return {"user": {"id": uid, "email": raw_email, "name": name, "avatar": meta.get("avatar_url") or meta.get("picture")},
                "plan": {"plan": "invite", "active": False, "until": None, "label": "Invite only"}, "invite_only": True}
    
    # Anti-abuse: only strictly block multiple trials per device if explicitly turned on in env
    strict_device = os.environ.get("STRICT_TRIAL_DEVICE", "0") == "1"
    device_key = (device_id or "").strip() or (client_ip or "").strip()
    prior_user = _TRIAL_DEVICES.get(device_key) if (device_key and strict_device) else None
    trial_blocked = bool(prior_user and prior_user != uid and effective_email not in owners())

    prof = store.profile(uid)
    if not prof:
        trial_end = now if trial_blocked else now + timedelta(days=trial_days())
        prof = store.create_profile({
            "user_id": uid, "email": raw_email, "name": name, "avatar": meta.get("avatar_url") or meta.get("picture"),
            "created_at": now.isoformat(), "trial_ends_at": trial_end.isoformat(), "visits": 0})
    if device_key:
        _TRIAL_DEVICES[device_key] = uid

    try:
        store.count_visit(uid, now.date().isoformat())
        store.update_profile(uid, {"last_seen_at": now.isoformat(), "email": raw_email, "name": name})
    except Exception:
        pass  # the numbers are informational; never block a sign-in on them
    plan = plan_of(prof, effective_email, now)
    out = {"user": {"id": uid, "email": raw_email, "name": name, "avatar": prof.get("avatar")}, "plan": plan,
           "created_at": prof.get("created_at"), "trial_ends_at": prof.get("trial_ends_at"), "pro_until": prof.get("pro_until")}
    if trial_blocked and plan["plan"] == "free":
        out["trial_blocked"] = True
    if plan["active"]:
        until = _t(plan["until"]) if plan["until"] else None
        out["pass"] = access.issue(f"u:{uid}", name, "pro", until, now.timestamp())
    return out


def signout(token: str, store: Store | None = None, now: datetime | None = None) -> bool:
    """Marks a user as logged out so they immediately show as offline in presence status."""
    now = now or datetime.now(timezone.utc)
    store = store or default_store()
    try:
        user = store.user_from_token(token)
    except Exception:
        try:
            ls = LocalStore()
            user = ls.user_from_token(token)
            store = ls
        except Exception:
            return False
    uid = user.get("id")
    if not uid:
        return False
    past = (now - timedelta(minutes=10)).isoformat()
    try:
        store.update_profile(uid, {"last_seen_at": past})
        return True
    except Exception:
        return False


def bearer(handler) -> str:
    h = handler.headers.get("Authorization") or ""
    if not h.lower().startswith("bearer "):
        raise PermissionError("Please sign in")
    return h[7:].strip()


# ------------------------------------------------------------------ Paystack
PAYSTACK = "https://api.paystack.co"


def _paystack(method: str, path: str, body: Any = None) -> dict:
    key = os.environ.get("PAYSTACK_SECRET_KEY", "")
    if not key:
        raise RuntimeError("Paystack is not configured: PAYSTACK_SECRET_KEY is missing from Vercel Environment Variables.")
    code, data = _http(method, PAYSTACK + path, body, {"Authorization": f"Bearer {key}"}, timeout=15)
    if code >= 300 or not isinstance(data, dict) or not data.get("status"):
        msg = (data or {}).get("message") if isinstance(data, dict) else f"HTTP {code}"
        raise RuntimeError(f"Paystack error: {msg}")
    return data["data"]


def checkout(user: dict, period: str, callback_url: str, now: float | None = None) -> dict[str, str]:
    """A Paystack payment page for a month or a year of Pro."""
    if period not in PERIOD_DAYS:
        raise ValueError(f"Unknown plan period '{period}'. Options: {', '.join(PERIOD_DAYS.keys())}")
    email = (user.get("email") or "").strip()
    if not email or "@" not in email:
        raise ValueError("A valid email address is required for Paystack checkout. Please ensure your account has an email.")
    amount = prices()[period] * 100  # kobo
    ref = f"ifx-{user.get('id', 'guest')[:8]}-{period[:2]}-{int((now or time.time()) * 1000)}"
    data = _paystack("POST", "/transaction/initialize", {
        "email": email, "amount": amount, "currency": "NGN", "reference": ref, "callback_url": callback_url,
        "metadata": {"user_id": user.get("id", ""), "period": period, "custom_fields": [
            {"display_name": "Plan", "variable_name": "plan", "value": f"ICT Flux Pro · {period}"}]}})
    return {"url": data["authorization_url"], "reference": ref}


def apply_payment(tx: dict, store: Store | None = None, now: datetime | None = None, expect_user: str | None = None) -> dict[str, Any]:
    """Credit a successful Paystack transaction: once per reference, the full price only, and (with
    `expect_user`) only to the account that is checking it."""
    store = store or Store()
    now = now or datetime.now(timezone.utc)
    if tx.get("status") != "success":
        raise ValueError(f"payment not successful ({tx.get('status')})")
    meta = tx.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            meta = {}
    uid, period = meta.get("user_id"), meta.get("period")
    if not uid or period not in PERIOD_DAYS:
        raise ValueError("payment is not for an ICT Flux plan")
    if expect_user and uid != expect_user:
        raise ValueError("this payment belongs to another account")
    if tx.get("currency") != "NGN" or int(tx.get("amount") or 0) < prices()[period] * 100:
        raise ValueError("amount paid does not match the plan price")
    if tx.get("domain") == "test" and not os.environ.get("PAYSTACK_SECRET_KEY", "").startswith("sk_test"):
        raise ValueError("a test-mode payment cannot buy a live plan")
    new = store.add_payment({"reference": tx["reference"], "user_id": uid, "email": (tx.get("customer") or {}).get("email"),
                             "amount_kobo": int(tx["amount"]), "period": period, "paid_at": tx.get("paid_at") or now.isoformat()})
    prof = store.profile(uid) or {}
    if new:
        base = max(now, _t(prof.get("pro_until")) or now)
        until = base + timedelta(days=PERIOD_DAYS[period])
        try:
            store.update_profile(uid, {"pro_until": until.isoformat()})
        except Exception:
            drop = getattr(store, "delete_payment", None)
            if drop:
                drop(tx["reference"])  # not credited, so not recorded: the next check credits it
            raise
        return {"applied": True, "pro_until": until.isoformat()}
    return {"applied": False, "pro_until": prof.get("pro_until")}


def verify_reference(reference: str) -> dict:
    return _paystack("GET", f"/transaction/verify/{quote(reference, safe='')}")


def webhook_ok(raw: bytes, signature: str) -> bool:
    key = os.environ.get("PAYSTACK_SECRET_KEY", "")
    want = hmac.new(key.encode(), raw, hashlib.sha512).hexdigest().encode()
    return bool(key) and hmac.compare_digest(want, (signature or "").encode("utf-8", "ignore"))


# ------------------------------------------------------------------ the owner's numbers
def stats(store: Any | None = None, now: datetime | None = None) -> dict[str, Any]:
    if store is None:
        try:
            store = Store()
        except Exception:
            store = default_store()
    now = now or datetime.now(timezone.utc)
    profs = dedupe_profiles(store.all_profiles() + LocalStore().all_profiles())
    pays = store.all_payments()
    usage = store.usage((now - timedelta(days=30)).date().isoformat())
    plans = [plan_of(p, p.get("email") or "", now)["plan"] for p in profs]
    day = lambda n: (now - timedelta(days=n)).date().isoformat()  # noqa: E731
    active = lambda n: len({u["user_id"] for u in usage if u["day"] >= day(n - 1)})  # noqa: E731
    signups: dict[str, int] = {}
    for p in profs:
        d = (_t(p.get("created_at")) or now).date().isoformat()
        if d >= day(29):
            signups[d] = signups.get(d, 0) + 1
    daily_active: dict[str, int] = {}
    for u in usage:
        daily_active[u["day"]] = daily_active.get(u["day"], 0) + 1
    month_ago = now - timedelta(days=30)
    revenue_30 = sum(p["amount_kobo"] for p in pays if (_t(p.get("paid_at")) or now) >= month_ago) // 100
    def user_status(p):
        ls = _t(p.get("last_seen_at"))
        sec = int((now - ls).total_seconds()) if ls else None
        return {
            "user_id": p.get("user_id"),
            "email": p.get("email"),
            "name": p.get("name"),
            "joined": p.get("created_at"),
            "last_seen": p.get("last_seen_at"),
            "seconds_ago": sec,
            "is_online": bool(sec is not None and sec <= 300),
            "visits": p.get("visits") or 0,
            "plan": plan_of(p, p.get("email") or "", now)["plan"],
        }

    return {
        "users": len(profs), "active_today": active(1), "active_7d": active(7), "active_30d": active(30),
        "on_trial": plans.count("trial"), "pro": plans.count("pro"), "free": plans.count("free"),
        "revenue_30d_ngn": revenue_30, "revenue_total_ngn": sum(p["amount_kobo"] for p in pays) // 100, "payments": len(pays),
        "signups_by_day": [{"day": day(n), "n": signups.get(day(n), 0)} for n in range(29, -1, -1)],
        "active_by_day": [{"day": day(n), "n": daily_active.get(day(n), 0)} for n in range(29, -1, -1)],
        "recent_users": [user_status(p) for p in profs[:1000]],
        "recent_payments": pays[:50],
        "server_time": now.isoformat(),
    }
