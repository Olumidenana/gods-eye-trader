"""The last finished scan, kept in Supabase so readers never wait for a fresh one.

The every-minute cron computes the scan once per 5-minute candle and saves it here; the
app and the MT5 feed read that copy back in well under a second instead of running the
whole engine again (which takes about 30 seconds and is what costs Vercel CPU).

Needs the table in db/snapshots.sql and the SUPABASE_URL / SUPABASE_SERVICE_KEY settings
the always-on bot already uses. Without them every call quietly does nothing, and the
scan is simply computed on request as before.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)
TABLE = "scan_snapshots"


def _conf() -> tuple[str, str] | None:
    url, key = os.environ.get("SUPABASE_URL", "").rstrip("/"), os.environ.get("SUPABASE_SERVICE_KEY", "")
    return (url, key) if url and key and os.environ.get("SNAPSHOTS", "1") != "0" else None


def _req(method: str, path: str, body: bytes | None = None, prefer: str | None = None, timeout: float = 8) -> bytes:
    url, key = _conf()  # type: ignore[misc]
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(f"{url}/rest/v1/{path}", data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def save(name: str, report: dict[str, Any]) -> bool:
    """Keep `report` as the latest `name` (e.g. "scan-5"). Never raises."""
    if not _conf():
        return False
    try:
        row = {"name": name, "body": json.dumps(report, separators=(",", ":")), "updated_at": datetime.now(timezone.utc).isoformat()}
        _req("POST", f"{TABLE}?on_conflict=name", json.dumps(row).encode(), prefer="resolution=merge-duplicates,return=minimal")
        return True
    except Exception as e:  # a missing table or a slow database must never break the scan
        log.warning("snapshot save failed: %s", e)
        return False


def load(name: str, max_age_s: float) -> str | None:
    """The saved report as JSON text, if it is younger than `max_age_s`; else None. Never raises."""
    if not _conf():
        return None
    try:
        rows = json.loads(_req("GET", f"{TABLE}?name=eq.{name}&select=body,updated_at"))
        if not rows:
            return None
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(rows[0]["updated_at"].replace("Z", "+00:00"))).total_seconds()
        return rows[0]["body"] if age <= max_age_s else None
    except Exception as e:
        log.warning("snapshot load failed: %s", e)
        return None
