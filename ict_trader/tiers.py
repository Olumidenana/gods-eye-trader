"""Alert tiers: how picky a phone (or the ntfy / Telegram channel) is about which trades it hears about.

  aplus  A+ trades only: every confluence lined up. The quietest setting.
  a      A+ and A trades (the default): the strong setups.
  b      A+, A and B trades, plus the heads-ups (a killzone opening, a setup forming).

Every event the scanner builds carries a `grade` (the trade's grade, or None for a heads-up) and the tier
decides who gets it. Each trade alert also carries a one-line suggestion that puts the grade and the
adviser's verdict together ("A+ trade, adviser says TAKE: the one to take").
"""

from __future__ import annotations

import os
from typing import Any

TIERS = ("aplus", "a", "b")
DEFAULT_TIER = "a"
LABELS = {"aplus": "A+ only", "a": "A and A+", "b": "B and up"}
_GRADES = {"aplus": ("A+",), "a": ("A+", "A"), "b": ("A+", "A", "B")}


def clean(tier: Any) -> str:
    """A tier name from outside (a request, the environment): the default when it is not one of ours."""
    t = str(tier or "").strip().lower().replace("+", "plus").replace(" ", "")
    return {"aplus": "aplus", "a+": "aplus", "aplusonly": "aplus", "a": "a", "b": "b"}.get(t, t if t in TIERS else DEFAULT_TIER)


def channel_tier() -> str:
    """The tier for the shared ntfy topic and Telegram: NOTIFY_MIN_GRADE (A+ / A / B), A by default."""
    g = os.environ.get("NOTIFY_MIN_GRADE", "A").strip().upper()
    return {"A+": "aplus", "A": "a", "B": "b", "C": "b"}.get(g, DEFAULT_TIER)


def allows(tier: Any, ev: dict[str, Any]) -> bool:
    """Does a phone on this tier get this event? Graded events by grade; heads-ups only on the widest tier.
    A cancelled limit follows the grade it was sent with; one sent before grades were kept goes to A and B."""
    t = clean(tier)
    g = ev.get("grade")
    if g:
        return g in _GRADES[t]
    if ev.get("kind") in ("headsup", "forming"):
        return t == "b"
    if ev.get("kind") in ("cancel", "news"):   # a news heads-up is for anyone who trades; the A+-only setting stays quiet
        return t != "aplus"
    return True


def suggestion(grade: str | None, advice: dict[str, Any] | None, models: int = 1, filled: bool = False) -> str:
    """The call to make on a trade, said plainly: the adviser's verdict (PULL THE TRIGGER / SIZE DOWN / STAND DOWN)
    with the grade and the score behind it. `filled`: the order is already in, so the line says how to carry it."""
    v = (advice or {}).get("verdict")
    score = (advice or {}).get("score")
    sc = f" {score}/100" if score is not None else ""
    agree = f" {models} models agree." if models > 1 else ""
    why = f"{grade} trade, adviser says {v}{sc}" if v else f"{grade} trade"
    if filled and grade in ("A+", "A", "B"):
        cue = {"SKIP": "the adviser leans SKIP, so keep the risk small", "CAUTION": "the adviser says CAUTION, so keep the risk small"}.get(v or "", "follow the plan to the stop and target")
        return f"💡 Filled: {grade} trade{', adviser ' + v + sc if v else ''}; {cue}.{agree}"
    if grade in ("A+", "A"):
        if v == "SKIP":
            return f"⛔ STAND DOWN: {grade} on confluence, but the adviser says SKIP{sc}. Pass, or wait for a cleaner one.{agree}"
        if v == "CAUTION":
            return f"⚠️ SIZE DOWN: {why}. Take it smaller{', and only if you have no A+ going' if grade == 'A' else ''}.{agree}"
        tail = "" if grade == "A+" else " An A+ would rank above it."
        return f"🎯 PULL THE TRIGGER: {why}.{tail}{agree}" if v == "TAKE" else f"🎯 {grade} trade, no adviser verdict yet: the strong setup to watch.{agree}"
    if grade == "B":
        return f"💡 OPTIONAL: B trade{', adviser ' + v + sc if v else ''}. Take it small, or save your risk for an A+ or A.{agree}"
    return ""
