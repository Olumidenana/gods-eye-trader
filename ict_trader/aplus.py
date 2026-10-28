"""ICT's A+ checklist: the boxes a trade has to tick before it is the one to take.

ICT never wrote an A+ formula. What he teaches, over and over, is a short list of things that have to line up
(rules/ict-models.json, the dossiers): the higher timeframes agree, the draw is a named pool with a clear road to it,
price is on the right side of the range, a real raid took resting stops, the shift is a displacement that leaves a clean
gap, and the reward pays for the risk. On a market with a clock (forex, gold) the time and day count too. On the Deriv
synthetics there is no clock, so those boxes simply do not exist: they are judged on price alone.

The boxes are read from the judgement matrix (advisor.py) that already holds the answers, so the checklist and the
matrix can never disagree. A trade is A+ when the three it cannot do without are ticked (higher timeframes agree, a
named draw with a clear road, a clean entry) and at least three quarters of all the boxes that apply to it are. A box
the model cannot read is left out of the count, and a trade that cannot show the three it needs is not A+.

This is now THE grade. The old confluence points in scanner.grade() (reward size, the daily bias, SMT, the midnight open) are
gone from the decision: reward size is not a reason to take a trade (a far target is a lower-probability one), and a grade that
says A+ while the checklist ticks one box in seven contradicts itself. The grade, the verdict (advisor.py) and the checklist
now come from the same three required conditions, so they can never disagree:

  A+  every required box ticked and at least three quarters of all the boxes that apply
  A   every required box ticked (the three he cannot do without), some extras missing
  B   one required box missing (or a model that cannot read some of them, with all it can read ticked)
  C   two or more required boxes missing: not a trade ICT would take
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from .narrative import EXTERNAL

SHARE = 0.75   # of the boxes that apply, the share an A+ has to tick (the three required ones included)
# key, short word for alerts, the question, the matrix row that answers it, the share of that row's points that counts as yes, required
BOXES = (
    ("htf", "HTF", "Higher timeframes agree", "Draw on liquidity", 1.0, True),
    ("draw", "Draw", "A named draw with a clear road", "Path to the target", 1.0, True),
    ("entry", "Entry", "A clean entry", "Displacement and gap", 0.7, True),
    ("location", "Range", "Right side of the range", "Premium / discount", 1.0, False),
    ("raid", "Raid", "A real raid on resting stops", "Liquidity taken", 0.8, False),
    ("reward", "3R", "3R or more to the draw", "Reward", 1.0, False),
    ("time", "Time", "Right time and day", "Time and day", 1.0, False),   # only on a market with a clock
)


GRADES_ORDER = ("A+", "A", "B", "C")   # best first
ROOM_MAX = 1.25   # the day's range used so far plus what the target needs, in average daily ranges
NAMED_DRAWS = EXTERNAL   # a draw on liquidity is a higher-timeframe pool: the prior day or week, an IPDA range, an opening gap (a session range or an invented measured-move target is not one)


def checklist(advice: dict[str, Any] | None, smt: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The boxes for a trade, from its judgement matrix. `smt`: the divergence read against a correlated market, when one
    was measured (a single-market test never has it, so it never adds a box there)."""
    if not advice or not advice.get("rows"):
        return None
    rows = {r["name"]: r for r in advice["rows"]}
    boxes: list[dict[str, Any]] = []
    for key, short, label, row, share, required in BOXES:
        r = rows.get(row)
        if r is None and key == "time":
            continue                                    # no clock on this market: the box does not exist
        if r is None:
            boxes.append({"key": key, "short": short, "label": label, "ok": False, "applies": False, "required": required,
                          "why": "this model does not measure it"})
            continue
        ok, why = r["points"] >= r["of"] * share - 1e-9, r["why"]
        if key == "draw":
            tgt = str(advice.get("target") or "")
            if ok and not tgt.startswith(NAMED_DRAWS):
                ok, why = False, f"the road is clear, but {tgt or 'the target'} is not a higher-timeframe pool"
            elif ok:
                why = f"{tgt} is the draw; {why}"
        boxes.append({"key": key, "short": short, "label": label, "ok": ok, "applies": True, "required": required, "why": why})
    tr = advice.get("htf_trend")
    if tr:   # the 1h and 4h structure: with the trade, against it, or neutral
        word = {1: "with the trade", -1: "AGAINST the trade", 0: "neutral"}
        boxes.append({"key": "trend", "short": "Trend", "label": "The 1h and 4h trend is with the trade", "ok": all(v >= 0 for v in tr.values()) and any(v > 0 for v in tr.values()),
                      "applies": True, "required": False, "why": "; ".join(f"{'1h' if int(k) == 60 else '4h' if int(k) == 240 else str(k) + 'm'} structure {word[v]}" for k, v in sorted(tr.items()))})
    poi = advice.get("poi")
    if poi:    # top-down: the higher timeframe says where price may turn; the entry should be there
        boxes.append({"key": "poi", "short": "HTF POI", "label": "At a higher-timeframe array", "ok": poi["state"] in ("array", "pool"), "applies": True, "required": False,
                      "why": poi["text"]})
    room = advice.get("room")
    room_ok = True
    if room:   # price only: an average daily range is measured on a synthetic index as on forex
        proj = room.get("total_proj")
        used, needs = room["used"], room["needs"]
        room_ok = (proj <= ROOM_MAX) if proj is not None else (used + needs <= ROOM_MAX)
        boxes.append({"key": "room", "short": "Room", "label": "Room left in the day's range", "ok": room_ok, "applies": True, "required": False,
                      "why": (f"the trade projects {proj:.0%} of the average daily range" if proj is not None else f"the day has used {used:.0%} of its average range and the target needs {needs:.0%} more")
                             + ("" if room_ok else ": the move is likely spent before it gets there")})
    if smt:
        boxes.append({"key": "smt", "short": "SMT", "label": "SMT divergence with a correlated market", "ok": bool(smt.get("divergence")),
                      "applies": True, "required": False, "why": smt.get("text") or ""})
    c = advice.get("c") or {}
    prob_audit = None
    if c and (c.get("entry_price") is not None or c.get("target_price") is not None or c.get("rr") is not None):
        try:
            from . import probabilistic_engine
            d = c.get("dir", 1)
            entry_px = float(c.get("entry_price") or 100.0)
            target_px = float(c.get("target_price") or (entry_px + (1.0 * d)))
            rr = float(c.get("rr") or 2.0)
            stop_dist = abs(target_px - entry_px) / rr if rr > 0 else 0.5
            stop_px = entry_px - (stop_dist * d)
            prob_audit = probabilistic_engine.evaluate_setup(
                direction=d,
                entry_price=entry_px,
                stop_price=stop_px,
                target_price=target_px,
                spread=float(c.get("spread") or 0.0),
                daily_bias=int(c.get("bias") or 0),
                weekly_bias=int(c.get("wbias") or 0),
                body_mult=float(c.get("body_mult")) if c.get("body_mult") is not None else None,
                gap_ratio=float(c.get("gap_ratio")) if c.get("gap_ratio") is not None else None,
                pd_location=float(c.get("pd") or 0.0),
                opposing_obstacles=int(c.get("obstacles") or 0),
                is_killzone=bool(c.get("window")),
                weekday=int(c.get("weekday") or 2),
                hour_ny=int(c.get("hour") or 10),
                target_name=str(advice.get("target") or ""),
                level=str(c.get("level") or ""),
                room_total_proj=float(proj) if (proj := (advice.get("room") or {}).get("total_proj")) is not None else None,
                market_key=str(c.get("market") or ""),
                daily_pnl_pct=float(c.get("daily_pnl_pct") or 0.0),
                is_synthetic=not bool(c.get("timed", True)),
            )
            if prob_audit:
                prob_ok = prob_audit.is_a_plus
                boxes.append({
                    "key": "probability",
                    "short": "HE-ARE",
                    "label": f"HE-ARE score ≥ 80 out of 100 ({prob_audit.probability * 100:.0f})",
                    "ok": prob_ok,
                    "applies": True,
                    "required": False,
                    "why": f"score {prob_audit.probability * 100:.0f}/100 (hand-weighted from the rows above: it is NOT a measured win rate); anti-ruin: {'approved' if prob_audit.anti_ruin_approved else 'blocked'}"
                           + (f" ({', '.join(prob_audit.rejection_reasons)})" if prob_audit.rejection_reasons else "")
                })
        except Exception:
            prob_audit = None

    app = [b for b in boxes if b["applies"]]
    ticked, of = sum(1 for b in app if b["ok"]), len(app)
    pd_row = rows.get("Premium / discount")
    loc_ok = pd_row is None or pd_row["points"] > 0
    aplus = all(b["ok"] for b in boxes if b["required"]) and room_ok and loc_ok and ticked >= math.ceil(SHARE * of)
    if prob_audit and (not prob_audit.is_a_plus or not prob_audit.anti_ruin_approved):
        aplus = False
    missing = [b["label"] for b in boxes if b["required"] and not b["ok"]] or [b["label"] for b in app if not b["ok"]]
    pa_dict = asdict(prob_audit) if prob_audit else None
    return {"aplus": aplus, "ticked": ticked, "of": of, "boxes": boxes, "missing": missing if not aplus else [], "loc_zero": not loc_ok, "prob_audit": pa_dict}


def grade_of(cl: dict[str, Any]) -> str:
    """A+ / A / B / C from the checklist alone (see the module doc)."""
    if cl["aplus"]:
        return "A+"
    room_b = next((b for b in cl["boxes"] if b["key"] == "room"), None)
    if (room_b is not None and not room_b["ok"]) or cl.get("loc_zero"):
        req_ok = sum(1 for b in cl["boxes"] if b["required"] and b["applies"] and b["ok"])
        req_tot = sum(1 for b in cl["boxes"] if b["required"] and b["applies"])
        return "B" if req_tot and req_ok == req_tot else "C"
    req = [b for b in cl["boxes"] if b["required"]]
    readable = [b for b in req if b["applies"]]
    ok = sum(1 for b in readable if b["ok"])
    if readable and ok == len(readable):
        return "A" if len(readable) == len(req) else "B"
    return "B" if len(readable) >= 2 and ok == len(readable) - 1 else "C"


def quality(advice: dict[str, Any] | None) -> dict[str, Any] | None:
    """The trade's grade in the shape the app and the alerts use ({grade, score, confluence}), from its checklist; None without one."""
    cl = (advice or {}).get("aplus")
    if not cl:
        return None
    res: dict[str, Any] = {"grade": grade_of(cl), "score": cl["ticked"], "of": cl["of"], "confluence": [b["label"] for b in cl["boxes"] if b["ok"]]}
    if cl.get("prob_audit"):
        pa = cl["prob_audit"]
        res["probability"] = pa.get("probability") if isinstance(pa, dict) else getattr(pa, "probability", None)
        res["anti_ruin"] = pa.get("anti_ruin_approved") if isinstance(pa, dict) else getattr(pa, "anti_ruin_approved", None)
    return res


def text(advice: dict[str, Any] | None) -> str:
    """The checklist as one alert line: A+ checklist 5/6 passes: ✅ HTF · ✅ Draw · ⬜ Raid …"""
    cl = (advice or {}).get("aplus")
    if not cl:
        return ""
    marks = " · ".join(("✅ " if b["ok"] else "⬜ ") + b["short"] for b in cl["boxes"] if b["applies"])
    return f"{grade_of(cl)} · A+ checklist {cl['ticked']}/{cl['of']} {'passes' if cl['aplus'] else 'not met'}: {marks}"


def line(advice: dict[str, Any] | None) -> list[str]:
    """The checklist as a rules-trail line the Research lab reads back (AP-00 checklist 5/6 A+: HTF, Draw, ...)."""
    cl = (advice or {}).get("aplus")
    return [f"AP-00 checklist {cl['ticked']}/{cl['of']} {grade_of(cl)}: " + ", ".join(b["short"] for b in cl["boxes"] if b["ok"])] if cl else []


def parse_grade(reasons: Any) -> str | None:
    """The grade a trade earned, from its rules trail; None when it carries no checklist."""
    for r in reasons or ():
        if r.startswith("AP-00 checklist "):
            g = r.split(" ", 3)[3].split(":", 1)[0]
            return g if g in ("A+", "A", "B", "C") else ("A+" if g == "A+" else "C")
    return None


def parse(reasons: Any) -> bool | None:
    """Did the trade pass the A+ checklist? From its rules trail; None when it carries no checklist."""
    g = parse_grade(reasons)
    return None if g is None else g == "A+"
