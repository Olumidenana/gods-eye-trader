"""The adviser: ICT's judgement as a scoring matrix.

ICT leaves a lot to judgement ("is the bias clear", "is that a real run on liquidity", "is the path to
the target clean", "is this a day to trade"). The video research (rules/ict-models.json) shows the
questions he asks. Each question is a row here with a weight, and a setup's answers add up to a score
out of 100 and a verdict: TAKE, CAUTION or SKIP. The weights are a judgement call written down so
they can be tested, not a result: the Research lab reports win rate and expectancy by verdict, which
is how the matrix is checked and tuned. Rows that do not apply to a market (time of day on a
24/7 synthetic index) are dropped and the score is scaled over the rows that remain.

A row looks like (name, points earned, points possible, why), and every why is shown to the trader.

The verdict is not the points. ICT's conditions are gates, not credits: no higher-timeframe agreement, no named draw with a
clear road, or no clean shift and he does not take the trade, however well it scores elsewhere. So the verdict comes from the
A+ checklist's three required boxes (aplus.py) first, and the score only moves a trade that has passed them between PULL THE
TRIGGER and SIZE DOWN. Two required boxes missing is STAND DOWN, one missing is SIZE DOWN at best.
"""

from __future__ import annotations

from typing import Any

from . import aplus

TAKE, CAUTION = 65, 40  # score thresholds for the verdicts when a model has no checklist
GATED_TAKE = 55         # once the three required conditions are met, the score a trade needs to be a plain TAKE
# What each verdict means to the trader, said plainly: this is what a model's trade is announced with.
CALL = {"TAKE": "PULL THE TRIGGER", "CAUTION": "SIZE DOWN", "SKIP": "STAND DOWN"}


def _row(name: str, got: float, top: float, why: str) -> dict[str, Any]:
    return {"name": name, "points": round(got, 1), "of": top, "why": why}


def assess(c: dict[str, Any]) -> dict[str, Any]:
    """`c` holds what the engine knew when the order was placed:
    dir (+1/-1), bias (+1/-1/0 daily), wbias (weekly), bias_text, level (name), body_mult, gap_ratio,
    pd (+1 right side of yesterday's range, -1 wrong side, 0 unknown), pd20 (same for the 20-day range or None; pd20_name says which range it was),
    obstacles (int), sd (bool, seek and destroy), rr, window (name or None), weekday (0 Mon), hour (NY),
    timed (bool: a clock market), midnight_ok (True/False/None), target (name), profile (narrative.weekly_profile or None),
    poi (ict2022._htf_poi: the top-down answer, or None), adr ((average daily range, today's range so far) or None), entry_price and target_price (to measure the room in the day).
    `unknown`: names of rows the caller could not read (a model that does not track them); those rows are left
    out and the score is scaled over the rows that remain, the same as a row that does not apply to a market."""
    rows: list[dict[str, Any]] = []
    d = c["dir"]
    unknown = set(c.get("unknown") or ())

    # 1. The draw: is the day's and week's direction with the trade? (25)
    b, w = c.get("bias", 0), c.get("wbias", 0)
    got = 12 if b == d else 5 if b == 0 else 0
    got += 8 if w == d else 3 if w == 0 else 0
    got += 5 if (b == d and w == d) else 0
    bn, wn = c.get("bias_name", "daily"), c.get("wbias_name", "weekly")
    why = (f"{bn} draw with the trade" if b == d else f"no clear {bn} draw" if b == 0 else f"AGAINST the {bn} draw") + \
          (f"; {wn} with it" if w == d else f"; {wn} neutral" if w == 0 else f"; AGAINST the {wn} draw")
    rows.append(_row("Draw on liquidity", got, 25, why))

    # 2. Location: sell in premium, buy in discount (15)
    pd, pd20 = c.get("pd", 0), c.get("pd20")
    got = (8 if pd >= 0.5 else 6 if pd > 0 else 3 if pd == 0 else 0) + (7 if pd20 is None and pd > 0 else 7 if pd20 and pd20 > 0 else 3 if pd20 is None and pd == 0 else 0)
    buy = d > 0
    if "Premium / discount" not in unknown:
        rows.append(_row("Premium / discount", got, 15,
                         (f"entry deep in the {'discount' if buy else 'premium'} of the dealing range, its outer quadrant: the best place to {'buy' if buy else 'sell'}" if pd >= 0.5 else
                          "entry on the right side of the dealing range" if pd > 0 else "entry location unclear" if pd == 0 else "entry on the WRONG side of the dealing range")
                         + ("" if pd20 is None else f"; right side of {c.get('pd20_name') or 'the 20-day range'}" if pd20 > 0 else f"; WRONG side of {c.get('pd20_name') or 'the 20-day range'}")))

    # 3. What was run: equal highs/lows and session ranges are real stops (15)
    lvl = c.get("level", "")
    kind = ("equal highs/lows" if lvl.startswith("EQ") else "an opening gap" if lvl.startswith(("NDOG", "NWOG", "ORG")) else
            "the prior day/week extreme" if lvl in ("PDH", "PDL", "PWH", "PWL") or lvl.startswith("IPDA") else
            "a session range" if lvl.endswith(("_H", "_L")) else "a minor swing" if lvl.startswith("SW") else "a reference level")
    top = {"equal highs/lows": 15, "an opening gap": 14, "the prior day/week extreme": 13, "a session range": 12, "a reference level": 10, "a minor swing": 6}[kind]
    rows.append(_row("Liquidity taken", top, 15, f"ran {kind} ({lvl})"))

    # 4. The shift: displacement and a gap of real size (15)
    bm, gr = c.get("body_mult"), c.get("gap_ratio")
    got = (8 if (bm or 0) >= 2.2 else 6 if (bm or 0) >= 1.4 else 2 if bm is None else 1)
    got += (7 if (gr or 0) >= 0.45 else 5 if (gr or 0) >= 0.18 else 3 if gr is None else 1)
    if "Displacement and gap" not in unknown:
        rows.append(_row("Displacement and gap", got, 15,
                         ("strong shift" if (bm or 0) >= 2.2 else "solid shift" if (bm or 0) >= 1.4 else "weak shift" if bm else "no displacement measure")
                         + ("; sizeable gap" if (gr or 0) >= 0.45 else "; clean gap" if (gr or 0) >= 0.18 else "; small gap" if gr else "")))

    # 5. The road to the target (15)
    ob = c.get("obstacles", 0)
    got = 15 if ob == 0 else 8 if ob == 1 else 0
    if c.get("sd"):
        got = 0
    if "Path to the target" not in unknown:
        rows.append(_row("Path to the target", got, 15,
                         "seek and destroy: both sides of the range are being stopped out" if c.get("sd") else
                         "clear runway (low resistance)" if ob == 0 else f"{ob} unmitigated opposing gap{'s' if ob > 1 else ''} in the way"))

    # 6. Reward (10)
    rr = c.get("rr", 0)
    rows.append(_row("Reward", 10 if rr >= 3 else 7 if rr >= 2 else 4 if rr >= 1.5 else 1, 10, f"{rr:.1f}R to {c.get('target', 'the target')}"))

    # 7. Time and day: only on markets with a clock (5): the window (3) and the day against the weekly profile (2)
    if c.get("timed"):
        wd, hr, win, prof = c.get("weekday", 2), c.get("hour", 10), c.get("window"), c.get("profile")
        got = 3 if win else 1
        notes = [f"inside the {win} window" if win else "outside the main windows"]
        fit = (prof or {}).get("fit")
        if prof is None:   # no profile to read (a model that does not keep one): the old rule of thumb
            got += 0 if (wd == 4 and hr >= 12) else 2 if wd in (1, 2, 3) else 1
            if wd == 0:
                notes.append("Monday: often sets the range, be selective")
            if wd == 4 and hr >= 12:
                notes.append("Friday afternoon: close before the weekend")
        else:
            got += 2 if fit > 0 else 1 if fit == 0 else 0
            notes.append(f"{prof['day']} in a {prof['week']} week ({prof['phase']}): " + ("fits the weekly profile" if fit > 0 else "neutral for the weekly profile" if fit == 0 else "does not fit the weekly profile"))
        if c.get("midnight_ok") is False:
            got -= 2; notes.append("the run is on the wrong side of the midnight open")
        rows.append(_row("Time and day", max(0, got), 5, "; ".join(notes)))

    top = sum(r["of"] for r in rows)
    score = round(sum(r["points"] for r in rows) / top * 100)
    adr = c.get("adr")
    room = None
    if adr and c.get("target_price") is not None and c.get("entry_price") is not None:
        target_px, entry_px = float(c["target_price"]), float(c["entry_price"])
        if len(adr) >= 4:
            adr_val, used_rng, today_hi, today_lo = adr[0], adr[1], adr[2], adr[3]
            proj_rng = max(today_hi, target_px) - min(today_lo, target_px)
            room = {"adr": adr_val, "used": used_rng / adr_val, "needs": max(0.0, proj_rng - used_rng) / adr_val, "total_proj": proj_rng / adr_val}
        else:
            room = {"adr": adr[0], "used": adr[1] / adr[0], "needs": abs(target_px - entry_px) / adr[0]}
    out: dict[str, Any] = {"score": score, "rows": rows, "target": c.get("target"), "room": room, "profile": c.get("profile"), "poi": c.get("poi"), "htf_trend": c.get("htf_trend"), "c": c}
    cl = aplus.checklist(out)
    out["aplus"] = cl
    verdict = "TAKE" if score >= TAKE else "CAUTION" if score >= CAUTION else "SKIP"
    if cl:
        req = [b for b in cl["boxes"] if b["required"]]
        failed = [b for b in req if b["applies"] and not b["ok"]]
        unread = [b for b in req if not b["applies"]]
        verdict = "SKIP" if len(failed) >= 2 else "CAUTION" if failed else ("TAKE" if (cl["aplus"] or score >= GATED_TAKE) else "CAUTION")
        if unread and verdict == "TAKE":
            verdict = "CAUTION"   # a model that cannot show all three required conditions is never a plain PULL THE TRIGGER
    if room and ((room.get("total_proj", room["used"] + room["needs"])) > aplus.ROOM_MAX) and verdict == "TAKE":
        verdict = "CAUTION"       # target requires more range than the day typically has left
    against = [r for r in rows if r["points"] == 0 and r["name"] in ("Draw on liquidity", "Premium / discount", "Path to the target")]
    if against and verdict == "TAKE":
        verdict = "CAUTION"  # a rule he treats as a gate is broken: never a plain TAKE
    weakest = min(rows, key=lambda r: r["points"] / r["of"])
    strongest = max(rows, key=lambda r: r["points"] / r["of"])
    miss = f" Missing: {', '.join(m.lower() for m in cl['missing'])}." if cl and cl["missing"] and verdict != "TAKE" else ""
    summary = {"TAKE": "All three of ICT's required conditions are met and the rest agree.",
               "CAUTION": "Mixed: trade it smaller, or wait for a better one.",
               "SKIP": "ICT would not take this: two or more of his required conditions are not met."}[verdict] + \
              f" Best: {strongest['name'].lower()}. Weakest: {weakest['name'].lower()} ({weakest['why']}).{miss}"
    out.update(verdict=verdict, call=CALL[verdict], summary=summary)
    return out


def second_try(a: dict[str, Any]) -> dict[str, Any]:
    """RE-01: the first trade on this idea was stopped out and price ran on, then a fresh shift came at the same level. ICT takes it, but it
    is a second try on an idea that has just failed once, so it is never a plain PULL THE TRIGGER: size it down, and say why."""
    a = dict(a)
    a["reentry"] = True
    note = "Second try: the first trade on this idea was stopped out and price ran on, so this is a fresh shift at the same level. Size it down."
    capped = a["verdict"] == "TAKE"
    if capped:
        a["verdict"], a["call"] = "CAUTION", CALL["CAUTION"]
    tail = a["summary"].split(". ", 1)[1] if capped and ". " in a["summary"] else "" if capped else a["summary"]
    a["summary"] = f"{note} {tail}".strip()
    return a


def line(a: dict[str, Any]) -> str:
    """One rules-trail line, parsed back by the Research lab (AD-00 advisor <score>/100 <verdict>)."""
    return f"AD-00 advisor {a['score']}/100 {a['verdict']}: {a['summary']}"
