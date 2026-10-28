"""The engine's live read of a chart, the way ICT marks one up.

After the ICT 2022 strategy has processed every closed candle, this turns its state
and the candles into everything a trader would draw and say about the chart right
now: the session ranges and killzones, the liquidity pools (prior-day, session,
developing HOD/LOD, and equal highs/lows) and which were swept, the 3-tier ICT
swing structure hierarchy (STH/STL, ITH/ITL, LTH/LTL), the NY midnight open, open
fair value gaps, the 8-tier PD-array matrix, the dealing range with premium/discount,
the bias and draw on liquidity, where the model is in its sequence (time -> sweep ->
shift -> gap -> limit order), and the trade plan with both take-profits.

Nothing here invents a setup: every mark is either a candle fact or an event the
strategy itself recorded. Times are unix seconds (bar open), prices are rounded.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from statistics import median
from typing import Any

from .config import Window
from .sessions import NY, Bar, to_ny
from .strategy import LONG, SHORT

LEVEL_NAMES = {
    "PDH": "Prior-day high", "PDL": "Prior-day low", "PWH": "Prior-week high", "PWL": "Prior-week low",
    "HOD": "Today's high (HOD)", "LOD": "Today's low (LOD)",
    "ASIA_H": "Asian range high", "ASIA_L": "Asian range low", "LONDON_H": "London high", "LONDON_L": "London low",
    "PREMARKET_H": "Pre-market high", "PREMARKET_L": "Pre-market low", "LUNCH_H": "Lunch high", "LUNCH_L": "Lunch low",
    "NDOG_H": "New Day Opening Gap high", "NDOG_L": "New Day Opening Gap low", "NDOG_CE": "NDOG 50% (CE)",
    "NWOG_H": "New Week Opening Gap high", "NWOG_L": "New Week Opening Gap low", "NWOG_CE": "NWOG 50% (CE)",
    "ORG_H": "Opening Range Gap high", "ORG_L": "Opening Range Gap low", "ORG_CE": "ORG 50% (CE)",
}


def _level_label(name: str) -> str:
    if name.startswith("IPDA") and name[4:].split("_")[0].isdigit():
        return f"{name[4:].split('_')[0]}-day range {'high' if name.endswith('_H') else 'low'} (IPDA)"
    if name.startswith(("CBDR+", "CBDR-", "ASIA+", "ASIA-")):
        return f"{name[:4]} {name[4:]} projection"
    return LEVEL_NAMES.get(name, name)


WINDOW_NAMES = {
    "asia": "Asian range", "london": "London killzone", "new_york": "New York killzone", "ny_am": "NY AM killzone",
    "ny_pm": "NY PM killzone", "premarket": "Pre-market", "lunch": "NY lunch",
}


def _names(msg: str) -> str:
    for code, name in LEVEL_NAMES.items():
        msg = msg.replace(code + " ", name + " ")
    return msg


def _ts(t: datetime | None) -> int | None:
    return int(t.timestamp()) if t else None


def _window_bounds(w: Window, day: datetime) -> tuple[datetime, datetime]:
    """UTC start/end of a NY-time window on the NY calendar `day` (00:00 NY); a window
    that wraps midnight (Asia 20:00-00:00) belongs to the evening before."""
    start = day.replace(hour=w.start.hour, minute=w.start.minute)
    end = day.replace(hour=w.end.hour, minute=w.end.minute)
    if w.start >= w.end:
        start -= timedelta(days=1)
    return start, end


def _swings(bars: list[Bar], start: int) -> list[tuple[int, str, float]]:
    out = []
    for i in range(max(1, start), len(bars) - 1):
        a, b, c = bars[i - 1], bars[i], bars[i + 1]
        if b.high > a.high and b.high > c.high:
            out.append((i, "high", b.high))
        if b.low < a.low and b.low < c.low:
            out.append((i, "low", b.low))
    return out


def _classify_swings(
    bars: list[Bar],
    start: int,
    rng: float,
    gaps: list[dict[str, Any]],
    sweep_times: set[int],
    px_fn: Any,
) -> list[dict[str, Any]]:
    """Classify ICT's 3-tier swing hierarchy (STH/STL, ITH/ITL, LTH/LTL).

    - Short-Term High/Low (STH/STL): 3-bar fractal with local prominence.
    - Intermediate-Term High/Low (ITH/ITL): an STH flanked by lower STHs on both
      sides (or an STL flanked by higher STLs), OR a swing that rebalances an
      opposing Fair Value Gap (ICT Rebalance Swing).
    - Long-Term High/Low (LTH/LTL): an ITH flanked by lower ITHs (or ITL flanked
      by higher ITLs), a swing that raided a major liquidity pool, or the
      extreme high/low of the active dealing range.
    """
    raw = _swings(bars, max(1, start))
    if not raw:
        return []
    span = bars[max(0, start):]
    hi_idx = max(range(max(0, start), len(bars)), key=lambda k: bars[k].high) if span else -1
    lo_idx = min(range(max(0, start), len(bars)), key=lambda k: bars[k].low) if span else -1

    highs: list[dict[str, Any]] = []
    lows: list[dict[str, Any]] = []
    min_prom = 0.20 * rng
    for idx, side, val in raw:
        w0, w1 = max(0, idx - 3), min(len(bars), idx + 4)
        prom = (val - min(b.low for b in bars[w0:w1])) if side == "high" else (max(b.high for b in bars[w0:w1]) - val)
        if prom < min_prom and idx not in (hi_idx, lo_idx):
            continue
        after = bars[idx + 1:]
        intact = not any((b.high > val) if side == "high" else (b.low < val) for b in after)
        # Check if this swing rebalanced an earlier opposing FVG
        rebalance = False
        for g in gaps:
            if g["i"] >= idx:
                continue
            if side == "high" and g["dir"] == "bear" and g["bottom"] <= val <= g["top"] + 0.15 * rng:
                rebalance = True
                break
            if side == "low" and g["dir"] == "bull" and g["bottom"] - 0.15 * rng <= val <= g["top"]:
                rebalance = True
                break
        item = {
            "idx": idx,
            "time": _ts(bars[idx].time),
            "side": side,
            "val": val,
            "price": px_fn(val),
            "intact": intact,
            "rebalance": rebalance,
            "tier": "STH" if side == "high" else "STL",
        }
        if side == "high":
            highs.append(item)
        else:
            lows.append(item)

    # Promote STH -> ITH (structural flanking or FVG rebalance)
    for k, h in enumerate(highs):
        flanked = 0 < k < len(highs) - 1 and highs[k - 1]["val"] < h["val"] and highs[k + 1]["val"] < h["val"]
        if flanked or h["rebalance"]:
            h["tier"] = "ITH"
    for k, l in enumerate(lows):
        flanked = 0 < k < len(lows) - 1 and lows[k - 1]["val"] > l["val"] and lows[k + 1]["val"] > l["val"]
        if flanked or l["rebalance"]:
            l["tier"] = "ITL"

    # Promote ITH -> LTH (flanked by lower ITHs, swept a reference pool, or dealing-range anchor)
    iths = [h for h in highs if h["tier"] == "ITH"]
    for k, h in enumerate(iths):
        if (0 < k < len(iths) - 1 and iths[k - 1]["val"] < h["val"] and iths[k + 1]["val"] < h["val"]) or h["idx"] == hi_idx or h["time"] in sweep_times:
            h["tier"] = "LTH"
    itls = [l for l in lows if l["tier"] == "ITL"]
    for k, l in enumerate(itls):
        if (0 < k < len(itls) - 1 and itls[k - 1]["val"] > l["val"] and itls[k + 1]["val"] > l["val"]) or l["idx"] == lo_idx or l["time"] in sweep_times:
            l["tier"] = "LTL"
    for h in highs:
        if h["idx"] == hi_idx or h["time"] in sweep_times:
            h["tier"] = "LTH"
    for l in lows:
        if l["idx"] == lo_idx or l["time"] in sweep_times:
            l["tier"] = "LTL"

    merged = sorted(highs + lows, key=lambda x: x["idx"])
    # Keep all LTH/LTL and ITH/ITL in the visible window + the most recent intact STH/STL
    important = [x for x in merged if x["tier"] in ("LTH", "LTL", "ITH", "ITL")]
    recent_st = [x for x in merged[-10:] if x["tier"] in ("STH", "STL") and x["intact"]][-4:]
    chosen = {x["idx"]: x for x in (important[-12:] + recent_st)}
    return [
        {
            "tier": x["tier"],
            "side": x["side"],
            "price": x["price"],
            "time": x["time"],
            "intact": x["intact"],
            "rebalance": x["rebalance"],
        }
        for x in sorted(chosen.values(), key=lambda s: s["idx"])
    ]


def _fvgs(bars: list[Bar], start: int, min_size: float) -> list[dict[str, Any]]:
    """3-candle gaps (FV-01) since `start`, with how far later candles have filled them."""
    out = []
    for i in range(max(2, start), len(bars)):
        c1, c3 = bars[i - 2], bars[i]
        if c1.high < c3.low and c3.low - c1.high >= min_size:
            g = {"dir": "bull", "bottom": c1.high, "top": c3.low, "i": i - 2}
        elif c1.low > c3.high and c1.low - c3.high >= min_size:
            g = {"dir": "bear", "bottom": c3.high, "top": c1.low, "i": i - 2}
        else:
            continue
        ce = (g["top"] + g["bottom"]) / 2
        state, end, inv = "open", None, None
        for j in range(i + 1, len(bars)):
            b = bars[j]
            if g["dir"] == "bull":
                if b.close < g["bottom"]:
                    state, end, inv = "filled", j, j  # closed through: now resistance (IFVG)
                    break
                if b.low <= g["bottom"]:
                    state, end = "filled", j
                    break
                if b.low <= ce:
                    state = "ce"
            else:
                if b.close > g["top"]:
                    state, end, inv = "filled", j, j  # closed through: now support (IFVG)
                    break
                if b.high >= g["top"]:
                    state, end = "filled", j
                    break
                if b.high >= ce:
                    state = "ce"
        g.update(ce=ce, state=state, end=end, inv=inv)
        out.append(g)
    return out


def analyze(strategy: Any, bars: list[Bar], cfg: Any, tf: int, dp: int, now: datetime,
            signals: list[dict[str, Any]] | None = None, news: Any = None) -> dict[str, Any]:
    st = cfg.strategy
    px = lambda v: round(v, dp) if v is not None else None  # noqa: E731
    last = bars[-1]
    price = last.close
    ny_now = to_ny(last.time)
    day0 = ny_now.replace(hour=0, minute=0, second=0, microsecond=0)  # NY midnight, today
    any_hour = any(w.start == time(0, 0) and w.end >= time(23, 0) for w in st.killzones)
    index = {b.time: i for i, b in enumerate(bars)}

    # ---- the window the read covers: today (intraday) or the last days/weeks (1h/4h)
    span_start = day0 - timedelta(hours=4) if tf < 60 else day0 - timedelta(days=5 if tf == 60 else 20)
    i0 = next((i for i, b in enumerate(bars) if b.time >= span_start), 0)
    rng = median([b.high - b.low for b in bars[-288:]]) or (abs(price) * 1e-4)

    # ---- sessions: killzones and reference ranges, with their highs/lows so far
    sessions = []
    if not any_hour:
        seen = set()
        for kind, wins in (("range", st.reference_ranges), ("killzone", st.killzones)):
            for w in wins:
                if w.name in seen:
                    continue
                seen.add(w.name)
                s_t, e_t = _window_bounds(w, day0)
                inside = [b for b in bars if s_t <= to_ny(b.time) < e_t]
                sessions.append({
                    "name": w.name, "label": WINDOW_NAMES.get(w.name, w.name), "kind": kind,
                    "start": _ts(s_t), "end": _ts(e_t), "active": s_t <= to_ny(last.time) < e_t,
                    "high": px(max(b.high for b in inside)) if inside else None,
                    "low": px(min(b.low for b in inside)) if inside else None,
                })
        sessions.sort(key=lambda x: x["start"])

    # ---- events the engine recorded since the read started
    evs = [e for e in getattr(strategy, "events", []) if to_ny(e["time"]) >= span_start]
    taken = {}
    for e in evs:
        if e["type"] == "sweep":
            taken[e["level"]] = e
        if e["type"] == "spent":
            name = e["message"].split(" ")[0]
            taken.setdefault(name, {"time": e["time"], "outside": True})

    # ---- liquidity pools: reference levels (RL-01/02/04), IPDA ranges, opening gaps, and CBDR projections
    tracker = getattr(strategy, "_tracker", None)
    origins = tracker.origin_times() if tracker and hasattr(tracker, "origin_times") else {}
    spent = getattr(strategy, "_spent", set())
    levels = []
    base = list(tracker.levels()) if tracker else []
    htf = getattr(strategy, "_htf_levels", None)   # the IPDA 20/40/60-day ranges and the prior week: marked on every chart, ICT's higher-timeframe pools
    gap_lvs = getattr(strategy, "_gap_levels", None)
    gap_mids = getattr(strategy, "_gap_midpoints", None)
    cbdr_lvs = getattr(strategy, "_cbdr_levels", None)
    extra_lvs = (
        (htf() if callable(htf) else [])
        + (gap_lvs() if callable(gap_lvs) else [])
        + (gap_mids() if callable(gap_mids) else [])
        + (cbdr_lvs() if callable(cbdr_lvs) else [])
    )
    have = {lv.name for lv in base}
    all_lvs = list(base)
    for x in extra_lvs:
        if x.name not in have:
            have.add(x.name)
            all_lvs.append(x)
    for lv in all_lvs:
        t = taken.get(lv.name)
        status = "swept" if t and not t.get("outside") else "spent" if lv.name in spent else "intact"
        levels.append({"name": lv.name, "label": _level_label(lv.name), "price": px(lv.price), "side": lv.side,
                       "pool": "BSL" if lv.side == "high" else "SSL", "status": status,
                       "origin_time": _ts(origins.get(lv.name)),
                       "swept_time": _ts(t["time"]) if t else None})
    # Add Today's Developing High (HOD) & Developing Low (LOD) anchored at the exact candle where they formed
    if tracker is not None and tf < 240 and getattr(tracker, "_day_high", float("-inf")) > float("-inf"):
        if "HOD" not in have:
            levels.append({
                "name": "HOD", "label": _level_label("HOD"), "price": px(tracker._day_high), "side": "high",
                "pool": "BSL", "status": "developing", "origin_time": _ts(origins.get("HOD")), "swept_time": None,
            })
        if "LOD" not in have and getattr(tracker, "_day_low", float("inf")) < float("inf"):
            levels.append({
                "name": "LOD", "label": _level_label("LOD"), "price": px(tracker._day_low), "side": "low",
                "pool": "SSL", "status": "developing", "origin_time": _ts(origins.get("LOD")), "swept_time": None,
            })

    # ---- equal highs / lows (SW-03): two swing points within a hair, not yet run
    tol = max(st.min_sweep_pips * cfg.instrument.pip_size, 0.12 * rng)
    sw = _swings(bars, len(bars) - 300)
    eq = []
    for side in ("high", "low"):
        pts = [s for s in sw if s[1] == side][-30:]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                ia, _, pa = pts[a]
                ib, _, pb = pts[b]
                if ib - ia < 3 or abs(pa - pb) > tol:
                    continue
                lvl = max(pa, pb) if side == "high" else min(pa, pb)
                after = bars[ib + 2:]
                if any((x.high > lvl) if side == "high" else (x.low < lvl) for x in after):
                    continue
                eq.append({"side": side, "pool": "BSL" if side == "high" else "SSL", "price": px(lvl),
                           "times": [_ts(bars[ia].time), _ts(bars[ib].time)]})
    eq = sorted(eq, key=lambda e: abs(e["price"] - price))
    eq = [e for k, e in enumerate(eq) if all(abs(e["price"] - f["price"]) > tol for f in eq[:k])][:4]

    # ---- fair value gaps (FV-01/03): open or half-filled, nearest price first
    gaps = _fvgs(bars, max(i0, len(bars) - 240), 0.25 * rng)
    fvgs = [{"dir": g["dir"], "top": px(g["top"]), "bottom": px(g["bottom"]), "ce": px(g["ce"]), "state": g["state"],
             "time": _ts(bars[g["i"]].time), "end": _ts(bars[g["end"]].time) if g["end"] is not None else None}
            for g in gaps if g["state"] != "filled" or (g["end"] is not None and g["end"] >= len(bars) - 12)]
    fvgs = sorted(fvgs, key=lambda g: min(abs(g["top"] - price), abs(g["bottom"] - price)))[:8]

    # ---- 3-tier ICT swing structure hierarchy (STH/STL, ITH/ITL, LTH/LTL)
    sweep_ts = {_ts(e["time"]) for e in evs if e["type"] == "sweep" and e.get("time")}
    swings = _classify_swings(bars, max(i0, len(bars) - 220), rng, gaps, {t for t in sweep_ts if t is not None}, px)

    # ---- PD-Array Matrix: IFVGs (IV-01), Order Blocks (OB-01), Breakers (MS-05), Mitigation, Rejection, Propulsion, BPR, VI, HTF
    pd_arrays = []
    for g in gaps:
        if g["inv"] is None or g["inv"] < len(bars) - 120:
            continue
        flip = "bear" if g["dir"] == "bull" else "bull"
        later = bars[g["inv"] + 1:]
        if any((x.close > g["top"]) if flip == "bear" else (x.close < g["bottom"]) for x in later):
            continue  # disrespected again: no longer valid
        pd_arrays.append({"kind": "ifvg", "dir": flip, "top": px(g["top"]), "bottom": px(g["bottom"]),
                          "time": _ts(bars[g["inv"]].time), "label": "IFVG"})
    sweeps = [e for e in getattr(strategy, "events", []) if e["type"] == "sweep" and to_ny(e["time"]) >= span_start]
    for m in [e for e in getattr(strategy, "events", []) if e["type"] == "mss" and to_ny(e["time"]) >= span_start]:
        sw_e = next((e for e in reversed(sweeps) if e["time"] <= m["time"] and e["level"] == m["level"]), None)
        if not sw_e or sw_e["time"] not in index or m["time"] not in index:
            continue
        i0_s, i1_s = index[sw_e["time"]], index[m["time"]]
        # the last opposite candle before the displacement: down-close for a bullish shift
        cands = [i for i in range(max(0, i0_s - 2), i1_s) if (bars[i].close < bars[i].open) == (m["dir"] == LONG)]
        if not cands:
            continue
        ob = bars[cands[-1]]
        mt = (ob.high + ob.low) / 2  # mean threshold (OB-01)
        # OB-02: a candle body closing through the mean threshold invalidates the block
        after = bars[i1_s + 1:]
        broken = any((min(x.open, x.close) < mt) if m["dir"] == LONG else (max(x.open, x.close) > mt) for x in after)
        if not broken:
            pd_arrays.append({"kind": "ob", "dir": "bull" if m["dir"] == LONG else "bear", "top": px(ob.high), "bottom": px(ob.low),
                              "mt": px(mt), "time": _ts(ob.time), "label": "+OB" if m["dir"] == LONG else "−OB"})
        # MS-05 breaker: the swing the displacement broke now acts against a return to it
        sw_t = m.get("swing_time")
        if sw_t in index:
            sb = bars[index[sw_t]]
            pd_arrays.append({"kind": "breaker", "dir": "bull" if m["dir"] == LONG else "bear", "top": px(sb.high), "bottom": px(sb.low),
                              "time": _ts(sb.time), "label": "Breaker"})
    pd_arrays = pd_arrays[-6:]
    # Enrich with full 8-tier PD-Array Matrix (scan_pd_arrays) and HTF arrays while keeping event arrays first
    try:
        from .pd_arrays import scan_pd_arrays
        pda_labels = {
            "ob": ("+OB", "−OB"), "breaker": ("+Brk", "−Brk"), "mitigation": ("+MB", "−MB"),
            "rejection": ("+RB", "−RB"), "propulsion": ("+PB", "−PB"), "bpr": ("+BPR", "−BPR"),
            "volume_imbalance": ("+VI", "−VI"),
        }
        seen_pda = {(a["kind"], a["top"], a["bottom"]) for a in pd_arrays}
        extra_pda = []
        for a in scan_pd_arrays(bars[-200:], min_displacement=0.25 * rng):
            if a.mitigated or a.kind == "fvg":
                continue
            d_str = "bull" if a.direction == LONG else "bear"
            top_v, bot_v = px(a.top), px(a.bottom)
            if (a.kind, top_v, bot_v) in seen_pda:
                continue
            seen_pda.add((a.kind, top_v, bot_v))
            lbl_pair = pda_labels.get(a.kind, (a.kind.upper(), a.kind.upper()))
            extra_pda.append({
                "kind": a.kind, "dir": d_str, "top": top_v, "bottom": bot_v,
                "mt": px(a.ce), "time": _ts(a.time), "label": lbl_pair[0] if d_str == "bull" else lbl_pair[1],
            })
        htf_fn = getattr(strategy, "_htf", None)
        if callable(htf_fn):
            for h in htf_fn():
                d_str = "bull" if h["dir"] == LONG else "bear"
                top_v, bot_v = px(h["top"]), px(h["bottom"])
                kind_h = f"htf_{h['kind']}"
                if (kind_h, top_v, bot_v) in seen_pda:
                    continue
                seen_pda.add((kind_h, top_v, bot_v))
                extra_pda.append({
                    "kind": kind_h, "dir": d_str, "top": top_v, "bottom": bot_v,
                    "mt": px((h["top"] + h["bottom"]) / 2), "time": _ts(h["time"]),
                    "label": f"{h['tf']} {'+' if d_str == 'bull' else '−'}{h['kind'].upper()}",
                })
        extra_pda.sort(key=lambda a: min(abs(a["top"] - price), abs(a["bottom"] - price)))
        pd_arrays = (pd_arrays + extra_pda)[:10]
    except Exception:
        pass

    # ---- dealing range (RL-06): the range since the read started, premium above 50%
    span = bars[i0:] or bars[-48:]
    hi_b = max(span, key=lambda b: b.high)
    lo_b = min(span, key=lambda b: b.low)
    eq_px = (hi_b.high + lo_b.low) / 2
    dealing = {"high": px(hi_b.high), "low": px(lo_b.low), "eq": px(eq_px), "high_time": _ts(hi_b.time),
               "low_time": _ts(lo_b.time), "zone": "premium" if price > eq_px else "discount",
               "label": "Today's range" if tf < 60 else "Recent range"}

    mo = getattr(strategy, "_midnight_open", None) if not any_hour else None

    # ---- where the model is (the ICT sequence) and the trade plan
    setups = list(getattr(strategy, "_setups", []))
    open_sig = next((s for s in (signals or []) if s.get("outcome", {}).get("status") == "open"), None)
    staged = next((s for s in setups if s.stage == "limit"), None)
    waiting = next((s for s in setups if s.stage == "await_mss"), None)
    in_window = any(w.contains(ny_now.time()) for w in st.killzones)
    nxt = None
    if not in_window and not any_hour:
        upcoming = [(_window_bounds(w, day0)[0], w) for w in st.killzones]
        upcoming += [(_window_bounds(w, day0 + timedelta(days=1))[0], w) for w in st.killzones]
        later = sorted((t, w) for t, w in upcoming if t > to_ny(last.time))
        nxt = later[0] if later else None

    def lv_label(name: str) -> str:
        return LEVEL_NAMES.get(name, name)

    intact = [lv for lv in levels if lv["status"] == "intact"]
    above = sorted((lv for lv in intact if lv["side"] == "high" and lv["price"] > price), key=lambda lv: lv["price"])
    below = sorted((lv for lv in intact if lv["side"] == "low" and lv["price"] < price), key=lambda lv: -lv["price"])

    plan = None
    checklist = []
    tick = lambda rule, label, state, detail: checklist.append({"rule": rule, "label": label, "state": state, "detail": detail})  # noqa: E731
    window_txt = (f"Inside the {WINDOW_NAMES.get(next((w.name for w in st.killzones if w.contains(ny_now.time())), ''), 'killzone')}" if in_window
                  else "Any hour (no time filter on this market/timeframe)" if any_hour
                  else f"Outside the killzones; next is the {WINDOW_NAMES.get(nxt[1].name, nxt[1].name)} at {nxt[0]:%H:%M} NY" if nxt else "Outside the killzones")
    if open_sig:
        stage, d = "in_trade", (SHORT if open_sig["direction"] == "short" else LONG)
        su = open_sig.get("setup") or {}
        plan = {"direction": open_sig["direction"], "entry": open_sig["entry"], "stop": open_sig["stop"], "tp2": open_sig["target"],
                "tp2_name": su.get("target_name"), "tp1": su.get("tp1"), "status": "open", "time": su.get("mss_time")}
    elif staged:
        stage, d = "limit", staged.direction
        plan = {"direction": "short" if d == SHORT else "long", "entry": px(staged.limit), "stop": px(staged.stop),
                "tp2": px(staged.target), "tp2_name": staged.target_name, "tp1": px(staged.tp1), "status": "pending",
                "time": _ts(staged.mss_time), "fvg": [px(staged.fvg[0]), px(staged.fvg[1])],
                "candles_left": strategy._expiry_bars() - staged.bars}
    elif waiting:
        stage, d = "await_mss", waiting.direction
    else:
        stage, d = ("watching" if (in_window or any_hour) else "waiting_time"), None
    if plan:
        risk = abs(plan["entry"] - plan["stop"]) or 1e-12
        plan["rr2"] = round(abs(plan["tp2"] - plan["entry"]) / risk, 2)
        plan["rr1"] = round(abs(plan["tp1"] - plan["entry"]) / risk, 2) if plan.get("tp1") is not None else None

    # the setup the checklist follows: in trade / staged / waiting for the shift
    cur = staged or waiting
    sweep_ev = next((e for e in reversed(evs) if e["type"] == "sweep" and (cur is None or e["level"] == cur.level.name)), None)
    tick("KZ", "Time", "done" if (in_window or any_hour or stage in ("in_trade", "limit", "await_mss")) else "pending", window_txt)
    if stage in ("in_trade", "limit", "await_mss") and sweep_ev:
        tick("SW-01", "Liquidity swept", "done",
             f"{lv_label(sweep_ev['level'])} {px(sweep_ev['level_price'])} run to {px(sweep_ev['extreme'])} at {to_ny(sweep_ev['time']):%H:%M} NY "
             f"({'buy-side' if sweep_ev['side'] == 'high' else 'sell-side'} liquidity taken)")
    else:
        near = ", ".join(f"{lv['label']} {lv['price']}" for lv in ([*above[:1], *below[:1]])) or "no fresh level nearby"
        tick("SW-01", "Liquidity swept", "pending", f"Waiting for a run on {near}")
    need = None
    if stage == "await_mss":
        found = strategy._last_swing(d, len(strategy._bars) - 1, max(0, waiting.sweep_index - st.swing_lookback))
        need = px(found[0]) if found else None
        left = st.mss_max_bars - waiting.bars
        tick("MS-02", "Shift in structure", "pending",
             (f"Needs a displacement candle to close {'below the swing low' if d == SHORT else 'above the swing high'} at {need}"
              if need is not None else "Waiting for a 3-candle swing to form") + f" ({max(0, left)} candles left)")
    elif stage in ("limit", "in_trade"):
        mss_ev = next((e for e in reversed(evs) if e["type"] == "mss"), None)
        tick("MS-02", "Shift in structure", "done",
             f"Displacement closed through {px(mss_ev['price'])} at {to_ny(mss_ev['time']):%H:%M} NY" if mss_ev else "Displacement through the swing")
    else:
        tick("MS-02", "Shift in structure", "pending", "After the sweep: a large-bodied candle through the last swing")
    if plan and plan.get("fvg"):
        tick("FV-01", "Fair value gap", "done", f"{plan['fvg'][0]}–{plan['fvg'][1]} in {'premium' if d == SHORT else 'discount'} of the leg (FV-02)")
    elif stage == "in_trade":
        tick("FV-01", "Fair value gap", "done", "Gap left by the displacement")
    else:
        tick("FV-01", "Fair value gap", "pending", "The displacement must leave a 3-candle gap in premium (sells) / discount (buys)")
    if plan:
        tick("EN-01", "Limit order", "done" if stage == "in_trade" else "pending",
             f"{'Filled' if stage == 'in_trade' else 'Resting'} at {plan['entry']}" + (f" · {plan['candles_left']} candles left (EN-04)" if plan.get("candles_left") is not None else ""))
        tick("TP-02", "Reward to risk", "done", f"Stop {plan['stop']} · TP1 {plan['tp1'] if plan['tp1'] is not None else '—'} · TP2 {plan['tp2']} ({plan['rr2']}R)")
    else:
        tick("EN-01", "Limit order", "pending", "Placed at the gap's edge once the gap forms")
        tick("TP-02", "Reward to risk", "pending", f"Needs at least {st.min_rr}R to the opposing liquidity")

    # ---- bias and draw on liquidity
    sides_taken = [e["side"] for e in evs if e["type"] in ("sweep",)] + \
                  [lv["side"] for lv in levels if lv["status"] == "spent"]
    if plan:
        bias = "bearish" if plan["direction"] == "short" else "bullish"
        draw = {"name": plan.get("tp2_name"), "label": lv_label(plan.get("tp2_name") or ""), "price": plan["tp2"]}
        src = cur.level.name if cur else (open_sig or {}).get("level", "")
        why = f"The model has a {plan['direction']} {'position' if stage == 'in_trade' else 'order'} from the {lv_label(src) if src else 'liquidity'} sweep."
    elif stage == "await_mss":
        bias = "bearish" if d == SHORT else "bullish"
        tgt = (below if d == SHORT else above)[:1]
        draw = {"name": tgt[0]["name"], "label": tgt[0]["label"], "price": tgt[0]["price"]} if tgt else None
        why = f"{lv_label(waiting.level.name)} was just swept; if structure shifts, price should seek the opposite liquidity."
    elif "low" in sides_taken and "high" not in sides_taken and (mo is None or price > mo):
        bias, draw = "bullish", ({"name": above[0]["name"], "label": above[0]["label"], "price": above[0]["price"]} if above else None)
        why = "Sell-side liquidity was taken first today" + (" and price is above the midnight open" if mo is not None else "") + ", so buy-side liquidity is the likely draw."
    elif "high" in sides_taken and "low" not in sides_taken and (mo is None or price < mo):
        bias, draw = "bearish", ({"name": below[0]["name"], "label": below[0]["label"], "price": below[0]["price"]} if below else None)
        why = "Buy-side liquidity was taken first today" + (" and price is below the midnight open" if mo is not None else "") + ", so sell-side liquidity is the likely draw."
    else:
        bias = "neutral"
        cands = [*above[:1], *below[:1]]
        near = min(cands, key=lambda lv: abs(lv["price"] - price)) if cands else None
        draw = {"name": near["name"], "label": near["label"], "price": near["price"]} if near else None
        if "low" in sides_taken and "high" in sides_taken:
            why = "Both buy-side and sell-side liquidity have been taken today: a two-sided day, no clean draw."
        elif sides_taken and mo is not None:
            side = "Sell-side" if "low" in sides_taken else "Buy-side"
            why = (f"{side} liquidity was taken, but price is still {'below' if side == 'Sell-side' else 'above'} the midnight open, "
                   "so the reversal is not confirmed.")
        else:
            why = "Neither side's liquidity has been run yet today. The nearer pool is the first draw."

    # ---- high-impact news today for this market's currencies (NF-01)
    news_today = []
    if news is not None:
        before = timedelta(minutes=news.config.minutes_before)
        after = timedelta(minutes=news.config.minutes_after)
        for e in news.events:
            if to_ny(e.time).date() == ny_now.date():
                news_today.append({"time": _ts(e.time), "time_ny": f"{to_ny(e.time):%H:%M}", "currency": e.currency,
                                   "title": e.title, "impact": e.impact,
                                   "from_ny": f"{to_ny(e.time - before):%H:%M}", "to_ny": f"{to_ny(e.time + after):%H:%M}",
                                   "active": e.time - before <= last.time <= e.time + after})
    off_day = news.no_trade(last.time) if news is not None else None
    if off_day:
        checklist.insert(1, {"rule": off_day.split()[0], "label": "No-trade day", "state": "fail", "detail": off_day.split(" ", 1)[1]})
    blackout = next((n for n in news_today if n["active"]), None)
    if blackout:
        checklist.insert(1, {"rule": "NF-01", "label": "News", "state": "fail",
                             "detail": f"{blackout['currency']} {blackout['title']} at {blackout['time_ny']} NY: no new entries {blackout['from_ny']}–{blackout['to_ny']}"})
    elif news_today:
        nxt_n = next((n for n in news_today if n["time"] > _ts(last.time)), None)
        if nxt_n:
            checklist.insert(1, {"rule": "NF-01", "label": "News", "state": "done",
                                 "detail": f"Clear now · next: {nxt_n['currency']} {nxt_n['title']} {nxt_n['time_ny']} NY (no entries {nxt_n['from_ny']}–{nxt_n['to_ny']})"})

    # ---- daily bias from the last completed days (RL-05)
    daily = None
    if hasattr(strategy, "daily_bias") and tf < 60:
        b_dir, b_why = strategy.daily_bias()
        daily = {"direction": "bullish" if b_dir == LONG else "bearish" if b_dir == SHORT else "neutral", "reason": b_why}
        if bias == "neutral" and b_dir:
            # nothing decisive today yet: the daily draw sets the side to look for
            bias = daily["direction"]
            pool = above if b_dir == LONG else below
            draw = {"name": pool[0]["name"], "label": pool[0]["label"], "price": pool[0]["price"]} if pool else draw
            why = f"Daily bias: {b_why}. Look for {'sell-side to be run first, then buys' if b_dir == LONG else 'buy-side to be run first, then sells'}."

    # ---- the narrative (narrative.py): the one story the strategy trades on, so the page never tells a different one
    nv = strategy.narrative() if hasattr(strategy, "narrative") and getattr(strategy, "_bars", None) else None
    story_json = nv.to_json(px) if nv else None
    if nv and getattr(strategy.config, "narrative_required", False) and not plan and stage != "await_mss":
        bias = story_json["bias"]
        why = nv.lines[-1]
        draw = {"name": nv.draw.name, "label": lv_label(nv.draw.name), "price": px(nv.draw.price)} if nv.draw else None

    # ---- narrative, in the order things happened
    story = []
    if nv and getattr(strategy.config, "narrative_required", False):
        story.extend(nv.lines[:-1])
    elif daily:
        story.append(f"Daily bias (RL-05): {daily['direction']} — {daily['reason']}.")
    for n in news_today:
        story.append(f"{n['time_ny']} — {n['currency']} {n['title']} ({n['impact']} impact): no entries {n['from_ny']}–{n['to_ny']} (NF-01).")
    for s_ in sessions:
        if s_["kind"] == "range" and s_["high"] is not None and s_["end"] <= _ts(last.time):
            story.append(f"{s_['label']}: {s_['low']}–{s_['high']}.")
    if mo is not None:
        story.append(f"NY midnight open {px(mo)}; price is {'above' if price > mo else 'below'} it, in the {dealing['zone']} of {dealing['label'].lower()} (50% at {dealing['eq']}).")
    for e in evs:
        t = f"{to_ny(e['time']):%H:%M}"
        if e["type"] == "sweep":
            story.append(f"{t} — {lv_label(e['level'])} {px(e['level_price'])} swept to {px(e['extreme'])}: {'buy-side' if e['side'] == 'high' else 'sell-side'} liquidity taken in the {WINDOW_NAMES.get(e['window'], e['window'])}.")
        elif e["type"] == "mss":
            story.append(f"{t} — Market structure shift: displacement through {px(e['price'])}.")
        elif e["type"] == "staged":
            story.append(f"{t} — {'Sell' if e['dir'] == SHORT else 'Buy'} limit at {px(e['limit'])} in the FVG {px(e['fvg'][0])}–{px(e['fvg'][1])}, stop {px(e['stop'])}, "
                         + (f"TP1 {px(e['tp1'])}, " if e.get("tp1") is not None else "") + f"TP2 {lv_label(e['target_name'])} {px(e['target'])} ({e['rr']:.1f}R).")
        elif e["type"] == "filled":
            story.append(f"{t} — Limit filled at {px(e['price'])}.")
        elif e["type"] in ("rejected", "cancelled", "ignored"):
            story.append(f"{t} — {_names(e['message'])} ({e['rule']}).")
        elif e["type"] == "spent":
            story.append(f"{t} — {_names(e['message'].replace('; spent', ''))}: spent for today (RL-07).")
    story.append(f"Bias: {bias}. {why}" + (f" Draw on liquidity: {draw['label']} {draw['price']}." if draw else ""))

    # ---- 20d, 40d, and 60d IPDA dealing ranges (HT-01)
    ipda_ranges = {}
    htf_ranges_fn = getattr(strategy, "_htf_ranges", None)
    if callable(htf_ranges_fn):
        try:
            r_all = htf_ranges_fn().get("ipda_all", {})
            for n in (20, 40, 60):
                if n in r_all and r_all[n]:
                    hi_n, lo_n = r_all[n]
                    eq_n = (hi_n + lo_n) / 2.0
                    ipda_ranges[f"{n}d"] = {
                        "high": px(hi_n),
                        "low": px(lo_n),
                        "eq": px(eq_n),
                        "zone": "premium" if price > eq_n else "discount",
                    }
        except Exception:
            pass

    # ---- weekly profile / bias (HT-02)
    weekly = None
    if hasattr(strategy, "weekly_bias"):
        try:
            w_dir, w_why = strategy.weekly_bias()
            weekly = {
                "direction": "bullish" if w_dir == LONG else "bearish" if w_dir == SHORT else "neutral",
                "reason": w_why,
            }
        except Exception:
            pass

    # ---- where to wait action plan
    matching_pds = []
    if pd_arrays:
        target_dir = "bull" if bias == "bullish" else "bear" if bias == "bearish" else None
        if target_dir:
            matching_pds = [p for p in pd_arrays if p.get("dir") == target_dir]
        if not matching_pds:
            matching_pds = pd_arrays

    best_poi = matching_pds[0] if matching_pds else None
    poi_label = best_poi.get("label", best_poi.get("kind", "").upper()) if best_poi else None
    poi_bounds = f"{best_poi['bottom']} – {best_poi['top']}" if best_poi else None

    if poi_label and poi_bounds:
        wait_instruction = f"Wait for {poi_label} [{poi_bounds}]"
    elif dealing.get("eq"):
        wait_instruction = f"Wait for {'Discount pullback (< ' if bias == 'bullish' else 'Premium rally (> '}{dealing['eq']})"
    elif draw:
        wait_instruction = f"Wait for session liquidity sweep before move to {draw.get('label', draw.get('name'))}"
    else:
        wait_instruction = "Wait for session liquidity sweep"

    session_window = (
        "Pure Price Action (24/7 continuous — algorithmic volatility, time zones inactive)"
        if any_hour else
        "London Killzone (02:00–05:00 NY) / NY AM (08:30–11:00 NY) / Silver Bullet (10:00–11:00 NY)"
    )

    where_to_wait = {
        "bias": bias,
        "conviction": (story_json.get("conviction") if story_json else "moderate") or "moderate",
        "daily_bias": daily,
        "weekly_bias": weekly,
        "draw": draw,
        "ipda_ranges": ipda_ranges,
        "poi": {
            "label": poi_label,
            "bounds": poi_bounds,
            "instruction": wait_instruction,
        },
        "session_window": session_window,
        "execution_sequence": [
            "1. Higher-Timeframe Alignment: Ensure trade direction matches the HTF Draw on Liquidity and IPDA 20/40/60-day bias.",
            "2. Intraday Liquidity Raid: Wait for price to sweep an active liquidity pool (Asian High/Low, London High/Low, or PDH/PDL).",
            "3. Market Structure Shift (MSS): Look for a decisive 5-minute displacement candle closing with a full body beyond recent swing structure.",
            "4. Fair Value Gap (FVG) Retest: Place a limit order at the newly formed FVG boundary or 50% Consequent Encroachment (CE).",
            "5. Titanium Risk Management: Set Stop Loss at swing invalidation; move Stop to Breakeven at +1.0R gain; harvest 50% partial at +1.5R (TP1); let runner ride to opposing DOL (TP2).",
        ],
    }

    def ev_json(e: dict) -> dict:
        out = {"type": e["type"], "time": _ts(e["time"])}
        for k, v in e.items():
            if k in ("type", "time"):
                continue
            if isinstance(v, datetime):
                out[k] = _ts(v)
            elif isinstance(v, float):
                out[k] = px(v)
            elif isinstance(v, tuple):
                out[k] = [px(x) if isinstance(x, float) else x for x in v]
            elif k == "dir":
                out[k] = "short" if v == SHORT else "long"
            else:
                out[k] = v
        return out

    return {
        "as_of": _ts(last.time), "price": px(price), "timeframe_minutes": tf, "any_hour": any_hour,
        "day_start": _ts(day0), "midnight_open": px(mo), "sessions": sessions, "levels": levels,
        "equal_levels": eq, "swings": swings, "fvgs": fvgs, "dealing_range": dealing,
        "events": [ev_json(e) for e in evs][-20:], "stage": stage, "need_mss": need,
        "checklist": checklist, "plan": plan, "bias": bias, "daily_bias": daily, "weekly_bias": weekly, "news": news_today, "no_trade": off_day, "pd_arrays": pd_arrays, "bias_reason": why, "draw": draw, "narrative": story[-14:], "story": story_json,
        "ipda_ranges": ipda_ranges, "where_to_wait": where_to_wait,
    }
