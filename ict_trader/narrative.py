"""The narrative: ICT's top-down read that every trade has to answer to.

ICT does not look for a pattern and then ask whether it is a good one. He builds the story first and lets price
show whether it is playing out:

  where is price coming from   the liquidity it just took, and what yesterday and last week did
  where is it going            the draw on liquidity: the next pool of resting stops (or imbalance) in that direction
  where is it now              premium or discount of the dealing range (yesterday's, the 20-day IPDA range)
  is the road clear            how many opposing arrays sit between price and the draw
  is there a story at all      no draw, a draw already fulfilled, or daily and weekly pulling opposite ways: stand aside

Then, and only then: a raid on a pool that fits the story, a displacement that confirms it, an entry in the array the
displacement leaves, the stop beyond the raid, the target at the draw with partials at the pools on the way.

This module is the single place that story is written. The strategy asks it before it takes a sweep seriously
(`bias`, `stand_aside`), aims its target with it (`draw_pool`), and the app shows the same words to the trader
(analysis.py), so the screen and the trades can never tell two different stories. It reads only what the strategy
already tracks (completed trading days, reference levels, liquidity pools), so it is price-only and also applies to
the synthetic indices, whose day is a fixed 24-hour bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .sessions import Level, to_ny

LONG, SHORT = 1, -1
EXTERNAL = ("PD", "PW", "IPDA", "NDOG", "NWOG", "ORG", "CBDR", "ASIA+", "ASIA-")   # prior day / week, the 20-60 day ranges, opening gaps, SD projections
SESSION_POOLS = EXTERNAL + ("LONDON_", "ASIA_", "PREMARKET_", "LUNCH_", "EQH", "EQL", "OC")


def draw_pool(st: Any, direction: int, ref: float) -> Level | None:
    """The draw for a trade in `direction` from price `ref`: the nearest unpurged higher-timeframe pool (prior day or
    week extreme, the IPDA ranges, an opening gap) ahead of it; when that pool is beyond today's remaining average
    daily range (or when none is left), the nearest reachable session or equal-highs/lows pool ahead of price."""
    side = "low" if direction == SHORT else "high"
    cands = st._all_levels()
    if st.config.htf_context:
        cands += st._htf_levels()
    if st.config.opening_gaps:
        cands += st._gap_midpoints()
    cands += st._cbdr_levels()
    opp = [lv for lv in cands if lv.side in (side, "mid") and (lv.price - ref) * direction > 0]
    if not opp:
        pip = getattr(st.instrument, "pip_size", 0.0001)
        return Level("DOL_EXPANSION", ref + direction * 25.0 * pip, side)
    unspent = [lv for lv in opp if lv.name not in st._spent]
    opp = unspent if unspent else opp
    ext = [lv for lv in opp if lv.name.startswith(EXTERNAL)]
    best_ext = min(ext, key=lambda lv: abs(lv.price - ref)) if ext else None
    adr_info = st._adr() if hasattr(st, "_adr") else None
    if best_ext is not None and adr_info is not None:
        adr, used = adr_info[0], adr_info[1]
        if adr > 0 and (used + abs(best_ext.price - ref)) / adr > 1.25:
            named = [lv for lv in opp if lv.name.startswith(SESSION_POOLS) and (used + abs(lv.price - ref)) / adr <= 1.25]
            if named:
                return min(named, key=lambda lv: abs(lv.price - ref))
    return best_ext or min(opp, key=lambda lv: abs(lv.price - ref))


@dataclass(frozen=True)
class Narrative:
    bias: int                       # +1 buy side, -1 sell side, 0 no story
    conviction: str                 # "strong" | "moderate" | "weak" | "none"
    came_from: str                  # where price is coming from
    going_to: str                   # where it is going, and how clear the road is
    location: str                   # premium / discount, in words
    draw: Level | None              # the objective
    stand_aside: str | None         # why there is no trade to take today, when that is the answer
    lines: tuple[str, ...]          # the story in the order a trader reads it
    profile: dict[str, Any] | None = None   # where the week is, on a market with a clock (weekly_profile)

    def to_json(self, fmt=lambda v: v) -> dict[str, Any]:
        return {"bias": "bullish" if self.bias > 0 else "bearish" if self.bias < 0 else "neutral", "conviction": self.conviction,
                "came_from": self.came_from, "going_to": self.going_to, "location": self.location,
                "draw": {"name": self.draw.name, "price": fmt(self.draw.price)} if self.draw else None,
                "stand_aside": self.stand_aside, "lines": list(self.lines), "profile": self.profile}


# ICT's weekly profile: a week has a shape. In a bullish week the low of the week forms early (Monday into Tuesday, a raid of
# sell-side liquidity below the weekly open), Wednesday is the pivot, Thursday the expansion toward the weekly draw and Friday
# the retrace and distribution. A bearish week is the mirror. Only on a market with a clock: a synthetic index has no week.
_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
_PHASE = {
    0: ("set-up", "Monday: the week is being set up, often as a range. The {extreme} of the week can form now or on Tuesday, so be selective and let a raid of {raided} liquidity show the hand first."),
    1: ("manipulation", "Tuesday: the classic day for the {extreme} of a {week} week, a raid of {raided} liquidity {where} the weekly open. An entry after that raid is high probability."),
    2: ("pivot", "Wednesday: the pivot of the week. If Monday and Tuesday moved against the weekly bias, this is where it reverses and the expansion begins."),
    3: ("expansion", "Thursday: the expansion toward the weekly draw. Follow it, and take profits at the draw."),
    4: ("distribution", "Friday: the expansion is mostly done. Expect a retrace or a range, take profits and avoid new trades late in the day."),
}


def weekly_profile(st: Any, d: int | None = None) -> dict[str, Any] | None:
    """Where the week is and whether a trade in direction `d` (default: the story's side) fits it: fit +1 (it does), 0 (neutral), -1 (it does not)."""
    if not st._bars or not st._time_based():
        return None
    now = to_ny(st._bars[-1].time)
    wd = now.weekday()
    if wd > 4:
        return None
    w_bias, _ = st.weekly_bias()
    bias = w_bias or st.daily_bias()[0]
    side = d if d is not None else bias
    week = "bullish" if bias > 0 else "bearish" if bias < 0 else "unclear"
    phase, text = _PHASE[wd]
    text = text.format(extreme="low" if bias >= 0 else "high", raided="sell-side" if bias >= 0 else "buy-side", week=week, where="below" if bias >= 0 else "above")
    monday = [k for k in sorted(st._days) if k >= (now.date() - timedelta(days=wd))]
    wopen = st._days[monday[0]][0] if monday else None
    price = st._bars[-1].close
    right_side = None if wopen is None or not side else (price < wopen if side > 0 else price > wopen)
    if not bias:
        fit, text = 0, f"{_DAYS[wd]}: no clear weekly lean, so no weekly profile to lean on."
    elif side != bias:
        fit = -1
        text = f"{_DAYS[wd]}: this goes against the {week} week. " + text
    elif wd == 0:
        fit = 0
    elif wd == 4:
        fit = -1 if now.hour >= 12 else 0
    else:
        fit = 1 if (right_side in (None, True) or wd in (2, 3)) else 0   # Tuesday manipulation on right side of weekly open, or Wednesday/Thursday expansion with the week
    if right_side is not None and bias and side == bias and wd in (1, 2, 3):
        text += f" Price is {'below' if price < wopen else 'above'} the weekly open {st._fmt(wopen)}, " + ("the right side for " + ("buys." if side > 0 else "sells.") if right_side else "the wrong side for " + ("buys." if side > 0 else "sells."))
    return {"day": _DAYS[wd], "week": week, "phase": phase, "text": text, "fit": fit, "weekly_open": wopen}


def read(st: Any) -> Narrative:
    """The story for the market as the strategy sees it right now."""
    f = st._fmt
    price = st._bars[-1].close if st._bars else 0.0
    # The bias comes from the rung above the execution timeframe (htf.ladder): 1m -> 1h, 5m -> 4h, 15m and up -> daily. On those
    # the primary is the daily bias and the secondary the weekly, exactly as before.
    lb = st.ladder_bias() if hasattr(st, "ladder_bias") else None
    if lb:
        (d_bias, d_why), (w_bias, w_why), (s_bias, s_why), (n1, n2) = lb["p"], lb["u"], lb["s"], lb["names"]
        pr, rname = lb["range"], lb["range_name"]
    else:
        (d_bias, d_why), (w_bias, w_why) = st.daily_bias(), st.weekly_bias()
        s_bias, s_why = st.structure_bias() if hasattr(st, "structure_bias") else (0, "")
        n1, n2, pr, rname = "daily", "weekly", st._prior_range(), "yesterday's range"
    lines: list[str] = []

    # where price is coming from: the weekly and daily draw, then the liquidity taken today
    came = [f"{n2.capitalize()}: {w_why}.", f"{n1.capitalize()}: {d_why}."]
    taken = [s for s in st._swept_sides]
    if taken:
        came.append(("Buy-side" if taken[0] == "BSL" else "Sell-side") + " liquidity was taken first today"
                    + (f", then {'buy-side' if taken[1] == 'BSL' else 'sell-side'}" if len(taken) > 1 else "") + ".")
    lines += came

    # where it is now: premium or discount of yesterday's range and of the 20-day range
    loc_parts, loc = [], 0
    if pr:
        eq = (pr[0] + pr[1]) / 2
        loc_parts.append(f"{'discount' if price < eq else 'premium'} of {rname} (50% at {f(eq)})")
        loc = 1 if price < eq else -1 if price > eq else 0
    # Michael J. Huddleston Interbank Price Delivery Algorithm (IPDA): 20, 40, and 60-day institutional dealing ranges
    ipda_all = st._htf_ranges().get("ipda_all", {}) if st.config.htf_context and hasattr(st, "_htf_ranges") else {}
    for n in (20, 40, 60):
        if n in ipda_all and ipda_all[n]:
            rng_n = ipda_all[n]
            eq_n = (rng_n[0] + rng_n[1]) / 2
            side_n = "discount" if price < eq_n else "premium"
            loc_parts.append(f"{side_n} of the {n}-day IPDA range (50% at {f(eq_n)})")
    if not ipda_all and st.config.htf_context and hasattr(st, "_htf_ranges"):
        ip = st._htf_ranges().get("ipda")
        if ip:
            eq20 = (ip[0] + ip[1]) / 2
            loc_parts.append(f"{'discount' if price < eq20 else 'premium'} of the 20-day range (50% at {f(eq20)})")
    location = "In the " + " and in the ".join(loc_parts) + "." if loc_parts else "Not enough completed days to place price in a range yet."
    lines.append(location)

    # the story. ICT "does not force a daily bias": he reads the likely weekly expansion and trades with it (koN1ge8bewI 20:41).
    # So the daily draw leads when it is clear, the weekly draw may veto it, and when the day itself is inside or two-sided
    # the weekly draw leads on its own (less sure of itself). No draw at all, or the two pulling opposite ways: stand aside.
    # The trend is read over many candles (the daily structure), not from one: yesterday's candle against the day before and last
    # week's against the week before are two-candle reads, and a bearish daily candle inside an uptrend is a pullback, not a new bias.
    if s_bias:
        lines.append(f"{n1.capitalize()} structure: {s_why}.")
    stand: str | None = None
    bias, weekly_led = d_bias, False
    if n1 == "daily" and s_bias and d_bias and w_bias and d_bias == w_bias != s_bias:
        lines.append("The daily and weekly candles run against the daily structure: a counter-trend idea, lower probability.")
    elif n1 == "daily" and s_bias and ((d_bias and d_bias != s_bias) or (w_bias and w_bias != s_bias)):
        bias = s_bias
        lines.append("A candle reads against the daily structure, so it is a pullback inside the trend, not a new bias: the trend decides.")
    elif n1 != "daily" and w_bias and d_bias != w_bias:
        bias, weekly_led = w_bias, True
        lines.append(f"The {n1} candle retraced against the {n2} draw ({w_why}), setting up an intraday manipulation leg toward the {n2} draw.")
    elif d_bias and w_bias and w_bias != d_bias:
        # A single-day candle against the weekly trend is a retracement/pullback in the weekly expansion.
        bias, weekly_led = w_bias, True
        lines.append(f"The daily candle retraced against the weekly draw ({w_why}); higher-timeframe weekly draw leads.")
    elif not d_bias and w_bias:
        bias, weekly_led = w_bias, True
        lines.append("The day is inside or two-sided, so the weekly draw leads.")
    elif not d_bias and s_bias:
        bias, weekly_led = s_bias, True
        lines.append("The day is inside or two-sided, so the daily structure leads.")
    elif not d_bias:
        if lb:
            wb_top, wb_why = st.weekly_bias()
            sb_top, sb_why = st.structure_bias() if hasattr(st, "structure_bias") else (0, "")
            if wb_top or sb_top:
                bias, weekly_led = (wb_top or sb_top), True
                lines.append(f"The {n1} and {n2} candles are inside, so the higher-timeframe {'weekly draw' if wb_top else 'daily structure'} leads ({wb_why if wb_top else sb_why}).")
            elif len(taken) == 1 and getattr(st, "_midnight_open", None) is not None:
                mo_ = st._midnight_open
                po3_dir = LONG if taken[0] == "SSL" and price >= mo_ else SHORT if taken[0] == "BSL" and price <= mo_ else 0
                if po3_dir:
                    bias, weekly_led = po3_dir, True
                    lines.append(f"Higher timeframes are inside, so intraday Power of 3 manipulation ({taken[0]} purged across Midnight Open {f(mo_)}) leads.")
        if not bias:
            wb_top, wb_why = st.weekly_bias()
            sb_top, sb_why = st.structure_bias() if hasattr(st, "structure_bias") else (0, "")
            if wb_top:
                bias, weekly_led = wb_top, True
                lines.append(f"Inside day: leaning on higher-timeframe weekly draw ({wb_why}).")
            elif sb_top:
                bias, weekly_led = sb_top, True
                lines.append(f"Inside day: leaning on higher-timeframe structure ({sb_why}).")
            else:
                stand, bias = "no clear daily or weekly draw (inside or two-sided): nothing to lean on", 0

    draw = draw_pool(st, bias, price) if bias else None
    if bias and draw is None:
        draw = Level("DOL_EXPANSION", price + bias * 30 * getattr(st.instrument, "pip_size", 0.0001), "high" if bias > 0 else "low")
    if bias and len(st._bars) > 48 and st._seek_destroy(len(st._bars) - 1):
        stand = "both sides of the range are being stopped out (seek and destroy)"

    agree = [x for x in (d_bias, w_bias, s_bias, loc) if x]
    conviction = "none" if not bias else "strong" if all(x == bias for x in agree) and w_bias == bias and not weekly_led else \
        "moderate" if all(x == bias for x in agree) and not weekly_led else "weak"
    if stand:
        conviction = "none"

    if bias:
        side = "buy-side" if bias > 0 else "sell-side"
        going = (f"The {side} draw is {draw.name} at {f(draw.price)}" if draw else f"The {side} draw is already fulfilled") + \
                (f", {abs(draw.price - price) / max(st.instrument.pip_size, 1e-12):.0f} pips away" if draw else "") + "."
        lines.append(f"Story: {'bullish' if bias > 0 else 'bearish'} ({conviction}). Look for {'sell-side' if bias > 0 else 'buy-side'} liquidity to be run first, then "
                     f"{'buys' if bias > 0 else 'sells'} toward the draw. {going}")
    else:
        going = "No draw to aim at: stand aside."
        lines.append(f"No story: {stand}. ICT stands aside here.")
    if stand and bias:
        lines.append(f"But stand aside: {stand}.")
    prof = weekly_profile(st, bias or None)
    if prof:
        lines.append("Week: " + prof["text"])
    return Narrative(bias, conviction, " ".join(came), going, location, draw, stand, tuple(lines), prof)
