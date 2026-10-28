"""How each trading style and model takes profit and moves the stop, the ICT way.

ICT's trade management (TP-03/TP-04, SL-04/SL-05, TP-01):
- take part of the position off at the first, short-term liquidity pool (TP1);
- at 50% of the way to the target, trim the stop's risk by 25% instead of jumping to
  breakeven, so routine noise doesn't stop the trade out (SL-04);
- at 75% of the way, or once the first partial is banked, move the stop to entry (SL-05);
- hold the rest, risk-free, to the draw on liquidity (TP-01), trailing the stop behind each new swing once the first
  partial is banked or the trade is half way (the trailing runner; TRAIL_RUNNER=0 switches it off).

How much comes off at TP1 depends on the style and the model:
- scalp (1-3m): scalps live off the first pool, so most of the position (75%) comes off there.
- intraday (5m/15m, the 2022 model): half at TP1, the runner to the day's draw.
- swing (1h): half at TP1, the runner to the prior day's or week's liquidity.
- long-term (4h): a quarter at TP1; the trade is about the weekly draw.
- Silver Bullet: a single objective, the next liquidity pool, so the whole position rides to it.
- Sweep & reclaim (turtle soup): a fade of a false break; half at the first pool (when the
  model has one), the rest to the opposite side of the range.
"""

from __future__ import annotations

import os
from dataclasses import replace

from .config import StrategyConfig

STYLE_PLANS = {
    "scalp": {"partial_at_tp1": 0.75, "trim_stop_at_50_pct": False, "move_be_at_75_pct": True},
    "intraday": {"partial_at_tp1": 0.5, "trim_stop_at_50_pct": True, "move_be_at_75_pct": True},
    "swing": {"partial_at_tp1": 0.5, "trim_stop_at_50_pct": True, "move_be_at_75_pct": True},
    "position": {"partial_at_tp1": 0.25, "trim_stop_at_50_pct": False, "move_be_at_75_pct": True},
}
MODEL_PLANS = {
    "silver_bullet": {"partial_at_tp1": 0.0},
    "sweep": {},
}


def style_of(tf: int) -> str:
    return "scalp" if tf <= 3 else "intraday" if tf <= 15 else "swing" if tf <= 60 else "position"


def plan(tf: int, model: str) -> dict:
    return {**STYLE_PLANS[style_of(tf)], **MODEL_PLANS.get(model, {})}


def trail_on(tf: int, model: str) -> bool:
    """Does the live plan trail the runner behind each new swing? Only when part of the position is taken off at TP1
    (there is a runner), and unless TRAIL_RUNNER=0. The Research lab keeps the plain plan as its baseline and tests the
    trail as its own part ("Trailing Runner")."""
    return os.environ.get("TRAIL_RUNNER", "1").strip().lower() not in ("0", "false", "off", "no") and plan(tf, model)["partial_at_tp1"] > 0


def apply(st: StrategyConfig, tf: int | None = None, live: bool = False) -> StrategyConfig:
    """The strategy with this style's and model's ICT trade management. `live`: the scan, which also trails the runner."""
    tf = tf or st.timeframe_minutes
    return replace(st, **plan(tf, st.model), **({"trail_structure": True} if live and trail_on(tf, st.model) else {}))


def describe(tf: int, model: str, trail: bool | None = None) -> dict:
    """The plan in plain words, for signals, the bots and the app. `trail`: say the runner trails (default: as the live plan does)."""
    p = plan(tf, model)
    trail = trail_on(tf, model) if trail is None else trail
    part = p["partial_at_tp1"]
    steps = []
    if part > 0:
        steps.append(f"Take {int(part * 100)}% off at TP1 and move the stop to entry")
    if p["trim_stop_at_50_pct"]:
        steps.append("halfway to the target, cut the stop's risk by 25%")
    if p["move_be_at_75_pct"]:
        steps.append("at 75% of the way, stop to entry")
    if trail and part > 0:
        steps.append("then trail the stop behind each new swing")
    steps.append("hold the rest to the final target" if part > 0 else "hold the whole trade to the target")
    text = "; ".join(steps)
    return {"style": style_of(tf), "partial": part, "trim_50": p["trim_stop_at_50_pct"], "be_75": p["move_be_at_75_pct"], "trail": bool(trail and part > 0),
            "text": text[0].upper() + text[1:] + "."}
