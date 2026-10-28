"""The trading models the engine runs, and how to build each one's strategy.

- ict2022:       the 2022 Mentorship model: killzone -> liquidity sweep -> displacement
                 MSS -> FVG -> limit order at the gap (ict_trader/ict2022.py)
- silver_bullet: the same sequence, but only inside ICT's three 60-minute Silver Bullet
                 windows (SB-01: 03:00-04:00, 10:00-11:00, 14:00-15:00 NY); the limit
                 sits at the first gap the displacement leaves (SB-02) and is cancelled
                 if price reaches the draw before filling (SB-04)
- judas_swing:   the 2022 sequence only in the first hour of the London (02:00-03:00 NY) and
                 New York (09:30-10:30 NY) opens: the false move at the open runs a level
                 against the day's bias, then reverses (KZ-01 midnight open, RL-05 bias)
- ote:           the 2022 sequence, but the limit sits at the 70.5% optimal trade entry of the
                 displacement leg (62-79% zone) instead of in the gap
- unicorn:       the 2022 sequence, but only when the gap overlaps the breaker block; the
                 limit rests in the overlap
- sweep:         Turtle Soup (sweep & reclaim): an old high or low is run inside a killzone
                 and a candle closes back inside it; entry at market on that close (strategy.py)

Newer ICT models (LAB_MODELS): switched on in the live scan, alerts and bots like the rest, and
marked "new" until the Research lab has confirmed them on real data:
- ifvg:          after the sweep, a displacement close through a gap formed the other way;
                 that inverted gap is the entry (inversion fair value gap)
- cisd:          after the sweep, a close back past the open of the run that made it (change in
                 state of delivery) confirms the turn; entry in the gap it leaves
- breaker:       the 2022 sequence with the limit at the breaker block's mean threshold
- ob:            the 2022 sequence with the limit at the order block's mean threshold
- narrative:     the 2022 model traded only toward the daily draw, selling in premium and
                 buying in discount of yesterday's range
- htf:           narrative plus the weekly draw and the 20-day (IPDA) range
- pools:         swing highs/lows and equal highs/lows as extra liquidity to sweep and target
- quality:       only fair value gaps of real size that price has not traded back into
- gaps:          opening gaps (new day, new week, 09:30 range gap) as levels to sweep and draws to target
- fpfvg:         the first fair value gap printed after the 01:30 / 07:00 / 09:30 NY open, entered at 50%
- clear:         skips high-resistance setups and seek-and-destroy conditions
- dayopen:       buys at or below the day's open, sells at or above it
- weekly:        sits out Monday and Friday afternoon, never against the weekly draw
- sdt:           measured-move targets when no liquidity gives enough reward
- reentry:       after a stop-out that price runs past, looks for a fresh shift and takes the idea again
- macro, osok, cbdr, trail, drawtarget, tightstop: one part each (time, week, targets, management, stop)
- stack_trend / stack_reversal / stack_open: parts chosen to work together for a kind of market
- custom:entry=...;stop=...;target=...;filter=...;mgmt=...: any combination, for the Research lab
- venom, atm, mm: pattern models with their own trigger, entry and stop (patterns.py)
- reaper:        an inversion gap nested with an opposing gap
- full:          all of the above together
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import time

from .config import InstrumentConfig, StrategyConfig, Window
from .ict2022 import ICT2022Strategy
from .institutional import ICTInstitutionalStrategy
from .opening import FirstPresentedStrategy
from .blk_mitigation import MitigationStrategy
from .blk_propulsion import PropulsionStrategy
from .blk_rejection import RejectionStrategy
from .blk_suspension import SuspensionStrategy
from .blk_vacuum import VacuumStrategy
from .blk_vimb import VolumeImbalanceStrategy
from .advisor import CAUTION
from .patterns import ATMStrategy, MarketMakerStrategy, VenomStrategy
from .strategy import TimeAndPriceStrategy

CORE_MODELS = ("ict2022", "silver_bullet", "judas_swing", "ote", "unicorn", "sweep")
# Newer models: not yet confirmed in the Research lab, but switched on everywhere (scan, alerts,
# bots, Desk) and marked "new" in the app. Confirm them with the Research lab, then keep or drop.
PATTERN_MODELS = {"venom": VenomStrategy, "atm": ATMStrategy, "mm": MarketMakerStrategy,
                  "mitigation": MitigationStrategy, "rejection": RejectionStrategy, "propulsion": PropulsionStrategy,
                  "suspension": SuspensionStrategy, "vimb": VolumeImbalanceStrategy, "vacuum": VacuumStrategy}
INSTITUTIONAL_MODELS = ("institutional", "ict_institutional")
LAB_MODELS = ("nonarrative", "dailybias", "ifvg", "cisd", "breaker", "ob", "pools", "quality", "gaps", "fpfvg", "clear", "dayopen",
              "weekly", "sdt", "reentry", "macro", "osok", "cbdr", "trail", "drawtarget", "tightstop",
              "stack_trend", "stack_reversal", "stack_open", "venom", "atm", "mm", "mitigation", "rejection", "propulsion", "suspension", "vimb", "vacuum",
              "reaper", "full", "institutional")  # ict_institutional is an alias of institutional, not scanned twice
MODELS = CORE_MODELS + LAB_MODELS
# Models whose detectors or evidence are still being reviewed against the lesson frames: they run in the Research lab and
# the tests, but stay out of the live scan, alerts and bots until the review is done (SCAN_MODELS=... can name them).
# Venom, the ATM method and the Market Maker join them: on random data (no edge by construction) they took 80 to 250 trades each in 40 days
# while the ICT 2022 family took none, so nothing yet shows they tell structure from noise. They go live when the Research lab shows a measured edge.
PREVIEW_MODELS = ("mitigation", "rejection", "propulsion", "suspension", "vimb", "vacuum", "venom", "atm", "mm")

# What each name is. A MODEL is a complete trade recipe: it finds the setup, rests an order with a stop and a target,
# and its trade goes through the adviser's judgement matrix, which says TAKE / CAUTION / SKIP. Everything else is a
# BUILDING BLOCK: a part a model (or a custom build) uses, or that the Research lab tests on top of the base model to
# see what it adds. Building blocks never alert on their own and never count as "a model agrees".
#   direction  which way to lean (the day's, the week's and the 20-day draw)
#   filter     which setups to leave out, or which levels to watch
#   zone       a PD array to enter from (order block, breaker, mitigation and the other blocks, inversion gaps)
#   target     where to aim
#   management how to carry the trade
#   stop       where to put the stop
KIND = {
    "nonarrative": "direction", "dailybias": "direction", "weekly": "direction", "dayopen": "direction",
    "pools": "filter", "quality": "filter", "clear": "filter", "macro": "filter", "gaps": "filter",
    "ifvg": "zone", "cisd": "zone", "breaker": "zone", "ob": "zone", "mitigation": "zone", "rejection": "zone",
    "propulsion": "zone", "suspension": "zone", "vimb": "zone", "vacuum": "zone",
    "sdt": "target", "drawtarget": "target", "cbdr": "target",
    "trail": "management", "reentry": "management",
    "tightstop": "stop",
}
KIND_LABEL = {"model": "Model", "direction": "Direction", "filter": "Filter", "zone": "Zone", "target": "Target", "management": "Management", "stop": "Stop"}


def kind_of(name: str) -> str:
    """'model' for a trade recipe (including every custom build), else the kind of building block it is."""
    return KIND.get(name, "model")


TRADE_MODELS = tuple(m for m in MODELS if kind_of(m) == "model")
BUILDING_BLOCKS = tuple(m for m in MODELS if kind_of(m) != "model")
# Which models the live scan runs: the trade models by default (building blocks are tested in the Research lab, not
# alerted on), SCAN_MODELS="core" for the original six only, or a comma list that may name any model or block.
_SCAN = os.environ.get("SCAN_MODELS", "all").strip().lower()
LIVE_MODELS = (tuple(m for m in TRADE_MODELS if m not in PREVIEW_MODELS) if _SCAN in ("", "all") else CORE_MODELS if _SCAN == "core"
               else tuple(m for m in MODELS if m in {x.strip() for x in _SCAN.split(",")}) or CORE_MODELS)
FPFVG_WINDOWS = (Window("london_open", time(1, 30), time(2, 30)), Window("ny_open_fx", time(7, 0), time(8, 0)),
                 Window("ny_open", time(9, 30), time(11, 0)))
JUDAS_WINDOWS = (Window("london_open", time(2, 0), time(3, 0)), Window("ny_open", time(9, 30), time(10, 30)))
TIME_MODELS = ("silver_bullet", "judas_swing", "gaps", "fpfvg", "weekly", "macro", "osok", "cbdr", "stack_open")
MODEL_INFO = {
    "ict2022": {"label": "ICT 2022 Model", "entry": "limit",
                "about": "Killzone, liquidity sweep, displacement shift in structure, fair value gap; a limit order rests in the gap."},
    "silver_bullet": {"label": "Silver Bullet", "entry": "limit",
                      "about": "The same sequence only inside the 03:00, 10:00 and 14:00 NY one-hour windows; limit at the first gap."},
    "judas_swing": {"label": "Judas Swing", "entry": "limit",
                    "about": "The false move at the London or New York open runs a level against the day's bias, then reverses; the 2022 entry on the reversal."},
    "ote": {"label": "OTE", "entry": "limit",
            "about": "After the sweep and shift, a limit at the 70.5% optimal trade entry of the displacement leg (62-79% zone)."},
    "unicorn": {"label": "Unicorn", "entry": "limit",
                "about": "The 2022 sequence, only when the fair value gap overlaps the breaker block; the limit rests in the overlap."},
    "sweep": {"label": "Turtle Soup", "entry": "market",
              "about": "An old high or low is run in a killzone and a candle closes back inside it (a false breakout); enter at market on that close."},
    "institutional": {"label": "ICT Institutional", "entry": "limit", "lab": True,
                      "about": "Institutional engine: HTF bias, PD array matrix (order blocks, breakers, FVGs), OTE 70.5% and SMT divergence."},
    "ict_institutional": {"label": "ICT Institutional", "entry": "limit", "lab": True,
                          "about": "Institutional engine: HTF bias, PD array matrix (order blocks, breakers, FVGs), OTE 70.5% and SMT divergence."},
    "ifvg": {"label": "Inversion FVG", "entry": "limit", "lab": True,
             "about": "After the sweep, price displaces through a gap that formed the other way; the inverted gap becomes the entry."},
    "cisd": {"label": "CISD", "entry": "limit", "lab": True,
             "about": "After the sweep, a close back past the open of the run that made it (change in state of delivery); entry in the gap it leaves."},
    "breaker": {"label": "Breaker", "entry": "limit", "lab": True,
                "about": "The 2022 sequence with the limit at the breaker block (the order block at the broken swing), at its mean threshold."},
    "ob": {"label": "Order block", "entry": "limit", "lab": True,
           "about": "The 2022 sequence with the limit at the order block (the last opposite candle before the displacement), at its mean threshold."},
    "nonarrative": {"label": "No narrative (control)", "entry": "limit", "lab": True,
                    "about": "The ICT 2022 model without the top-down story: any sweep and shift in the window is taken whichever way the draw points, and the matrix does not gate it. It is the control that shows what the narrative adds to the model."},
    "dailybias": {"label": "Daily bias (the old read)", "entry": "limit", "lab": True,
                  "about": "The ICT 2022 recipe with the bias always from yesterday's daily candle and last week's, whatever timeframe it trades: the read the engine had before the bias followed the execution timeframe (1m from the 1-hour, 5m from the 4-hour). Run it beside ICT 2022 on the same market, timeframe and window: the difference is what reading the right rung is worth."},
    "pools": {"label": "Liquidity Pools", "entry": "limit", "lab": True,
              "about": "The 2022 model that also sweeps and targets unswept swing highs/lows and equal highs/lows, not only the prior day and sessions."},
    "quality": {"label": "Clean FVG", "entry": "limit", "lab": True,
                "about": "The 2022 model that only enters gaps of real size that price has not already traded back into."},
    "gaps": {"label": "Opening Gaps", "entry": "limit", "lab": True,
             "about": "The 2022 model that also sweeps and targets the opening gaps: new-day (17:00 to 18:00), new-week (Friday close to Sunday open) and, on indices, the 09:30 range gap. Their midpoints are draws."},
    "fpfvg": {"label": "First Presented FVG", "entry": "limit", "lab": True,
              "about": "The first fair value gap printed in the first half hour after the 01:30, 07:00 or 09:30 NY open; a limit at its 50% with the daily draw, cancelled if a body closes through the midpoint."},
    "clear": {"label": "Clear Path", "entry": "limit", "lab": True,
              "about": "The 2022 model that skips high-resistance setups (two or more unmitigated opposing gaps before the target) and seek-and-destroy conditions."},
    "dayopen": {"label": "Day Open", "entry": "limit", "lab": True,
                "about": "The 2022 model that only buys at or below the day's open and sells at or above it."},
    "weekly": {"label": "Weekly Profile", "entry": "limit", "lab": True,
               "about": "The 2022 model that sits out Monday and Friday afternoon and never trades against the weekly draw."},
    "sdt": {"label": "Measured Targets", "entry": "limit", "lab": True,
            "about": "The 2022 model that, when no liquidity gives enough reward, projects the displacement leg (0.5 to 2 times) as the target."},
    "reentry": {"label": "Re-entry", "entry": "limit", "lab": True,
                "about": "The 2022 model that, after a stop-out where price runs on past the stop, looks for a fresh shift and takes the same idea again (up to two more tries a day)."},
    "macro": {"label": "Macro Windows", "entry": "limit", "lab": True,
              "about": "The 2022 model that only takes a shift that closes inside an algorithmic macro window (10 minutes either side of the hour, NY)."},
    "osok": {"label": "One Shot One Kill", "entry": "limit", "lab": True,
             "about": "One trade a week: Monday to Wednesday, a run beyond the weekly open against the weekly draw, then the 2022 shift and entry."},
    "cbdr": {"label": "CBDR Targets", "entry": "limit", "lab": True,
             "about": "The 2022 model with the prior day's 2-8 PM NY range (CBDR) projected 1 to 3 times as targets."},
    "trail": {"label": "Trailing Runner", "entry": "limit", "lab": True,
              "about": "The 2022 model whose runner trails behind each new swing once the trade is in profit (research and backtests; the live alerts keep the plain plan)."},
    "drawtarget": {"label": "HTF Draw Target", "entry": "limit", "lab": True,
                   "about": "The 2022 model aiming at the higher-timeframe draw (prior day/week extremes, the 20/40/60-day ranges, opening gaps) instead of the nearest pool."},
    "tightstop": {"label": "Tight Stop", "entry": "limit", "lab": True,
                  "about": "The 2022 model with the stop just beyond the first candle of the entry gap instead of beyond the sweep: smaller risk, more stop-outs."},
    "stack_trend": {"label": "Stack: Trend", "entry": "limit", "lab": True,
                    "about": "Models working together for a trending day: weekly and 20-day context, the daily draw, an OTE entry, clean path, the higher-timeframe draw as the target and a trailing runner. With the trend it also takes the retracement: price drops into a higher-timeframe array on the side of the bias, runs the stops just beyond it, and shifts back with the trend."},
    "stack_reversal": {"label": "Stack: Reversal", "entry": "limit", "lab": True,
                       "about": "Models working together for a reversal: liquidity pools to run, an inversion gap entry that must be clean, a tight stop and the higher-timeframe draw as the target."},
    "stack_open": {"label": "Stack: Opening", "entry": "limit", "lab": True,
                   "about": "Models working together for the open: opening gaps, a macro-window shift, the day-open rule, clean gaps, and the higher-timeframe draw as the target."},
    "venom": {"label": "Venom", "entry": "limit", "lab": True,
              "about": "One candle runs a liquidity pool, the next displaces back; a limit at the second candle's close with a stop just past the sweep. Void if price trades back through its midpoint."},
    "atm": {"label": "ATM Method", "entry": "limit", "lab": True,
            "about": "A short-term high (low) is run and structure breaks straight away; a limit at the retest of the broken level, the stop just beyond the rejection extreme."},
    "mm": {"label": "Market Maker", "entry": "limit", "lab": True,
           "about": "An original consolidation, a stepped run away from it, then a smart money reversal that sweeps liquidity and shifts structure; a low-risk entry in the first retrace, aiming back through the consolidation."},
    "mitigation": {"label": "Mitigation Block", "entry": "limit", "lab": True,
                 "about": "A failure swing (a lower high that does not take the old high) then a break of the swing low: the last down candle at that low becomes a sell zone on the return."},
    "rejection": {"label": "Rejection Block", "entry": "limit", "lab": True,
                "about": "The body-to-wick zone of a swing high or low with long wicks: a limit at the body edge, the stop beyond the wick extreme."},
    "propulsion": {"label": "Propulsion Block", "entry": "limit", "lab": True,
                 "about": "A second candle that trades into an order block and then proves itself: a limit at its open, the stop under its mean threshold."},
    "suspension": {"label": "Suspension Block", "entry": "limit", "lab": True,
                 "about": "Three same-direction candles with a volume imbalance either side of the middle one: the retest of the far half of the block."},
    "vimb": {"label": "Volume Imbalance", "entry": "limit", "lab": True,
           "about": "Two candles whose bodies do not overlap while their wicks do: price returns to fill the close-to-open gap and reacts."},
    "vacuum": {"label": "Vacuum Block", "entry": "limit", "lab": True,
             "about": "A true gap between candles (a session open, the weekend, news): a limit at the gap edge when price comes back and bounces at once."},
    "reaper": {"label": "Reaper", "entry": "limit", "lab": True,
               "about": "An inversion gap nested with an opposing gap in the same displacement: the 2022 sequence with that pair as the entry zone."},
    "full": {"label": "Full Narrative", "entry": "limit", "lab": True,
             "about": "Every context rule together: daily and weekly draw, 20-day range, liquidity pools and clean gaps."},
}

for _k, _v in MODEL_INFO.items():
    _v["kind"] = kind_of(_k)  # model, or the kind of building block it is


# ---- build your own: custom:entry=ote;stop=zone;target=draw;filter=narrative,regime;mgmt=trail,reentry -------------
CUSTOM_ENTRY = {"fvg": {}, "ce": {"entry_mode": "ce"}, "ote": {"entry_mode": "ote"}, "ob": {"entry_mode": "ob"},
                "breaker": {"entry_mode": "breaker"}, "unicorn": {"entry_mode": "unicorn"}, "bpr": {"entry_mode": "bpr", "enable_bpr": True},
                "ifvg": {"trigger": "ifvg"}, "cisd": {"trigger": "cisd"}}
CUSTOM_STOP = {"sweep": {"stop_mode": "sweep"}, "candle1": {"stop_mode": "candle1"}, "zone": {"stop_mode": "zone"}}
CUSTOM_TARGET = {"nearest": {"target_pref": "nearest"}, "draw": {"target_pref": "draw", "htf_context": True, "opening_gaps": True},
                 "sd": {"sd_targets": True}, "cbdr": {"cbdr_targets": True, "target_pref": "draw"}}
CUSTOM_FILTER = {"narrative": {"narrative_filter": True}, "htf": {"htf_context": True}, "regime": {"regime_filter": True},
                 "quality": {"fvg_quality": True}, "pools": {"liquidity_pools": True}, "gaps": {"opening_gaps": True},
                 "dayopen": {"day_open_rule": True}, "weekly": {"weekly_profile": True}, "macro": {"macro_only": True},
                 "osok": {"osok": True}}
CUSTOM_MGMT = {"trail": {"trail_structure": True}, "reentry": {"reentry": True}, "sd": {"sd_targets": True}}
CUSTOM_PARTS = {"entry": CUSTOM_ENTRY, "stop": CUSTOM_STOP, "target": CUSTOM_TARGET, "filter": CUSTOM_FILTER, "mgmt": CUSTOM_MGMT}


def custom_changes(spec: str) -> dict:
    """The StrategyConfig changes for 'custom:entry=ote;stop=zone;...' (ValueError for anything unknown)."""
    out: dict = {}
    for part in spec.removeprefix("custom:").split(";"):
        if not part.strip():
            continue
        key, _, vals = part.partition("=")
        table = CUSTOM_PARTS.get(key.strip())
        if table is None:
            raise ValueError(f"unknown part {key!r}; use entry, stop, target, filter or mgmt")
        for v in (x.strip() for x in vals.split(",") if x.strip()):
            if v not in table:
                raise ValueError(f"unknown {key} {v!r}; choose from {', '.join(table)}")
            out.update(table[v])
    return out


def custom(strategy: StrategyConfig) -> StrategyConfig:
    return replace(strategy, **custom_changes(strategy.model))


# Names that used to be separate models and are now simply the ICT 2022 model, which always reads the narrative.
ALIASES = {"narrative": "ict2022", "htf": "ict2022"}


def is_model(name: str) -> bool:
    if name in MODELS or name in ALIASES:
        return True
    if name.startswith("custom:"):
        try:
            custom_changes(name)
            return True
        except ValueError:
            return False
    return False


def label_of(name: str) -> str:
    return MODEL_INFO[name]["label"] if name in MODEL_INFO else ("Custom: " + name[7:].replace(";", " · ")) if name.startswith("custom:") else name


def any_hour(strategy: StrategyConfig) -> bool:
    return any(w.start.hour == 0 and w.end.hour == 23 for w in strategy.killzones)


def model_available(strategy: StrategyConfig, model: str) -> bool:
    """The Silver Bullet is a time model: it does not exist where the rules run at any hour."""
    return not (model in TIME_MODELS and any_hour(strategy))


# The models run by the ICT 2022 state machine. ICT builds the story first (narrative.py): every one of them answers
# to it unless it is the "No narrative" control, and the judgement matrix rules out SKIP setups before they are placed.
SPINE_MODELS = tuple(m for m in MODELS if m not in PATTERN_MODELS and m not in INSTITUTIONAL_MODELS and m not in ("sweep", "fpfvg", "nonarrative"))


def resolve(strategy: StrategyConfig) -> StrategyConfig:
    """The concrete rules for a model: its own parts, then (for the ICT 2022 family) the narrative and the matrix gate."""
    strategy = replace(strategy, model=ALIASES.get(strategy.model, strategy.model))
    out = _resolve_parts(strategy)
    if strategy.model in SPINE_MODELS or strategy.model.startswith("custom:"):
        out = replace(out, narrative_filter=True, narrative_required=True, htf_context=True, htf_arrays=out.htf_arrays,
                      opening_gaps=True, liquidity_pools=True, fvg_quality=True, day_open_rule=out.day_open_rule,
                      weekly_profile=True, array_levels=out.array_levels, trail_structure=True,
                      target_pref="draw" if out.target_pref == "nearest" else out.target_pref,
                      min_score=out.min_score or CAUTION,
                      mss_max_bars=out.mss_max_bars,
                      max_obstacles=max(out.max_obstacles, 5))
    return out


def _resolve_parts(strategy: StrategyConfig) -> StrategyConfig:
    """The model's own rules: the Silver Bullet swaps the killzones for its windows."""
    if strategy.model == "dailybias":
        return replace(strategy, bias_ladder=False)   # the control for the top-down ladder: the daily rung for every timeframe
    if strategy.model == "nonarrative":
        # the control: no direction at all, so a run of this model measures what the story adds
        return replace(strategy, narrative_required=False, narrative_filter=False, htf_context=False, daily_bias_filter=False, regime_filter=False, min_score=0, liquidity_pools=True)
    if strategy.model in PATTERN_MODELS and KIND.get(strategy.model) == "zone" and not strategy.liquidity_pools:
        # ICT aims at old highs and lows (equal highs, swing pools), not only the named session levels: without them
        # most block setups have "no opposing liquidity" to target and are thrown away.
        strategy = replace(strategy, liquidity_pools=True)
    if strategy.model == "silver_bullet" and not any_hour(strategy):
        return replace(strategy, killzones=strategy.silver_bullet_windows)
    if strategy.model == "judas_swing" and not any_hour(strategy):
        return replace(strategy, killzones=JUDAS_WINDOWS, midnight_open_filter=True, daily_bias_filter=True)
    if strategy.model in ("ote", "unicorn", "breaker", "ob"):
        return replace(strategy, entry_mode=strategy.model)
    if strategy.model in ("ifvg", "cisd"):
        return replace(strategy, trigger=strategy.model)
    if strategy.model == "pools":
        return replace(strategy, liquidity_pools=True)
    if strategy.model == "quality":
        return replace(strategy, fvg_quality=True)
    if strategy.model == "gaps":
        return replace(strategy, opening_gaps=True)
    if strategy.model == "fpfvg":
        return replace(strategy, opening_gaps=True, killzones=FPFVG_WINDOWS if not any_hour(strategy) else strategy.killzones)
    if strategy.model == "reaper":
        return replace(strategy, trigger="ifvg", reaper=True)
    if strategy.model == "macro":
        return replace(strategy, macro_only=True)
    if strategy.model == "osok":
        return replace(strategy, osok=True, max_signals_per_day=1)
    if strategy.model == "cbdr":
        return replace(strategy, cbdr_targets=True, target_pref="draw")
    if strategy.model == "trail":
        return replace(strategy, trail_structure=True)
    if strategy.model == "drawtarget":
        return replace(strategy, target_pref="draw", htf_context=True, opening_gaps=True)
    if strategy.model == "tightstop":
        return replace(strategy, stop_mode="candle1")
    if strategy.model == "stack_trend":
        return replace(strategy, narrative_filter=True, htf_context=True, regime_filter=True, entry_mode="ote",
                       target_pref="draw", sd_targets=True, trail_structure=True, array_levels=True)
    if strategy.model == "stack_reversal":
        return replace(strategy, trigger="ifvg", liquidity_pools=True, fvg_quality=True, stop_mode="candle1",
                       target_pref="draw", htf_context=True)
    if strategy.model == "stack_open":
        return replace(strategy, opening_gaps=True, macro_only=True, day_open_rule=True, fvg_quality=True, target_pref="draw", htf_context=True)
    if strategy.model.startswith("custom:"):
        return custom(strategy)
    if strategy.model == "reentry":
        return replace(strategy, reentry=True)
    if strategy.model == "weekly":
        return replace(strategy, weekly_profile=True)
    if strategy.model == "sdt":
        return replace(strategy, sd_targets=True)
    if strategy.model == "clear":
        return replace(strategy, regime_filter=True)
    if strategy.model == "dayopen":
        return replace(strategy, day_open_rule=True)
    if strategy.model == "full":
        return replace(strategy, narrative_filter=True, htf_context=True, liquidity_pools=True, fvg_quality=True, opening_gaps=True,
                       regime_filter=True, day_open_rule=True, weekly_profile=True, sd_targets=True, reentry=True)
    return strategy


def make_strategy(strategy: StrategyConfig, instrument: InstrumentConfig):
    strategy = replace(strategy, model=ALIASES.get(strategy.model, strategy.model))
    if strategy.model == "fpfvg":
        return FirstPresentedStrategy(resolve(strategy), instrument)
    if strategy.model in PATTERN_MODELS:
        return PATTERN_MODELS[strategy.model](resolve(strategy), instrument)
    if strategy.model in INSTITUTIONAL_MODELS:
        return ICTInstitutionalStrategy(strategy, instrument)
    if strategy.model.startswith("custom:") or strategy.model in ("ict2022", "silver_bullet", "judas_swing", "ote", "unicorn") + LAB_MODELS:
        return ICT2022Strategy(resolve(strategy), instrument)
    if strategy.model == "sweep":
        return TimeAndPriceStrategy(strategy, instrument)
    raise ValueError(f"unknown model {strategy.model!r}; use one of {MODELS}")
