"""Configuration for the Time-and-Price system.

Every tunable rule lives here so the trader can confirm each value against
their own methodology (see RULES.md). All session times are New York local
time, the ICT convention; bar timestamps everywhere else are UTC.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from datetime import time
from pathlib import Path
from typing import Any


def parse_hhmm(value: str | time) -> time:
    if isinstance(value, time):
        return value
    hh, mm = value.strip().split(":")
    return time(int(hh), int(mm))


@dataclass(frozen=True)
class Window:
    """A New York time-of-day window. `end` is exclusive; may wrap midnight."""

    name: str
    start: time
    end: time

    def contains(self, t: time) -> bool:
        if self.start <= self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str = "EURUSD"
    pip_size: float = 0.0001
    # Typical spread charged on entry/exit (prices in data are assumed to be bid).
    spread_pips: float = 0.8
    # Units per 1.0 lot, used for commission and MT5 lot conversion.
    contract_size: float = 100_000
    # Round-turn commission per 1.0 lot, in account currency.
    commission_per_lot: float = 7.0
    # Account-currency value of 1.0 in the quote currency (1.0 for EURUSD on a
    # USD account). Live trading uses the broker's tick value instead.
    quote_to_account_rate: float = 1.0
    volume_step: float = 0.01
    min_lots: float = 0.01
    # Currencies whose high-impact news blocks this symbol. Empty = derive
    # from a 6-letter FX symbol (EURUSD -> EUR, USD).
    news_currencies: tuple[str, ...] = ()
    asset_class: str = "forex"  # "forex", "commodity", "index"
    is_cent_account: bool = False  # USD Cent (USC) account where $10 = 1,000 cents

    def pips(self, n: float) -> float:
        return n * self.pip_size


@dataclass(frozen=True)
class StrategyConfig:
    timeframe_minutes: int = 5
    # Trading-day rollover in NY time; 17:00 matches the forex daily candle.
    day_rollover: time = time(17, 0)
    # Only these windows may produce entries (rule KZ-1).
    killzones: tuple[Window, ...] = (
        Window("london", time(2, 0), time(5, 0)),
        Window("new_york", time(7, 0), time(10, 0)),
    )
    # Ranges whose high/low become reference levels once the window closes (rule TP-1).
    reference_ranges: tuple[Window, ...] = (Window("asia", time(20, 0), time(0, 0)),)
    # Use prior trading day's high/low as reference levels (rule TP-1).
    use_prior_day_levels: bool = True
    # Period for the "prior" high/low levels: "day" (PDH/PDL) for scalp, intraday and swing;
    # "week" (PWH/PWL, RL-04) for long-term trading on 4-hour candles.
    level_period: str = "day"
    # A level already traded through outside a killzone today is spent (rule TP-2).
    invalidate_levels_taken_outside_killzone: bool = True
    # Price must exceed the level by at least this much to count as a sweep (TP-3).
    min_sweep_pips: float = 0.5
    # Bars allowed after the sweep for a close back inside the level (TP-4).
    max_confirm_bars: int = 3
    # Stop goes beyond the sweep extreme by this buffer (RK-1).
    stop_buffer_pips: float = 1.0
    min_stop_pips: float = 3.0
    max_stop_pips: float = 25.0
    # "rr" = fixed reward:risk; "opposite_level" = nearest opposing liquidity (TG-1).
    target_mode: str = "opposite_level"
    target_rr: float = 2.0
    # Minimum reward:risk; a setup with less is skipped (TG-2).
    min_rr: float = 1.5
    max_signals_per_day: int = 1
    # Flatten any open trade at this NY time (EX-1). None disables.
    flatten_time: time | None = time(16, 0)

    # Which entry model drives signals: "ict2022" = sweep -> displacement MSS -> FVG limit
    # entry (ict_trader/ict2022.py); "sweep" = the original sweep-and-reclaim model.
    model: str = "sweep"
    mss_max_bars: int = 12            # MS-02: candles allowed after the sweep for the shift
    swing_lookback: int = 20          # MS-01: how far back to look for the swing to break
    displacement_body_mult: float = 1.5  # MS-02: displacement body vs recent median body
    entry_mode: str = "edge"          # EN-01 "edge" (near boundary) or EN-02 "ce" (50% of gap);
                                      # models set "ote", "unicorn", "breaker" or "ob"
    # What confirms the reversal after the sweep: "mss" (MS-02, a displacement close through the
    # last swing), "cisd" (a close back past the open of the run that made the sweep: change in
    # state of delivery) or "ifvg" (a displacement close through a gap formed the other way,
    # which then becomes the entry: inversion fair value gap).
    trigger: str = "mss"
    # Narrative: trade only toward the daily draw (the bias must point the trade's way) and
    # only sell in premium / buy in discount of yesterday's range. Price-only, so it also
    # applies where the time rules don't (synthetics, 1h and up).
    narrative_filter: bool = False
    # The narrative (narrative.py), ICT's top-down story: no sweep is taken seriously unless the daily draw (vetoed by the
    # weekly draw) points the trade's way, the draw on liquidity is still ahead, and the day is not seek-and-destroy.
    # The models set it; leave it off to see what the story adds (the "No narrative" control in the Research lab).
    narrative_required: bool = False
    # The judgement matrix as a gate: a setup whose adviser score is under this is not taken (0 = off; 40 is the
    # CAUTION line, so only SKIP setups are dropped).
    min_score: int = 40
    # Higher-timeframe context, read from the same candles: the weekly bias must not oppose the
    # trade, the entry sits in discount (buys) / premium (sells) of the last `ipda_days` days'
    # range (the IPDA look-back), and the weekly and IPDA extremes count as targets.
    htf_context: bool = False
    # Top-down: the entry zone must sit at an unmitigated 1h / 4h / daily / weekly array (gap, order block, breaker, void, rejection block) of
    # the right direction (htf.py), unless the raid was on a higher-timeframe pool (prior day / week, the IPDA ranges). Only the timeframes
    # above the execution one count for 1h / 4h; the daily and weekly come from the completed days.
    htf_arrays: bool = False
    # The first partial comes off at the near edge of the first opposing PD array on the road to the target, when that is sooner than the leg's extreme.
    tp1_before_wall: bool = True
    # Continuation: a retracement into a higher-timeframe array on the side of the bias is the raid (ict2022._array_levels); needs htf_arrays.
    array_levels: bool = False
    # The bias comes from the rung above the execution timeframe (htf.ladder): 1m -> 1h, 5m -> 4h, 15m and up -> daily. False: always the daily.
    bias_ladder: bool = True
    htf_array_tfs: tuple[int, ...] = (60, 240)
    ipda_days: int = 20
    # The three look-back ranges ICT calls the interbank price delivery data ranges: the highs and lows of the last
    # 20, 40 and 60 days all count as levels and targets (the 20-day one also sets premium / discount).
    ipda_lookbacks: tuple[int, ...] = (20, 40, 60)
    # More liquidity: unswept swing highs/lows, and equal highs/lows (within `pool_equal_range`
    # of a median candle range), are pools price can sweep and run to.
    liquidity_pools: bool = False
    pool_equal_range: float = 0.1
    # Fair value gap quality: at least `fvg_min_range` of a median candle range tall, and not
    # yet traded back to its midpoint (unmitigated) when the order is placed.
    fvg_quality: bool = False
    fvg_min_range: float = 0.25
    # Opening gaps (time markets only): the New Day Opening Gap (17:00 close to 18:00 open), the
    # New Week Opening Gap (Friday close to the Sunday open) and, on indices, the RTH Opening Range
    # Gap (prior 16:14 settlement to the 09:30 open). Their edges are levels to sweep, and their
    # midpoints (consequent encroachment) and edges are draws to target.
    opening_gaps: bool = False
    # Market conditions (low vs high resistance, seek and destroy): skip a setup when `max_obstacles`
    # or more unmitigated opposing gaps sit between the entry and its target, or when price has
    # been stopping out both sides of a recent range (alternating failed sweeps).
    regime_filter: bool = True
    max_obstacles: int = 2
    # Day-open rule: buy at or below the day's opening price, sell at or above it.
    day_open_rule: bool = False
    # Re-entry: when a trade is stopped out and price keeps going past its stop, treat that as a deeper run on the same
    # liquidity and look for a fresh shift in the same direction (up to `max_reentries` more tries a day).
    reentry: bool = False
    max_reentries: int = 2
    # ---- the parts a trade is built from (each a switch, so models can be combined) ----
    # Stop model: "sweep" (beyond the sweep extreme, the default), "candle1" (beyond the first candle of the
    # entry gap, tighter) or "zone" (beyond the far side of the entry zone: order block, breaker, gap).
    stop_mode: str = "sweep"
    # Target model: "nearest" (nearest opposing liquidity, the default) or "draw" (the higher-timeframe draw:
    # prior day / week extremes, the 20/40/60-day ranges, opening gaps, when one gives `min_rr`; else the nearest).
    target_pref: str = "nearest"
    # Management model: trail the runner behind each new swing once it is in profit (the backtester's exit).
    trail_structure: bool = False
    # Time model: only trade a displacement that closes inside an algorithmic macro window (xx:50 to xx:10 NY).
    macro_only: bool = False
    # One Shot One Kill: one trade a week, Monday to Wednesday, a sweep beyond the weekly open against the weekly draw.
    osok: bool = False
    # CBDR (2 to 8 PM NY of the day before): its width projected 1 to 3 times beyond its body high and low are targets.
    cbdr_targets: bool = False
    # Reaper: the inversion gap must be nested with an opposing-polarity gap in the same displacement.
    reaper: bool = False
    # IV-02 Balanced Price Range (BPR): overlap of an opposing FVG and the displacement FVG.
    enable_bpr: bool = False
    # Weekly profile (clock markets): sit out Monday and Friday afternoon, and never trade against the weekly draw.
    weekly_profile: bool = False
    # Measured-move targets: when no opposing liquidity gives `min_rr`, project the displacement leg
    # (0.5, 1, 1.5, 2 times its size) beyond its extreme and use the nearest projection that does.
    sd_targets: bool = False
    opening_gap_min_range: float = 0.25   # a gap smaller than this share of a median candle range is ignored
    # OB-01: when the displacement leaves no fair value gap, rest the limit at the order
    # block's mean threshold (50% of the last opposite candle before the shift) instead.
    ob_entry_fallback: bool = True
    entry_expiry_bars: int = 3        # EN-04: candles the limit order may rest unfilled (3 on 5m)
    entry_expiry_minutes: int = 15    # EN-04: ...and at least this long on 1-3 minute charts
    # KZ-01: the sweep must form above the NY midnight (00:00) open for shorts, below it for longs.
    midnight_open_filter: bool = True
    # RL-05: trade only toward the daily draw read from the last two completed days
    # (a close beyond the prior day's range, or a run on it that closed back inside).
    daily_bias_filter: bool = True
    # The daily draw must back the trade and the weekly draw must not oppose it: no clear draw is no trade. For the models that have no story of
    # their own (Turtle Soup); the ICT 2022 family reads the whole story (narrative_required) and the pattern models ask it too.
    bias_required: bool = False
    # TP-04: take this fraction off at TP1 (the displacement leg's extreme) and move the
    # rest to breakeven. 0 disables partials.
    partial_at_tp1: float = 0.5

    # Fair Value Gap and Displacement filters (FV-01, FV-02)
    require_fvg: bool = False
    fvg_in_premium_discount: bool = False
    # Order of Liquidity Purging filter (SW-02)
    enforce_liquidity_purge_order: bool = False

    # Silver Bullet 60-minute execution windows (SB-01)
    enable_silver_bullet: bool = False
    silver_bullet_windows: tuple[Window, ...] = (
        Window("london_sb", time(3, 0), time(4, 0)),
        Window("ny_am_sb", time(10, 0), time(11, 0)),
        Window("ny_pm_sb", time(14, 0), time(15, 0)),
    )

    # Algorithmic Time Macros (MC-01)
    enable_macros: bool = False
    macro_windows: tuple[Window, ...] = (
        Window("macro_pre_open", time(8, 50), time(9, 10)),
        Window("macro_am_spool", time(9, 50), time(10, 10)),
        Window("macro_morning_finale", time(10, 50), time(11, 10)),
        Window("macro_pm_spool", time(13, 50), time(14, 10)),
        Window("macro_moc_spool", time(14, 50), time(15, 10)),
    )

    # Stop loss trimming milestones (SL-04, SL-05)
    trim_stop_at_50_pct: bool = False
    move_be_at_75_pct: bool = False
    move_be_at_1r: bool = False   # lab only: a stop at breakeven after +1R scratches most winners on a noisy chart; ICT moves it after the first partial or at 75% of the way

    # Small account asymmetric R:R filter (SA-03)
    small_account_mode: bool = False
    small_account_min_rr: float = 3.0


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_weekly_loss_pct: float = 4.0
    max_consecutive_losses: int = 2  # RK-05
    max_trades_per_day: int = 2
    max_open_positions: int = 1
    max_lots: float = 5.0

    # Dynamic risk reduction on consecutive losses (RK-02, RK-03)
    enable_dynamic_risk_reduction: bool = False
    loss_recovery_threshold_pct: float = 50.0

    # Small account compounding mode (SA-02, SA-04)
    small_account_mode: bool = False
    fixed_risk_dollars: float | None = None
    compounding_milestones: tuple[tuple[float, float], ...] = (
        (10.0, 0.01),
        (25.0, 0.01),
        (50.0, 0.02),
        (100.0, 0.03),
        (200.0, 0.05),
    )


@dataclass(frozen=True)
class NewsConfig:
    enabled: bool = True
    impacts: tuple[str, ...] = ("high",)
    minutes_before: int = 30
    minutes_after: int = 30
    # Close an open trade this many minutes before a blocking event. 0 disables.
    flatten_minutes_before: int = 5
    # No-trade days (from the calendar): NF-03 no afternoon trading on FOMC days,
    # NF-04 stop at Wednesday 12:00 NY in NFP week (no Thursday/Friday), NF-05 bank
    # holidays and the Friday afternoon before a holiday Monday.
    fomc_blackout: bool = True
    nfp_week_cutoff: bool = True
    holiday_blackout: bool = True


@dataclass(frozen=True)
class LiveConfig:
    # "mt5" (official Python API) or "mt4" (file bridge to the ICTBridge EA).
    platform: str = "mt5"
    # MT4 only: the terminal's shared Common\Files folder, where ICTBridge reads/writes.
    bridge_dir: str = ""
    # MT4 only: seconds to wait for the EA to confirm an order.
    bridge_timeout_seconds: float = 15.0
    # Timezone of MetaTrader server timestamps: an IANA name, or "NY+7" for the common
    # broker convention of New York time + 7h (UTC+2 winter / UTC+3 summer).
    server_timezone: str = "NY+7"
    poll_seconds: float = 5.0
    magic: int = 20260901
    # Slippage tolerance for market orders, in points.
    deviation_points: int = 10
    # Ignore a signal whose bar closed longer ago than this (e.g. after a reconnect).
    max_signal_age_seconds: int = 120
    # Create this file to flatten everything and stop the live runner.
    kill_switch_file: str = "STOP"
    journal_path: str = "journal.sqlite"


@dataclass(frozen=True)
class Config:
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    initial_equity: float = 10_000.0


def _windows(items: list[dict[str, str]]) -> tuple[Window, ...]:
    return tuple(Window(i["name"], parse_hhmm(i["start"]), parse_hhmm(i["end"])) for i in items)


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in ("killzones", "reference_ranges", "silver_bullet_windows", "macro_windows"):
            value = _windows(value)
        elif key in ("day_rollover",):
            value = parse_hhmm(value)
        elif key == "flatten_time":
            value = parse_hhmm(value) if value else None
        elif key == "compounding_milestones" and isinstance(value, list):
            value = tuple(tuple(float(x) for x in item) for item in value)
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return Config(
        instrument=_build(InstrumentConfig, raw.get("instrument", {})),
        strategy=_build(StrategyConfig, raw.get("strategy", {})),
        risk=_build(RiskConfig, raw.get("risk", {})),
        news=_build(NewsConfig, raw.get("news", {})),
        live=_build(LiveConfig, raw.get("live", {})),
        initial_equity=float(raw.get("initial_equity", 10_000.0)),
    )


def with_strategy(config: Config, **overrides: Any) -> Config:
    return replace(config, strategy=replace(config.strategy, **overrides))
