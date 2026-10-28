"""The Comprehensive ICT Institutional Strategy Engine.

Integrates the complete institutional trading model synthesized from the 903 ICT video
corpus:
  1. Time & Institutional Sessions:
     - Killzones (London Open, NY AM, NY PM)
     - 20-minute algorithmic Macro Windows (08:50-09:10, 09:50-10:10, 10:50-11:10, 13:50-14:10)
     - Power of Three (AMD: Accumulation -> Manipulation -> Distribution)
  2. HTF Liquidity Sweeps:
     - Previous Day High / Low (PDH/PDL), Asian Session High/Low, Prior Week High/Low
  3. Market Structure Shift (MSS) with Real Displacement:
     - Energetic candle closes beyond opposite 3-candle swing point with body expansion
  4. Institutional PD Array Confluence:
     - Order Blocks with 50% Mean Threshold (MT)
     - Breaker Blocks (structural inversion of failed highs/lows)
     - Fair Value Gaps with 50% Consequent Encroachment (CE)
     - Inversion FVGs (IFVGs)
  5. Optimal Trade Entry (OTE) & Standard Deviation (SD) Projections:
     - Fibonacci 61.8%, 70.5% (Sweet Spot), 79.0% retracement
     - -1.0 SD, -1.5 SD, -2.0 SD algorithmic delivery expansion targets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median
from typing import Any

from .config import InstrumentConfig, StrategyConfig
from .ote import DealingRange, find_dealing_range
from .pd_arrays import PDArray, scan_pd_arrays
from .sessions import Bar, Level, LevelTracker, in_any
from .sessions_advanced import get_amd_phase, in_macro
from .strategy import LONG, SHORT, Decision, Signal


@dataclass
class _InstitutionalSetup:
    level: Level
    direction: int
    extreme: float
    sweep_index: int
    bars: int = 0
    stage: str = "await_mss"  # await_mss -> limit
    limit: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    tp1: float | None = None
    target_name: str = ""
    fvg: tuple[float, float] = (0.0, 0.0)
    mss_time: datetime | None = None
    pd_array: PDArray | None = None
    dealing_range: DealingRange | None = None
    window: str = ""
    macro_name: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass
class ICTInstitutionalStrategy:
    config: StrategyConfig
    instrument: InstrumentConfig
    decisions: list[Decision] = field(default_factory=list)
    staged: list[dict] = field(default_factory=list)  # pending limit orders, for display
    _tracker: LevelTracker = field(init=False)
    _bars: list[Bar] = field(default_factory=list)
    _spent: set[str] = field(default_factory=set)
    _setups: list[_InstitutionalSetup] = field(default_factory=list)
    _swept_sides: list[str] = field(default_factory=list)
    _signals_today: int = 0

    def __post_init__(self) -> None:
        self._tracker = LevelTracker(self.config)

    # ---------------------------------------------------------------- helpers
    def _log(self, ts: datetime, rule: str, msg: str) -> None:
        self.decisions.append(Decision(ts, rule, msg))

    def _fmt(self, p: float) -> str:
        decimals = max(0, len(f"{self.instrument.pip_size:.10f}".rstrip("0").split(".")[1]) + 1)
        return f"{p:.{decimals}f}"

    def _median_body(self) -> float:
        bodies = [abs(b.close - b.open) for b in self._bars[-21:-1]]
        return median(bodies) if bodies else 0.0

    def _last_swing(self, direction: int, before: int, after: int) -> float | None:
        """Most recent 3-candle swing low (for shorts) / high (for longs) in (after, before)."""
        b = self._bars
        for i in range(min(before, len(b) - 2), max(after, 1) - 1, -1):
            if i - 1 < 0 or i + 1 >= len(b):
                continue
            if direction == SHORT and b[i].low < b[i - 1].low and b[i].low < b[i + 1].low:
                return b[i].low
            if direction == LONG and b[i].high > b[i - 1].high and b[i].high > b[i + 1].high:
                return b[i].high
        return None

    def _find_confluent_pd_array(self, direction: int, start: int, end: int) -> PDArray | None:
        """Finds the best institutional PD Array formed in or around the displacement leg."""
        recent_bars = self._bars[:end + 1]
        arrays = scan_pd_arrays(recent_bars, lookback=max(20, end - start + 10))
        # Filter for matching direction and unmitigated
        matching = [a for a in arrays if a.direction == direction and a.bar_index >= max(0, start - 2)]
        if not matching:
            return None
        # Priority: Breaker > Order Block > Inversion FVG > FVG
        priority = {"breaker": 4, "order_block": 3, "ifvg": 2, "fvg": 1, "volume_imbalance": 0, "rejection_block": 0}
        matching.sort(key=lambda a: (priority.get(a.kind, 0), a.time), reverse=True)
        return matching[0]

    def _target(self, direction: int, entry: float) -> Level | None:
        side = "low" if direction == SHORT else "high"
        opp = [lv for lv in self._tracker.levels() if lv.side == side and (lv.price - entry) * direction > 0]
        return min(opp, key=lambda lv: abs(lv.price - entry)) if opp else None

    # ---------------------------------------------------------------- main event loop
    def on_bar(self, bar: Bar) -> Signal | None:
        cfg, pip = self.config, self.instrument.pip_size
        close_time = bar.time + timedelta(minutes=cfg.timeframe_minutes)
        self._bars.append(bar)
        idx = len(self._bars) - 1

        if self._tracker.update(bar):
            for s in self._setups:
                if s.stage == "limit":
                    self._log(close_time, "EN-04", f"{s.level.name} limit cancelled: new trading day")
            self._setups.clear()
            self._spent.clear()
            self._swept_sides.clear()
            self._signals_today = 0

        levels = [lv for lv in self._tracker.levels() if lv.name not in self._spent]
        win = in_any(cfg.killzones, bar.time)
        macro = in_macro(bar.time)
        amd_phase = get_amd_phase(bar.time)

        fired: Signal | None = None
        keep: list[_InstitutionalSetup] = []

        for s in self._setups:
            s.bars += 1
            d = s.direction

            if s.stage == "await_mss":
                if win is None:
                    self._log(close_time, "KZ-1", f"{s.level.name} setup cancelled: window closed before MSS")
                    continue

                s.extreme = max(s.extreme, bar.high) if d == SHORT else min(s.extreme, bar.low)
                if s.bars > cfg.mss_max_bars:
                    self._log(close_time, "MS-02", f"{s.level.name} setup cancelled: no MSS within {cfg.mss_max_bars} bars")
                    continue

                # MSS Check: candle body closes beyond opposite swing
                swing = self._last_swing(d, before=idx, after=s.sweep_index)
                if swing is None:
                    keep.append(s)
                    continue

                closed_past = (bar.close < swing) if d == SHORT else (bar.close > swing)
                if not closed_past:
                    keep.append(s)
                    continue

                # Displacement Check: energetic candle body
                med_body = self._median_body()
                is_disp = abs(bar.close - bar.open) >= (med_body * cfg.displacement_body_mult) if med_body > 0 else True
                if not is_disp:
                    self._log(close_time, "MS-01", f"{s.level.name} break ignored: body below displacement threshold")
                    keep.append(s)
                    continue

                # PD Array & OTE Detection
                pd_arr = self._find_confluent_pd_array(d, s.sweep_index, idx)
                dr = find_dealing_range(
                    swing_low=s.extreme if d == LONG else bar.low,
                    swing_high=bar.high if d == LONG else s.extreme,
                    direction=d,
                )

                # Determine limit entry price
                if pd_arr:
                    # Near edge of PD Array or 50% CE/MT
                    limit = pd_arr.midpoint if cfg.entry_mode == "ce" else (pd_arr.top if d == SHORT else pd_arr.bottom)
                elif dr and dr.is_in_ote(bar.close):
                    limit = dr.ote_sweet_spot
                else:
                    # Fallback to leg midpoint / equilibrium
                    limit = dr.equilibrium if dr else (bar.close)

                # Stop loss beyond sweep extreme + buffer
                buf = cfg.stop_buffer_pips * pip
                stop = s.extreme + buf if d == SHORT else s.extreme - buf
                dist = abs(limit - stop)

                if dist < cfg.min_stop_pips * pip or dist > cfg.max_stop_pips * pip:
                    self._log(close_time, "SL-03", f"{s.level.name} setup cancelled: stop distance {dist/pip:.1f} pips outside allowed bounds")
                    continue

                # Target determination: Opposing Liquidity or Standard Deviation
                opp_target = self._target(d, limit)
                sd_target = dr.standard_deviation_target(2.0) if dr else 0.0

                if opp_target:
                    target = opp_target.price
                    target_name = opp_target.name
                elif sd_target > 0:
                    target = sd_target
                    target_name = "-2.0 SD Target"
                else:
                    target = limit + d * dist * cfg.min_rr
                    target_name = f"Fixed {cfg.min_rr:.1f}R"

                rr = abs(target - limit) / dist if dist > 0 else 0.0
                if rr < cfg.min_rr:
                    self._log(close_time, "TP-02", f"{s.level.name} setup cancelled: planned {rr:.2f}R < min {cfg.min_rr}R")
                    continue

                # Transition to staged limit order
                s.stage = "limit"
                s.bars = 0
                s.limit = limit
                s.stop = stop
                s.target = target
                s.target_name = target_name
                s.pd_array = pd_arr
                s.dealing_range = dr
                s.macro_name = macro.name if macro else ""

                reasons = [
                    f"KZ-1 inside {s.window} killzone",
                    f"TP-3 swept {s.level.name} to {self._fmt(s.extreme)}",
                    f"MS-01 confirmed MSS displacement through {self._fmt(swing)}",
                ]
                if pd_arr:
                    reasons.append(f"PD-01 institutional {pd_arr.kind.upper()} ({self._fmt(pd_arr.bottom)} - {self._fmt(pd_arr.top)})")
                if dr:
                    reasons.append(f"OTE-01 dealing range OTE sweet spot at {self._fmt(dr.ote_sweet_spot)}")
                if macro:
                    reasons.append(f"MC-01 algorithmic macro window: {macro.name}")
                reasons.append(f"PO3-01 Power of 3 phase: {amd_phase.upper()}")
                reasons.append(f"RK-1 stop {self._fmt(stop)} beyond sweep extreme")
                reasons.append(f"TG-1 target {target_name} at {self._fmt(target)} ({rr:.2f}R)")

                s.reasons = reasons
                self._log(close_time, "EN-01", f"{s.level.name} staged {s.target_name} limit at {self._fmt(limit)} (SL {self._fmt(stop)}, TP {self._fmt(target)})")
                keep.append(s)

            elif s.stage == "limit":
                # Check cancellation: stop or target reached before entry fill
                hit_target = (bar.high >= s.target) if d == LONG else (bar.low <= s.target)
                if hit_target:
                    self._log(close_time, "EN-03", f"{s.level.name} limit cancelled: target reached before fill")
                    continue

                hit_stop = (bar.low <= s.stop) if d == LONG else (bar.high >= s.stop)
                if hit_stop:
                    self._log(close_time, "EN-03", f"{s.level.name} limit cancelled: stop price hit before fill")
                    continue

                if s.bars > cfg.entry_expiry_bars:
                    self._log(close_time, "EN-04", f"{s.level.name} limit expired after {cfg.entry_expiry_bars} bars without fill")
                    continue

                # Check fill: price traded through the limit level
                filled = (bar.low <= s.limit <= bar.high)
                if filled and fired is None and self._signals_today < cfg.max_signals_per_day:
                    dist = abs(s.limit - s.stop)
                    rr = abs(s.target - s.limit) / dist if dist > 0 else 0.0

                    fired = Signal(
                        time=close_time,
                        direction=d,
                        entry_ref=s.limit,
                        stop=s.stop,
                        target=s.target,
                        rr=round(rr, 2),
                        level=s.level,
                        killzone=s.window,
                        reasons=tuple(s.reasons),
                        entry_type="limit",
                    )
                    self._spent.add(s.level.name)
                    self._signals_today += 1
                    self._log(close_time, "EN-02", f"{s.level.name} limit filled at {self._fmt(s.limit)}")
                    continue

                keep.append(s)

        self._setups = keep

        # Update staged limit orders for visual display
        self.staged = [
            {
                "level": s.level.name,
                "direction": "long" if s.direction == LONG else "short",
                "limit": s.limit,
                "stop": s.stop,
                "target": s.target,
                "window": s.window,
                "bars_waiting": s.bars,
                "bars_left": max(0, cfg.entry_expiry_bars - s.bars),
                "fvg": (s.pd_array.bottom, s.pd_array.top) if s.pd_array else (s.limit, s.limit),
                "pd_array": s.pd_array.kind if s.pd_array else "ote",
                "macro": s.macro_name,
            }
            for s in self._setups if s.stage == "limit"
        ]

        if fired is not None or win is None:
            return fired

        # Search for new sweeps of reference levels
        sweep_buf = cfg.min_sweep_pips * pip
        for lv in levels:
            if lv.side == "high" and bar.high > lv.price + sweep_buf:
                if self._can_sweep(lv, "high", bar):
                    self._swept_sides.append("high")
                    self._setups.append(_InstitutionalSetup(
                        level=lv,
                        direction=SHORT,
                        extreme=bar.high,
                        sweep_index=idx,
                        window=win.name,
                    ))
                    self._log(close_time, "TP-3", f"swept {lv.name} ({self._fmt(lv.price)}) to {self._fmt(bar.high)}")

            elif lv.side == "low" and bar.low < lv.price - sweep_buf:
                if self._can_sweep(lv, "low", bar):
                    self._swept_sides.append("low")
                    self._setups.append(_InstitutionalSetup(
                        level=lv,
                        direction=LONG,
                        extreme=bar.low,
                        sweep_index=idx,
                        window=win.name,
                    ))
                    self._log(close_time, "TP-3", f"swept {lv.name} ({self._fmt(lv.price)}) to {self._fmt(bar.low)}")

        return fired

    def _can_sweep(self, lv: Level, side: str, bar: Bar) -> bool:
        if any(s.level.name == lv.name for s in self._setups):
            return False
        # Sweep order protection
        if not self.config.enforce_liquidity_purge_order:
            return True
        if self._swept_sides and self._swept_sides[-1] == side:
            return False
        return True
