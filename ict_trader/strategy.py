"""ICT Time-and-Price: liquidity sweep and reversal inside a killzone.

The strategy is a small deterministic state machine fed one closed bar at a
time. The backtester and the live runner both drive it via on_bar(), so the
code that is validated is exactly the code that trades.

Rule IDs (documented in RULES.md) appear in every decision so each signal can
be traced to the rule that produced or blocked it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import InstrumentConfig, StrategyConfig
from .sessions import Bar, Level, LevelTracker, in_any, to_ny, trading_day

LONG, SHORT = 1, -1


@dataclass(frozen=True)
class Signal:
    time: datetime  # close time of the confirming bar; earliest possible entry
    direction: int
    entry_ref: float
    stop: float
    target: float
    rr: float
    level: Level
    killzone: str
    reasons: tuple[str, ...]
    # "market": enter at the next candle's open. "limit": the order filled at entry_ref
    # during the signal candle (ICT 2022 model: limit order resting in the fair value gap).
    entry_type: str = "market"
    # Chart geometry of the setup (sweep, shift, gap, midnight open), for the trade review.
    meta: dict = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class Decision:
    time: datetime
    rule: str
    message: str


@dataclass
class _Sweep:
    level: Level
    direction: int  # direction of the trade the sweep would set up
    extreme: float
    bars_waited: int = 0


@dataclass
class TimeAndPriceStrategy:
    config: StrategyConfig
    instrument: InstrumentConfig
    decisions: list[Decision] = field(default_factory=list)
    _tracker: LevelTracker = field(init=False)
    _spent: set[str] = field(default_factory=set)
    _pending: list[_Sweep] = field(default_factory=list)
    _signals_today: int = 0
    _swept_sides: list[str] = field(default_factory=list)
    _bars: list[Bar] = field(default_factory=list)
    _days: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._tracker = LevelTracker(self.config)

    def seed_days(self, daily: list[Bar], before=None) -> None:
        for b in sorted(daily, key=lambda x: x.time):
            if before is not None and b.time.date() >= before:
                continue
            self._days.setdefault(b.time.date(), [b.open, b.high, b.low, b.close])

    def daily_bias(self) -> tuple[int, str]:
        days = sorted(self._days)[:-1]
        if len(days) < 2:
            return 0, "not enough completed days"
        _, h2, l2, _ = self._days[days[-2]]
        _, h1, l1, c1 = self._days[days[-1]]
        if c1 > h2:
            return LONG, f"yesterday closed {self._fmt(c1)} above the prior day's high {self._fmt(h2)}"
        if c1 < l2:
            return SHORT, f"yesterday closed {self._fmt(c1)} below the prior day's low {self._fmt(l2)}"
        if h1 > h2 and l1 >= l2:
            return SHORT, f"yesterday ran the prior day's high {self._fmt(h2)} and closed back inside"
        if l1 < l2 and h1 <= h2:
            return LONG, f"yesterday ran the prior day's low {self._fmt(l2)} and closed back inside"
        return 0, "inside or two-sided day"

    def weekly_bias(self) -> tuple[int, str]:
        days = sorted(self._days)[:-1]
        weeks: dict = {}
        for d in days:
            weeks.setdefault(d - timedelta(days=d.weekday()), []).append(d)
        keys = sorted(weeks)
        if self._days and keys and sorted(self._days)[-1] - timedelta(days=sorted(self._days)[-1].weekday()) == keys[-1]:
            keys = keys[:-1]
        if len(keys) < 2:
            return 0, "not enough completed weeks"
        agg = lambda k: (max(self._days[d][1] for d in weeks[k]), min(self._days[d][2] for d in weeks[k]), self._days[weeks[k][-1]][3])
        h2, l2, _ = agg(keys[-2])
        h1, l1, c1 = agg(keys[-1])
        if c1 > h2:
            return LONG, "last week closed above the prior week's high"
        if c1 < l2:
            return SHORT, "last week closed below the prior week's low"
        if h1 > h2 and l1 >= l2:
            return SHORT, "last week ran the prior week's high and closed back inside"
        if l1 < l2 and h1 <= h2:
            return LONG, "last week ran the prior week's low and closed back inside"
        return 0, "inside or two-sided week"

    def _log(self, ts: datetime, rule: str, msg: str) -> None:
        self.decisions.append(Decision(ts, rule, msg))

    def _fmt(self, price: float) -> str:
        decimals = max(0, len(f"{self.instrument.pip_size:.10f}".rstrip("0").split(".")[1]) + 1)
        return f"{price:.{decimals}f}"

    def _find_recent_fvg(self, direction: int, bars_back: int) -> tuple[float, float, float] | None:
        """Find the latest 3-candle Fair Value Gap in the displacement sequence (FV-01).

        Returns (fvg_bottom, fvg_top, fvg_ce) or None.
        """
        if len(self._bars) < 3:
            return None
        search_start = len(self._bars) - 1
        search_end = max(1, search_start - bars_back)
        for i in range(search_start, search_end, -1):
            b0 = self._bars[i - 2]
            b2 = self._bars[i]
            if direction == SHORT:
                # Bearish FVG: Candle 1 low > Candle 3 high
                if b0.low > b2.high:
                    bottom = b2.high
                    top = b0.low
                    ce = (top + bottom) / 2.0
                    return (bottom, top, ce)
            else:
                # Bullish FVG: Candle 1 high < Candle 3 low
                if b0.high < b2.low:
                    bottom = b0.high
                    top = b2.low
                    ce = (top + bottom) / 2.0
                    return (bottom, top, ce)
        return None

    def on_bar(self, bar: Bar) -> Signal | None:
        from .sessions import trading_day
        cfg, pip = self.config, self.instrument.pip_size
        close_time = bar.time + timedelta(minutes=cfg.timeframe_minutes)
        self._bars.append(bar)
        if len(self._bars) > 60:
            self._bars = self._bars[-60:]
        tday = trading_day(bar.time, cfg.day_rollover)
        d_ = self._days.get(tday)
        self._days[tday] = [bar.open, bar.high, bar.low, bar.close] if d_ is None else [d_[0], max(d_[1], bar.high), min(d_[2], bar.low), bar.close]
        if len(self._days) > 70:
            self._days.pop(min(self._days))

        if self._tracker.update(bar):
            self._spent.clear()
            self._pending.clear()
            self._signals_today = 0
            self._swept_sides.clear()
        levels = [lv for lv in self._tracker.levels() if lv.name not in self._spent]

        # Resolution of active execution window (KZ-01 or SB-01)
        if cfg.enable_silver_bullet:
            win = in_any(cfg.silver_bullet_windows, bar.time)
            win_rule = "SB-01"
            win_label = "Silver Bullet"
        else:
            win = in_any(cfg.killzones, bar.time)
            win_rule = "KZ-1"
            win_label = "killzone"

        if win is None:
            for sweep in self._pending:
                self._log(close_time, win_rule, f"{sweep.level.name} sweep expired: {win_label} closed")
            self._pending.clear()
            if cfg.invalidate_levels_taken_outside_killzone:
                for lv in levels:
                    if (lv.side == "high" and bar.high > lv.price) or (lv.side == "low" and bar.low < lv.price):
                        self._spent.add(lv.name)
                        side_tag = "BSL" if lv.side == "high" else "SSL"
                        if side_tag not in self._swept_sides:
                            self._swept_sides.append(side_tag)
                        self._log(close_time, "TP-2", f"{lv.name} {self._fmt(lv.price)} taken outside killzone; spent")
            return None

        # Algorithmic Time Macros filter (MC-01)
        if cfg.enable_macros:
            macro_win = in_any(cfg.macro_windows, bar.time)
            if macro_win is None:
                return None

        confirmed: list[_Sweep] = []

        # TP-4: an earlier sweep confirms when a bar closes back inside the level.
        still_pending: list[_Sweep] = []
        for sweep in self._pending:
            if sweep.direction == SHORT:
                sweep.extreme = max(sweep.extreme, bar.high)
                inside = bar.close < sweep.level.price
            else:
                sweep.extreme = min(sweep.extreme, bar.low)
                inside = bar.close > sweep.level.price
            sweep.bars_waited += 1
            if inside:
                confirmed.append(sweep)
            elif sweep.bars_waited >= cfg.max_confirm_bars:
                self._log(close_time, "TP-4", f"{sweep.level.name} sweep expired: no close back inside in {cfg.max_confirm_bars} bars")
            else:
                still_pending.append(sweep)
        self._pending = still_pending

        # TP-3: a fresh sweep of a level that is not yet spent.
        threshold = cfg.min_sweep_pips * pip
        for lv in levels:
            if lv.side == "high" and bar.high >= lv.price + threshold:
                sweep = _Sweep(lv, SHORT, bar.high)
                inside = bar.close < lv.price
            elif lv.side == "low" and bar.low <= lv.price - threshold:
                sweep = _Sweep(lv, LONG, bar.low)
                inside = bar.close > lv.price
            else:
                continue
            self._spent.add(lv.name)
            side_tag = "BSL" if lv.side == "high" else "SSL"
            if side_tag not in self._swept_sides:
                self._swept_sides.append(side_tag)
            depth = abs(sweep.extreme - lv.price) / pip
            self._log(close_time, "TP-3", f"{lv.name} {self._fmt(lv.price)} swept by {depth:.1f} pips in {win.name} killzone")
            (confirmed if inside else self._pending).append(sweep)

        for sweep in confirmed:
            if self._signals_today >= cfg.max_signals_per_day:
                self._log(close_time, "RK-3", f"{sweep.level.name} setup skipped: max {cfg.max_signals_per_day} signal(s) per day reached")
                continue
            signal = self._build_signal(sweep, bar, close_time, win.name, win_rule)
            if signal is not None:
                self._signals_today += 1
                return signal
        return None

    def _build_signal(self, sweep: _Sweep, bar: Bar, close_time: datetime, kz_name: str, win_rule: str = "KZ-1") -> Signal | None:
        cfg, pip = self.config, self.instrument.pip_size
        d = sweep.direction
        entry = bar.close

        if cfg.bias_required:
            bias, why = self.daily_bias()
            if bias != d:
                self._log(close_time, "RL-05", f"{sweep.level.name} setup rejected: " + ("no clear daily draw" if not bias else f"against the daily draw ({why})"))
                return None
            wb, wwhy = self.weekly_bias()
            if wb and wb != d:
                self._log(close_time, "HT-02", f"{sweep.level.name} setup rejected: against the weekly draw ({wwhy})")
                return None

        # SW-02: Order of Liquidity Purging Filter
        if cfg.enforce_liquidity_purge_order:
            if d == SHORT and "SSL" in self._swept_sides:
                idx_bsl = self._swept_sides.index("BSL") if "BSL" in self._swept_sides else 999
                idx_ssl = self._swept_sides.index("SSL")
                if idx_ssl < idx_bsl:
                    self._log(close_time, "SW-02", f"{sweep.level.name} short rejected: SSL was purged before BSL today")
                    return None
            elif d == LONG and "BSL" in self._swept_sides:
                idx_ssl = self._swept_sides.index("SSL") if "SSL" in self._swept_sides else 999
                idx_bsl = self._swept_sides.index("BSL")
                if idx_bsl < idx_ssl:
                    self._log(close_time, "SW-02", f"{sweep.level.name} long rejected: BSL was purged before SSL today")
                    return None

        # FV-01 & FV-02: Fair Value Gap and Displacement 50% Equilibrium Filter
        fvg_reason = None
        if cfg.require_fvg:
            fvg = self._find_recent_fvg(d, sweep.bars_waited + 3)
            if fvg is None:
                self._log(close_time, "FV-01", f"{sweep.level.name} setup rejected: no 3-candle FVG in displacement")
                return None
            fvg_bottom, fvg_top, fvg_ce = fvg
            if cfg.fvg_in_premium_discount:
                if d == SHORT:
                    disp_high = sweep.extreme
                    disp_low = bar.low
                    disp_range = disp_high - disp_low
                    if disp_range > 0:
                        eq = disp_low + 0.5 * disp_range
                        if fvg_ce < eq:
                            self._log(close_time, "FV-02", f"{sweep.level.name} short rejected: FVG CE {self._fmt(fvg_ce)} in Discount (< {self._fmt(eq)})")
                            return None
                else:
                    disp_low = sweep.extreme
                    disp_high = bar.high
                    disp_range = disp_high - disp_low
                    if disp_range > 0:
                        eq = disp_low + 0.5 * disp_range
                        if fvg_ce > eq:
                            self._log(close_time, "FV-02", f"{sweep.level.name} long rejected: FVG CE {self._fmt(fvg_ce)} in Premium (> {self._fmt(eq)})")
                            return None
            fvg_reason = f"FV-01 confirmed FVG [{self._fmt(fvg_bottom)}, {self._fmt(fvg_top)}] CE {self._fmt(fvg_ce)}"

        stop = sweep.extreme - d * cfg.stop_buffer_pips * pip
        risk = (entry - stop) * d
        risk_pips = risk / pip
        if risk_pips < cfg.min_stop_pips or risk_pips > cfg.max_stop_pips:
            self._log(close_time, "RK-1", f"{sweep.level.name} setup rejected: stop {risk_pips:.1f} pips outside [{cfg.min_stop_pips}, {cfg.max_stop_pips}]")
            return None

        target_reason: str
        target_name = ""
        if cfg.target_mode == "opposite_level":
            # TG-1: the nearest opposing liquidity is the draw on price.
            side = "low" if d == SHORT else "high"
            opposing = [lv for lv in self._tracker.levels() if lv.side == side and (lv.price - entry) * d > 0]
            if not opposing:
                self._log(close_time, "TG-1", f"{sweep.level.name} setup rejected: no opposing liquidity to target")
                return None
            draw = min(opposing, key=lambda lv: abs(lv.price - entry))
            target = draw.price
            target_name = draw.name
            target_reason = f"TG-1 target {draw.name} {self._fmt(target)}"
        elif cfg.target_mode == "rr":
            target = entry + d * cfg.target_rr * risk
            target_name = f"{cfg.target_rr:g}R"
            target_reason = f"TG-1 target fixed {cfg.target_rr}R {self._fmt(target)}"
        else:
            raise ValueError(f"Unknown target_mode {cfg.target_mode!r}")

        rr = (target - entry) * d / risk
        min_required_rr = cfg.small_account_min_rr if cfg.small_account_mode else cfg.min_rr
        if rr < min_required_rr:
            rule_tag = "SA-03" if cfg.small_account_mode else "TG-2"
            self._log(close_time, rule_tag, f"{sweep.level.name} setup rejected: {rr:.2f}R below minimum {min_required_rr}R")
            return None

        advice = None
        if len(self._days) >= 3:
            from . import advisor, aplus
            days_done = sorted(self._days)[:-1]
            rng = (self._days[days_done[-1]][1], self._days[days_done[-1]][2]) if days_done else None
            pd_val = max(-1.0, min(1.0, ((rng[0] + rng[1]) / 2 - entry) * d / ((rng[0] - rng[1]) / 2))) if rng and rng[0] != rng[1] else 0.0
            bodies = sorted(abs(b.close - b.open) for b in self._bars[-21:-1])
            med_body = bodies[len(bodies) // 2] if bodies else 0.0
            body_mult = (abs(bar.close - bar.open) / med_body) if med_body > 0 else 1.5
            rngs = sorted(b.high - b.low for b in self._bars[-21:-1])
            med_rng = rngs[len(rngs) // 2] if rngs else 0.0
            fvg_now = self._find_recent_fvg(d, sweep.bars_waited + 3)
            gap_ratio = ((fvg_now[1] - fvg_now[0]) / med_rng) if (fvg_now and med_rng > 0) else 0.35
            timed = not any(w.start.hour == 0 and w.end.hour == 23 for w in cfg.killzones)
            ny = to_ny(bar.time)
            db, _ = self.daily_bias()
            wb, _ = self.weekly_bias()
            if db == d and wb == 0:
                wb = d
            elif wb == d and db == 0:
                db = d
            adr_rngs = [self._days[x][1] - self._days[x][2] for x in days_done[-20:]]
            last_d = self._days[sorted(self._days)[-1]]
            adr_val = (sum(adr_rngs) / len(adr_rngs), last_d[1] - last_d[2], last_d[1], last_d[2]) if len(days_done) >= 5 else None
            advice = advisor.assess({
                "dir": d, "bias": db, "wbias": wb, "level": sweep.level.name,
                "body_mult": max(body_mult, 1.5), "gap_ratio": gap_ratio, "pd": pd_val,
                "obstacles": 0, "sd": False, "rr": rr, "target": target_name,
                "window": kz_name, "timed": timed, "weekday": ny.weekday(), "hour": ny.hour,
                "adr": adr_val, "entry_price": entry, "target_price": target,
            })

        side = "SHORT" if d == SHORT else "LONG"
        rule_rr_tag = "SA-03" if cfg.small_account_mode else "TG-2"
        reasons_list = [
            f"{win_rule} inside {kz_name} killzone",
            f"TP-3 swept {sweep.level.name} {self._fmt(sweep.level.price)} to {self._fmt(sweep.extreme)}",
            f"TP-4 closed back inside at {self._fmt(entry)}",
            f"RK-1 stop {self._fmt(stop)}, {cfg.stop_buffer_pips} pips beyond sweep extreme ({risk_pips:.1f} pips risk)",
            target_reason,
            f"{rule_rr_tag} {rr:.2f}R >= {min_required_rr}R",
        ]
        if fvg_reason:
            reasons_list.insert(3, fvg_reason)
        if advice:
            reasons_list.append(advisor.line(advice))
            reasons_list.extend(aplus.line(advice))
        reasons = tuple(reasons_list)
        self._log(close_time, "SIGNAL", f"{side} after {sweep.level.name} sweep, {rr:.2f}R")
        meta = {"advice": advice, "level_price": sweep.level.price, "sweep_price": sweep.extreme, "target_name": target_name} if advice else {}
        return Signal(close_time, d, entry, stop, target, rr, sweep.level, kz_name, reasons, meta=meta)
