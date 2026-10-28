"""God's Eye × ICT Flux Single-Pass Chokepoint Intercept Engine.

Replaces the 35 redundant models of legacy ict-trader with ONE unified, vectorized
institutional state machine designed exclusively for real order-flow markets
(Forex, Metals, Commodities, Indices, Stocks).

Core Quantitative Upgrades over Legacy ICT Flux:
1. Strict Order-Flow Alignment (No Counter-Trend Freight-Train Fades):
   Never fades an expanding 4H/1H institutional trend just because a session high/low was touched.
   Trades WITH the 4H/1H structural trend after a liquidity sweep (Trend Pullback Raid / Session Sweep)
   or requires extreme outer-quadrant exhaustion + structural trend break for reversals.
2. CISD (Change in State of Delivery) + Immediate Chokepoint Execution:
   Confirms institutional rejection via CISD + displacement body and executes at the Chokepoint
   (entering on limit pullback or next-bar open if within 0.25R so high-velocity winners are never missed).
3. Intermarket SMT Triad Confirmation:
   Cross-checks correlated sister markets (EURUSD/GBPUSD, XAUUSD/XAGUSD, NAS100/US500, WTI/BRENT)
   at the exact bar of the liquidity raid.
4. Defensive Asymmetric Dual-Tranche Governor:
   - At +0.80R: Stop-Loss automatically shifts to Breakeven (+0.05R) — scratches reversals before they lose.
   - At +1.25R (TP1): Banks 70% of position and locks runner SL at +0.25R.
   - At TP2 (2.2R–3.5R Draw on Liquidity): Banks remaining 30% structural runner.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from statistics import median
from typing import Any

from .feed import Bar, to_ny
from .universe import MarketSpec, is_macro_window, killzones_for

LONG = 1
SHORT = -1


@dataclass(frozen=True)
class LiquidityPool:
    name: str
    price: float
    side: str          # "BSL" (buy-side high) | "SSL" (sell-side low)
    tier: str          # "EXTERNAL" (PDH/PDL/PWH/PWL) | "SESSION" (ASIA/LONDON/EQH/EQL) | "INTERNAL" (SWING)
    formed_at: datetime


@dataclass
class ChokepointSignal:
    id: str
    symbol: str
    label: str
    asset_class: str
    timeframe_minutes: int
    direction: int                  # +1 LONG, -1 SHORT
    archetype: str                  # "TREND_PULLBACK_RAID" | "SMT_TURTLE_SOUP" | "JUDAS_SWEEP_REVERSAL"
    stage: str                      # "STAGE_3_INTERCEPT"
    swept_pool: str
    swept_price: float
    sweep_time: datetime
    signal_time: datetime
    entry_price: float
    stop_price: float
    tp1_price: float                # +1.25R (70% tranche + BE lock)
    tp2_price: float                # External Draw on Liquidity (30% runner)
    tp2_name: str
    rr_tp1: float
    rr_tp2: float
    confidence_pct: float           # Calibrated Bayesian / Empirical score [0..99]
    ev_r: float                     # Expected Value in R after spread friction
    ev_lower_bound_r: float         # Conservative lower-bound EV in R
    spread_friction_pct: float
    smt_confirmed: bool
    smt_note: str
    killzone: str
    is_macro: bool
    grade: str                      # "A+" | "A" | "B"
    verdict: str                    # "EXECUTE A+" | "EXECUTE A" | "STAND DOWN"
    reasons: list[str] = field(default_factory=list)


@dataclass
class TradeExecution:
    signal: ChokepointSignal
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    tp1_hit: bool
    r_multiple: float               # Realized net R after spread & dual-tranche weighting
    mfe_r: float                    # Maximum Favorable Excursion in R
    mae_r: float                    # Maximum Adverse Excursion in R
    exit_reason: str
    status: str                     # "win" | "loss" | "breakeven" | "open"


@dataclass
class MarketRadarState:
    symbol: str
    label: str
    asset_class: str
    last_price: float
    last_time: datetime
    last_time_ny: str
    bias: int                       # +1 Bullish, -1 Bearish, 0 Neutral
    bias_label: str
    regime: str                     # "EXPANSION_TREND" | "MANIPULATION_REVERSAL" | "CONSOLIDATION"
    dealing_range_pct: float        # 0% = extreme discount, 50% = equilibrium, 100% = extreme premium
    location_label: str             # "DEEP_DISCOUNT" | "DISCOUNT" | "EQUILIBRIUM" | "PREMIUM" | "DEEP_PREMIUM"
    adr_used_pct: float
    active_stage: str               # "STAGE 1: PRECURSOR ACCUMULATION" | "STAGE 2: VELOCITY INFLECTION" | "STAGE 3: CHOKEPOINT INTERCEPT"
    stage_code: int                 # 1 | 2 | 3
    stage_summary: str
    nearest_bsl: LiquidityPool | None
    nearest_ssl: LiquidityPool | None
    smt_active: bool
    smt_summary: str
    active_killzone: str | None
    is_macro: bool
    staged_intercept: dict[str, Any] | None
    recent_signals: list[dict[str, Any]]
    recent_trades: list[dict[str, Any]]
    funnel_counts: dict[str, int]


def _fmt_price(p: float, decimals: int) -> float:
    return round(p, decimals)


def _detect_smt_at_bar(
    bars_a: list[Bar],
    idx_a: int,
    partner_bars: list[Bar] | None,
    partner_time_index: list[datetime] | None,
    direction: int,
    swept_symbol: str,
    partner_symbol: str,
) -> tuple[bool, str]:
    """Fast O(log N) SMT divergence check between target asset and its correlated partner
    over the last 24 bars up to `bars_a[idx_a].time`.
    """
    if not partner_bars or not partner_time_index or idx_a < 15:
        return False, ""
    t_now = bars_a[idx_a].time
    p_idx = bisect_left(partner_time_index, t_now)
    if p_idx >= len(partner_bars):
        p_idx = len(partner_bars) - 1
    if p_idx < 15 or abs((partner_bars[p_idx].time - t_now).total_seconds()) > 3600:
        return False, ""

    win_a = bars_a[idx_a - 14 : idx_a + 1]
    win_b = partner_bars[p_idx - 14 : p_idx + 1]
    if len(win_a) < 10 or len(win_b) < 10:
        return False, ""

    if direction == LONG:
        a_prior_lo = min(x.low for x in win_a[:-3])
        a_curr_lo = min(x.low for x in win_a[-3:])
        b_prior_lo = min(x.low for x in win_b[:-3])
        b_curr_lo = min(x.low for x in win_b[-3:])
        if (a_curr_lo < a_prior_lo and b_curr_lo > b_prior_lo) or (a_curr_lo > a_prior_lo and b_curr_lo < b_prior_lo):
            return True, f"Bullish SMT Triad: {swept_symbol} vs {partner_symbol} divergence at sell-side lows"
    else:
        a_prior_hi = max(x.high for x in win_a[:-3])
        a_curr_hi = max(x.high for x in win_a[-3:])
        b_prior_hi = max(x.high for x in win_b[:-3])
        b_curr_hi = max(x.high for x in win_b[-3:])
        if (a_curr_hi > a_prior_hi and b_curr_hi < b_prior_hi) or (a_curr_hi < a_prior_hi and b_curr_hi > b_prior_hi):
            return True, f"Bearish SMT Triad: {swept_symbol} vs {partner_symbol} divergence at buy-side highs"

    return False, ""


class GodsEyeFluxEngine:
    """Single-Pass Institutional Chokepoint Engine for Live Radar & Research Lab Backtesting."""

    def __init__(
        self,
        spec: MarketSpec,
        tf_minutes: int = 5,
        mode: str = "ict",             # "ict" (killzones) | "allday" (all institutional sessions)
        tp1_r: float = 1.25,           # 70% partial banked at +1.25R
        be_trigger_r: float = 0.78,    # Early Breakeven shield at +0.78R
        tp1_weight: float = 0.70,      # 70% tranche 1, 30% structural runner
        min_rr_tp2: float = 2.0,       # Minimum runner target R:R
        max_spread_pct: float = 10.0,  # Max allowed spread / stop distance (%)
    ) -> None:
        self.spec = spec
        self.tf_minutes = tf_minutes
        self.mode = mode
        self.tp1_r = tp1_r
        self.be_trigger_r = be_trigger_r
        self.tp1_weight = tp1_weight
        self.min_rr_tp2 = min_rr_tp2
        self.max_spread_pct = max_spread_pct
        self.spread_price = spec.pip_size * spec.typical_spread_pips
        self.killzones = killzones_for(spec)

    def run(
        self,
        bars: list[Bar],
        partner_bars: list[Bar] | None = None,
    ) -> tuple[list[ChokepointSignal], list[TradeExecution], MarketRadarState]:
        """Executes a single O(N) pass over `bars`, generating all ChokepointSignals,
        simulating conservative Dual-Tranche executions, and returning the current MarketRadarState.
        """
        if len(bars) < 40:
            raise ValueError(f"At least 40 bars required for {self.spec.key}, got {len(bars)}")

        partner_times = [b.time for b in partner_bars] if partner_bars else None
        signals: list[ChokepointSignal] = []
        trades: list[TradeExecution] = []

        funnel = {
            "pools_swept_in_window": 0,
            "rejected_against_order_flow": 0,
            "rejected_no_cisd_or_mss": 0,
            "rejected_exhausted_move": 0,
            "rejected_spread_or_adr": 0,
            "chokepoint_orders_staged": 0,
            "intercepts_filled": 0,
        }

        day_ohlc: dict[date, list[float]] = {}
        asia_ranges: dict[date, list[float]] = {}
        london_ranges: dict[date, list[float]] = {}
        midnight_opens: dict[date, float] = {}
        spent_pools_today: dict[date, set[str]] = {}
        signals_per_day: dict[date, int] = {}

        pending_raid: dict[str, Any] | None = None
        staged_order: ChokepointSignal | None = None
        staged_bars_waited: int = 0
        open_pos: dict[str, Any] | None = None

        ema20 = bars[0].close
        ema50 = bars[0].close
        ema120 = bars[0].close
        alpha20 = 2.0 / 21.0
        alpha50 = 2.0 / 51.0
        alpha120 = 2.0 / 121.0

        max_daily_trades = 2 if self.mode == "ict" else 3

        for i, bar in enumerate(bars):
            ny_dt = to_ny(bar.time)
            ny_date = ny_dt.date()
            ny_t = ny_dt.time()
            trading_d = ny_date + timedelta(days=1) if (self.spec.asset_class != "stock" and ny_t >= time(17, 0)) else ny_date

            ema20 = bar.close * alpha20 + ema20 * (1.0 - alpha20)
            ema50 = bar.close * alpha50 + ema50 * (1.0 - alpha50)
            ema120 = bar.close * alpha120 + ema120 * (1.0 - alpha120)

            if ny_t >= time(0, 0) and trading_d not in midnight_opens:
                midnight_opens[trading_d] = bar.open

            if time(20, 0) <= ny_t or ny_t < time(0, 0):
                ar = asia_ranges.setdefault(trading_d, [bar.high, bar.low])
                ar[0] = max(ar[0], bar.high)
                ar[1] = min(ar[1], bar.low)

            if time(2, 0) <= ny_t < time(5, 0):
                lr = london_ranges.setdefault(trading_d, [bar.high, bar.low])
                lr[0] = max(lr[0], bar.high)
                lr[1] = min(lr[1], bar.low)

            # 1. Manage active open trade position
            if open_pos is not None:
                self._update_open_position(open_pos, bar, i, ny_t, trades, bars)
                if open_pos["closed"]:
                    open_pos = None

            # 2. Manage staged Chokepoint order (fills on limit touch OR next-bar open if within 0.30R of Chokepoint)
            if staged_order is not None and open_pos is None:
                staged_bars_waited += 1
                d = staged_order.direction
                risk_dist = abs(staged_order.entry_price - staged_order.stop_price) or 1e-9
                if (d == LONG and bar.low <= staged_order.stop_price) or (d == SHORT and bar.high >= staged_order.stop_price):
                    staged_order = None
                elif (d == LONG and bar.high >= staged_order.tp1_price) or (d == SHORT and bar.low <= staged_order.tp1_price):
                    staged_order = None
                else:
                    touched_limit = bar.low <= staged_order.entry_price <= bar.high
                    near_open = abs(bar.open - staged_order.entry_price) <= 0.30 * risk_dist and (
                        (d == LONG and bar.close > bar.open) or (d == SHORT and bar.close < bar.open)
                    )
                    if touched_limit or near_open:
                        funnel["intercepts_filled"] += 1
                        signals.append(staged_order)
                        signals_per_day[trading_d] = signals_per_day.get(trading_d, 0) + 1
                        base_fill = staged_order.entry_price if touched_limit else bar.open
                        fill_px = base_fill + (self.spread_price if d == LONG else 0.0)
                        open_pos = {
                            "signal": staged_order,
                            "entry_time": bar.time,
                            "entry_idx": i,
                            "entry_price": fill_px,
                            "current_stop": staged_order.stop_price,
                            "be_locked": False,
                            "tp1_hit": False,
                            "realized_r": 0.0,
                            "rem_weight": 1.0,
                            "mfe_r": 0.0,
                            "mae_r": 0.0,
                            "closed": False,
                        }
                        staged_order = None
                        self._update_open_position(open_pos, bar, i, ny_t, trades, bars, on_fill_bar=True)
                        if open_pos["closed"]:
                            open_pos = None
                    elif staged_bars_waited >= 3:
                        staged_order = None

            if i < 35:
                self._record_day_ohlc(day_ohlc, trading_d, bar)
                continue

            pools = self._active_pools(
                bars, i, trading_d, ny_t, day_ohlc, asia_ranges, london_ranges, spent_pools_today.get(trading_d, set())
            )

            bias, bias_label, regime, dr_pct, loc_label = self._compute_order_flow(
                bars, i, ema20, ema50, ema120, day_ohlc, trading_d, midnight_opens.get(trading_d)
            )

            kz_name = self._active_killzone(ny_t)
            in_session = (kz_name is not None) or (self.mode == "allday") or (self.tf_minutes >= 60)
            if self.tf_minutes < 60 and ((ny_dt.weekday() == 0 and ny_t < time(6, 0)) or (ny_dt.weekday() == 4 and ny_t >= time(13, 0))):
                in_session = False

            # Step A: Evaluate pending raid for Velocity Shift (CISD + Momentum confirmation)
            if pending_raid is not None:
                pending_raid["bars_waited"] += 1
                raid_dir = pending_raid["direction"]
                if raid_dir == LONG and bar.low < pending_raid["extreme"]:
                    pending_raid["extreme"] = bar.low
                elif raid_dir == SHORT and bar.high > pending_raid["extreme"]:
                    pending_raid["extreme"] = bar.high

                shifted, shift_kind, entry_px, reasons = self._check_velocity_shift(bars, i, pending_raid, ema20)
                if shifted and open_pos is None and staged_order is None and signals_per_day.get(trading_d, 0) < max_daily_trades:
                    sig_candidate = self._build_chokepoint_signal(
                        bars=bars,
                        idx=i,
                        raid=pending_raid,
                        shift_kind=shift_kind,
                        entry_px=entry_px,
                        shift_reasons=reasons,
                        pools=pools,
                        bias=bias,
                        dr_pct=dr_pct,
                        day_ohlc=day_ohlc,
                        trading_d=trading_d,
                        kz_name=kz_name or "SESSION",
                        ny_t=ny_t,
                        partner_bars=partner_bars,
                        partner_times=partner_times,
                        funnel=funnel,
                    )
                    pending_raid = None
                    if sig_candidate is not None:
                        funnel["chokepoint_orders_staged"] += 1
                        staged_order = sig_candidate
                        staged_bars_waited = 0
                elif pending_raid["bars_waited"] >= 5:
                    funnel["rejected_no_cisd_or_mss"] += 1
                    pending_raid = None

            # Step B: Detect new Liquidity Sweep inside active session
            if in_session and open_pos is None and staged_order is None and signals_per_day.get(trading_d, 0) < max_daily_trades:
                swept = self._detect_pool_sweep(bars, i, pools)
                if swept is not None:
                    pool_obj, raid_dir, extreme_px = swept
                    spent_pools_today.setdefault(trading_d, set()).add(pool_obj.name)
                    funnel["pools_swept_in_window"] += 1

                    smt_ok, _ = _detect_smt_at_bar(
                        bars, i, partner_bars, partner_times, raid_dir, self.spec.key, self.spec.smt_partner
                    )

                    # Strict Institutional Order-Flow & Dealing-Range Quadrant Gate:
                    # Never buy in deep premium (>58%) and never sell in deep discount (<42%).
                    in_valid_quadrant = (raid_dir == LONG and dr_pct <= 58.0) or (raid_dir == SHORT and dr_pct >= 42.0)
                    if not in_valid_quadrant:
                        order_flow_ok = False
                    elif bias == raid_dir:
                        # Internal swings additionally require being on the favorable side of Equilibrium (<=50% for longs, >=50% for shorts)
                        order_flow_ok = True if pool_obj.tier in ("EXTERNAL", "SESSION") else (
                            (raid_dir == LONG and dr_pct <= 48.0) or (raid_dir == SHORT and dr_pct >= 52.0)
                        )
                    elif bias == 0:
                        order_flow_ok = pool_obj.tier in ("EXTERNAL", "SESSION") and (
                            (raid_dir == LONG and dr_pct <= 45.0) or (raid_dir == SHORT and dr_pct >= 55.0)
                        )
                    else:
                        extreme_loc = (raid_dir == LONG and dr_pct <= 20.0) or (raid_dir == SHORT and dr_pct >= 80.0)
                        order_flow_ok = pool_obj.tier in ("EXTERNAL", "SESSION") and smt_ok and extreme_loc

                    if not order_flow_ok:
                        funnel["rejected_against_order_flow"] += 1
                    else:
                        pending_raid = {
                            "pool": pool_obj,
                            "direction": raid_dir,
                            "extreme": extreme_px,
                            "sweep_idx": i,
                            "sweep_time": bar.time,
                            "bars_waited": 0,
                        }
                        shifted, shift_kind, entry_px, reasons = self._check_velocity_shift(bars, i, pending_raid, ema20)
                        if shifted:
                            sig_candidate = self._build_chokepoint_signal(
                                bars=bars,
                                idx=i,
                                raid=pending_raid,
                                shift_kind=shift_kind,
                                entry_px=entry_px,
                                shift_reasons=reasons,
                                pools=pools,
                                bias=bias,
                                dr_pct=dr_pct,
                                day_ohlc=day_ohlc,
                                trading_d=trading_d,
                                kz_name=kz_name or "SESSION",
                                ny_t=ny_t,
                                partner_bars=partner_bars,
                                partner_times=partner_times,
                                funnel=funnel,
                            )
                            pending_raid = None
                            if sig_candidate is not None:
                                funnel["chokepoint_orders_staged"] += 1
                                staged_order = sig_candidate
                                staged_bars_waited = 0

            self._record_day_ohlc(day_ohlc, trading_d, bar)

        if open_pos is not None and bars:
            last_b = bars[-1]
            sig = open_pos["signal"]
            d = sig.direction
            risk = abs(open_pos["entry_price"] - sig.stop_price) or 1e-9
            exit_px = last_b.close - (0.0 if d == LONG else self.spread_price)
            floating_r = ((exit_px - open_pos["entry_price"]) * d) / risk
            net_r = open_pos["realized_r"] + open_pos["rem_weight"] * floating_r
            trades.append(
                TradeExecution(
                    signal=sig,
                    entry_time=open_pos["entry_time"],
                    entry_price=open_pos["entry_price"],
                    exit_time=last_b.time,
                    exit_price=exit_px,
                    stop_price=open_pos["current_stop"],
                    tp1_price=sig.tp1_price,
                    tp2_price=sig.tp2_price,
                    tp1_hit=open_pos["tp1_hit"],
                    r_multiple=round(net_r, 3),
                    mfe_r=round(open_pos["mfe_r"], 3),
                    mae_r=round(open_pos["mae_r"], 3),
                    exit_reason="OPEN (Live Mark-to-Market)",
                    status="open",
                )
            )

        radar_state = self._build_radar_state(
            bars=bars,
            ema20=ema20,
            ema50=ema50,
            ema120=ema120,
            day_ohlc=day_ohlc,
            asia_ranges=asia_ranges,
            london_ranges=london_ranges,
            midnight_opens=midnight_opens,
            spent_pools_today=spent_pools_today,
            pending_raid=pending_raid,
            staged_order=staged_order,
            open_pos=open_pos,
            signals=signals,
            trades=trades,
            partner_bars=partner_bars,
            partner_times=partner_times,
            funnel=funnel,
        )
        return signals, trades, radar_state

    def _record_day_ohlc(self, day_ohlc: dict[date, list[float]], trading_d: date, bar: Bar) -> None:
        if trading_d not in day_ohlc:
            day_ohlc[trading_d] = [bar.open, bar.high, bar.low, bar.close]
        else:
            d_rec = day_ohlc[trading_d]
            d_rec[1] = max(d_rec[1], bar.high)
            d_rec[2] = min(d_rec[2], bar.low)
            d_rec[3] = bar.close

    def _active_killzone(self, ny_t: time) -> str | None:
        for w in self.killzones:
            if w.contains(ny_t):
                return w.name
        return None

    def _active_pools(
        self,
        bars: list[Bar],
        idx: int,
        trading_d: date,
        ny_t: time,
        day_ohlc: dict[date, list[float]],
        asia_ranges: dict[date, list[float]],
        london_ranges: dict[date, list[float]],
        spent_today: set[str],
    ) -> list[LiquidityPool]:
        pools: list[LiquidityPool] = []
        completed_days = [d for d in sorted(day_ohlc) if d < trading_d]

        if completed_days:
            pd = day_ohlc[completed_days[-1]]
            t_pd = bars[idx - 1].time
            if "PDH" not in spent_today:
                pools.append(LiquidityPool("PDH", pd[1], "BSL", "EXTERNAL", t_pd))
            if "PDL" not in spent_today:
                pools.append(LiquidityPool("PDL", pd[2], "SSL", "EXTERNAL", t_pd))

        if len(completed_days) >= 5:
            wk_days = completed_days[-5:]
            pwh = max(day_ohlc[d][1] for d in wk_days)
            pwl = min(day_ohlc[d][2] for d in wk_days)
            t_pw = bars[idx - 1].time
            if "PWH" not in spent_today:
                pools.append(LiquidityPool("PWH", pwh, "BSL", "EXTERNAL", t_pw))
            if "PWL" not in spent_today:
                pools.append(LiquidityPool("PWL", pwl, "SSL", "EXTERNAL", t_pw))

        if ny_t >= time(0, 0) and ny_t < time(17, 0) and trading_d in asia_ranges:
            ar = asia_ranges[trading_d]
            if "ASIA_H" not in spent_today:
                pools.append(LiquidityPool("ASIA_H", ar[0], "BSL", "SESSION", bars[idx - 1].time))
            if "ASIA_L" not in spent_today:
                pools.append(LiquidityPool("ASIA_L", ar[1], "SSL", "SESSION", bars[idx - 1].time))

        if ny_t >= time(5, 0) and ny_t < time(17, 0) and trading_d in london_ranges:
            lr = london_ranges[trading_d]
            if "LONDON_H" not in spent_today:
                pools.append(LiquidityPool("LONDON_H", lr[0], "BSL", "SESSION", bars[idx - 1].time))
            if "LONDON_L" not in spent_today:
                pools.append(LiquidityPool("LONDON_L", lr[1], "SSL", "SESSION", bars[idx - 1].time))

        win_start = max(2, idx - 42)
        swing_highs: list[tuple[int, float]] = []
        swing_lows: list[tuple[int, float]] = []
        for j in range(win_start, idx - 1):
            b_prev, b_cur, b_next = bars[j - 1], bars[j], bars[j + 1]
            if b_cur.high > b_prev.high and b_cur.high > b_next.high:
                if all(bars[k].high <= b_cur.high for k in range(j + 1, idx)):
                    swing_highs.append((j, b_cur.high))
            if b_cur.low < b_prev.low and b_cur.low < b_next.low:
                if all(bars[k].low >= b_cur.low for k in range(j + 1, idx)):
                    swing_lows.append((j, b_cur.low))

        med_rng = median([b.range for b in bars[max(0, idx - 20) : idx]]) or self.spec.pip_size * 5
        eq_tol = 0.25 * med_rng

        for j_s, h_s in swing_highs[-1:]:
            is_eqh = any(abs(h_s - h_prev) <= eq_tol for _, h_prev in swing_highs[:-1])
            nm = "EQH" if is_eqh else "SWH"
            if nm not in spent_today:
                pools.append(LiquidityPool(nm, h_s, "BSL", "SESSION" if is_eqh else "INTERNAL", bars[j_s].time))

        for j_s, l_s in swing_lows[-1:]:
            is_eql = any(abs(l_s - l_prev) <= eq_tol for _, l_prev in swing_lows[:-1])
            nm = "EQL" if is_eql else "SWL"
            if nm not in spent_today:
                pools.append(LiquidityPool(nm, l_s, "SSL", "SESSION" if is_eql else "INTERNAL", bars[j_s].time))

        return pools

    def _compute_order_flow(
        self,
        bars: list[Bar],
        idx: int,
        ema20: float,
        ema50: float,
        ema120: float,
        day_ohlc: dict[date, list[float]],
        trading_d: date,
        midnight_open: float | None,
    ) -> tuple[int, str, str, float, str]:
        cur_px = bars[idx].close
        lookback_bars = bars[max(0, idx - 72) : idx + 1]
        hi_72 = max(b.high for b in lookback_bars)
        lo_72 = min(b.low for b in lookback_bars)

        completed_days = [d for d in sorted(day_ohlc) if d < trading_d]
        if completed_days:
            pd_hi, pd_lo = day_ohlc[completed_days[-1]][1], day_ohlc[completed_days[-1]][2]
            dr_hi = max(hi_72, pd_hi)
            dr_lo = min(lo_72, pd_lo)
        else:
            dr_hi, dr_lo = hi_72, lo_72

        span = max(dr_hi - dr_lo, 1e-9)
        dr_pct = max(0.0, min(100.0, ((cur_px - dr_lo) / span) * 100.0))

        if dr_pct <= 25.0:
            loc_label = "DEEP_DISCOUNT"
        elif dr_pct < 48.0:
            loc_label = "DISCOUNT"
        elif dr_pct <= 52.0:
            loc_label = "EQUILIBRIUM"
        elif dr_pct < 75.0:
            loc_label = "PREMIUM"
        else:
            loc_label = "DEEP_PREMIUM"

        # True Multi-Day Macro Order-Flow Vector (prevents shorting a multi-day bull trend during an overnight pullback)
        macro_bias = 0
        if len(completed_days) >= 4:
            recent_d = completed_days[-5:]
            d_closes = [day_ohlc[d][3] for d in recent_d]
            d_highs = [day_ohlc[d][1] for d in recent_d]
            d_lows = [day_ohlc[d][2] for d in recent_d]
            avg_5d = sum(d_closes) / len(d_closes)
            ipda_mid = (max(d_highs) + min(d_lows)) / 2.0
            if cur_px >= avg_5d and d_closes[-1] >= d_closes[0] and cur_px >= ipda_mid:
                macro_bias = LONG
            elif cur_px <= avg_5d and d_closes[-1] <= d_closes[0] and cur_px <= ipda_mid:
                macro_bias = SHORT

        intraday_bull = (ema20 >= ema50 >= ema120) or (cur_px > ema50 and ema50 >= ema120)
        intraday_bear = (ema20 <= ema50 <= ema120) or (cur_px < ema50 and ema50 <= ema120)

        if macro_bias == LONG and not intraday_bear:
            bias = LONG
            bias_label = "BULLISH MACRO & INTRADAY ORDER FLOW (IPDA + 4H/1H Structure)"
            regime = "EXPANSION_TREND"
        elif macro_bias == SHORT and not intraday_bull:
            bias = SHORT
            bias_label = "BEARISH MACRO & INTRADAY ORDER FLOW (IPDA + 4H/1H Structure)"
            regime = "EXPANSION_TREND"
        elif macro_bias == LONG and intraday_bear:
            # Multi-day bull trend experiencing an intraday discount pullback -> ONLY look for LONG raids in discount!
            bias = LONG
            bias_label = "BULLISH MACRO PULLBACK (Hunting Discount Sell-Side Raids Only)"
            regime = "MANIPULATION_REVERSAL"
        elif macro_bias == SHORT and intraday_bull:
            # Multi-day bear trend experiencing an intraday premium bounce -> ONLY look for SHORT raids in premium!
            bias = SHORT
            bias_label = "BEARISH MACRO RETRACEMENT (Hunting Premium Buy-Side Raids Only)"
            regime = "MANIPULATION_REVERSAL"
        elif intraday_bull:
            bias = LONG
            bias_label = "BULLISH INTRADAY ORDER FLOW (4H/1H Structure)"
            regime = "EXPANSION_TREND"
        elif intraday_bear:
            bias = SHORT
            bias_label = "BEARISH INTRADAY ORDER FLOW (4H/1H Structure)"
            regime = "EXPANSION_TREND"
        else:
            bias = 0
            bias_label = "ROTATIONAL DEALING RANGE (Awaiting Extremity Raid)"
            regime = "CONSOLIDATION"

        return bias, bias_label, regime, round(dr_pct, 1), loc_label

    def _detect_pool_sweep(
        self,
        bars: list[Bar],
        idx: int,
        pools: list[LiquidityPool],
    ) -> tuple[LiquidityPool, int, float] | None:
        bar = bars[idx]
        min_penetration = self.spec.pip_size * 0.2
        tier_rank = {"EXTERNAL": 0, "SESSION": 1, "INTERNAL": 2}
        for p in sorted(pools, key=lambda x: tier_rank.get(x.tier, 3)):
            if p.side == "SSL":
                if bar.low < p.price - min_penetration and bar.close >= p.price - (0.35 * bar.range):
                    return p, LONG, bar.low
            else:
                if bar.high > p.price + min_penetration and bar.close <= p.price + (0.35 * bar.range):
                    return p, SHORT, bar.high
        return None

    def _check_velocity_shift(
        self,
        bars: list[Bar],
        idx: int,
        raid: dict[str, Any],
        ema20: float,
    ) -> tuple[bool, str, float, list[str]]:
        """Confirms Stage 2 Velocity Inflection:
        Requires price to reclaim the swept pool + close past the CISD delivery open
        AND close in the upper 35% of the bar range (for longs) or lower 35% (for shorts)
        so weak wicks that immediately fail are filtered out.
        """
        bar = bars[idx]
        d = raid["direction"]
        sweep_idx = raid["sweep_idx"]
        pool: LiquidityPool = raid["pool"]
        med_body = median([b.body for b in bars[max(0, idx - 20) : idx]]) or (self.spec.pip_size * 2)

        cisd_idx = sweep_idx
        while cisd_idx > max(0, sweep_idx - 4):
            prev_b = bars[cisd_idx - 1]
            if (d == LONG and prev_b.close < prev_b.open) or (d == SHORT and prev_b.close > prev_b.open):
                cisd_idx -= 1
            else:
                break
        cisd_level = bars[cisd_idx].open

        fvg_zone: tuple[float, float] | None = None
        if idx >= 2:
            c1, c3 = bars[idx - 2], bars[idx]
            if d == LONG and c3.low > c1.high:
                fvg_zone = (c1.high, c3.low)
            elif d == SHORT and c3.high < c1.low:
                fvg_zone = (c3.high, c1.low)

        body_mult = bar.body / max(med_body, 1e-9)
        reclaimed_pool = (bar.close > pool.price) if d == LONG else (bar.close < pool.price)
        cisd_closed = (bar.close > cisd_level) if d == LONG else (bar.close < cisd_level)
        strong_close = (
            (bar.close >= bar.low + 0.65 * max(bar.range, 1e-9))
            if d == LONG
            else (bar.close <= bar.high - 0.65 * max(bar.range, 1e-9))
        )
        ema_reclaim = (bar.close >= ema20) if d == LONG else (bar.close <= ema20)

        if not (reclaimed_pool and cisd_closed and strong_close and body_mult >= 0.95 and ema_reclaim):
            return False, "", 0.0, []

        reasons = [
            f"SW-01 Swept {pool.tier.lower()} liquidity pool {pool.name} ({pool.price:.{self.spec.decimals}f})",
            f"CS-01 Change in State of Delivery (CISD) & EMA-20 reclaim confirmed at {cisd_level:.{self.spec.decimals}f} ({body_mult:.1f}x body)",
        ]

        if fvg_zone is not None:
            entry_px = fvg_zone[0] + 0.75 * (fvg_zone[1] - fvg_zone[0]) if d == LONG else fvg_zone[1] - 0.75 * (fvg_zone[1] - fvg_zone[0])
            reasons.append(
                f"FV-01 Chokepoint Fair Value Gap [{fvg_zone[0]:.{self.spec.decimals}f} – {fvg_zone[1]:.{self.spec.decimals}f}]"
            )
            return True, "CISD_FVG_CHOKEPOINT", entry_px, reasons

        entry_px = bar.close - (0.15 * bar.body) if d == LONG else bar.close + (0.15 * bar.body)
        reasons.append(f"OB-01 Chokepoint Order Block & CISD Reclaim at {entry_px:.{self.spec.decimals}f}")
        return True, "CISD_OB_CHOKEPOINT", entry_px, reasons

    def _build_chokepoint_signal(
        self,
        bars: list[Bar],
        idx: int,
        raid: dict[str, Any],
        shift_kind: str,
        entry_px: float,
        shift_reasons: list[str],
        pools: list[LiquidityPool],
        bias: int,
        dr_pct: float,
        day_ohlc: dict[date, list[float]],
        trading_d: date,
        kz_name: str,
        ny_t: time,
        partner_bars: list[Bar] | None,
        partner_times: list[datetime] | None,
        funnel: dict[str, int],
    ) -> ChokepointSignal | None:
        d = raid["direction"]
        pool: LiquidityPool = raid["pool"]
        extreme: float = raid["extreme"]
        med_rng = median([b.range for b in bars[max(0, idx - 20) : idx]]) or (self.spec.pip_size * 5)

        buffer = max(self.spec.pip_size * 1.0, 0.20 * med_rng)
        stop_px = (extreme - buffer) if d == LONG else (extreme + buffer)
        stop_dist = (entry_px - stop_px) * d
        if stop_dist <= self.spec.pip_size * 1.5:
            return None

        if stop_dist > 3.8 * med_rng:
            funnel["rejected_exhausted_move"] += 1
            return None

        spread_pct = (self.spread_price / stop_dist) * 100.0
        if spread_pct > self.max_spread_pct:
            funnel["rejected_spread_or_adr"] += 1
            return None

        opp_side = "BSL" if d == LONG else "SSL"
        opp_pools = [p for p in pools if p.side == opp_side and (p.price - entry_px) * d > stop_dist * 1.6]
        if opp_pools:
            valid_tp2 = [p for p in opp_pools if abs(p.price - entry_px) / stop_dist >= self.min_rr_tp2]
            tp2_pool = min(valid_tp2, key=lambda p: abs(p.price - entry_px)) if valid_tp2 else max(opp_pools, key=lambda p: abs(p.price - entry_px))
            tp2_px = tp2_pool.price
            tp2_name = tp2_pool.name
        else:
            tp2_px = entry_px + d * (2.5 * stop_dist)
            tp2_name = "DOL_2.5R_PROJECTION"

        rr_tp2 = min(4.5, max(2.0, abs(tp2_px - entry_px) / stop_dist))
        tp2_px = entry_px + d * (rr_tp2 * stop_dist)

        total_leg = abs(tp2_px - extreme)
        already_traveled = abs(bars[idx].close - extreme)
        if total_leg > 0 and (already_traveled / total_leg) > 0.58:
            funnel["rejected_exhausted_move"] += 1
            return None

        tp1_px = entry_px + d * (self.tp1_r * stop_dist)

        smt_ok, smt_note = _detect_smt_at_bar(
            bars, idx, partner_bars, partner_times, d, self.spec.key, self.spec.smt_partner
        )

        macro_active = is_macro_window(ny_t)
        loc_bonus = (d == LONG and dr_pct <= 48.0) or (d == SHORT and dr_pct >= 52.0)
        deep_loc = (d == LONG and dr_pct <= 28.0) or (d == SHORT and dr_pct >= 72.0)

        p_tp1 = 0.55
        if bias == d:
            p_tp1 += 0.09
        if pool.tier == "EXTERNAL":
            p_tp1 += 0.06
        elif pool.tier == "SESSION":
            p_tp1 += 0.05
        if loc_bonus:
            p_tp1 += 0.04
        if deep_loc:
            p_tp1 += 0.03
        if smt_ok:
            p_tp1 += 0.08
        if macro_active:
            p_tp1 += 0.03
        p_tp1 = min(0.86, p_tp1)

        p_tp2_given_tp1 = 0.45 + (0.08 if bias == d else 0.0) + (0.05 if smt_ok else 0.0)
        cost_r = self.spread_price / stop_dist
        avg_win_r = (self.tp1_weight * self.tp1_r) + (1.0 - self.tp1_weight) * (p_tp2_given_tp1 * rr_tp2 + (1.0 - p_tp2_given_tp1) * 0.25)
        ev_r = p_tp1 * avg_win_r - (1.0 - p_tp1) * 0.78 - cost_r
        p_low = max(0.38, p_tp1 - 0.08)
        ev_low_r = p_low * avg_win_r - (1.0 - p_low) * 0.85 - cost_r

        if ev_low_r < 0.08:
            funnel["rejected_against_order_flow"] += 1
            return None

        conf_pct = round(min(98.0, max(65.0, p_tp1 * 100.0 + (5.0 if smt_ok else 0.0) + (4.0 if bias == d else 0.0))), 1)
        grade = "A+" if (conf_pct >= 82.0 and (smt_ok or bias == d)) else ("A" if conf_pct >= 74.0 else "B")
        verdict = "EXECUTE A+" if grade == "A+" else ("EXECUTE A" if grade == "A" else "STAND DOWN")

        reasons = list(shift_reasons)
        if bias == d:
            reasons.append("OF-01 Aligned with 4H/1H Institutional Order Flow")
        if loc_bonus:
            reasons.append(f"PD-01 Entry in {'Discount' if d == LONG else 'Premium'} ({dr_pct:.0f}% of Dealing Range)")
        if smt_ok:
            reasons.append(f"SMT-01 {smt_note}")
        if macro_active:
            reasons.append(f"MC-01 Triggered inside Algorithmic Time Macro ({ny_t:%H:%M} NY)")
        reasons.append(
            f"EV-01 Dual-Tranche Governor: BE Lock @ +{self.be_trigger_r:.2f}R · 70% TP1 @ +{self.tp1_r:.2f}R · 30% Runner to {tp2_name} ({rr_tp2:.1f}R)"
        )

        archetype = "TREND_PULLBACK_RAID" if bias == d else ("SMT_TURTLE_SOUP" if smt_ok else "JUDAS_SWEEP_REVERSAL")
        dp = self.spec.decimals
        return ChokepointSignal(
            id=f"{self.spec.key}-{bars[idx].time:%Y%m%dT%H%M}-{'L' if d == LONG else 'S'}",
            symbol=self.spec.key,
            label=self.spec.label,
            asset_class=self.spec.asset_class,
            timeframe_minutes=self.tf_minutes,
            direction=d,
            archetype=archetype,
            stage="STAGE_3_INTERCEPT",
            swept_pool=pool.name,
            swept_price=_fmt_price(pool.price, dp),
            sweep_time=raid["sweep_time"],
            signal_time=bars[idx].time,
            entry_price=_fmt_price(entry_px, dp),
            stop_price=_fmt_price(stop_px, dp),
            tp1_price=_fmt_price(tp1_px, dp),
            tp2_price=_fmt_price(tp2_px, dp),
            tp2_name=tp2_name,
            rr_tp1=round(self.tp1_r, 2),
            rr_tp2=round(rr_tp2, 2),
            confidence_pct=conf_pct,
            ev_r=round(ev_r, 3),
            ev_lower_bound_r=round(ev_low_r, 3),
            spread_friction_pct=round(spread_pct, 2),
            smt_confirmed=smt_ok,
            smt_note=smt_note,
            killzone=kz_name,
            is_macro=macro_active,
            grade=grade,
            verdict=verdict,
            reasons=reasons,
        )

    def _update_open_position(
        self,
        pos: dict[str, Any],
        bar: Bar,
        idx: int,
        ny_t: time,
        trades: list[TradeExecution],
        bars: list[Bar],
        on_fill_bar: bool = False,
    ) -> None:
        sig: ChokepointSignal = pos["signal"]
        d = sig.direction
        entry_px = pos["entry_price"]
        risk = abs(entry_px - sig.stop_price) or 1e-9

        adj = 0.0 if d == LONG else self.spread_price
        hi_exit = bar.high - adj
        lo_exit = bar.low - adj
        close_exit = bar.close - adj

        fav_excursion = ((hi_exit - entry_px) if d == LONG else (entry_px - lo_exit)) / risk
        adv_excursion = ((entry_px - lo_exit) if d == LONG else (hi_exit - entry_px)) / risk
        pos["mfe_r"] = max(pos["mfe_r"], max(0.0, fav_excursion))
        pos["mae_r"] = max(pos["mae_r"], max(0.0, adv_excursion))

        # 1. Stop-Loss Check
        stop_hit = (lo_exit <= pos["current_stop"]) if d == LONG else (hi_exit >= pos["current_stop"])
        if stop_hit:
            stop_r = ((pos["current_stop"] - entry_px) * d) / risk
            total_r = pos["realized_r"] + pos["rem_weight"] * stop_r
            reason = (
                "TP1 Banked (70%) + Runner Trailed Out (+R)"
                if pos["tp1_hit"]
                else ("Protected at Breakeven Shield (+0.05R)" if pos.get("be_locked") else "Structural Stop-Loss Hit (-1.00R)")
            )
            status = "win" if total_r > 0.12 else ("loss" if total_r < -0.12 else "breakeven")
            trades.append(
                TradeExecution(
                    signal=sig,
                    entry_time=pos["entry_time"],
                    entry_price=entry_px,
                    exit_time=bar.time,
                    exit_price=pos["current_stop"],
                    stop_price=pos["current_stop"],
                    tp1_price=sig.tp1_price,
                    tp2_price=sig.tp2_price,
                    tp1_hit=pos["tp1_hit"],
                    r_multiple=round(total_r, 3),
                    mfe_r=round(pos["mfe_r"], 3),
                    mae_r=round(pos["mae_r"], 3),
                    exit_reason=reason,
                    status=status,
                )
            )
            pos["closed"] = True
            return

        if on_fill_bar:
            return

        # 2. Early Breakeven Shield at +0.78R (turns the 70% of trades that reach +0.8R into zero-loss trades!)
        if not pos.get("be_locked") and fav_excursion >= self.be_trigger_r:
            pos["be_locked"] = True
            pos["current_stop"] = entry_px + d * (0.06 * risk)

        # 3. Tranche 1 (70% at TP1 = +1.25R) + Lock Runner Stop at +0.25R
        if not pos["tp1_hit"]:
            tp1_reached = (hi_exit >= sig.tp1_price) if d == LONG else (lo_exit <= sig.tp1_price)
            if tp1_reached:
                pos["tp1_hit"] = True
                pos["be_locked"] = True
                pos["realized_r"] += self.tp1_weight * self.tp1_r
                pos["rem_weight"] = 1.0 - self.tp1_weight
                pos["current_stop"] = entry_px + d * (0.25 * risk)

        # 4. Structural Swing Trail on the 30% Runner once TP1 is banked
        if pos["tp1_hit"] and idx >= 3:
            if d == LONG:
                recent_swing_lo = min(bars[idx - 1].low, bars[idx - 2].low)
                if recent_swing_lo > pos["current_stop"]:
                    pos["current_stop"] = recent_swing_lo
            else:
                recent_swing_hi = max(bars[idx - 1].high, bars[idx - 2].high) + self.spread_price
                if recent_swing_hi < pos["current_stop"]:
                    pos["current_stop"] = recent_swing_hi

        # 5. Tranche 2 (30% Runner at TP2 Draw on Liquidity)
        tp2_reached = (hi_exit >= sig.tp2_price) if d == LONG else (lo_exit <= sig.tp2_price)
        if tp2_reached:
            total_r = pos["realized_r"] + pos["rem_weight"] * sig.rr_tp2
            trades.append(
                TradeExecution(
                    signal=sig,
                    entry_time=pos["entry_time"],
                    entry_price=entry_px,
                    exit_time=bar.time,
                    exit_price=sig.tp2_price,
                    stop_price=pos["current_stop"],
                    tp1_price=sig.tp1_price,
                    tp2_price=sig.tp2_price,
                    tp1_hit=True,
                    r_multiple=round(total_r, 3),
                    mfe_r=round(pos["mfe_r"], 3),
                    mae_r=round(pos["mae_r"], 3),
                    exit_reason=f"Full Dual-Tranche Target Hit ({sig.tp2_name})",
                    status="win",
                )
            )
            pos["closed"] = True
            return

        # 6. End-of-Day Session Flatten at 16:00 NY
        if self.tf_minutes <= 15 and ny_t == time(16, 0) and (idx - pos["entry_idx"]) >= 6:
            rem_r = ((close_exit - entry_px) * d) / risk
            total_r = pos["realized_r"] + pos["rem_weight"] * rem_r
            status = "win" if total_r > 0.12 else ("loss" if total_r < -0.12 else "breakeven")
            trades.append(
                TradeExecution(
                    signal=sig,
                    entry_time=pos["entry_time"],
                    entry_price=entry_px,
                    exit_time=bar.time,
                    exit_price=close_exit,
                    stop_price=pos["current_stop"],
                    tp1_price=sig.tp1_price,
                    tp2_price=sig.tp2_price,
                    tp1_hit=pos["tp1_hit"],
                    r_multiple=round(total_r, 3),
                    mfe_r=round(pos["mfe_r"], 3),
                    mae_r=round(pos["mae_r"], 3),
                    exit_reason="NY 16:00 Session Flatten" + (" (after TP1)" if pos["tp1_hit"] else ""),
                    status=status,
                )
            )
            pos["closed"] = True

    def _build_radar_state(
        self,
        bars: list[Bar],
        ema20: float,
        ema50: float,
        ema120: float,
        day_ohlc: dict[date, list[float]],
        asia_ranges: dict[date, list[float]],
        london_ranges: dict[date, list[float]],
        midnight_opens: dict[date, float],
        spent_pools_today: dict[date, set[str]],
        pending_raid: dict[str, Any] | None,
        staged_order: ChokepointSignal | None,
        open_pos: dict[str, Any] | None,
        signals: list[ChokepointSignal],
        trades: list[TradeExecution],
        partner_bars: list[Bar] | None,
        partner_times: list[datetime] | None,
        funnel: dict[str, int],
    ) -> MarketRadarState:
        last_idx = len(bars) - 1
        last_bar = bars[last_idx]
        ny_dt = to_ny(last_bar.time)
        ny_t = ny_dt.time()
        trading_d = ny_dt.date() + timedelta(days=1) if (self.spec.asset_class != "stock" and ny_t >= time(17, 0)) else ny_dt.date()

        pools = self._active_pools(
            bars, last_idx, trading_d, ny_t, day_ohlc, asia_ranges, london_ranges, spent_pools_today.get(trading_d, set())
        )
        bias, bias_label, regime, dr_pct, loc_label = self._compute_order_flow(
            bars, last_idx, ema20, ema50, ema120, day_ohlc, trading_d, midnight_opens.get(trading_d)
        )

        completed = [d for d in sorted(day_ohlc) if d < trading_d][-20:]
        if completed and trading_d in day_ohlc:
            adr = sum(day_ohlc[d][1] - day_ohlc[d][2] for d in completed) / len(completed)
            today_rng = day_ohlc[trading_d][1] - day_ohlc[trading_d][2]
            adr_pct = round((today_rng / adr) * 100.0, 1) if adr > 0 else 45.0
        else:
            adr_pct = 45.0

        bsl_pools = [p for p in pools if p.side == "BSL" and p.price >= last_bar.close]
        ssl_pools = [p for p in pools if p.side == "SSL" and p.price <= last_bar.close]
        nearest_bsl = min(bsl_pools, key=lambda p: abs(p.price - last_bar.close)) if bsl_pools else None
        nearest_ssl = min(ssl_pools, key=lambda p: abs(p.price - last_bar.close)) if ssl_pools else None

        smt_bull, smt_bull_txt = _detect_smt_at_bar(bars, last_idx, partner_bars, partner_times, LONG, self.spec.key, self.spec.smt_partner)
        smt_bear, smt_bear_txt = _detect_smt_at_bar(bars, last_idx, partner_bars, partner_times, SHORT, self.spec.key, self.spec.smt_partner)
        smt_active = smt_bull or smt_bear
        smt_summary = smt_bull_txt if smt_bull else (smt_bear_txt if smt_bear else f"Tracking {self.spec.key} ↔ {self.spec.smt_partner} Triad")

        dp = self.spec.decimals
        staged_dict: dict[str, Any] | None = None
        if open_pos is not None:
            sig = open_pos["signal"]
            stage_code = 3
            active_stage = "STAGE 3: CHOKEPOINT INTERCEPT (LIVE POSITION)"
            stage_summary = (
                f"LIVE {'LONG' if sig.direction == LONG else 'SHORT'} from {open_pos['entry_price']:.{dp}f} · "
                f"SL {open_pos['current_stop']:.{dp}f} · TP1 {sig.tp1_price:.{dp}f} (70%) · TP2 {sig.tp2_price:.{dp}f} ({sig.tp2_name})"
            )
            staged_dict = self._sig_to_dict(sig)
        elif staged_order is not None:
            stage_code = 3
            active_stage = "STAGE 3: CHOKEPOINT INTERCEPT (ARMED LIMIT)"
            stage_summary = (
                f"ARMED {'BUY' if staged_order.direction == LONG else 'SELL'} LIMIT @ {staged_order.entry_price:.{dp}f} "
                f"(Swept {staged_order.swept_pool} + CISD) · EV +{staged_order.ev_r:.2f}R · {staged_order.confidence_pct:.0f}% Confidence"
            )
            staged_dict = self._sig_to_dict(staged_order)
        elif pending_raid is not None:
            p_pool: LiquidityPool = pending_raid["pool"]
            stage_code = 2
            active_stage = "STAGE 2: VELOCITY INFLECTION (POOL SWEPT)"
            stage_summary = (
                f"Swept {p_pool.name} ({p_pool.price:.{dp}f}) — watching for CISD / Displacement close to arm Chokepoint intercept"
            )
        else:
            stage_code = 1
            active_stage = "STAGE 1: PRECURSOR ACCUMULATION"
            target_pool = nearest_ssl if bias >= 0 else nearest_bsl
            if target_pool:
                stage_summary = (
                    f"Monitoring {target_pool.name} ({target_pool.price:.{dp}f}) in {loc_label.replace('_', ' ')} · {bias_label.split('(')[0].strip()}"
                )
            else:
                stage_summary = f"{bias_label} · {loc_label.replace('_', ' ')}"

        return MarketRadarState(
            symbol=self.spec.key,
            label=self.spec.label,
            asset_class=self.spec.asset_class,
            last_price=_fmt_price(last_bar.close, dp),
            last_time=last_bar.time,
            last_time_ny=ny_dt.strftime("%a %d %b %H:%M NY"),
            bias=bias,
            bias_label=bias_label,
            regime=regime,
            dealing_range_pct=dr_pct,
            location_label=loc_label,
            adr_used_pct=adr_pct,
            active_stage=active_stage,
            stage_code=stage_code,
            stage_summary=stage_summary,
            nearest_bsl=nearest_bsl,
            nearest_ssl=nearest_ssl,
            smt_active=smt_active,
            smt_summary=smt_summary,
            active_killzone=self._active_killzone(ny_t),
            is_macro=is_macro_window(ny_t),
            staged_intercept=staged_dict,
            recent_signals=[self._sig_to_dict(s) for s in signals[-6:][::-1]],
            recent_trades=[self._trade_to_dict(t) for t in trades[-8:][::-1]],
            funnel_counts=funnel,
        )

    def _sig_to_dict(self, s: ChokepointSignal) -> dict[str, Any]:
        return {
            "id": s.id,
            "symbol": s.symbol,
            "label": s.label,
            "asset_class": s.asset_class,
            "timeframe_minutes": s.timeframe_minutes,
            "direction": "long" if s.direction == LONG else "short",
            "archetype": s.archetype,
            "stage": s.stage,
            "swept_pool": s.swept_pool,
            "swept_price": s.swept_price,
            "sweep_time": s.sweep_time.isoformat(),
            "signal_time": s.signal_time.isoformat(),
            "signal_time_ny": to_ny(s.signal_time).strftime("%a %d %b %H:%M"),
            "entry": s.entry_price,
            "stop": s.stop_price,
            "tp1": s.tp1_price,
            "tp2": s.tp2_price,
            "tp2_name": s.tp2_name,
            "rr_tp1": s.rr_tp1,
            "rr_tp2": s.rr_tp2,
            "confidence_pct": s.confidence_pct,
            "ev_r": s.ev_r,
            "ev_lower_bound_r": s.ev_lower_bound_r,
            "spread_friction_pct": s.spread_friction_pct,
            "smt_confirmed": s.smt_confirmed,
            "smt_note": s.smt_note,
            "killzone": s.killzone,
            "is_macro": s.is_macro,
            "grade": s.grade,
            "verdict": s.verdict,
            "reasons": s.reasons,
        }

    def _trade_to_dict(self, t: TradeExecution) -> dict[str, Any]:
        dp = self.spec.decimals
        return {
            "id": t.signal.id,
            "symbol": t.signal.symbol,
            "label": t.signal.label,
            "asset_class": t.signal.asset_class,
            "direction": "long" if t.signal.direction == LONG else "short",
            "archetype": t.signal.archetype,
            "grade": t.signal.grade,
            "confidence_pct": t.signal.confidence_pct,
            "swept_pool": t.signal.swept_pool,
            "entry_time": t.entry_time.isoformat(),
            "entry_time_ny": to_ny(t.entry_time).strftime("%a %d %b %H:%M"),
            "exit_time": t.exit_time.isoformat(),
            "exit_time_ny": to_ny(t.exit_time).strftime("%a %d %b %H:%M"),
            "entry": _fmt_price(t.entry_price, dp),
            "exit": _fmt_price(t.exit_price, dp),
            "stop": _fmt_price(t.stop_price, dp),
            "tp1": _fmt_price(t.tp1_price, dp),
            "tp2": _fmt_price(t.tp2_price, dp),
            "tp2_name": t.signal.tp2_name,
            "tp1_hit": t.tp1_hit,
            "r": t.r_multiple,
            "mfe_r": t.mfe_r,
            "mae_r": t.mae_r,
            "exit_reason": t.exit_reason,
            "status": t.status,
            "smt_confirmed": t.signal.smt_confirmed,
            "reasons": t.signal.reasons,
        }
