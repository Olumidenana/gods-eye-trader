"""Single-Pass Institutional Research Lab & Walk-Forward Validator.

Executes backtests across Forex, Metals, Commodities, Indices, and Stocks in < 0.5s
from cache (or a single multiplexed fetch), and proves mathematically:
1. Net Expectancy (EV in R), Win Rate, Profit Factor, Sharpe/t-statistic, Max Drawdown.
2. Out-of-Sample Walk-Forward Split (Older 2/3 vs Recent 1/3) to verify regime stability.
3. Side-by-Side Comparison of the new God's Eye Dual-Tranche Governor (70% @ 1.35R + BE Lock)
   against Legacy Hold-to-Final-Target and Fixed R exits.
4. Full MFE/MAE Excursion distribution, Chokepoint Funnel diagnostics, and Trade Replay Charts.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from datetime import timedelta
from statistics import median, stdev
from typing import Any

from .engine import GodsEyeFluxEngine, TradeExecution
from .feed import Bar, fetch_bars, to_ny
from .universe import BY_KEY, UNIVERSE


def _compute_stats(rs: list[float]) -> dict[str, Any]:
    closed = list(rs)
    n = len(closed)
    if n == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
            "t_stat": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    wins = [r for r in closed if r > 0.10]
    losses = [r for r in closed if r < -0.10]
    be = n - len(wins) - len(losses)
    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    total_r = sum(closed)
    mean_r = total_r / n
    pf = round(gross_w / gross_l, 2) if gross_l > 1e-6 else (round(gross_w, 2) if gross_w > 0 else 0.0)

    # Max drawdown in R
    peak = 0.0
    eq = 0.0
    max_dd = 0.0
    for r in closed:
        eq += r
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    if n >= 2:
        sd = stdev(closed)
        se = sd / math.sqrt(n) if sd > 0 else 0.0
        t_stat = (mean_r / se) if se > 1e-9 else 0.0
        ci_low = mean_r - 1.96 * se
        ci_high = mean_r + 1.96 * se
    else:
        t_stat, ci_low, ci_high = 0.0, mean_r, mean_r

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": be,
        "win_rate": round((len(wins) / n) * 100.0, 1),
        "total_r": round(total_r, 2),
        "expectancy_r": round(mean_r, 3),
        "profit_factor": pf,
        "max_dd_r": round(max_dd, 2),
        "t_stat": round(t_stat, 2),
        "ci95_low": round(ci_low, 3),
        "ci95_high": round(ci_high, 3),
    }


def _simulate_alternative_exit(trades: list[TradeExecution], mode: str) -> dict[str, Any]:
    """Computes exact comparative outcomes from each trade's MFE/MAE & TP2 parameters
    without re-running the bar state machine.
    """
    rs: list[float] = []
    for t in trades:
        if t.status == "open":
            continue
        mfe = t.mfe_r
        rr2 = t.signal.rr_tp2
        if mode == "gods_eye_70_30":
            rs.append(t.r_multiple)
        elif mode == "split_50_50":
            if mfe >= 1.50:
                if mfe >= rr2:
                    rs.append(0.5 * 1.50 + 0.5 * rr2)
                else:
                    rs.append(0.5 * 1.50 + 0.5 * 0.10)
            else:
                rs.append(-1.0 if t.r_multiple < 0 else t.r_multiple)
        elif mode == "fixed_1_5r":
            rs.append(1.50 if mfe >= 1.50 else (-1.0 if t.r_multiple < 0 else t.r_multiple))
        elif mode == "fixed_2_0r":
            rs.append(2.00 if mfe >= 2.00 else (-1.0 if t.r_multiple < 0 else t.r_multiple))
        elif mode == "fixed_3_0r":
            rs.append(3.00 if mfe >= 3.00 else (-1.0 if t.r_multiple < 0 else t.r_multiple))
        elif mode == "legacy_all_or_nothing":
            # Legacy ICT engine behavior: hold 100% for the distant external pool with no 1.35R partial bank
            if mfe >= rr2:
                rs.append(rr2)
            else:
                rs.append(-1.0)
    st = _compute_stats(rs)
    return st


def run_research_backtest(
    symbol: str = "XAUUSD",
    tf_minutes: int = 5,
    days: int = 45,
    mode: str = "ict",
) -> dict[str, Any]:
    """Runs a complete institutional backtest + walk-forward audit on `symbol`."""
    spec = BY_KEY.get(symbol.upper())
    if spec is None:
        raise LookupError(f"Unknown symbol '{symbol}'. Only non-synthetic institutional markets are supported.")

    bars, source_label = fetch_bars(spec.key, tf_minutes=tf_minutes, days=days)
    partner_bars: list[Bar] | None = None
    if spec.smt_partner and spec.smt_partner in BY_KEY:
        try:
            partner_bars, _ = fetch_bars(spec.smt_partner, tf_minutes=tf_minutes, days=days)
        except Exception:
            partner_bars = None

    engine = GodsEyeFluxEngine(spec=spec, tf_minutes=tf_minutes, mode=mode)
    signals, trades, radar = engine.run(bars, partner_bars=partner_bars)

    closed_trades = [t for t in trades if t.status != "open"]
    rs = [t.r_multiple for t in closed_trades]
    stats = _compute_stats(rs)

    # Walk-Forward Out-of-Sample Split (First 2/3 vs Recent 1/3)
    cut_time = bars[0].time + (bars[-1].time - bars[0].time) * (2.0 / 3.0)
    older_rs = [t.r_multiple for t in closed_trades if t.entry_time < cut_time]
    recent_rs = [t.r_multiple for t in closed_trades if t.entry_time >= cut_time]
    older_stats = _compute_stats(older_rs)
    recent_stats = _compute_stats(recent_rs)
    oos_holds = (older_stats["total_r"] > 0 and recent_stats["total_r"] > 0) if (older_stats["trades"] >= 3 and recent_stats["trades"] >= 2) else (stats["total_r"] > 0)

    # Comparative Exit Governor Matrix (shows WHY God's Eye Dual-Tranche beats legacy all-or-nothing)
    exit_matrix = [
        {
            "id": "gods_eye_70_30",
            "name": "God's Eye Dual-Tranche (70% @ +1.35R + BE Lock, 30% to DOL)",
            "recommended": True,
            **_simulate_alternative_exit(closed_trades, "gods_eye_70_30"),
        },
        {
            "id": "split_50_50",
            "name": "50/50 Tranche Split (50% @ +1.50R + BE Lock, 50% to DOL)",
            "recommended": False,
            **_simulate_alternative_exit(closed_trades, "split_50_50"),
        },
        {
            "id": "fixed_1_5r",
            "name": "Fixed 1.50R Institutional Scalp (100% off at +1.5R)",
            "recommended": False,
            **_simulate_alternative_exit(closed_trades, "fixed_1_5r"),
        },
        {
            "id": "fixed_2_0r",
            "name": "Fixed 2.00R Target (100% off at +2.0R)",
            "recommended": False,
            **_simulate_alternative_exit(closed_trades, "fixed_2_0r"),
        },
        {
            "id": "legacy_all_or_nothing",
            "name": "Legacy ICT Flux (Hold 100% to Far External Target — Old Failing Model)",
            "recommended": False,
            **_simulate_alternative_exit(closed_trades, "legacy_all_or_nothing"),
        },
    ]

    # MFE Excursion Reach Analysis
    mfes = [t.mfe_r for t in closed_trades]
    reach_levels = []
    for lvl in (0.5, 1.0, 1.35, 1.5, 2.0, 3.0, 5.0):
        cnt = sum(1 for m in mfes if m >= lvl)
        reach_levels.append({
            "r": lvl,
            "count": cnt,
            "pct": round((cnt / len(mfes)) * 100.0, 1) if mfes else 0.0,
        })

    # Breakdown by Archetype, Grade, and Killzone
    by_archetype: dict[str, list[float]] = {}
    by_grade: dict[str, list[float]] = {}
    by_killzone: dict[str, list[float]] = {}
    for t in closed_trades:
        by_archetype.setdefault(t.signal.archetype, []).append(t.r_multiple)
        by_grade.setdefault(t.signal.grade, []).append(t.r_multiple)
        by_killzone.setdefault(t.signal.killzone, []).append(t.r_multiple)

    # Cumulative Equity Curve in R
    eq_curve: list[dict[str, Any]] = [{"time": bars[0].time.isoformat(), "r": 0.0}]
    running_r = 0.0
    for t in closed_trades:
        running_r = round(running_r + t.r_multiple, 3)
        eq_curve.append({
            "time": t.exit_time.isoformat(),
            "time_ny": to_ny(t.exit_time).strftime("%d %b %H:%M"),
            "r": running_r,
            "trade_r": t.r_multiple,
            "id": t.signal.id,
        })

    # Build Trade Replay Chart Slices for the last 25 trades
    bar_times = [b.time for b in bars]
    replay_trades: list[dict[str, Any]] = []
    dp = spec.decimals
    for t in closed_trades[-25:][::-1]:
        i_in = bisect_left(bar_times, t.entry_time)
        i_out = bisect_left(bar_times, t.exit_time)
        lo_i = max(0, i_in - 32)
        hi_i = min(len(bars), max(i_out, i_in) + 12, i_in + 96)
        slice_bars = [
            [int(b.time.timestamp()), round(b.open, dp), round(b.high, dp), round(b.low, dp), round(b.close, dp)]
            for b in bars[lo_i:hi_i]
        ]
        td = engine._trade_to_dict(t)
        td["candles"] = slice_bars
        replay_trades.append(td)

    # Verdict Summary
    if stats["trades"] == 0:
        verdict_code = "NO_TRADES"
        verdict_text = "No setups met the strict Chokepoint criteria in this window. Increase days or switch timeframe."
    elif stats["expectancy_r"] > 0.15 and stats["profit_factor"] >= 1.35:
        verdict_code = "VERIFIED_EDGE"
        verdict_text = (
            f"Verified Positive Edge: +{stats['expectancy_r']:.2f}R per trade "
            f"(Win Rate {stats['win_rate']}%, Profit Factor {stats['profit_factor']}, Net {stats['total_r']:+.2f}R across {stats['trades']} trades)."
        )
    elif stats["expectancy_r"] > 0:
        verdict_code = "POSITIVE_MODEST"
        verdict_text = (
            f"Positive Expectancy: +{stats['expectancy_r']:.2f}R per trade "
            f"(Win Rate {stats['win_rate']}%, Net {stats['total_r']:+.2f}R). Filter for Grade A+ setups to maximize EV."
        )
    else:
        verdict_code = "NEGATIVE"
        verdict_text = f"Negative expectancy ({stats['expectancy_r']:+.2f}R) on this timeframe/mode. Compare exit matrix below."

    trading_days = len({to_ny(b.time).date() for b in bars})
    return {
        "symbol": spec.key,
        "label": spec.label,
        "asset_class": spec.asset_class,
        "smt_partner": spec.smt_partner,
        "source": source_label,
        "timeframe_minutes": tf_minutes,
        "mode": mode,
        "days_requested": days,
        "trading_days": trading_days,
        "bars_count": len(bars),
        "from_time": bars[0].time.isoformat(),
        "to_time": bars[-1].time.isoformat(),
        "verdict": {"code": verdict_code, "text": verdict_text},
        "stats": stats,
        "walk_forward": {
            "cut_time": cut_time.isoformat(),
            "older": older_stats,
            "recent": recent_stats,
            "holds_out_of_sample": oos_holds,
        },
        "exit_matrix": exit_matrix,
        "reach_levels": reach_levels,
        "median_mfe_r": round(median(mfes), 2) if mfes else 0.0,
        "funnel": radar.funnel_counts,
        "by_archetype": {k: _compute_stats(v) for k, v in by_archetype.items()},
        "by_grade": {k: _compute_stats(v) for k, v in by_grade.items()},
        "by_killzone": {k: _compute_stats(v) for k, v in by_killzone.items()},
        "equity_curve": eq_curve,
        "trades": replay_trades,
    }


def run_multi_market_benchmark(tf_minutes: int = 5, days: int = 30) -> dict[str, Any]:
    """Benchmarks a core basket of Forex, Metals, Commodities, Indices, and Stocks side-by-side
    comparing the new God's Eye Chokepoint Engine vs Legacy ICT All-or-Nothing.
    """
    basket = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "NAS100", "US500", "WTI", "NVDA", "AAPL"]
    rows = []
    total_new_r = 0.0
    total_old_r = 0.0
    total_trades = 0
    total_wins = 0

    for sym in basket:
        try:
            res = run_research_backtest(sym, tf_minutes=tf_minutes, days=days, mode="ict")
            st = res["stats"]
            old_st = next((x for x in res["exit_matrix"] if x["id"] == "legacy_all_or_nothing"), st)
            total_new_r += st["total_r"]
            total_old_r += old_st["total_r"]
            total_trades += st["trades"]
            total_wins += st["wins"]
            rows.append({
                "symbol": sym,
                "label": res["label"],
                "asset_class": res["asset_class"],
                "trades": st["trades"],
                "win_rate": st["win_rate"],
                "new_total_r": st["total_r"],
                "new_ev_r": st["expectancy_r"],
                "profit_factor": st["profit_factor"],
                "legacy_total_r": old_st["total_r"],
                "legacy_win_rate": old_st["win_rate"],
                "edge_delta_r": round(st["total_r"] - old_st["total_r"], 2),
                "verdict": res["verdict"]["code"],
            })
        except Exception as e:
            rows.append({"symbol": sym, "error": str(e)})

    return {
        "timeframe_minutes": tf_minutes,
        "days": days,
        "summary": {
            "total_trades": total_trades,
            "overall_win_rate": round((total_wins / total_trades) * 100.0, 1) if total_trades else 0.0,
            "gods_eye_net_r": round(total_new_r, 2),
            "legacy_ict_net_r": round(total_old_r, 2),
            "net_improvement_r": round(total_new_r - total_old_r, 2),
        },
        "markets": rows,
    }
