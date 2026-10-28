"""Performance statistics, including an honest check of sample size."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, stdev
from typing import Any


@dataclass(frozen=True)
class Trade:
    symbol: str
    direction: int
    entry_time: datetime
    entry_price: float
    stop: float
    target: float
    lots: float
    risk_amount: float
    exit_time: datetime
    exit_price: float
    exit_reason: str
    pnl: float
    r_multiple: float
    reasons: tuple[str, ...]


def max_drawdown_pct(equity: list[float]) -> float:
    peak, worst = float("-inf"), 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            worst = max(worst, (peak - e) / peak * 100)
    return worst


_T95 = ((1, 12.71), (2, 4.30), (3, 3.18), (4, 2.78), (5, 2.57), (6, 2.45), (7, 2.36), (8, 2.31), (9, 2.26), (10, 2.23),
        (12, 2.18), (15, 2.13), (20, 2.09), (25, 2.06), (30, 2.04), (60, 2.00))


def t_crit95(df: int) -> float:
    """Two-sided 95% Student-t value: a handful of trades needs a far wider range than the textbook 1.96."""
    return next((v for d, v in _T95 if df <= d), 1.96)


def summarize(trades: list[Trade], equity_curve: list[tuple[datetime, float]], initial_equity: float) -> dict[str, Any]:
    n = len(trades)
    rs = [t.r_multiple for t in trades]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    final = equity_curve[-1][1] if equity_curve else initial_equity

    streak = longest = 0
    for t in trades:
        streak = streak + 1 if t.pnl <= 0 else 0
        longest = max(longest, streak)

    expectancy = mean(rs) if rs else 0.0
    sd = stdev(rs) if n >= 2 else 0.0
    se = sd / math.sqrt(n) if n >= 2 else 0.0
    t_stat = expectancy / se if se > 0 else 0.0
    # How sure the average is: luck alone could put the true expectancy anywhere in this 95% range, and the number of
    # trades it would take for an edge of this size to clear t = 2 (None when there is no positive edge to confirm).
    crit = t_crit95(n - 1) if n >= 2 else 0.0
    ci95 = [expectancy - crit * se, expectancy + crit * se]
    trades_needed = math.ceil((2 * sd / expectancy) ** 2) if expectancy > 0 and sd > 0 else None

    return {
        "trades": n,
        "win_rate_pct": len(wins) / n * 100 if n else 0.0,
        "avg_win_r": mean(t.r_multiple for t in wins) if wins else 0.0,
        "avg_loss_r": mean(t.r_multiple for t in losses) if losses else 0.0,
        "expectancy_r": expectancy,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0),
        "net_pnl": final - initial_equity,
        "return_pct": (final - initial_equity) / initial_equity * 100,
        "max_drawdown_pct": max_drawdown_pct([e for _, e in equity_curve]),
        "longest_losing_streak": longest,
        "expectancy_t_stat": t_stat,
        "expectancy_se": se,
        "expectancy_ci95": ci95,
        "trades_needed": trades_needed,
        # Rough bar for "this edge is probably not luck": at least 30 trades and
        # a mean R more than ~2 standard errors above zero.
        "statistically_meaningful": n >= 30 and t_stat > 2.0,
    }


def format_summary(stats: dict[str, Any]) -> str:
    lines = [
        f"Trades               {stats['trades']}",
        f"Win rate             {stats['win_rate_pct']:.1f}%",
        f"Avg win / loss       {stats['avg_win_r']:+.2f}R / {stats['avg_loss_r']:+.2f}R",
        f"Expectancy           {stats['expectancy_r']:+.3f}R per trade",
        f"Profit factor        {stats['profit_factor']:.2f}",
        f"Net P&L              {stats['net_pnl']:+,.2f} ({stats['return_pct']:+.2f}%)",
        f"Max drawdown         {stats['max_drawdown_pct']:.2f}%",
        f"Longest losing run   {stats['longest_losing_streak']}",
        f"Expectancy t-stat    {stats['expectancy_t_stat']:.2f}",
    ]
    if not stats["statistically_meaningful"]:
        lines.append("WARNING: sample too small or edge not distinguishable from zero; do not go live on this result.")
    return "\n".join(lines)
