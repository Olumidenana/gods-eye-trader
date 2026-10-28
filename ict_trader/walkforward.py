"""Walk-forward validation: tune on one period, judge on the next, unseen one.

Only the out-of-sample (test) results count. A strategy whose in-sample
numbers look great but whose out-of-sample numbers collapse is curve-fit.
"""

from __future__ import annotations

import itertools
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .backtest import Backtester
from .config import Config, with_strategy
from .metrics import Trade, summarize
from .news import NewsFilter
from .sessions import Bar

WARMUP = timedelta(days=4)


def add_months(ts: datetime, months: int) -> datetime:
    m = ts.month - 1 + months
    return ts.replace(year=ts.year + m // 12, month=m % 12 + 1, day=1, hour=0, minute=0, second=0, microsecond=0)


@dataclass
class WindowResult:
    train_start: datetime
    test_start: datetime
    test_end: datetime
    params: dict[str, Any] | None
    train_stats: dict[str, Any] | None
    test_stats: dict[str, Any] | None


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]
    oos_trades: list[Trade]
    oos_stats: dict[str, Any]


def _slice(bars: list[Bar], times: list[datetime], start: datetime, end: datetime) -> list[Bar]:
    return bars[bisect_left(times, start - WARMUP): bisect_left(times, end)]


def param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))] or [{}]


def walk_forward(
    bars: list[Bar],
    config: Config,
    grid: dict[str, list[Any]],
    train_months: int = 6,
    test_months: int = 1,
    min_trades: int = 20,
    news: NewsFilter | None = None,
) -> WalkForwardResult:
    if not bars:
        raise ValueError("No bars")
    times = [b.time for b in bars]
    combos = param_grid(grid)
    windows: list[WindowResult] = []
    oos: list[Trade] = []

    # Start at the first whole month of data.
    first = bars[0].time
    train_start = add_months(first, 0 if first == add_months(first, 0) else 1)
    while True:
        test_start = add_months(train_start, train_months)
        test_end = add_months(test_start, test_months)
        if test_end > bars[-1].time + timedelta(days=1):
            break

        best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
        train_bars = _slice(bars, times, train_start, test_start)
        for params in combos:
            bt = Backtester(with_strategy(config, **params), news)
            stats = bt.run(train_bars, trade_from=train_start, trade_until=test_start).stats
            if stats["trades"] < min_trades:
                continue
            if best is None or stats["expectancy_r"] > best[0]:
                best = (stats["expectancy_r"], params, stats)

        if best is None:
            windows.append(WindowResult(train_start, test_start, test_end, None, None, None))
        else:
            _, params, train_stats = best
            bt = Backtester(with_strategy(config, **params), news)
            test = bt.run(_slice(bars, times, test_start, test_end), trade_from=test_start, trade_until=test_end)
            windows.append(WindowResult(train_start, test_start, test_end, params, train_stats, test.stats))
            oos.extend(test.trades)

        train_start = add_months(train_start, test_months)

    equity, curve = config.initial_equity, []
    for t in oos:
        equity += t.pnl
        curve.append((t.exit_time, equity))
    return WalkForwardResult(windows, oos, summarize(oos, curve, config.initial_equity))


def format_walk_forward(result: WalkForwardResult) -> str:
    lines = ["Window (test month)   Params                                   IS exp R   OOS trades  OOS exp R"]
    for w in result.windows:
        last = w.test_end - timedelta(days=1)
        label = f"{w.test_start:%Y-%m}" + ("" if (last.year, last.month) == (w.test_start.year, w.test_start.month) else f" .. {last:%Y-%m}")
        if w.params is None:
            lines.append(f"{label:<21} no parameter set met the minimum trade count; window skipped")
            continue
        params = ", ".join(f"{k}={v}" for k, v in w.params.items()) or "(defaults)"
        lines.append(
            f"{label:<21} {params[:40]:<40} {w.train_stats['expectancy_r']:+8.3f}   {w.test_stats['trades']:>10}  {w.test_stats['expectancy_r']:+8.3f}"
        )
    return "\n".join(lines)
