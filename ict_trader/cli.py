"""Command line: demo, backtest, walkforward, live."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .backtest import Backtester
from .config import Config, load_config
from .data import load_bars_csv, synthetic_bars
from .metrics import format_summary
from .news import NewsFilter, load_calendar_csv
from .report import render_report
from .sessions import utc
from .walkforward import format_walk_forward, walk_forward

DEFAULT_GRID = {"min_sweep_pips": [0.5, 1.5], "max_confirm_bars": [2, 4], "min_rr": [1.5, 2.0]}


def _config(args: argparse.Namespace) -> Config:
    return load_config(args.config) if args.config else Config()


def _news(args: argparse.Namespace, config: Config) -> NewsFilter | None:
    if not args.news:
        if config.news.enabled:
            print("note: no --news calendar given; backtest ignores news (NF-1 inactive)", file=sys.stderr)
        return None
    return NewsFilter(config.news, config.instrument, load_calendar_csv(args.news))


def _write_report(path: str | None, title: str, result) -> None:  # type: ignore[no-untyped-def]
    if path:
        Path(path).write_text(render_report(title, result.stats, result.trades, result.equity_curve, result.decisions), encoding="utf-8")
        print(f"report written to {path}")


def cmd_demo(args: argparse.Namespace) -> int:
    config = _config(args)
    bars = synthetic_bars(utc(2025, 1, 5), days=args.days, timeframe_minutes=config.strategy.timeframe_minutes)
    print(f"Synthetic random-walk data: {len(bars)} bars. It has no edge by design, so these numbers only prove the pipeline runs.\n")
    result = Backtester(config).run(bars)
    print(format_summary(result.stats))
    _write_report(args.report, "Demo backtest (synthetic data)", result)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    config = _config(args)
    bars = load_bars_csv(args.data, tz=args.tz)
    result = Backtester(config, _news(args, config)).run(bars)
    print(f"{config.instrument.symbol}: {len(bars)} bars, {bars[0].time:%Y-%m-%d} to {bars[-1].time:%Y-%m-%d}\n")
    print(format_summary(result.stats))
    _write_report(args.report, f"{config.instrument.symbol} backtest", result)
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    config = _config(args)
    bars = load_bars_csv(args.data, tz=args.tz)
    grid = json.loads(Path(args.grid).read_text()) if args.grid else DEFAULT_GRID
    result = walk_forward(bars, config, grid, args.train_months, args.test_months, args.min_trades, _news(args, config))
    print(format_walk_forward(result))
    print("\nOut-of-sample only (the numbers that matter):")
    print(format_summary(result.oos_stats))
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .live import Journal, LiveRunner

    config = _config(args)
    if args.platform:
        config = replace(config, live=replace(config.live, platform=args.platform))
    if not args.news and config.news.enabled:
        print("error: live trading requires --news calendar (or disable [news] in config)", file=sys.stderr)
        return 2
    news = NewsFilter(config.news, config.instrument, load_calendar_csv(args.news)) if args.news else None
    if config.live.platform == "mt4":
        from .broker.mt4 import MT4BridgeBroker

        broker = MT4BridgeBroker(config)
    elif config.live.platform == "mt5":
        from .broker.mt5 import MT5Broker

        broker = MT5Broker(config)  # type: ignore[assignment]
    else:
        print(f"error: unknown live.platform {config.live.platform!r} (use mt4 or mt5)", file=sys.stderr)
        return 2
    broker.connect()
    if broker.is_real_account() and not args.allow_real_account:
        print("error: connected to a REAL-money account. Validate on demo first; pass --allow-real-account to override.", file=sys.stderr)
        return 2
    runner = LiveRunner(config, broker, news, Journal(config.live.journal_path), dry_run=not args.execute)
    runner.run()
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    import tomllib
    from .broker.mt5 import MT5Broker
    from .live import Journal
    from .portfolio import PortfolioRunner
    from .notifier import Notifier

    portfolio_path = args.config or "examples/portfolio.toml"
    with open(portfolio_path, "rb") as f:
        data = tomllib.load(f)
    symbol_files = data.get("symbols", [])
    if not symbol_files:
        print("error: portfolio config must list 'symbols' paths", file=sys.stderr)
        return 2
    configs = [load_config(s) for s in symbol_files]
    news = NewsFilter(configs[0].news, configs[0].instrument, load_calendar_csv(args.news)) if args.news else None
    journal = Journal(configs[0].live.journal_path)
    notifier = Notifier()

    def broker_factory(cfg: Config):
        return MT5Broker(cfg)

    first_broker = MT5Broker(configs[0])
    first_broker.connect()
    if first_broker.is_real_account() and not args.allow_real_account:
        print("error: connected to a REAL-money account. Validate on demo first; pass --allow-real-account to override.", file=sys.stderr)
        return 2

    runner = PortfolioRunner(
        configs=configs,
        broker_factory=broker_factory,
        news=news,
        journal=journal,
        dry_run=not args.execute,
        notifier=notifier,
        max_portfolio_open_positions=data.get("max_portfolio_open_positions", 3),
    )
    runner.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="ict_trader", description="ICT Time-and-Price trading system")
    parser.add_argument("--config", help="TOML config file (defaults built in)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="run the full pipeline on synthetic data")
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--report", help="write an HTML report here")
    p.set_defaults(func=cmd_demo)

    for name, func, help_ in (("backtest", cmd_backtest, "backtest on a CSV of bars"), ("walkforward", cmd_walkforward, "walk-forward validation")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--data", required=True, help="CSV of OHLC bars")
        p.add_argument("--tz", default="UTC", help="timezone of naive timestamps in the CSV")
        p.add_argument("--news", help="economic calendar CSV")
        p.set_defaults(func=func)
        if name == "backtest":
            p.add_argument("--report", help="write an HTML report here")
        else:
            p.add_argument("--grid", help="JSON file mapping strategy params to candidate values")
            p.add_argument("--train-months", type=int, default=6)
            p.add_argument("--test-months", type=int, default=1)
            p.add_argument("--min-trades", type=int, default=20)

    p = sub.add_parser("live", help="run against MetaTrader 4 or 5 (dry run unless --execute)")
    p.add_argument("--platform", choices=["mt4", "mt5"], help="override live.platform from the config")
    p.add_argument("--news", help="economic calendar CSV")
    p.add_argument("--execute", action="store_true", help="actually send orders (otherwise log what it would do)")
    p.add_argument("--allow-real-account", action="store_true", help="permit a real-money account")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("portfolio", help="run autonomous multi-market scanner across all instruments")
    p.add_argument("--news", help="economic calendar CSV")
    p.add_argument("--execute", action="store_true", help="actually send orders (otherwise log what it would do)")
    p.add_argument("--allow-real-account", action="store_true", help="permit a real-money account")
    p.set_defaults(func=cmd_portfolio)

    args = parser.parse_args(argv)
    return args.func(args)
