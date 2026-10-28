"""Multi-Market Portfolio Engine for concurrent multi-asset live execution.

Runs a unified event loop over multiple markets (e.g. GBPUSD, EURUSD, XAUUSD, US30),
tracking individual liquidity sweeps, session windows, and FVGs in parallel,
governed by a shared institutional risk manager and real-time mobile notifier.
"""

from __future__ import annotations

import logging
import time as systime
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .broker.base import Broker, BrokerPosition
from .config import Config
from .live import Journal
from .news import NewsFilter
from .notifier import Notifier
from .risk import RiskManager, lots_for_risk
from .sessions import Bar, flatten_deadline
from .models import make_strategy
from .strategy import LONG, Decision, Signal, TimeAndPriceStrategy

log = logging.getLogger("ict_trader.portfolio")


@dataclass
class SymbolContext:
    config: Config
    broker: Broker
    strategy: TimeAndPriceStrategy
    last_bar_time: datetime | None = None
    known: dict[int, BrokerPosition] = field(default_factory=dict)
    flushed_decisions: int = 0
    tf: timedelta = field(default_factory=lambda: timedelta(minutes=5))


class PortfolioRunner:
    def __init__(
        self,
        configs: list[Config],
        broker_factory: Callable[[Config], Broker],
        news: NewsFilter | None,
        journal: Journal,
        dry_run: bool = True,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        notifier: Notifier | None = None,
        max_portfolio_open_positions: int = 3,
    ):
        if not configs:
            raise ValueError("PortfolioRunner requires at least one symbol config")
        self.configs = configs
        self.news = news
        self.journal = journal
        self.dry_run = dry_run
        self.clock = clock
        self.notifier = notifier or Notifier()
        self.max_portfolio_open_positions = max_portfolio_open_positions

        # Shared risk manager anchored to primary config
        primary_cfg = configs[0]
        self.primary_cfg = primary_cfg

        # Initialize symbol contexts
        self.contexts: dict[str, SymbolContext] = {}
        for cfg in configs:
            broker = broker_factory(cfg)
            strat = make_strategy(cfg.strategy, cfg.instrument)
            tf = timedelta(minutes=cfg.strategy.timeframe_minutes)
            self.contexts[cfg.instrument.symbol] = SymbolContext(
                config=cfg,
                broker=broker,
                strategy=strat,
                tf=tf,
            )

        self.risk: RiskManager | None = None

    def start(self, warmup_bars: int = 1500) -> None:
        log.info("Starting Multi-Market Portfolio Engine across %d symbol(s)...", len(self.contexts))
        first_broker = next(iter(self.contexts.values())).broker
        first_broker.connect()
        initial_equity = first_broker.equity()
        self.risk = RiskManager(self.primary_cfg.risk, initial_equity, self.primary_cfg.strategy.day_rollover)

        for sym, ctx in self.contexts.items():
            if ctx.broker != first_broker:
                ctx.broker.connect()
            history = ctx.broker.closed_bars(warmup_bars)
            for bar in history:
                ctx.strategy.on_bar(bar)
            ctx.flushed_decisions = len(ctx.strategy.decisions)
            ctx.last_bar_time = history[-1].time if history else None
            ctx.known = {p.ticket: p for p in ctx.broker.open_positions()}
            log.info("Symbol %s warmed up on %d bars (%d active position(s))", sym, len(history), len(ctx.known))

        mode = "DRY RUN (Paper Trading)" if self.dry_run else "REAL AUTOMATED EXECUTION"
        self._note("PORTFOLIO_START", f"{mode} across symbols: {list(self.contexts.keys())}; Equity: ${initial_equity:,.2f}")

    def _note(self, rule: str, msg: str) -> None:
        self.journal.decision(Decision(self.clock(), rule, msg))

    def _total_open_positions(self) -> int:
        return sum(len(ctx.broker.open_positions()) for ctx in self.contexts.values())

    def poll(self) -> bool:
        """One multi-symbol scan cycle. Returns False if kill switch fired."""
        assert self.risk is not None, "call start() first"
        now = self.clock()

        # Check Kill Switch
        if Path(self.primary_cfg.live.kill_switch_file).exists():
            for ctx in self.contexts.values():
                for p in ctx.broker.open_positions():
                    self._close(ctx, p, "KILL kill switch")
            self._note("KILL", "kill switch file found; portfolio flattened and halting")
            return False

        first_broker = next(iter(self.contexts.values())).broker
        current_equity = first_broker.equity()
        self.risk.mark(now, current_equity)

        # 1. Manage existing positions & book exits across all symbols
        for sym, ctx in self.contexts.items():
            positions = ctx.broker.open_positions()
            active_tickets = {p.ticket for p in positions}
            # Check closed positions
            for ticket in [t for t in ctx.known if t not in active_tickets]:
                pnl = ctx.broker.realized_pnl(ticket)
                if pnl is not None:
                    self.risk.record_close(pnl)
                    self.journal.closed(ticket, now, pnl)
                    self._note("CLOSED", f"{sym} ticket {ticket} closed, P&L {pnl:+.2f}")
                    self.notifier.trade_closed(sym, ticket, pnl, "Broker exit (SL/TP)")
                del ctx.known[ticket]
            for p in positions:
                ctx.known.setdefault(p.ticket, p)

            # Check scheduled flatten times & news blackouts
            for p in positions:
                deadline = flatten_deadline(p.open_time, ctx.config.strategy.flatten_time)
                if deadline is not None and now >= deadline:
                    self._close(ctx, p, "EX-1 flatten time")
                elif self.news and (ev := self.news.should_flatten(now)):
                    self._close(ctx, p, f"NF-2 flatten before {ev.currency} {ev.title}")

        # 2. Scan new bars and evaluate setups across all symbols
        for sym, ctx in self.contexts.items():
            new_bars = self._get_new_bars(ctx, now)
            for bar in new_bars:
                signal = ctx.strategy.on_bar(bar)
                for d in ctx.strategy.decisions[ctx.flushed_decisions:]:
                    self.journal.decision(d)
                ctx.flushed_decisions = len(ctx.strategy.decisions)

                if signal is not None:
                    self._handle_signal(ctx, signal, now)

        return True

    def _get_new_bars(self, ctx: SymbolContext, now: datetime) -> list[Bar]:
        if ctx.last_bar_time is None:
            bars = ctx.broker.closed_bars(10)
        else:
            missed = int((now - ctx.last_bar_time) / ctx.tf) + 2
            bars = [b for b in ctx.broker.closed_bars(min(max(missed, 3), 5000)) if b.time > ctx.last_bar_time]
        if bars:
            ctx.last_bar_time = bars[-1].time
        return bars

    def _close(self, ctx: SymbolContext, p: BrokerPosition, reason: str) -> None:
        sym = ctx.config.instrument.symbol
        if self.dry_run:
            self._note("DRY-RUN", f"would close {sym} ticket {p.ticket}: {reason}")
            return
        ctx.broker.close_position(p, reason.split()[0])
        self._note("EXIT", f"closed {sym} ticket {p.ticket}: {reason}")

    def _handle_signal(self, ctx: SymbolContext, sig: Signal, now: datetime) -> None:
        cfg = ctx.config
        sym = cfg.instrument.symbol

        # Check stale signal
        age = (now - sig.time).total_seconds()
        if age > cfg.live.max_signal_age_seconds:
            self._note("STALE", f"{sym} signal expired: {age:.0f}s old")
            return

        # Check Portfolio-wide open positions cap
        total_open = self._total_open_positions()
        if total_open >= self.max_portfolio_open_positions:
            self._note("PORTFOLIO_CAP", f"{sym} setup skipped: portfolio open position cap ({total_open}/{self.max_portfolio_open_positions}) reached")
            return

        # Check Portfolio Risk Manager
        assert self.risk is not None
        decision = self.risk.check(open_positions=total_open)
        if not decision.allowed:
            self._note(decision.reason.split()[0], f"{sym} entry blocked: {decision.reason}")
            return

        # Check News Blackout for this symbol's currencies
        if self.news and (ev := self.news.blocking_event(now)):
            self._note("NF-1", f"{sym} entry blocked: {ev.impact} {ev.currency} {ev.title} at {ev.time:%H:%M}Z")
            return

        spec = ctx.broker.symbol_spec()
        fill = spec.ask if sig.direction == LONG else spec.bid
        stop_dist = (fill - sig.stop) * sig.direction
        reward = (sig.target - fill) * sig.direction
        min_required_rr = cfg.strategy.small_account_min_rr if cfg.strategy.small_account_mode else cfg.strategy.min_rr
        if stop_dist <= 0 or reward <= 0 or reward / stop_dist < min_required_rr:
            self._note("TG-2", f"{sym} entry skipped: price moved; {reward / stop_dist if stop_dist > 0 else 0:.2f}R at {fill}")
            return

        lots = lots_for_risk(
            decision.risk_amount,
            stop_dist,
            spec.value_per_price_unit_per_lot,
            spec.volume_step,
            spec.min_lots,
            min(spec.max_lots, cfg.risk.max_lots),
            small_account_mode=cfg.risk.small_account_mode,
            compounding_milestones=cfg.risk.compounding_milestones,
            equity=ctx.broker.equity(),
        )
        if lots <= 0:
            self._note("RK-1", f"{sym} entry skipped: minimum lot exceeds risk budget")
            return

        # Broadcast signal to mobile channels
        self.notifier.signal_detected(sym, sig)

        side_str = "BUY" if sig.direction == LONG else "SELL"
        if self.dry_run:
            self._note("DRY-RUN", f"would {side_str} {lots} lots {sym} at {fill}, SL {sig.stop}, TP {sig.target}; {decision.reason}")
            return

        # Live Order Placement on MetaTrader
        ticket = ctx.broker.market_order(sig.direction, lots, sig.stop, sig.target, f"ICT {sig.level.name}")
        self.risk.record_open()
        self.journal.opened(ticket, now, sig, lots, fill)
        self._note("ENTRY", f"{side_str} {lots} lots {sym}, ticket {ticket}; {decision.reason}")
        self.notifier.order_executed(sym, ticket, sig.direction, lots, fill, sig.stop, sig.target, decision.risk_amount, decision.reason)

    def close(self) -> None:
        self.journal.close()

    def run(self) -> None:  # pragma: no cover - continuous daemon loop
        self.start()
        while True:
            try:
                if not self.poll():
                    return
            except Exception:
                log.exception("Portfolio poll cycle encountered exception; retrying in %ds", self.primary_cfg.live.poll_seconds)
            systime.sleep(self.primary_cfg.live.poll_seconds)
