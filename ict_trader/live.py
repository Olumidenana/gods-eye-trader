"""Live / demo runner: the same strategy and risk code as the backtester, on a broker.

Each poll it: honours the kill switch, books P&L for positions the broker
closed, enforces the flatten-time and news exits, and feeds newly closed bars
to the strategy. Every decision goes to a SQLite journal for review.
"""

from __future__ import annotations

import logging
import sqlite3
import time as systime
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .broker.base import Broker, BrokerPosition
from .config import Config
from .news import NewsFilter
from .notifier import Notifier
from .risk import RiskManager, lots_for_risk
from .sessions import Bar, flatten_deadline
from .models import make_strategy
from .strategy import LONG, Decision, Signal, TimeAndPriceStrategy

log = logging.getLogger("ict_trader.live")


class Journal:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS decisions (time TEXT, rule TEXT, message TEXT);
            CREATE TABLE IF NOT EXISTS trades (
                ticket INTEGER PRIMARY KEY, opened TEXT, direction INTEGER, lots REAL,
                entry REAL, stop REAL, target REAL, reasons TEXT, closed TEXT, pnl REAL
            );
            """
        )

    def decision(self, d: Decision) -> None:
        log.info("%s %s %s", d.time.isoformat(), d.rule, d.message)
        self.db.execute("INSERT INTO decisions VALUES (?, ?, ?)", (d.time.isoformat(), d.rule, d.message))
        self.db.commit()

    def opened(self, ticket: int, ts: datetime, sig: Signal, lots: float, entry: float) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO trades (ticket, opened, direction, lots, entry, stop, target, reasons) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ticket, ts.isoformat(), sig.direction, lots, entry, sig.stop, sig.target, "\n".join(sig.reasons)),
        )
        self.db.commit()

    def closed(self, ticket: int, ts: datetime, pnl: float) -> None:
        self.db.execute("UPDATE trades SET closed = ?, pnl = ? WHERE ticket = ?", (ts.isoformat(), pnl, ticket))
        self.db.commit()

    def close(self) -> None:
        self.db.close()


class LiveRunner:
    def __init__(
        self,
        config: Config,
        broker: Broker,
        news: NewsFilter | None,
        journal: Journal,
        dry_run: bool = True,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        notifier: Notifier | None = None,
    ):
        if news is None and config.news.enabled:
            raise ValueError("News filter is enabled but no calendar was loaded; refusing to trade blind (NF-1)")
        self.config = config
        self.broker = broker
        self.news = news
        self.journal = journal
        self.dry_run = dry_run
        self.clock = clock
        self.notifier = notifier or Notifier()
        self.strategy = make_strategy(config.strategy, config.instrument)
        self.risk: RiskManager | None = None
        self.last_bar_time: datetime | None = None
        self.known: dict[int, BrokerPosition] = {}
        self._flushed = 0
        self.tf = timedelta(minutes=config.strategy.timeframe_minutes)

    def _note(self, rule: str, message: str) -> None:
        self.journal.decision(Decision(self.clock(), rule, message))

    def _flush_strategy_log(self) -> None:
        for d in self.strategy.decisions[self._flushed:]:
            self.journal.decision(d)
        self._flushed = len(self.strategy.decisions)

    def start(self, warmup_bars: int = 2000) -> None:
        self.broker.connect()
        self.risk = RiskManager(self.config.risk, self.broker.equity(), self.config.strategy.day_rollover)
        history = self.broker.closed_bars(warmup_bars)
        for bar in history:
            self.strategy.on_bar(bar)  # warm-up only; historical signals are never traded
        self._flushed = len(self.strategy.decisions)
        self.last_bar_time = history[-1].time if history else None
        self.known = {p.ticket: p for p in self.broker.open_positions()}
        self._note("START", f"{'DRY RUN' if self.dry_run else 'TRADING'} {self.config.instrument.symbol}; warmed up on {len(history)} bars; {len(self.known)} open position(s)")

    def poll(self) -> bool:
        """One iteration. Returns False once the kill switch has fired."""
        assert self.risk is not None, "call start() first"
        now = self.clock()

        if Path(self.config.live.kill_switch_file).exists():
            for p in self.broker.open_positions():
                self._close(p, "KILL kill switch")
            self._note("KILL", "kill switch file found; flattened and stopping")
            return False

        positions = self.broker.open_positions()
        current = {p.ticket for p in positions}
        for ticket in [t for t in self.known if t not in current]:
            pnl = self.broker.realized_pnl(ticket)
            if pnl is not None:
                self.risk.record_close(pnl)
                self.journal.closed(ticket, now, pnl)
                self._note("CLOSED", f"ticket {ticket} closed, P&L {pnl:+.2f}")
                self.notifier.trade_closed(self.config.instrument.symbol, ticket, pnl, "SL/TP exit")
            del self.known[ticket]
        for p in positions:
            self.known.setdefault(p.ticket, p)

        self.risk.mark(now, self.broker.equity())

        for p in positions:
            deadline = flatten_deadline(p.open_time, self.config.strategy.flatten_time)
            if deadline is not None and now >= deadline:
                self._close(p, "EX-1 flatten time")
            elif self.news and (ev := self.news.should_flatten(now)):
                self._close(p, f"NF-2 flatten before {ev.currency} {ev.title}")

        for bar in self._new_bars(now):
            signal = self.strategy.on_bar(bar)
            self._flush_strategy_log()
            if signal is not None:
                self._enter(signal, now)
        return True

    def _new_bars(self, now: datetime) -> list[Bar]:
        if self.last_bar_time is None:
            bars = self.broker.closed_bars(10)
        else:
            missed = int((now - self.last_bar_time) / self.tf) + 2
            bars = [b for b in self.broker.closed_bars(min(max(missed, 3), 5000)) if b.time > self.last_bar_time]
        if bars:
            self.last_bar_time = bars[-1].time
        return bars

    def _close(self, p: BrokerPosition, reason: str) -> None:
        if self.dry_run:
            self._note("DRY-RUN", f"would close ticket {p.ticket}: {reason}")
            return
        self.broker.close_position(p, reason.split()[0])
        self._note("EXIT", f"closed ticket {p.ticket}: {reason}")

    def _enter(self, sig: Signal, now: datetime) -> None:
        assert self.risk is not None
        cfg = self.config
        if (now - sig.time).total_seconds() > cfg.live.max_signal_age_seconds:
            self._note("STALE", f"signal from {sig.time.isoformat()} is too old to act on")
            return
        if self.news and (ev := self.news.blocking_event(now)):
            self._note("NF-1", f"entry blocked: {ev.impact} {ev.currency} {ev.title} at {ev.time:%H:%M}Z")
            return
        decision = self.risk.check(open_positions=len(self.broker.open_positions()))
        if not decision.allowed:
            self._note(decision.reason.split()[0], f"entry blocked: {decision.reason}")
            return
        spec = self.broker.symbol_spec()
        fill = spec.ask if sig.direction == LONG else spec.bid
        stop_dist = (fill - sig.stop) * sig.direction
        reward = (sig.target - fill) * sig.direction
        if stop_dist <= 0 or reward <= 0 or reward / stop_dist < cfg.strategy.min_rr:
            self._note("TG-2", f"entry skipped: price moved; {reward / stop_dist if stop_dist > 0 else 0:.2f}R at {fill}")
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
            equity=self.broker.equity(),
        )
        if lots <= 0:
            self._note("RK-1", "entry skipped: minimum lot would exceed risk budget")
            return
        self.notifier.signal_detected(cfg.instrument.symbol, sig)
        side = "BUY" if sig.direction == LONG else "SELL"
        if self.dry_run:
            self._note("DRY-RUN", f"would {side} {lots} lots at {fill}, SL {sig.stop}, TP {sig.target}; {decision.reason}")
            return
        ticket = self.broker.market_order(sig.direction, lots, sig.stop, sig.target, f"ICT {sig.level.name}")
        self.risk.record_open()
        self.journal.opened(ticket, now, sig, lots, fill)
        self._note("ENTRY", f"{side} {lots} lots, ticket {ticket}; {decision.reason}")
        self.notifier.order_executed(cfg.instrument.symbol, ticket, sig.direction, lots, fill, sig.stop, sig.target, decision.risk_amount, decision.reason)

    def close(self) -> None:
        self.journal.close()

    def run(self) -> None:  # pragma: no cover - long-running loop
        self.start()
        while True:
            try:
                if not self.poll():
                    return
            except Exception:
                log.exception("poll failed; retrying")
            systime.sleep(self.config.live.poll_seconds)
