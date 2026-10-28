"""Bar-by-bar backtester.

Conventions, chosen to be conservative rather than flattering:
- Data prices are bid. Longs fill at ask (bid + spread) and exit at bid;
  shorts fill at bid and exit at ask.
- A signal on bar N fills at the open of bar N+1. Nothing sees the future.
- If a bar touches both stop and target, the stop is assumed to hit first.
- A gap through the stop fills at the (worse) open price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .config import Config
from .metrics import Trade, summarize
from .news import NewsFilter
from .risk import RiskManager, lots_for_risk
from .sessions import Bar, flatten_deadline
from .models import make_strategy
from .strategy import LONG, Decision, Signal


@dataclass
class _Position:
    signal: Signal
    entry_time: datetime
    entry_price: float
    lots: float
    risk_amount: float
    deadline: datetime | None
    current_stop: float = 0.0
    realized: float = 0.0       # profit already booked by a TP1 partial (TP-04)
    partial_done: bool = False
    closed: bool = False
    lots0: float = 0.0          # size at entry
    seen: list = field(default_factory=list)   # candles since entry, for the structural trail


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[tuple[datetime, float]]
    decisions: list[Decision]
    stats: dict[str, Any] = field(default_factory=dict)


class Backtester:
    def __init__(self, config: Config, news: NewsFilter | None = None):
        self.config = config
        self.news = news
        inst = config.instrument
        self.spread = inst.pips(inst.spread_pips)
        self.value_per_unit = inst.contract_size * inst.quote_to_account_rate

    def _exit_price(self, d: int, bid: float) -> float:
        return bid if d == LONG else bid + self.spread

    def _pnl(self, pos: _Position, exit_price: float) -> float:
        d = pos.signal.direction
        gross = (exit_price - pos.entry_price) * d * pos.lots * self.value_per_unit
        return gross - self.config.instrument.commission_per_lot * pos.lots

    def run(self, bars: list[Bar], trade_from: datetime | None = None, trade_until: datetime | None = None,
            daily_bars: list[Bar] | None = None,
            precomputed: dict[datetime, Signal] | None = None) -> BacktestResult:
        """Run over bars. Bars before trade_from only warm up the levels.

        When `precomputed` (`{bar.time: Signal}`) is provided, the strategy is not re-run
        bar-by-bar, cutting live/history multi-model backtest time by ~90%.
        """
        cfg = self.config
        strategy = None if precomputed is not None else make_strategy(cfg.strategy, cfg.instrument)
        if strategy is not None and daily_bars and hasattr(strategy, "seed_days"):
            strategy.seed_days(daily_bars, before=bars[0].time.date() if bars else None)
        risk = RiskManager(cfg.risk, cfg.initial_equity, cfg.strategy.day_rollover)
        log: list[Decision] = []
        trades: list[Trade] = []
        curve: list[tuple[datetime, float]] = []
        balance = cfg.initial_equity
        pos: _Position | None = None
        pending: Signal | None = None
        tf = timedelta(minutes=cfg.strategy.timeframe_minutes)

        def close(bar_time: datetime, price: float, reason: str) -> None:
            nonlocal pos, balance
            assert pos is not None
            rest = self._pnl(pos, price)
            balance += rest
            pnl = pos.realized + rest  # a TP1 partial was already added to the balance
            risk.record_close(pnl)
            s = pos.signal
            trades.append(Trade(
                cfg.instrument.symbol, s.direction, pos.entry_time, pos.entry_price, s.stop, s.target,
                pos.lots0 or pos.lots, pos.risk_amount, bar_time, price, reason + (" · TP1 partial taken" if pos.partial_done else ""), pnl,
                pnl / pos.risk_amount if pos.risk_amount else 0.0, s.reasons,
            ))
            pos.closed = True
            pos = None

        def partial(price: float) -> None:
            nonlocal balance
            assert pos is not None
            part = pos.lots * cfg.strategy.partial_at_tp1
            booked = self._pnl(_Position(pos.signal, pos.entry_time, pos.entry_price, part, 0, None), price)
            balance += booked
            pos.realized += booked
            pos.lots -= part
            pos.partial_done = True

        def equity_at(bid: float) -> float:
            return balance + (self._pnl(pos, self._exit_price(pos.signal.direction, bid)) if pos else 0.0)

        for bar in bars:
            risk.mark(bar.time, equity_at(bar.open))

            # Forced exits at the open: end-of-day flatten (EX-1) and news (NF-2).
            if pos is not None:
                if pos.deadline is not None and bar.time >= pos.deadline:
                    close(bar.time, self._exit_price(pos.signal.direction, bar.open), "EX-1 flatten time")
                elif self.news and (ev := self.news.should_flatten(bar.time)):
                    close(bar.time, self._exit_price(pos.signal.direction, bar.open), f"NF-2 flatten before {ev.currency} {ev.title}")

            if pending is not None:
                if pos is None:
                    pos = self._try_open(pending, bar, risk, log)
                pending = None

            if pos is not None:
                self._intrabar_exit(pos, bar, close, partial)

            signal = precomputed.get(bar.time) if precomputed is not None else strategy.on_bar(bar)  # type: ignore[union-attr]
            if signal is not None:
                if (trade_from and signal.time < trade_from) or (trade_until and signal.time >= trade_until):
                    pass
                elif pos is not None:
                    log.append(Decision(signal.time, "RK-3", "signal ignored: position already open"))
                elif signal.entry_type == "limit":
                    # The limit order filled inside this candle at the limit price.
                    pos = self._open_limit(signal, bar, risk, log)
                    if pos is not None:
                        d = signal.direction
                        # Conservative: if this same candle also reached the stop, count the loss.
                        if (d == LONG and bar.low <= signal.stop) or (d != LONG and bar.high + self.spread >= signal.stop):
                            close(signal.time, signal.stop, "SL stop loss (same candle as fill)")
                else:
                    pending = signal

            curve.append((bar.time + tf, equity_at(bar.close)))

        if pos is not None and bars:
            last = bars[-1]
            close(last.time + tf, self._exit_price(pos.signal.direction, last.close), "END end of data")
            curve.append((last.time + tf, balance))

        decisions = sorted((strategy.decisions if strategy is not None else []) + log, key=lambda d: d.time)
        if trade_from:
            curve = [(t, e) for t, e in curve if t >= trade_from]
        result = BacktestResult(trades, curve, decisions)
        result.stats = summarize(trades, curve, curve[0][1] if curve else cfg.initial_equity)
        return result

    def _try_open(self, sig: Signal, bar: Bar, risk: RiskManager, log: list[Decision]) -> _Position | None:
        d = sig.direction
        if self.news and (why := self.news.no_trade(bar.time)):
            log.append(Decision(bar.time, why.split()[0], f"entry blocked: {why}"))
            return None
        if self.news and (ev := self.news.blocking_event(bar.time)):
            log.append(Decision(bar.time, "NF-1", f"entry blocked: {ev.impact} {ev.currency} {ev.title} at {ev.time:%H:%M}Z"))
            return None
        decision = risk.check(open_positions=0)
        if not decision.allowed:
            log.append(Decision(bar.time, decision.reason.split()[0], f"entry blocked: {decision.reason}"))
            return None
        fill = bar.open + self.spread if d == LONG else bar.open
        stop_dist = (fill - sig.stop) * d
        reward = (sig.target - fill) * d
        if stop_dist <= 0 or reward <= 0:
            log.append(Decision(bar.time, "RK-1", "entry skipped: price gapped past stop or target"))
            return None
        rr = reward / stop_dist
        if rr < self.config.strategy.min_rr:
            log.append(Decision(bar.time, "TG-2", f"entry skipped: {rr:.2f}R at fill below minimum"))
            return None
        inst = self.config.instrument
        lots = lots_for_risk(
            decision.risk_amount,
            stop_dist,
            self.value_per_unit,
            inst.volume_step,
            inst.min_lots,
            self.config.risk.max_lots,
            small_account_mode=self.config.risk.small_account_mode,
            compounding_milestones=self.config.risk.compounding_milestones,
            equity=risk.equity,
        )
        if lots <= 0:
            log.append(Decision(bar.time, "RK-1", "entry skipped: minimum lot would exceed risk budget"))
            return None
        risk.record_open()
        log.append(Decision(bar.time, "ENTRY", f"{'LONG' if d == LONG else 'SHORT'} {lots} lots at {fill:.5f}; {decision.reason}"))
        return _Position(
            sig,
            bar.time,
            fill,
            lots,
            lots * stop_dist * self.value_per_unit,
            flatten_deadline(bar.time, self.config.strategy.flatten_time),
            current_stop=sig.stop,
            lots0=lots,
        )

    def _open_limit(self, sig: Signal, bar: Bar, risk: RiskManager, log: list[Decision]) -> _Position | None:
        """Open at the limit price. Prices are bid: a buy limit fills at ask, so pay the spread."""
        if self.news and (why := self.news.no_trade(bar.time)):
            log.append(Decision(sig.time, why.split()[0], f"limit fill blocked: {why}"))
            return None
        if self.news and (ev := self.news.blocking_event(bar.time)):
            log.append(Decision(sig.time, "NF-1", f"limit fill blocked: {ev.impact} {ev.currency} {ev.title}"))
            return None
        decision = risk.check(open_positions=0)
        if not decision.allowed:
            log.append(Decision(sig.time, decision.reason.split()[0], f"limit fill blocked: {decision.reason}"))
            return None
        d = sig.direction
        fill = sig.entry_ref + self.spread if d == LONG else sig.entry_ref
        stop_dist = (fill - sig.stop) * d
        if stop_dist <= 0:
            return None
        inst = self.config.instrument
        lots = lots_for_risk(decision.risk_amount, stop_dist, self.value_per_unit, inst.volume_step, inst.min_lots,
                             self.config.risk.max_lots, small_account_mode=self.config.risk.small_account_mode,
                             compounding_milestones=self.config.risk.compounding_milestones, equity=risk.equity)
        if lots <= 0:
            log.append(Decision(sig.time, "RK-1", "limit fill skipped: minimum lot would exceed risk budget"))
            return None
        risk.record_open()
        log.append(Decision(sig.time, "ENTRY", f"{'LONG' if d == LONG else 'SHORT'} {lots} lots, limit filled at {fill:.5f}; {decision.reason}"))
        return _Position(sig, sig.time, fill, lots, lots * stop_dist * self.value_per_unit,
                         flatten_deadline(sig.time, self.config.strategy.flatten_time), current_stop=sig.stop, lots0=lots)

    def _intrabar_exit(self, pos: _Position, bar: Bar, close, partial=None) -> None:  # type: ignore[no-untyped-def]
        s, d = pos.signal, pos.signal.direction
        # Exit-side prices for this bar: bid for longs, ask for shorts.
        adj = 0.0 if d == LONG else self.spread
        o, hi, lo = bar.open + adj, bar.high + adj, bar.low + adj
        stop_label = ("SL stop loss" if pos.current_stop == s.stop else
                      "SL breakeven after TP1 (TP-04)" if pos.partial_done and pos.current_stop == pos.entry_price else
                      "SL breakeven stop (SL-05)" if pos.current_stop == pos.entry_price else "SL trimmed stop (SL-04)")

        if d == LONG:
            if o <= pos.current_stop:
                close(bar.time, o, "SL gap through stop")
            elif o >= s.target:
                close(bar.time, o, "TP gap through target")
            elif lo <= pos.current_stop:
                close(bar.time, pos.current_stop, stop_label)
            elif hi >= s.target:
                close(bar.time, s.target, "TP target")
        else:
            if o >= pos.current_stop:
                close(bar.time, o, "SL gap through stop")
            elif o <= s.target:
                close(bar.time, o, "TP gap through target")
            elif hi >= pos.current_stop:
                close(bar.time, pos.current_stop, stop_label)
            elif lo <= s.target:
                close(bar.time, s.target, "TP target")

        # TP-04: take the partial at TP1 and move the rest to breakeven. Checked after the
        # stop, so a candle that touched both is counted at the stop (conservative); if the
        # same candle also reached TP2, the rest closes there.
        tp1 = s.meta.get("tp1") if s.meta else None
        if partial and not pos.closed and not pos.partial_done and tp1 is not None and self.config.strategy.partial_at_tp1 > 0:
            if (d == LONG and hi >= tp1) or (d != LONG and lo <= tp1):
                partial(tp1)
                if (pos.entry_price - pos.current_stop) * d > 0:
                    pos.current_stop = pos.entry_price
                if (d == LONG and hi >= s.target) or (d != LONG and lo <= s.target):
                    close(bar.time, s.target, "TP target")
        if pos.closed:
            return

        # Stop milestones (SL-04, SL-05), applied after this bar's exit checks so a bar that
        # touched both the milestone and the stop is still counted at the stop it had.
        cfg_strat = self.config.strategy
        initial_risk = abs(pos.entry_price - s.stop)
        target_dist = abs(s.target - pos.entry_price)
        if target_dist > 0 and initial_risk > 0:
            progress = (hi - pos.entry_price) / target_dist if d == LONG else (pos.entry_price - lo) / target_dist
            r_progress = (hi - pos.entry_price) / initial_risk if d == LONG else (pos.entry_price - lo) / initial_risk
            if (cfg_strat.move_be_at_75_pct and progress >= 0.75) or (cfg_strat.move_be_at_1r and r_progress >= 1.0):
                new_stop = pos.entry_price  # SL-05: breakeven
            elif cfg_strat.trim_stop_at_50_pct and (progress >= 0.50 or (cfg_strat.move_be_at_1r and r_progress >= 0.75)):
                new_stop = pos.entry_price - d * 0.75 * initial_risk  # SL-04: risk trimmed by 25%
            else:
                new_stop = pos.current_stop
            if (new_stop - pos.current_stop) * d > 0:
                pos.current_stop = new_stop
        if cfg_strat.trail_structure:
            # Management model: once the trade is past breakeven territory (a partial taken, or half way to the
            # target), ratchet the stop behind each new 3-candle swing low (longs) / high (shorts). Never loosens.
            pos.seen.append(bar)
            if len(pos.seen) >= 3 and (pos.partial_done or (target_dist > 0 and ((hi - pos.entry_price) if d == LONG else (pos.entry_price - lo)) >= 0.5 * target_dist)):
                a, m, c = pos.seen[-3], pos.seen[-2], pos.seen[-1]
                if d == LONG and m.low < a.low and m.low < c.low and m.low > pos.current_stop and m.low < c.low:
                    pos.current_stop = m.low
                elif d != LONG and m.high > a.high and m.high > c.high and (m.high + adj) < pos.current_stop:
                    pos.current_stop = m.high + adj
