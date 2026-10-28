"""Hard risk limits and position sizing engine. Nothing upstream can override these.

Daily and weekly limits are enforced as equity floors: a new trade may only
risk the smaller of the per-trade budget and the distance to each floor, so
even a full stop-out cannot push equity through a limit (gaps and slippage
aside). Once a floor is touched, trading halts until the period rolls.

Supports:
- Institutional percentage risk (RK-01)
- Dynamic risk reduction schedule after losses (RK-02, RK-03 from Ep 41)
- Small Account & Cent Account Compounding Engine (SA-01 to SA-04)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time

from .config import RiskConfig
from .sessions import trading_day


@dataclass
class RiskDecision:
    allowed: bool
    risk_amount: float
    reason: str


class RiskManager:
    def __init__(self, config: RiskConfig, equity: float, day_rollover: time = time(17, 0)):
        self.config = config
        self.day_rollover = day_rollover
        self.equity = equity
        self._day: date | None = None
        self._week: tuple[int, int] | None = None
        self.day_start_equity = equity
        self.week_start_equity = equity
        self.trades_today = 0
        self.consecutive_losses = 0
        # Losses in a row within the current trading day: drives the RK-4/RK-05 halt and
        # resets each day. consecutive_losses (above) persists for dynamic risk (RK-02).
        self.streak_today = 0
        self.loss_recovery_peak: float = equity
        self.drawdown_trough: float = equity
        self.is_in_reduced_risk: bool = False

    def mark(self, ts: datetime, equity: float) -> None:
        """Record current (mark-to-market) equity and roll day/week."""
        day = trading_day(ts, self.day_rollover)
        week = tuple(day.isocalendar())[:2]
        if day != self._day:
            self._day = day
            self.day_start_equity = equity
            self.trades_today = 0
            self.streak_today = 0
            # Note: consecutive losses persist across day boundary for dynamic risk tracking
        if week != self._week:
            self._week = week  # type: ignore[assignment]
            self.week_start_equity = equity
        self.equity = equity

    @property
    def daily_floor(self) -> float:
        return self.day_start_equity * (1 - self.config.max_daily_loss_pct / 100)

    @property
    def weekly_floor(self) -> float:
        return self.week_start_equity * (1 - self.config.max_weekly_loss_pct / 100)

    def check(self, open_positions: int) -> RiskDecision:
        c = self.config
        if open_positions >= c.max_open_positions:
            return RiskDecision(False, 0.0, f"RK-3 max {c.max_open_positions} open position(s)")
        if self.trades_today >= c.max_trades_per_day:
            return RiskDecision(False, 0.0, f"RK-3 max {c.max_trades_per_day} trade(s) per day reached")
        if self.streak_today >= c.max_consecutive_losses:
            return RiskDecision(False, 0.0, f"RK-4 {self.streak_today} consecutive losses today; halted for the day")
        if self.equity <= self.daily_floor:
            return RiskDecision(False, 0.0, f"RK-2 daily loss limit {c.max_daily_loss_pct}% hit; halted for the day")
        if self.equity <= self.weekly_floor:
            return RiskDecision(False, 0.0, f"RK-2 weekly loss limit {c.max_weekly_loss_pct}% hit; halted for the week")

        # Determine effective risk percentage (Ep 41 RK-02 dynamic reduction)
        risk_pct = c.risk_per_trade_pct
        reason_tag = "RK-1"
        if c.enable_dynamic_risk_reduction and self.is_in_reduced_risk:
            if self.consecutive_losses == 1:
                risk_pct = c.risk_per_trade_pct / 2.0
                reason_tag = "RK-02 (50% risk reduction after 1 loss)"
            elif self.consecutive_losses >= 2:
                risk_pct = c.risk_per_trade_pct / 4.0
                reason_tag = "RK-02 (75% risk reduction after 2+ losses)"

        # Small account mode can utilize fixed dollar risk
        if c.small_account_mode and c.fixed_risk_dollars is not None:
            budget = min(
                c.fixed_risk_dollars,
                max(0.0, self.equity - self.daily_floor),
                max(0.0, self.equity - self.weekly_floor),
            )
            return RiskDecision(True, budget, f"SA-02 small account fixed risk {budget:.2f}")

        budget = min(
            self.equity * risk_pct / 100.0,
            max(0.0, self.equity - self.daily_floor),
            max(0.0, self.equity - self.weekly_floor),
        )
        return RiskDecision(True, budget, f"{reason_tag} risking {budget:.2f} ({budget / self.equity * 100:.2f}% of equity)")

    def record_open(self) -> None:
        self.trades_today += 1

    def record_close(self, pnl: float) -> None:
        if pnl < 0:
            if not self.is_in_reduced_risk:
                self.loss_recovery_peak = self.equity - pnl
                self.drawdown_trough = self.equity
            else:
                self.drawdown_trough = min(self.drawdown_trough, self.equity)
            self.consecutive_losses += 1
            self.streak_today += 1
            self.is_in_reduced_risk = True
        else:
            self.streak_today = 0
            if self.is_in_reduced_risk:
                # Ep 41 RK-03: recover 50% of the loss from drawdown trough to restore base risk
                loss_amount = self.loss_recovery_peak - self.drawdown_trough
                recovered = self.equity - self.drawdown_trough
                if loss_amount > 0 and (recovered / loss_amount) >= (self.config.loss_recovery_threshold_pct / 100.0):
                    self.consecutive_losses = 0
                    self.is_in_reduced_risk = False
                    self.loss_recovery_peak = self.equity
            else:
                self.consecutive_losses = 0


def lots_for_risk(
    risk_amount: float,
    stop_distance: float,
    value_per_price_unit_per_lot: float,
    volume_step: float,
    min_lots: float,
    max_lots: float,
    small_account_mode: bool = False,
    compounding_milestones: tuple[tuple[float, float], ...] = (),
    equity: float = 0.0,
) -> float:
    """Calculate the lot size rounded down to the broker step.

    Supports:
    - Milestone compounding tiers for small accounts (SA-04).
    - Small account micro-lot safety floor: prevents skipping trades on $10-$50 accounts.
    - Standard institutional sizing: skips trade if even min_lots exceeds budget.
    """
    if stop_distance <= 0 or value_per_price_unit_per_lot <= 0:
        return 0.0

    # Milestone compounding tier evaluation for small accounts (SA-04)
    if small_account_mode and compounding_milestones and equity > 0:
        target_lots = None
        for threshold, size in sorted(compounding_milestones, key=lambda x: x[0]):
            if equity >= threshold:
                target_lots = size
        if target_lots is not None:
            lots = math.floor(target_lots / volume_step + 1e-9) * volume_step
            return round(min(max(lots, min_lots), max_lots), 8)

    raw = risk_amount / (stop_distance * value_per_price_unit_per_lot)
    lots = math.floor(raw / volume_step + 1e-9) * volume_step
    lots = min(lots, max_lots)

    if lots < min_lots:
        # Small account mode allows min_lots rather than skipping the trade (SA-02)
        return min_lots if small_account_mode else 0.0

    return round(lots, 8)
