"""MetaTrader 5 adapter using the official `MetaTrader5` Python package.

That package runs on Windows alongside an installed MT5 terminal (a Windows
VPS is the usual deployment). Credentials come from environment variables:
MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, and optionally MT5_TERMINAL_PATH.
"""

from __future__ import annotations

import os
from typing import Any

from ..config import Config
from ..sessions import Bar
from ..strategy import LONG
from .base import BrokerError, BrokerPosition, SymbolSpec, server_time_to_utc


class MT5Broker:
    def __init__(self, config: Config):
        try:
            import MetaTrader5 as mt5  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - platform specific
            raise BrokerError("The MetaTrader5 package is not installed (pip install MetaTrader5; Windows only)") from exc
        self.mt5: Any = mt5
        self.config = config
        self.symbol = config.instrument.symbol
        tf = config.strategy.timeframe_minutes
        name = f"TIMEFRAME_M{tf}" if tf < 60 else f"TIMEFRAME_H{tf // 60}"
        if (tf >= 60 and tf % 60) or not hasattr(mt5, name):
            raise BrokerError(f"Unsupported timeframe {tf} minutes")
        self.timeframe = getattr(mt5, name)

    def _check(self, result: Any, what: str) -> Any:
        if result is None:
            raise BrokerError(f"{what} failed: {self.mt5.last_error()}")
        return result

    def connect(self) -> None:
        kwargs: dict[str, Any] = {}
        if path := os.environ.get("MT5_TERMINAL_PATH"):
            kwargs["path"] = path
        if login := os.environ.get("MT5_LOGIN"):
            kwargs.update(login=int(login), password=os.environ.get("MT5_PASSWORD", ""), server=os.environ.get("MT5_SERVER", ""))
        if not self.mt5.initialize(**kwargs):
            raise BrokerError(f"MT5 initialize failed: {self.mt5.last_error()}")
        if not self.mt5.symbol_select(self.symbol, True):
            raise BrokerError(f"Symbol {self.symbol} not available: {self.mt5.last_error()}")

    def is_real_account(self) -> bool:
        info = self._check(self.mt5.account_info(), "account_info")
        return info.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_REAL

    def closed_bars(self, count: int) -> list[Bar]:
        # Position 0 is the bar still forming; start at 1 for closed bars only.
        rates = self._check(self.mt5.copy_rates_from_pos(self.symbol, self.timeframe, 1, count), "copy_rates_from_pos")
        tz = self.config.live.server_timezone
        return [Bar(server_time_to_utc(int(r["time"]), tz), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["tick_volume"])) for r in rates]

    def equity(self) -> float:
        return float(self._check(self.mt5.account_info(), "account_info").equity)

    def symbol_spec(self) -> SymbolSpec:
        info = self._check(self.mt5.symbol_info(self.symbol), "symbol_info")
        tick = self._check(self.mt5.symbol_info_tick(self.symbol), "symbol_info_tick")
        return SymbolSpec(
            bid=tick.bid, ask=tick.ask,
            value_per_price_unit_per_lot=info.trade_tick_value / info.trade_tick_size,
            volume_step=info.volume_step, min_lots=info.volume_min, max_lots=info.volume_max, digits=info.digits,
        )

    def open_positions(self) -> list[BrokerPosition]:
        positions = self.mt5.positions_get(symbol=self.symbol) or ()
        tz = self.config.live.server_timezone
        return [
            BrokerPosition(p.ticket, LONG if p.type == self.mt5.POSITION_TYPE_BUY else -LONG, p.volume, p.price_open, p.sl, p.tp, server_time_to_utc(int(p.time), tz))
            for p in positions
            if p.magic == self.config.live.magic
        ]

    def _filling(self) -> int:
        mode = self._check(self.mt5.symbol_info(self.symbol), "symbol_info").filling_mode
        if mode & 1:
            return self.mt5.ORDER_FILLING_FOK
        if mode & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _send(self, request: dict[str, Any]) -> Any:
        result = self.mt5.order_send(request)
        if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
            detail = result.comment if result is not None else self.mt5.last_error()
            raise BrokerError(f"order_send rejected: {detail}")
        return result

    def market_order(self, direction: int, lots: float, stop: float, target: float, comment: str) -> int:
        spec = self.symbol_spec()
        buy = direction == LONG
        result = self._send({
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lots,
            "type": self.mt5.ORDER_TYPE_BUY if buy else self.mt5.ORDER_TYPE_SELL,
            "price": spec.ask if buy else spec.bid,
            "sl": round(stop, spec.digits),
            "tp": round(target, spec.digits),
            "deviation": self.config.live.deviation_points,
            "magic": self.config.live.magic,
            "comment": comment[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(),
        })
        return int(result.order)

    def close_position(self, position: BrokerPosition, comment: str) -> None:
        spec = self.symbol_spec()
        buy_to_close = position.direction != LONG
        self._send({
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "position": position.ticket,
            "volume": position.lots,
            "type": self.mt5.ORDER_TYPE_BUY if buy_to_close else self.mt5.ORDER_TYPE_SELL,
            "price": spec.ask if buy_to_close else spec.bid,
            "deviation": self.config.live.deviation_points,
            "magic": self.config.live.magic,
            "comment": comment[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(),
        })

    def realized_pnl(self, ticket: int) -> float | None:
        deals = self.mt5.history_deals_get(position=ticket)
        if not deals:
            return None
        return float(sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0.0) for d in deals))
