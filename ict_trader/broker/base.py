"""The broker interface the live runner depends on."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol
from zoneinfo import ZoneInfo

from ..sessions import NY, Bar


class BrokerError(RuntimeError):
    pass


def server_time_to_utc(epoch_seconds: int, server_timezone: str) -> datetime:
    """MetaTrader reports times as server-local wall clock encoded as epoch seconds."""
    wall = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).replace(tzinfo=None)
    if server_timezone.upper() == "NY+7":
        return (wall - timedelta(hours=7)).replace(tzinfo=NY).astimezone(timezone.utc)
    return wall.replace(tzinfo=ZoneInfo(server_timezone)).astimezone(timezone.utc)


@dataclass(frozen=True)
class BrokerPosition:
    ticket: int
    direction: int
    lots: float
    entry_price: float
    stop: float
    target: float
    open_time: datetime


@dataclass(frozen=True)
class SymbolSpec:
    bid: float
    ask: float
    # Account-currency value of a 1.0 price move on 1.0 lot.
    value_per_price_unit_per_lot: float
    volume_step: float
    min_lots: float
    max_lots: float
    digits: int


class Broker(Protocol):
    def connect(self) -> None: ...
    def is_real_account(self) -> bool: ...
    def closed_bars(self, count: int) -> list[Bar]: ...
    def equity(self) -> float: ...
    def symbol_spec(self) -> SymbolSpec: ...
    def open_positions(self) -> list[BrokerPosition]: ...
    def market_order(self, direction: int, lots: float, stop: float, target: float, comment: str) -> int: ...
    def close_position(self, position: BrokerPosition, comment: str) -> None: ...
    def realized_pnl(self, ticket: int) -> float | None: ...
