"""MetaTrader 4 adapter: talks to the ICTBridge EA (mql4/ICTBridge.mq4) through files.

MT4 has no official Python API, so the EA publishes account, bar, position
and history files to the terminal's shared Common\\Files folder, and executes
open/close commands this adapter drops there. Every file is written to a
temp name and renamed, so neither side ever reads a half-written file.
"""

from __future__ import annotations

import csv
import os
import time as systime
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..sessions import Bar
from ..strategy import LONG
from .base import BrokerError, BrokerPosition, SymbolSpec, server_time_to_utc

HEARTBEAT_MAX_AGE_SECONDS = 30


class MT4BridgeBroker:
    def __init__(self, config: Config, clock=lambda: datetime.now(timezone.utc)):  # type: ignore[no-untyped-def]
        if not config.live.bridge_dir:
            raise BrokerError("Set live.bridge_dir to the MT4 terminal's Common\\Files folder")
        self.config = config
        self.dir = Path(config.live.bridge_dir)
        self.symbol = config.instrument.symbol
        self.clock = clock

    def _path(self, what: str) -> Path:
        return self.dir / f"ictb_{self.symbol}_{what}"

    def _read(self, what: str) -> str:
        path = self._path(what)
        try:
            return path.read_text(encoding="latin-1")
        except FileNotFoundError as exc:
            raise BrokerError(f"{path} not found: is ICTBridge attached to a {self.symbol} chart?") from exc

    def _rows(self, what: str) -> list[dict[str, str]]:
        return list(csv.DictReader(self._read(what).splitlines()))

    def _account(self) -> dict[str, str]:
        out = {}
        for line in self._read("account.txt").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    def connect(self) -> None:
        acct = self._account()
        age = self.clock().timestamp() - int(acct["gmt_time"])
        if abs(age) > HEARTBEAT_MAX_AGE_SECONDS:
            raise BrokerError(f"ICTBridge heartbeat is {age:.0f}s old: is MT4 running with the EA attached and AutoTrading on?")
        if int(acct["timeframe_minutes"]) != self.config.strategy.timeframe_minutes:
            raise BrokerError(f"ICTBridge chart is M{acct['timeframe_minutes']} but strategy expects M{self.config.strategy.timeframe_minutes}")
        if acct.get("trade_allowed") != "1":
            raise BrokerError("MT4 reports trading not allowed: enable AutoTrading and 'Allow live trading' on the EA")

    def is_real_account(self) -> bool:
        return self._account()["demo"] != "1"

    def closed_bars(self, count: int) -> list[Bar]:
        tz = self.config.live.server_timezone
        rows = self._rows("bars.csv")[-count:]
        return [
            Bar(server_time_to_utc(int(r["time"]), tz), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows
        ]

    def equity(self) -> float:
        return float(self._account()["equity"])

    def symbol_spec(self) -> SymbolSpec:
        a = self._account()
        return SymbolSpec(
            bid=float(a["bid"]), ask=float(a["ask"]),
            value_per_price_unit_per_lot=float(a["tick_value"]) / float(a["tick_size"]),
            volume_step=float(a["lot_step"]), min_lots=float(a["min_lot"]), max_lots=float(a["max_lot"]), digits=int(a["digits"]),
        )

    def open_positions(self) -> list[BrokerPosition]:
        tz = self.config.live.server_timezone
        return [
            BrokerPosition(int(r["ticket"]), LONG if r["type"] == "0" else -LONG, float(r["lots"]), float(r["open_price"]),
                           float(r["sl"]), float(r["tp"]), server_time_to_utc(int(r["open_time"]), tz))
            for r in self._rows("positions.csv")
        ]

    def realized_pnl(self, ticket: int) -> float | None:
        for r in self._rows("history.csv"):
            if int(r["ticket"]) == ticket:
                return float(r["pnl"])
        return None

    def _command(self, action: str, *args: object) -> int:
        cmd_id = uuid.uuid4().hex[:12]
        tmp = self._path(f"cmd_{cmd_id}.part")
        tmp.write_text("\n".join([cmd_id, action, *(str(a) for a in args)]) + "\n", encoding="latin-1")
        os.replace(tmp, self._path(f"cmd_{cmd_id}.txt"))

        res = self._path(f"res_{cmd_id}.txt")
        deadline = systime.monotonic() + self.config.live.bridge_timeout_seconds
        while systime.monotonic() < deadline:
            if res.exists():
                reply = res.read_text(encoding="latin-1").strip()
                res.unlink(missing_ok=True)
                if reply.startswith("OK"):
                    return int(reply.split()[1])
                raise BrokerError(f"MT4 rejected {action}: {reply}")
            systime.sleep(0.1)
        raise BrokerError(f"No reply from ICTBridge to {action} within {self.config.live.bridge_timeout_seconds}s; check MT4 before restarting")

    def market_order(self, direction: int, lots: float, stop: float, target: float, comment: str) -> int:
        return self._command("OPEN", direction, lots, stop, target, comment[:31].replace("\n", " "))

    def close_position(self, position: BrokerPosition, comment: str) -> None:
        self._command("CLOSE", position.ticket)
