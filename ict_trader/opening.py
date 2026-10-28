"""The First Presented FVG model (ICT, 2024-2026), as a deterministic state machine.

In his words (research/ict-videos, model card "first presented FVG"):
- the anchor is the clock: the 09:30 NY open, also the 01:30 London and 07:00 New York windows;
  "the first fair value gap presented between 9:30 and 10:00 is the entry framework"
- the gap is the first three-candle imbalance printed after the anchor, completed inside the first
  30 minutes (Zm9Q0NDRxoY, uC4-1SYXJFg, NxwfXE6Zyns)
- a gap formed inside a balanced price range, "not breaking away from anything", is not valid (y63zsrLCcCw)
- "Wait for that first fair value gap to form and have your limit order sitting in that fair value
  gap" (NMCy1H2_Wr8); the consequent encroachment (50%) is the level that should hold, judged by
  bodies not wicks (Ipjk9S8HTbA, AxLXI00L8zM)
- "If it's weak, it would trade through consequent encroachment ... and then lay bodies down inside
  this portion here. Once it does that, then this should act as inversion fair value gap": the idea
  is cancelled when a body closes through the CE into the far half (AxLXI00L8zM)
- direction follows the draw on liquidity (rXgEMIJ1fQg); if both sides can be argued there is no
  trade (AxLXI00L8zM)

Rules the dossiers leave subjective are fixed here and written in each signal's reasons:
the stop sits beyond the candle that started the move, the target is the nearest opposing
liquidity, and the order rests until 90 minutes after the anchor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

from . import advisor, aplus, conditions, htf
from .config import InstrumentConfig, StrategyConfig
from .ict2022 import ICT2022Strategy, _Setup
from .sessions import Bar, Level, to_ny, trading_day
from .strategy import LONG, SHORT, Signal

ANCHORS = (("london_open", time(1, 30)), ("ny_open_fx", time(7, 0)), ("ny_open", time(9, 30)))
WINDOW_MIN = 30     # the gap must complete inside the first half hour
EXPIRY_MIN = 90     # the order rests until the end of the morning window
LOOKBACK = 6        # candles before the gap checked for a balanced range


@dataclass
class FirstPresentedStrategy(ICT2022Strategy):
    widen_tight_stops = False   # a block strategy sets this: a stop that is inside the market's noise is moved out, not rejected
    _done: set = field(default_factory=set)        # (day, anchor) pairs already used
    _window_open: dict = field(default_factory=dict)

    def _anchor_for(self, ts: datetime):
        t = to_ny(ts).time()
        for name, a in ANCHORS:
            start = datetime.combine(to_ny(ts).date(), a, tzinfo=to_ny(ts).tzinfo)
            if a <= t and to_ny(ts) < start + timedelta(minutes=WINDOW_MIN):
                return name, start
        return None

    def on_bar(self, bar: Bar) -> Signal | None:
        cfg, pip = self.config, self.instrument.pip_size
        step = timedelta(minutes=cfg.timeframe_minutes)
        close_time = bar.time + step
        self._bars.append(bar)
        idx = len(self._bars) - 1
        if self._htf() is not None:
            self._htfa.feed(bar)
        if self._rung()["tf"] < 1440:
            if self._lad is None:
                self._lad = htf.HtfArrays((60, 240), clock=self._time_based())
            self._lad.feed(bar)
        ny = to_ny(bar.time)
        if ny.date() != self._midnight_date:
            self._midnight_date, self._midnight_open = ny.date(), bar.open
        iso = ny.isocalendar()
        if (iso[0], iso[1]) != self._week_id:
            self._week_id, self._week_open = (iso[0], iso[1]), bar.open
        tday = trading_day(bar.time, cfg.day_rollover)
        d_ = self._days.get(tday)
        self._days[tday] = [bar.open, bar.high, bar.low, bar.close] if d_ is None else [d_[0], max(d_[1], bar.high), min(d_[2], bar.low), bar.close]
        if len(self._days) > max(65, self.config.ipda_days + 5, max(self.config.ipda_lookbacks) + 5):
            self._days.pop(min(self._days))
        if cfg.opening_gaps:
            self._update_gaps(bar)
        self._prev = bar
        if self._tracker.update(bar):
            self._setups.clear()
            self._spent.clear()
            self._swept_sides.clear()
            self._signals_today = 0
        for lv in self._all_levels():           # a level traded through is spent, as in the other models
            if (lv.side == "high" and bar.high >= lv.price) or (lv.side == "low" and bar.low <= lv.price):
                self._spent.add(lv.name)

        fired: Signal | None = None
        keep: list[_Setup] = []
        for s in self._setups:                  # orders resting in a presented gap
            s.bars += 1
            d = s.direction
            ce = (s.fvg[0] + s.fvg[1]) / 2
            hit_target = bar.low <= s.target if d == SHORT else bar.high >= s.target
            hit_stop = bar.high >= s.stop if d == SHORT else bar.low <= s.stop
            filled = bar.high >= s.limit if d == SHORT else bar.low <= s.limit
            if filled and fired is None and self._signals_today < cfg.max_signals_per_day:
                rr = (s.target - s.limit) * d / ((s.stop - s.limit) * -d)
                reasons = (
                    *(s.pre or (
                        f"FP-01 first fair value gap presented after the {s.window} anchor: [{self._fmt(s.fvg[0])}, {self._fmt(s.fvg[1])}]",
                        "FP-03 the gap breaks away from the candles before it (not inside a balanced range)",
                        f"RL-05 not against the daily draw ({self.daily_bias()[1]})",
                    )),
                    f"EN-01 limit at {self._fmt(s.limit)} ({s.name}), filled",
                    f"SL-01 stop {self._fmt(s.stop)} beyond the pattern's extreme",
                    f"TP-01 target {s.target_name} {self._fmt(s.target)}",
                    f"TP-02 {rr:.2f}R >= {cfg.min_rr}R",
                    *([advisor.line(s.advice)] if s.advice else []),
                    *aplus.line(s.advice),
                    *([conditions.tag(s.conditions)] if s.conditions else []),
                )
                self._signals_today += 1
                self._log(close_time, "SIGNAL", f"{'SHORT' if d == SHORT else 'LONG'} limit filled in the {s.name}, {rr:.2f}R")
                self._ev("filled", bar.time, level=s.level.name, dir=d, price=s.limit)
                meta = {"level_price": s.level.price, "sweep_price": s.extreme, "sweep_time": s.sweep_time,
                        "mss_level": s.mss_level, "mss_time": s.mss_time, "fvg": s.fvg, "fvg_time": s.fvg_time,
                        "midnight_open": None, "target_name": s.target_name, "entry_mode": cfg.model if cfg.model != "ict2022" else "fpfvg", "swing_time": s.swing_time,
                        "tp1": None, "levels": list(s.levels), "entry_kind": "fvg", "advice": s.advice, "conditions": s.conditions}
                fired = Signal(close_time, d, s.limit, s.stop, s.target, rr, s.level, s.window, reasons,
                               entry_type="limit", meta=meta)
                continue
            if hit_target or hit_stop:
                self._log(close_time, "EN-04", f"{s.window} {s.name} limit cancelled: price reached the {'target' if hit_target else 'stop'} before filling")
                continue
            if s.ce_cancel and ((bar.close < ce) if d == LONG else (bar.close > ce)):  # judged on the close, and only if the order did not fill
                self._log(close_time, "FP-02", f"{s.window} {s.name} cancelled: a body closed through the consequent encroachment {self._fmt(ce)} (it becomes an inversion FVG)")
                continue
            if s.bars >= (s.expiry or max(1, EXPIRY_MIN // cfg.timeframe_minutes)):
                self._log(close_time, "EN-04", f"{s.window} {s.name} limit cancelled: not filled before it expired")
                continue
            keep.append(s)
        self._setups = keep

        self._detect(idx, close_time, pip)
        if cfg.liquidity_pools:
            self._update_pools(idx)
        self.staged = [{"direction": "long" if s.direction == LONG else "short", "limit": s.limit, "stop": s.stop,
                        "target": s.target, "level": s.level.name, "fvg": s.fvg, "bars_left": (s.expiry or self._expiry_bars()) - s.bars,
                        "midnight_open": self._midnight_open, "tp1": None, "target_name": s.target_name,
                        "mss_level": s.mss_level, "fvg_time": s.fvg_time, "staged_time": s.mss_time}
                       for s in self._setups]
        return fired

    def _detect(self, idx: int, close_time: datetime, pip: float) -> None:
        """Look for the first presented gap: the third candle of the triple has just closed."""
        if idx >= 2:
            anchor = self._anchor_for(self._bars[idx - 2].time)
            if anchor and close_time <= anchor[1] + timedelta(minutes=WINDOW_MIN):  # the third candle completes inside the half hour
                key = (to_ny(self._bars[idx].time).date(), anchor[0])
                if key not in self._done:
                    self._try_gap(idx, anchor[0], key, close_time, pip)

    def _stage(self, d: int, level: Level, entry: float, stop: float, zone: tuple[float, float], idx: int, close_time: datetime,
               window: str, name: str, pre: tuple = (), ce_cancel: bool = True, expiry: int = 0, extreme: float | None = None,
               sweep_time: datetime | None = None, swing_time: datetime | None = None, target: Level | None = None) -> bool:
        """Check the stop and the reward, then rest a limit order for a pattern. True when it was staged."""
        cfg, pip, b = self.config, self.instrument.pip_size, self._bars
        if not all(math.isfinite(v) for v in (entry, stop, *zone)):
            self._log(close_time, "SL-03", f"{window} {name} rejected: a price in the setup is not a number")
            return False
        risk = (entry - stop) * d
        risk_pips = risk / pip
        if self.widen_tight_stops and risk > 0 and risk_pips < cfg.min_stop_pips:
            # A stop just past a small block is inside the market's own noise and gets run. ICT puts it beyond the
            # array with room; here that room is the market's minimum stop, so a good block is not thrown away.
            stop = entry - d * cfg.min_stop_pips * pip
            risk, risk_pips = (entry - stop) * d, float(cfg.min_stop_pips)
            self._log(close_time, "SL-03", f"{window} {name}: stop widened to the {cfg.min_stop_pips} pip minimum, beyond the noise")
        if risk <= 0 or risk_pips < cfg.min_stop_pips or risk_pips > cfg.max_stop_pips:
            self._log(close_time, "SL-03", f"{window} {name} rejected: stop {risk_pips:.1f} pips outside [{cfg.min_stop_pips}, {cfg.max_stop_pips}]")
            return False
        tgt = target if target is not None and (target.price - entry) * d / risk >= cfg.min_rr else self._target(d, entry)
        if tgt is None:
            self._log(close_time, "TP-01", f"{window} {name} rejected: no opposing liquidity to target")
            return False
        if not math.isfinite(tgt.price):
            self._log(close_time, "TP-01", f"{window} {name} rejected: the target is not a number")
            return False
        rr = (tgt.price - entry) * d / risk
        if rr < cfg.min_rr:
            self._log(close_time, "TP-02", f"{window} {name} rejected: {rr:.2f}R to {tgt.name} is below {cfg.min_rr}R")
            return False
        s = _Setup(level, d, extreme if extreme is not None else stop, idx, stage="limit", window=window, sweep_time=sweep_time or b[idx].time)
        s.limit, s.stop, s.target, s.target_name, s.fvg, s.mss_level = entry, stop, tgt.price, tgt.name, zone, entry
        s.mss_time, s.fvg_time, s.swing_time, s.entry_kind = b[idx].time, b[max(0, idx - 1)].time, swing_time or b[idx].time, "fvg"
        s.levels = tuple({"name": lv.name, "price": lv.price, "side": lv.side, "spent": lv.name in self._spent} for lv in self._all_levels())
        s.name, s.pre, s.ce_cancel, s.expiry = name, pre, ce_cancel, expiry
        mb = self._median_body()
        bm = max(abs(b[k].close - b[k].open) for k in range(max(0, idx - 3), idx + 1)) / mb if mb else None
        mr = self._median_range() or 1e-12
        adv_zone = zone if (zone[1] - zone[0]) >= 0.2 * mr else (entry - 0.15 * mr, entry + 0.15 * mr)
        tday = trading_day(b[idx].time, cfg.day_rollover)
        s.advice = self._advise(d, level, entry, tgt, rr, adv_zone, s.extreme, self._anchor_win(window), idx, bm, tday)
        s.conditions = self.read_conditions(d, self._obstacles(d, entry, tgt.price, idx))
        self._setups.append(s)
        self._ev("staged", b[idx].time, level=level.name, dir=d, limit=entry, stop=stop, target=tgt.price, target_name=tgt.name,
                 tp1=None, fvg=zone, fvg_time=s.fvg_time, rr=rr, rr1=None)
        self._log(close_time, "EN-01", f"{'BUY' if d == LONG else 'SELL'} LIMIT staged at {self._fmt(entry)} ({name}), stop {self._fmt(stop)}, target {tgt.name} {self._fmt(tgt.price)} ({rr:.2f}R)")
        return True

    def _anchor_win(self, name: str):
        from .config import Window
        return Window(name, time(0, 0), time(0, 0))

    def _expiry_bars(self) -> int:
        return max(1, EXPIRY_MIN // self.config.timeframe_minutes)

    def _try_gap(self, idx: int, window: str, key, close_time: datetime, pip: float) -> None:
        cfg, b = self.config, self._bars
        c1, c2, c3 = b[idx - 2], b[idx - 1], b[idx]
        if c3.low > c1.high:
            d, lo, hi = LONG, c1.high, c3.low
        elif c3.high < c1.low:
            d, lo, hi = SHORT, c3.high, c1.low
        else:
            return
        self._done.add(key)                      # the first gap is the only one; later gaps are not "first presented"
        ev_time = c2.time
        before = b[max(0, idx - 2 - LOOKBACK):idx - 2]
        if before and lo >= min(x.low for x in before) and hi <= max(x.high for x in before):
            self._log(close_time, "FP-03", f"{window} first presented FVG rejected: the gap sits inside the range of the candles before it (not breaking away)")
            return
        why = self._story_against(d)
        if why:
            self._log(close_time, "NR-01", f"{window} first presented FVG ignored: the story does not back it ({why})")
            return
        ce = (lo + hi) / 2
        stop = (c1.low - cfg.stop_buffer_pips * pip) if d == LONG else (c1.high + cfg.stop_buffer_pips * pip)
        risk = (ce - stop) * d
        risk_pips = risk / pip
        if risk <= 0 or risk_pips < cfg.min_stop_pips or risk_pips > cfg.max_stop_pips:
            self._log(close_time, "SL-03", f"{window} first presented FVG rejected: stop {risk_pips:.1f} pips outside [{cfg.min_stop_pips}, {cfg.max_stop_pips}]")
            return
        tgt = self._target(d, ce)
        if tgt is None:
            self._log(close_time, "TP-01", f"{window} first presented FVG rejected: no opposing liquidity to target")
            return
        rr = (tgt.price - ce) * d / risk
        if rr < cfg.min_rr:
            self._log(close_time, "TP-02", f"{window} first presented FVG rejected: {rr:.2f}R to {tgt.name} is below {cfg.min_rr}R")
            return
        level = Level(f"{window} FPFVG", ce, "low" if d == LONG else "high")
        s = _Setup(level, d, c1.low if d == LONG else c1.high, idx - 2, stage="limit", window=window, sweep_time=c1.time)
        s.limit, s.stop, s.target, s.target_name, s.fvg, s.mss_level = ce, stop, tgt.price, tgt.name, (lo, hi), ce
        s.mss_time, s.fvg_time, s.swing_time, s.entry_kind = close_time - timedelta(minutes=cfg.timeframe_minutes), c2.time, c1.time, "fvg"
        s.levels = tuple({"name": lv.name, "price": lv.price, "side": lv.side, "spent": lv.name in self._spent} for lv in self._all_levels())
        mb = self._median_body()
        bm = abs(c2.close - c2.open) / mb if mb else None
        tday = trading_day(b[idx].time, cfg.day_rollover)
        s.advice = self._advise(d, level, ce, tgt, rr, (lo, hi), s.extreme, self._anchor_win(window), idx, bm, tday)
        s.conditions = self.read_conditions(d, self._obstacles(d, ce, tgt.price, idx))
        if s.conditions.get("phase") in ("off", "killzone"):
            s.conditions["phase"] = "open"  # this model only fires in the first half hour of an open
        self._setups.append(s)
        self._ev("staged", ev_time, level=level.name, dir=d, limit=ce, stop=stop, target=tgt.price, target_name=tgt.name,
                 tp1=None, fvg=(lo, hi), fvg_time=c2.time, rr=rr, rr1=None)
        self._log(close_time, "EN-01", f"{'BUY' if d == LONG else 'SELL'} LIMIT staged at {self._fmt(ce)} (50% of the first presented FVG [{self._fmt(lo)}, {self._fmt(hi)}]), stop {self._fmt(stop)}, target {tgt.name} {self._fmt(tgt.price)} ({rr:.2f}R)")
