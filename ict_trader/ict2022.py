"""The ICT 2022 Mentorship model, as a deterministic state machine.

This is the full model from RULES.md, not just the sweep: every step must happen in
order, inside the time window, or the setup is cancelled.

  1. Time       (KZ-04/05/06/09) only inside a killzone (any hour on 1h/4h and synthetics);
                 (KZ-01)       the sweep (Judas swing) is above the NY midnight open for sells, below it for buys
  2. Liquidity  (SW-01, SW-02) price runs beyond a reference level (PDH/PDL, Asia, prior
                               week) by the minimum, in the right purge order
  3. MSS        (MS-01, MS-02) within `mss_max_bars`, a DISPLACEMENT candle (body at least
                               `displacement_body_mult` x the recent median body) closes
                               through the most recent opposite 3-candle swing point
  4. FVG        (FV-01, FV-02) the displacement leg leaves a 3-candle fair value gap, and
                               it sits in premium (shorts) or discount (longs) of the leg
  5. Entry      (EN-01/EN-02)  a LIMIT order at the gap's near edge (or 50% CE), valid for
                               3 candles / 15 minutes (EN-04); cancelled if price reaches the
                               target or the stop first
  6. Stop       (SL-01, SL-03) beyond the sweep extreme plus buffer, within min/max
  7. Target     (TP-01, TP-02) nearest opposing liquidity, at least `min_rr` from the limit

The signal is emitted when the limit order FILLS (Signal.entry_type == "limit", entry_ref
is the fill price). While waiting, the order is visible in `staged` so the app can show it
as a pending limit, exactly like a trader who has placed the order ahead of price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from datetime import date, datetime, time, timedelta, timezone
from statistics import median

from . import advisor, aplus, conditions, htf, narrative
from .pd_arrays import scan_pd_arrays
from .config import InstrumentConfig, StrategyConfig
from .sessions import Bar, Level, LevelTracker, in_any, to_ny, trading_day
from .strategy import LONG, SHORT, Decision, Signal


@dataclass
class _Setup:
    level: Level
    direction: int
    extreme: float          # sweep extreme (stop anchor)
    sweep_index: int
    bars: int = 0
    stage: str = "await_mss"  # await_mss -> limit
    limit: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    target_name: str = ""
    fvg: tuple[float, float] = (0.0, 0.0)
    mss_level: float = 0.0
    window: str = ""
    sweep_time: datetime | None = None
    mss_time: datetime | None = None
    fvg_time: datetime | None = None   # open time of the gap's middle candle
    swing_time: datetime | None = None  # the 3-candle swing the shift broke
    tp1: float | None = None            # TP-04 partial: the displacement leg's extreme
    entry_kind: str = "fvg"             # "fvg" (FV-01) or "ob" (OB-01 mean threshold)
    levels: tuple = ()                  # reference levels known when the order was placed
    advice: dict | None = None          # the adviser's score and reasons, from when the order was placed
    conditions: dict | None = None      # market conditions when the order was placed (conditions.py)
    reentry: bool = False               # RE-01: a second try after the first trade on this idea was stopped out
    name: str = "first presented FVG"   # what pattern placed it (pattern models)
    pre: tuple = ()                     # the pattern's own rules-trail lines, shown first when it fills
    ce_cancel: bool = True              # cancel if a body closes through the zone's midpoint before the fill
    expiry: int = 0                     # candles the order may rest (0: the model's default)
    story: str = ""                     # the narrative line the order was placed under (NR-01)


@dataclass
class ICT2022Strategy:
    config: StrategyConfig
    instrument: InstrumentConfig
    decisions: list[Decision] = field(default_factory=list)
    staged: list[dict] = field(default_factory=list)  # pending limit orders, for display
    _tracker: LevelTracker = field(init=False)
    _bars: list[Bar] = field(default_factory=list)
    _spent: set[str] = field(default_factory=set)
    _setups: list[_Setup] = field(default_factory=list)
    _swept_sides: list[str] = field(default_factory=list)
    _signals_today: int = 0
    _midnight_open: float | None = None
    _midnight_date: date | None = None
    # Every ICT event as it happens (sweep, shift, gap, order, fill, cancel), with candle
    # times and prices, so the app can mark the chart up the way ICT does.
    events: list[dict] = field(default_factory=list)
    _days: dict = field(default_factory=dict)  # trading day -> [open, high, low, close]
    _seeded: bool = False
    _week_open: float | None = None
    _week_id: tuple | None = None
    _osok_weeks: set = field(default_factory=set)
    _cbdr_acc: list | None = None     # [top, bottom] building for today's 14:00-20:00 NY window
    _cbdr: tuple | None = None        # the finished (top, bottom) of the last full window
    _live: list = field(default_factory=list)   # filled trades still open: {d, stop, target, level, extreme}
    _extra_signals: int = 0                      # tries added by re-entries today
    _reentries: int = 0
    _pools: list = field(default_factory=list)  # unswept swing / equal highs and lows: (Level, born)
    _gaps: list = field(default_factory=list)   # opening gaps: {"kind", "date", "hi", "lo"}
    _prev: Bar | None = None
    _lad: Any = None                  # 1h and 4h candles for the bias when the bias comes from below the daily (htf.ladder)
    _htfa: Any = None                 # the higher-timeframe candles and arrays built from these candles (htf.py)
    _fixed_key: Any = None            # which completed days the daily and weekly candles handed to _htfa were built from
    _pre17: float | None = None       # close of the last bar before 17:00 NY (the daily-open gap's anchor)
    _rth_close: float | None = None   # close of the last regular-hours bar (before 16:15 NY)

    def __post_init__(self) -> None:
        self._tracker = LevelTracker(self.config)

    # ---------------------------------------------------------------- helpers
    def _log(self, ts: datetime, rule: str, msg: str) -> None:
        self.decisions.append(Decision(ts, rule, msg))
        if " rejected" in msg or " cancelled" in msg or " ignored" in msg or "; spent" in msg:
            kind = "spent" if "; spent" in msg else "rejected" if " rejected" in msg else "ignored" if " ignored" in msg else "cancelled"
            bar_t = ts - timedelta(minutes=self.config.timeframe_minutes)
            self._ev(kind, bar_t, rule=rule, message=msg)

    def _ev(self, kind: str, t: datetime, **data) -> None:
        self.events.append({"type": kind, "time": t, **data})

    def _fmt(self, p: float) -> str:
        decimals = max(0, len(f"{self.instrument.pip_size:.10f}".rstrip("0").split(".")[1]) + 1)
        return f"{p:.{decimals}f}"

    def _median_body(self) -> float:
        bodies = [abs(b.close - b.open) for b in self._bars[-21:-1]]
        return median(bodies) if bodies else 0.0

    def _last_swing(self, direction: int, before: int, after: int) -> tuple[float, int] | None:
        """Most recent 3-candle swing low (for shorts) / high (for longs) in (after, before): (price, index)."""
        b = self._bars
        for i in range(min(before, len(b) - 2), max(after, 1) - 1, -1):
            if i - 1 < 0 or i + 1 >= len(b):
                continue
            if direction == SHORT and b[i].low < b[i - 1].low and b[i].low < b[i + 1].low:
                return b[i].low, i
            if direction == LONG and b[i].high > b[i - 1].high and b[i].high > b[i + 1].high:
                return b[i].high, i
        return None

    def _breaker(self, direction: int, swing_i: int) -> tuple[float, float] | None:
        """The breaker block: the order block at the swing that the displacement broke. For a sell,
        the last down-close candle at the swing low (the bullish order block that failed); for a
        buy, the last up-close candle at the swing high."""
        for i in range(swing_i, max(-1, swing_i - 6), -1):
            b = self._bars[i]
            if (b.close < b.open) if direction == SHORT else (b.close > b.open):
                return (b.low, b.high)
        return None

    def _fvg_in_leg(self, direction: int, start: int, end: int) -> tuple[float, float, int] | None:
        """Latest 3-candle FVG formed by candles in [start, end]: (low, high, middle candle index)."""
        b = self._bars
        min_size = self.config.fvg_min_range * self._median_range() if self.config.fvg_quality else 0.0
        for i in range(end, max(start + 2, 2) - 1, -1):
            c1, c3 = b[i - 2], b[i]
            if direction == SHORT and c1.low > c3.high:
                gap = (c3.high, c1.low, i - 1)
            elif direction == LONG and c1.high < c3.low:
                gap = (c1.high, c3.low, i - 1)
            else:
                continue
            if self.config.fvg_quality:
                ce = (gap[0] + gap[1]) / 2
                if gap[1] - gap[0] < min_size:
                    continue  # too small to be a real imbalance
                if any((x.high >= ce) if direction == SHORT else (x.low <= ce) for x in b[i + 1:end + 1]):
                    continue  # already traded back to its midpoint: mitigated
            return gap
        return None

    def _cisd(self, direction: int, sweep_i: int) -> tuple[float, int] | None:
        """Change in state of delivery: the open of the run of candles that delivered price into
        the sweep (for a sell, the first of the up-close candles that rallied into the high).
        A close back past it is the change. Returns (price, candle index)."""
        b, i = self._bars, sweep_i
        run = (lambda c: c.close > c.open) if direction == SHORT else (lambda c: c.close < c.open)
        if not run(b[i]):
            i -= 1
        first = None
        while i >= 0 and run(b[i]) and sweep_i - i < 12:
            first, i = i, i - 1
        return (b[first].open, first) if first is not None else None

    def _inverted_gap(self, direction: int, start: int, end: int, close: float) -> tuple[float, float, int] | None:
        """The latest gap formed the OTHER way in [start, end] that this close has gone through:
        for a sell, a bullish gap (support) closed below; for a buy, a bearish gap closed above.
        That gap inverts and becomes the entry (IFVG): (low, high, middle index)."""
        b = self._bars
        for i in range(end, max(start + 2, 2) - 1, -1):
            c1, c3 = b[i - 2], b[i]
            if direction == SHORT and c1.high < c3.low and close < c1.high:
                return (c1.high, c3.low, i - 1)
            if direction == LONG and c1.low > c3.high and close > c1.low:
                return (c3.high, c1.low, i - 1)
        return None

    def _prior_range(self) -> tuple[float, float] | None:
        """Yesterday's high and low: the dealing range for premium and discount."""
        days = sorted(self._days)[:-1]
        if not days:
            return None
        _, h, lo, _ = self._days[days[-1]]
        return h, lo

    def narrative(self) -> "narrative.Narrative":
        """ICT's top-down story for this market right now (narrative.py); read once per candle."""
        key = (len(self._bars), self._bars[-1].time if self._bars else None)
        if getattr(self, "_story_key", None) != key:
            self._story, self._story_key = narrative.read(self), key
        return self._story

    def daily_bias(self) -> tuple[int, str]:
        """RL-05: the daily draw from the last two completed trading days (D-1 against D-2).

        Bullish when yesterday closed above the prior day's high, or ran its low and closed
        back inside; bearish in mirror. Anything else is an inside day: no clear draw (0).
        """
        days = sorted(self._days)[:-1]  # the current day is still forming
        if len(days) < 2:
            return 0, "not enough completed days yet"
        o2, h2, l2, c2 = self._days[days[-2]]
        o1, h1, l1, c1 = self._days[days[-1]]
        f = self._fmt
        if c1 > h2:
            return LONG, f"yesterday closed {f(c1)} above the prior day's high {f(h2)}"
        if c1 < l2:
            return SHORT, f"yesterday closed {f(c1)} below the prior day's low {f(l2)}"
        if h1 > h2 and l1 >= l2:
            return SHORT, f"yesterday ran the prior day's high {f(h2)} and closed back inside (buy-side taken)"
        if l1 < l2 and h1 <= h2:
            return LONG, f"yesterday ran the prior day's low {f(l2)} and closed back inside (sell-side taken)"
        return 0, "inside or two-sided day: no clear daily draw"

    def _note_gap(self, kind: str, a: float, b: float, d: date) -> None:
        hi, lo = max(a, b), min(a, b)
        if hi - lo < self.config.opening_gap_min_range * self._median_range():
            return  # not a real gap
        self._gaps.append({"kind": kind, "date": d, "hi": hi, "lo": lo})
        keep = {"NDOG": 3, "NWOG": 2, "ORG": 1}
        same = [g for g in self._gaps if g["kind"] == kind]
        for g in same[:-keep[kind]]:
            self._gaps.remove(g)
        self._ev("gap", self._bars[-1].time, gap=kind, hi=hi, lo=lo)

    def _update_gaps(self, bar: Bar) -> None:
        """Find the opening gaps as bars arrive (clock markets only, never from the future)."""
        ny, prev = to_ny(bar.time), self._prev
        t = ny.time()
        if prev is not None:
            pt = to_ny(prev.time).time()
            if bar.time - prev.time > timedelta(hours=24):
                self._note_gap("NWOG", prev.close, bar.open, ny.date())      # the weekend break
            elif t >= time(18, 0) and (pt < time(18, 0) or to_ny(prev.time).date() != ny.date()) and self._pre17 is not None:
                self._note_gap("NDOG", self._pre17, bar.open, ny.date())     # 17:00 close to 18:00 open
            if self.instrument.asset_class == "index" and t >= time(9, 30) and (pt < time(9, 30) or to_ny(prev.time).date() != ny.date()) \
                    and self._rth_close is not None:
                self._note_gap("ORG", self._rth_close, bar.open, ny.date())  # prior settlement to the 09:30 open
        if t < time(17, 0):
            self._pre17 = bar.close
        if time(9, 30) <= t < time(16, 15):
            self._rth_close = bar.close

    def _cbdr_levels(self) -> list[Level]:
        if not (self.config.cbdr_targets and self._time_based()):
            return []
        if self._cbdr:
            top, bot = self._cbdr
            w = top - bot
            max_w = 40.0 * self.instrument.pip_size if self.instrument.asset_class == "forex" else float("inf")
            if 0 < w <= max_w:
                return [Level(f"CBDR+{k}SD", top + k * w, "high") for k in (1, 2, 3)] + [Level(f"CBDR-{k}SD", bot - k * w, "low") for k in (1, 2, 3)]
        lvs = {lv.name: lv.price for lv in self._tracker.levels()}
        if "ASIA_H" in lvs and "ASIA_L" in lvs and lvs["ASIA_H"] > lvs["ASIA_L"]:
            top, bot = lvs["ASIA_H"], lvs["ASIA_L"]
            w = top - bot
            return [Level(f"ASIA+{k}SD", top + k * w, "high") for k in (1, 2, 3)] + [Level(f"ASIA-{k}SD", bot - k * w, "low") for k in (1, 2, 3)]
        return []

    def _gap_levels(self) -> list[Level]:
        if not (self.config.opening_gaps and self._time_based()):
            return []
        out = []
        for g in self._gaps:
            tag = f"{g['kind']} {g['date']:%m-%d}"
            out += [Level(f"{tag} H", g["hi"], "high"), Level(f"{tag} L", g["lo"], "low")]
        return out

    def _gap_midpoints(self) -> list[Level]:
        """Consequent encroachment of each opening gap: a draw for either side, named for the level."""
        if not (self.config.opening_gaps and self._time_based()):
            return []
        return [Level(f"{g['kind']} {g['date']:%m-%d} CE", (g["hi"] + g["lo"]) / 2, "mid") for g in self._gaps]

    def _first_partial(self, d: int, limit: float, risk: float, target: float, leg_x: float, idx: int) -> float | None:
        """TP-04, the first partial: the displacement leg's extreme when it lies between the entry and the target, and never beyond the first
        opposing array on the road (ICT pays himself before the wall, not into it) when that is at least half an R away."""
        tp1 = leg_x if (limit - leg_x) * d < 0 and (target - leg_x) * d > 0 else None
        if self.config.tp1_before_wall:
            walls = self._walls(d, limit, target, idx)
            edge = (walls[0][0] if d == LONG else walls[-1][1]) if walls else None
            if edge is not None and (edge - limit) * d >= 0.5 * risk and (tp1 is None or (tp1 - edge) * d > 0):
                return edge
        return tp1

    def _obstacles(self, direction: int, entry: float, target: float, idx: int) -> int:
        """How many opposing PD arrays stand between the entry and its target (see _walls): ICT takes a trade only when the road is clear."""
        return len(self._walls(direction, entry, target, idx))

    def _walls(self, direction: int, entry: float, target: float, idx: int) -> list[tuple[float, float]]:
        """The opposing PD arrays standing between the entry and its target, nearest the low end first, as (low, high) zones.
        Counted: the fair value gaps price has not traded back into, the order blocks, breakers, inversion gaps, volume imbalances and
        rejection blocks that are unmitigated and unbroken, and (top-down) the 1h / 4h / daily / weekly arrays. Arrays that overlap are
        one wall, not two."""
        b, zones = self._bars, []
        lo_e, hi_e = sorted((entry, target))
        rng = sorted(x.high - x.low for x in b[max(0, idx - 19):idx + 1])
        med_rng = rng[len(rng) // 2] if rng else 0.0
        floor = 0.3 * med_rng                      # a sliver (a body gap a third of a candle high) is noise, not a wall
        for i in range(max(2, idx - 150), idx + 1):
            c1, c3 = b[i - 2], b[i]
            if direction == LONG and c1.low > c3.high:
                lo, hi = c3.high, c1.low           # a bearish gap
                if entry < lo and hi < target and not any(x.high >= (lo + hi) / 2 for x in b[i + 1:idx + 1]):
                    zones.append((lo, hi))
            elif direction == SHORT and c1.high < c3.low:
                lo, hi = c1.high, c3.low           # a bullish gap
                if target < lo and hi < entry and not any(x.low <= (lo + hi) / 2 for x in b[i + 1:idx + 1]):
                    zones.append((lo, hi))
        base = max(0, idx - 150)
        try:
            arrays = scan_pd_arrays(b[base:idx + 1], lookback=150)
        except Exception:
            arrays = []
        for a in arrays:
            if a.kind == "fvg" or a.direction != -direction or a.broken or a.top - a.bottom < floor or not (lo_e < a.bottom and a.top < hi_e):
                continue
            later = b[base + a.bar_index + 1:idx + 1]
            if a.mitigated or any((x.high >= a.midpoint if direction == LONG else x.low <= a.midpoint) for x in later):
                continue                           # price has already traded back into it: it no longer stands in the way
            zones.append((a.bottom, a.top))
        h = self._htf()
        if h is not None:
            zones += [(a.lo, a.hi) for tf in h.all_tfs for a in h.arrays(tf) if a.dir == -direction and lo_e < a.lo and a.hi < hi_e]
        out: list[tuple[float, float]] = []
        for lo, hi in sorted(zones):               # overlapping zones are one wall
            if out and lo <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], hi))
            else:
                out.append((lo, hi))
        return out

    def _seek_destroy(self, idx: int) -> bool:
        """Both sides of the recent range stopped out in turn: price runs a 12-bar high by at least
        min_sweep_pips and closes back under it, then runs the low and closes back over it (or the
        reverse), at least three times in 36 candles. ICT's rule is to stand aside on such a day."""
        b, seq = self._bars, []
        tol = max(self.config.min_sweep_pips * self.instrument.pip_size, 0.0)
        for i in range(max(12, idx - 35), idx + 1):
            prev = b[i - 12:i]
            hi, lo = max(x.high for x in prev), min(x.low for x in prev)
            tag = "U" if (b[i].high > hi + tol and b[i].close < hi) else "D" if (b[i].low < lo - tol and b[i].close > lo) else None
            if tag and (not seq or seq[-1] != tag):
                seq.append(tag)
        return len(seq) >= 3

    def read_conditions(self, d: int = 0, obstacles: int | None = None, sd: bool | None = None) -> dict:
        """Market conditions now (see conditions.py). With a trade direction `d`, the draw and the path
        are read against that trade; without one they describe the market."""
        b, idx = self._bars, len(self._bars) - 1
        cfg = self.config
        bias = self.daily_bias()[0]
        draw = ("with" if bias == d else "against" if bias else "none") if d else ("up" if bias > 0 else "down" if bias < 0 else "none")
        if not self._time_based():
            phase = "24h"
        else:
            ny = to_ny(b[idx].time)
            mins = ny.hour * 60 + ny.minute
            phase = "open" if any(a <= mins < a + 30 for a in (90, 420, 570)) else "killzone" if in_any(cfg.killzones, b[idx].time) else "off"
        if sd is None:
            sd = self._seek_destroy(idx) if idx >= 48 else False
        path = "seek" if sd else (None if obstacles is None else "clear" if obstacles == 0 else "busy" if obstacles >= 2 else "some")
        return {"trend": conditions.label_trend([x.close for x in b[-49:]]), "state": conditions.label_state([x.close for x in b]),
                "vol": conditions.label_vol([x.high - x.low for x in b[-300:]]),
                "draw": draw, "phase": phase, "path": path}

    def _advise(self, d: int, level: Level, limit: float, tgt: Level, rr: float, zone: tuple[float, float],
                extreme: float, win, idx: int, body_mult: float | None, tday) -> dict:
        """The adviser's read of this setup (see advisor.py): what the engine knows, scored."""
        def side(rng):
            """How far into the right half of the range the entry is: +1 at the extreme (the deepest discount for a buy, the
            highest premium for a sell, ICT's outer octant), above +0.5 in the outer quadrant, 0 at equilibrium, negative on the wrong side."""
            if not rng or rng[1] == rng[0]:
                return 0
            return max(-1.0, min(1.0, ((rng[0] + rng[1]) / 2 - limit) * d / (abs(rng[0] - rng[1]) / 2)))
        ip, ip_name = self._gate_range() if self.config.htf_context else (None, "")
        timed, ny = self._time_based(), to_ny(self._bars[idx].time)
        mo = self._midnight_open
        mr = self._median_range() or 1e-12
        cfg_htf = self.config.htf_arrays
        lb = self.ladder_bias()
        return advisor.assess({
            "dir": d, "bias": lb["p"][0], "wbias": lb["u"][0], "bias_name": lb["names"][0], "wbias_name": lb["names"][1], "level": level.name,
            "body_mult": body_mult, "gap_ratio": (zone[1] - zone[0]) / mr,
            "pd": side(lb["range"]), "pd20": (side(ip) if ip else None), "pd20_name": ip_name,
            "obstacles": self._obstacles(d, limit, tgt.price, idx), "sd": self._seek_destroy(idx),
            "rr": rr, "target": tgt.name, "window": win.name if win else None, "timed": timed,
            "weekday": ny.weekday(), "hour": ny.hour,
            "midnight_ok": ((extreme - mo) * d <= 0) if (timed and mo is not None) else None,
            "profile": narrative.weekly_profile(self, d), "adr": self._adr(), "entry_price": limit, "target_price": tgt.price, "htf_trend": self._htf_trend(d), "poi": self._htf_poi(d, level, zone) if cfg_htf else None,
        })

    def _adr(self) -> tuple[float, float, float, float] | None:
        """(the average daily range of the last 20 completed days, the range today has made so far, today's high, today's low): price only, so it holds on a
        synthetic index too. ICT sizes his expectation by it: once a day has spent its range, the expansion is usually done."""
        days = sorted(self._days)
        if len(days) < 6:
            return None
        rngs = [self._days[x][1] - self._days[x][2] for x in days[:-1][-20:]]
        adr = sum(rngs) / len(rngs)
        today = self._days[days[-1]]
        return (adr, today[1] - today[2], today[1], today[2]) if adr > 0 else None

    def _median_range(self) -> float:
        rs = [b.high - b.low for b in self._bars[-21:-1]]
        return median(rs) if rs else 0.0

    def seed_days(self, daily: list[Bar], before=None) -> None:
        """Start the higher-timeframe record from daily candles (the live scan only loads a few days of
        intraday candles, so without this the weekly and 20/40/60-day rules would have nothing to read).
        Call before feeding any bar. Days already covered by the intraday candles are left to them."""
        for b in sorted(daily, key=lambda x: x.time):
            if before is not None and b.time.date() >= before:
                continue
            self._days.setdefault(b.time.date(), [b.open, b.high, b.low, b.close])
        self._seeded = True

    def _htf_ranges(self) -> dict:
        """Completed higher-timeframe context from the trading days seen so far: the IPDA look-back
        range and the prior week's range (None while there is not enough history)."""
        days = sorted(self._days)[:-1]
        out: dict = {"ipda": None, "week": None, "ipda_all": {}}
        for n in sorted({self.config.ipda_days, *self.config.ipda_lookbacks}):
            if len(days) >= min(n, 5):
                recent = days[-n:]
                out["ipda_all"][n] = (max(self._days[d][1] for d in recent), min(self._days[d][2] for d in recent))
        out["ipda"] = out["ipda_all"].get(self.config.ipda_days)
        weeks: dict = {}
        for d in days:
            weeks.setdefault(d - timedelta(days=d.weekday()), []).append(d)
        keys = sorted(weeks)
        if self._days and keys and sorted(self._days)[-1] - timedelta(days=sorted(self._days)[-1].weekday()) == keys[-1]:
            keys = keys[:-1]  # the current week is still forming
        if keys:
            ds = weeks[keys[-1]]
            out["week"] = (max(self._days[d][1] for d in ds), min(self._days[d][2] for d in ds))
        return out

    def weekly_bias(self) -> tuple[int, str]:
        """The weekly draw: the last completed week against the one before it, read like the
        daily bias (a close beyond the prior week's range, or a run on it that closed back inside)."""
        days = sorted(self._days)[:-1]
        weeks: dict = {}
        for d in days:
            weeks.setdefault(d - timedelta(days=d.weekday()), []).append(d)
        keys = sorted(weeks)
        if self._days:
            cur = sorted(self._days)[-1]
            if keys and cur - timedelta(days=cur.weekday()) == keys[-1]:
                keys = keys[:-1]
        if len(keys) < 2:
            return 0, "not enough completed weeks yet"

        def agg(k):
            ds = weeks[k]
            return self._days[ds[0]][0], max(self._days[d][1] for d in ds), min(self._days[d][2] for d in ds), self._days[ds[-1]][3]
        _, h2, l2, _c2 = agg(keys[-2])
        _, h1, l1, c1 = agg(keys[-1])
        f = self._fmt
        if c1 > h2:
            return LONG, f"last week closed {f(c1)} above the prior week's high {f(h2)}"
        if c1 < l2:
            return SHORT, f"last week closed {f(c1)} below the prior week's low {f(l2)}"
        if h1 > h2 and l1 >= l2:
            return SHORT, f"last week ran the prior week's high {f(h2)} and closed back inside"
        if l1 < l2 and h1 <= h2:
            return LONG, f"last week ran the prior week's low {f(l2)} and closed back inside"
        return 0, "inside or two-sided week: no clear weekly draw"

    def _rung(self) -> dict:
        return htf.ladder(self.config.timeframe_minutes if self.config.bias_ladder else 60)

    def ladder_bias(self) -> dict:
        """The top-down bias for THIS execution timeframe (htf.ladder): the primary bias from the rung above it, the secondary from the rung
        above that, the structure trend of the primary timeframe, and its dealing range. On 15m and up this is exactly the daily / weekly /
        daily-structure read; on the 1m it is the 1h / 4h, on the 5m the 4h / daily."""
        r = self._rung()
        words = {1: "higher highs and higher lows", -1: "lower highs and lower lows", 0: "two-sided, no clear trend"}
        if r["tf"] >= 1440:
            s, sw = self.structure_bias()
            return {"p": self.daily_bias(), "u": self.weekly_bias(), "s": (s, sw), "names": (r["name"], r["up_name"]), "range": self._prior_range(), "range_name": "yesterday's range"}
        done = self._lad._done if self._lad is not None else {}
        c = done.get(r["tf"], [])
        sw = htf.trend_of([x.high for x in c[-60:]], [x.low for x in c[-60:]]) if len(c) >= 10 else 0
        u = htf.candle_bias(done.get(r["up"], []), self._fmt) if r["up"] in done else self.daily_bias()
        p = htf.candle_bias(c, self._fmt)
        if p[0] == 0 and sw != 0 and u[0] == sw:
            p = (sw, f"inside candle with {words[sw]}")
        last = c[-12:]
        return {"p": p, "u": u, "s": (sw, words[sw] if len(c) >= 10 else "not enough closed candles to read the structure"),
                "names": (r["name"], r["up_name"]), "range": (max(x.high for x in last), min(x.low for x in last)) if len(last) >= 6 else None,
                "range_name": f"the {r['name']} range"}

    def structure_bias(self) -> tuple[int, str]:
        """The DAILY structure over the last 40 completed days: higher highs and higher lows (+1), lower highs and lower lows (-1), or
        two-sided (0). A bias from yesterday's candle against the day before is two candles; this is the trend they sit inside."""
        days = sorted(self._days)[:-1][-40:]
        if len(days) < 10:
            return 0, "not enough completed days to read the daily structure"
        t = htf.trend_of([self._days[d][1] for d in days], [self._days[d][2] for d in days])
        return t, {1: "higher highs and higher lows", -1: "lower highs and lower lows", 0: "two-sided, no clear trend"}[t]

    def _all_levels(self) -> list[Level]:
        """Reference levels plus (when switched on) liquidity pools and the higher-timeframe extremes."""
        out = self._tracker.levels()  # a fresh list each call
        if self.config.liquidity_pools:
            out += [lv for lv, _ in self._pools]
        if self.config.opening_gaps:
            out += self._gap_levels()
        return out

    def _htf_levels(self) -> list[Level]:
        if not self.config.htf_context:
            return []
        r, out = self._htf_ranges(), []
        for n, (hi, lo) in r["ipda_all"].items():
            out += [Level(f"IPDA{n}_H", hi, "high"), Level(f"IPDA{n}_L", lo, "low")]
        if r["week"]:
            out += [Level("PWH", r["week"][0], "high"), Level("PWL", r["week"][1], "low")]
        return out

    def _update_pools(self, idx: int) -> None:
        """Track unswept swing highs/lows as pools; two within `pool_equal_range` of a median
        candle range make one equal-highs / equal-lows pool. Called after the bar is processed,
        so a pool never includes the bar that forms it."""
        b, f = self._bars, self._fmt
        taken = {n for n in self._spent}
        self._pools = [(lv, t) for lv, t in self._pools if lv.name not in taken and (b[-1].time - t) <= timedelta(hours=72)]
        if idx < 2:
            return
        tol = self.config.pool_equal_range * self._median_range()
        for side, price, is_swing in (("high", b[idx - 1].high, b[idx - 1].high > b[idx - 2].high and b[idx - 1].high > b[idx].high),
                                      ("low", b[idx - 1].low, b[idx - 1].low < b[idx - 2].low and b[idx - 1].low < b[idx].low)):
            if not is_swing:
                continue
            same = [(lv, t) for lv, t in self._pools if lv.side == side]
            near = [x for x in same if abs(x[0].price - price) <= tol]
            if near:
                best = max([price] + [x[0].price for x in near]) if side == "high" else min([price] + [x[0].price for x in near])
                self._pools = [x for x in self._pools if x not in near]
                lv = Level(f"{'EQH' if side == 'high' else 'EQL'} {f(best)}", best, side)
            else:
                lv = Level(f"{'SWH' if side == 'high' else 'SWL'} {f(price)}", price, side)
            self._pools.append((lv, b[idx - 1].time))
            mine = [x for x in self._pools if x[0].side == side]
            if len(mine) > 8:
                self._pools.remove(mine[0])

    def _story_against(self, d: int) -> str | None:
        """Why the story does not back a trade in direction d (None: it does). ICT builds the story first and a model only takes a raid that
        fits it: no clear draw is no trade. The spine asks at the sweep (NR-01); the models with their own trigger (Venom, the ATM method, the
        Market Maker, the first presented gap) ask here. Read once per candle."""
        if getattr(self, "_story_key", None) != len(self._bars):
            self._story, self._story_key = self.narrative(), len(self._bars)
        st = self._story
        if st.stand_aside:
            return f"stand aside, {st.stand_aside}"
        if st.bias != d:
            return "no clear draw" if not st.bias else f"against the {'bullish' if st.bias > 0 else 'bearish'} story"
        return None

    def _htf(self) -> "htf.HtfArrays | None":
        """The higher-timeframe arrays, when the top-down rule is on and there is a timeframe above this one to look at."""
        if self._htfa is None:
            tfs = tuple(t for t in self.config.htf_array_tfs if t > self.config.timeframe_minutes) if self.config.htf_arrays else ()
            self._htfa = htf.HtfArrays(tfs, clock=self._time_based()) if tfs else False
        h = self._htfa or None
        if h is not None:
            self._sync_fixed(h)
        return h

    def _sync_fixed(self, h: "htf.HtfArrays") -> None:
        """Hand the completed days and weeks to the higher-timeframe arrays, so the daily and weekly arrays are read like the 1h and 4h
        (the nesting test: an entry that sits in a 4h array inside a daily one is stronger than one in the 4h alone)."""
        today = max(self._days) if self._days else None
        days = sorted(self._days)[:-1]
        key = (len(days), days[-1] if days else None)
        if key == self._fixed_key:
            return
        self._fixed_key = key
        at = lambda d: datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        h.set_fixed(1440, [Bar(at(d), *self._days[d]) for d in days[-60:]])
        weeks: dict = {}
        for d in days:
            if today is not None and d - timedelta(days=d.weekday()) != today - timedelta(days=today.weekday()):   # not the week still forming
                weeks.setdefault(d - timedelta(days=d.weekday()), []).append(d)
        h.set_fixed(10080, [Bar(at(k), self._days[v[0]][0], max(self._days[d][1] for d in v), min(self._days[d][2] for d in v), self._days[v[-1]][3])
                            for k, v in sorted(weeks.items())])

    def _array_levels(self) -> list[Level]:
        """Continuation (`array_levels`): with the trend ICT does not wait for a named level to be raided. He waits for price to retrace into
        a higher-timeframe array on the side of the bias and run the stops just beyond it. These are those stops: the low of each bullish
        array below price when the bias is up, the high of each bearish one above it when the bias is down (the nearest three)."""
        h = self._htf() if self.config.array_levels else None
        if h is None or not self._bars:
            return []
        p, px, near = self.ladder_bias()["p"][0], self._bars[-1].close, []
        if not p:
            return []
        for tf in h.all_tfs:
            for a in h.arrays(tf):
                if a.dir != p or not (a.hi < px if p > 0 else a.lo > px):
                    continue
                tag = f"{htf.TF_NAMES.get(tf, tf)} {a.kind} {a.time:%m-%d %H}"
                near.append((abs(px - (a.hi if p > 0 else a.lo)), Level(tag + (" low" if p > 0 else " high"), a.lo if p > 0 else a.hi, "low" if p > 0 else "high")))
        return [lv for _, lv in sorted(near, key=lambda x: x[0])[:3]]

    def _gate_range(self) -> tuple[tuple[float, float] | None, str]:
        """The range HT-01 asks about, and its name. The IPDA look-back on the daily scale; a 1 to 3 minute chart is one rung under the
        4-hour, so there the 20-day range is two rungs away and the 4-hour range is asked instead."""
        if self._rung()["tf"] < 240:
            c = self._lad._done.get(240, [])[-12:] if self._lad is not None else []
            return ((max(x.high for x in c), min(x.low for x in c)) if len(c) >= 6 else None), "the 4-hour range"
        ranges = self._htf_ranges().get("ipda_all", {})
        for n in (20, 40, 60):
            if n in ranges and ranges[n]:
                return ranges[n], f"the {n}-day IPDA range"
        return self._htf_ranges().get("ipda"), f"the {self.config.ipda_days}-day range"

    def _htf_trend(self, d: int) -> dict | None:
        """The 1h and 4h structure trend against a trade in direction d: +1 with it, -1 against, 0 neutral (None: no higher timeframe to read)."""
        h = self._htf()
        return {tf: h.trend(tf) * d for tf in h.tfs} if h is not None else None

    def _htf_poi(self, d: int, level: Level, zone: tuple[float, float]) -> dict | None:
        """Is this entry where the higher timeframe says price may turn? None: no higher timeframe to ask (the rule does not apply).
        Otherwise {"state": "pool"} when the raid was on a higher-timeframe pool (prior day or week, an IPDA range), {"state": "array", ...}
        when the zone sits at an unmitigated 1h / 4h fair value gap or order block, {"state": "none"} when it sits at neither."""
        h = self._htf()
        if h is None:
            return None
        if level.name.startswith(narrative.EXTERNAL):
            return {"state": "pool", "text": f"the raid was on {level.name}, a higher-timeframe pool"}
        a = h.at(d, min(zone), max(zone))
        if a is not None:
            nest = h.nested(d, min(zone), max(zone))
            return {"state": "array", "tf": a.tf, "kind": a.kind, "lo": a.lo, "hi": a.hi, "nested": nest,
                    "text": "entry at a " + a.text(self._fmt) + (f" (arrays on {len(nest)} timeframes stack here: {', '.join(nest)})" if len(nest) > 1 else "")}
        return {"state": "none", "text": "the entry is at no unmitigated 1h, 4h, daily or weekly array (gap, order block, breaker, void or rejection block) of the right direction"}

    def _expiry_bars(self) -> int:
        cfg = self.config
        return max(cfg.entry_expiry_bars, math.ceil(cfg.entry_expiry_minutes / cfg.timeframe_minutes))

    def _time_based(self) -> bool:
        """False when the rules run at any hour (1h/4h, synthetics): no time theory then."""
        if not self.config.killzones:
            return False
        return not any(w.name in ('24h', 'any') or (w.start.hour == 0 and w.end.hour >= 23) for w in self.config.killzones)

    def draw_pool(self, direction: int, ref: float) -> Level | None:
        return narrative.draw_pool(self, direction, ref)

    def _target(self, direction: int, entry: float) -> Level | None:
        side = "low" if direction == SHORT else "high"
        cands = self._all_levels()
        if self.config.htf_context:
            cands += self._htf_levels()
        if self.config.opening_gaps:
            cands += self._gap_midpoints()
        cands += self._cbdr_levels()
        opp = [lv for lv in cands if lv.side in (side, "mid") and (lv.price - entry) * direction > 0 and lv.name not in self._spent]
        if not opp:
            return None
        if self.config.target_pref == "draw":
            return narrative.draw_pool(self, direction, entry)  # the draw on liquidity: the same pool the story names
        return min(opp, key=lambda lv: abs(lv.price - entry))

    # ---------------------------------------------------------------- main
    def on_bar(self, bar: Bar) -> Signal | None:
        cfg, pip = self.config, self.instrument.pip_size
        close_time = bar.time + timedelta(minutes=cfg.timeframe_minutes)
        self._bars.append(bar)  # kept whole: setup indices point into it
        idx = len(self._bars) - 1
        if self._htf() is not None:
            self._htfa.feed(bar)
        if self._rung()["tf"] < 1440:
            if self._lad is None:
                self._lad = htf.HtfArrays((60, 240), clock=self._time_based())
            self._lad.feed(bar)
        if cfg.opening_gaps:
            self._update_gaps(bar)
        self._prev = bar
        ny_day = to_ny(bar.time).date()
        if ny_day != self._midnight_date:  # first candle of the New York calendar day (KZ-01)
            self._midnight_date, self._midnight_open = ny_day, bar.open

        iso = to_ny(bar.time).isocalendar()
        if (iso[0], iso[1]) != self._week_id:  # the first candle of the New York week is the weekly open
            self._week_id, self._week_open = (iso[0], iso[1]), bar.open
        if cfg.cbdr_targets and self._time_based():
            hm = to_ny(bar.time).hour * 60 + to_ny(bar.time).minute
            body_hi, body_lo = max(bar.open, bar.close), min(bar.open, bar.close)
            if 14 * 60 <= hm < 20 * 60:
                self._cbdr_acc = [body_hi, body_lo] if self._cbdr_acc is None else [max(self._cbdr_acc[0], body_hi), min(self._cbdr_acc[1], body_lo)]
            elif hm >= 20 * 60 and self._cbdr_acc is not None:
                self._cbdr, self._cbdr_acc = (self._cbdr_acc[0], self._cbdr_acc[1]), None

        tday = trading_day(bar.time, cfg.day_rollover)
        d_ = self._days.get(tday)
        self._days[tday] = [bar.open, bar.high, bar.low, bar.close] if d_ is None else [d_[0], max(d_[1], bar.high), min(d_[2], bar.low), bar.close]
        if len(self._days) > max(65, self.config.ipda_days + 5, max(self.config.ipda_lookbacks) + 5):
            self._days.pop(min(self._days))
        if self._tracker.update(bar):
            for s in self._setups:
                if s.stage == "limit":
                    self._log(close_time, "EN-04", f"{s.level.name} limit cancelled: new trading day")
            self._setups.clear()
            self._spent.clear()
            self._swept_sides.clear()
            self._signals_today = 0
            self._extra_signals = self._reentries = 0
            self._live.clear()
        levels = [lv for lv in self._all_levels() + self._array_levels() if lv.name not in self._spent]
        win = in_any(cfg.killzones, bar.time)

        fired: Signal | None = None
        keep: list[_Setup] = []
        if self._live:
            for t in list(self._live):
                if t["bar"] == idx:
                    continue
                d = t["d"]
                if (bar.low <= t["target"]) if d == SHORT else (bar.high >= t["target"]):
                    self._live.remove(t)            # reached its target
                elif (bar.high >= t["stop"]) if d == SHORT else (bar.low <= t["stop"]):
                    self._live.remove(t)            # stopped out: price ran on past the stop
                    if win is not None and self._reentries < cfg.max_reentries:
                        self._reentries += 1
                        self._extra_signals += 1
                        ext = bar.high if d == SHORT else bar.low
                        self._log(close_time, "RE-01", f"{t['level'].name} trade stopped out at {self._fmt(t['stop'])}; price ran on to {self._fmt(ext)}, so watching for a fresh shift to re-enter")
                        self._ev("sweep", bar.time, level=t["level"].name, level_price=t["level"].price, side=t["level"].side, dir=d, extreme=ext, window=win.name)
                        self._setups.append(_Setup(t["level"], d, ext, idx, window=win.name, sweep_time=bar.time, reentry=True))
        for s in self._setups:
            s.bars += 1
            d = s.direction
            if s.stage == "await_mss":
                if win is None:
                    self._log(close_time, "KZ-1", f"{s.level.name} setup cancelled: window closed before a shift in structure")
                    continue
                # A further run beyond the sweep moves the stop anchor (still the same sweep).
                new_ext = max(s.extreme, bar.high) if d == SHORT else min(s.extreme, bar.low)
                if new_ext != s.extreme:
                    s.extreme, s.sweep_time = new_ext, bar.time
                mss_max = cfg.mss_max_bars
                if s.bars > mss_max:
                    self._log(close_time, "MS-02", f"{s.level.name} setup cancelled: no displacement shift within {mss_max} candles")
                    continue
                body = abs(bar.close - bar.open)
                candle_rng = bar.high - bar.low
                body_ratio = body / candle_rng if candle_rng > 0 else 0.0
                displaced = body > 0 and body >= cfg.displacement_body_mult * self._median_body() and body_ratio >= 0.45
                inverted = None
                if cfg.trigger == "ifvg":
                    # IFVG: a displacement close through a gap formed the other way (support for a
                    # sell, resistance for a buy); that gap inverts and is where the order rests.
                    inverted = self._inverted_gap(d, max(0, s.sweep_index - cfg.swing_lookback), idx - 1, bar.close)
                    if inverted is None or not displaced:
                        keep.append(s)
                        continue
                    if cfg.reaper:
                        pair = self._fvg_in_leg(d, max(0, s.sweep_index - 1), idx)   # the gap this displacement leaves
                        tol = 0.25 * self._median_range()
                        if pair is None or pair[0] > inverted[1] + tol or pair[1] < inverted[0] - tol:
                            self._log(close_time, "RP-01", f"{s.level.name} setup rejected: the inversion gap has no opposing gap nested with it (not a reaper)")
                            continue
                    swing, swing_i = (inverted[0] if d == SHORT else inverted[1]), inverted[2]
                else:
                    found = self._cisd(d, s.sweep_index) if cfg.trigger == "cisd" else self._last_swing(d, idx - 1, max(0, s.sweep_index - cfg.swing_lookback))
                    if found is None:
                        keep.append(s)
                        continue
                    swing, swing_i = found
                    broke = bar.close < swing if d == SHORT else bar.close > swing
                    if not broke:
                        keep.append(s)
                        continue
                    if not displaced and cfg.trigger != "cisd":  # CISD needs the close, not a displacement candle
                        self._log(close_time, "MS-02", f"{s.level.name}: structure broken at {self._fmt(swing)} but without displacement; waiting")
                        keep.append(s)
                        continue
                s.swing_time = self._bars[swing_i].time
                self._ev("mss", bar.time, level=s.level.name, dir=d, price=swing, swing_time=s.swing_time)
                fvg = inverted or self._fvg_in_leg(d, max(0, s.sweep_index - 1), idx)
                if inverted:
                    s.entry_kind = "ifvg"
                leg_hi0 = s.extreme if d == SHORT else max(b.high for b in self._bars[s.sweep_index:idx + 1])
                leg_lo0 = min(b.low for b in self._bars[s.sweep_index:idx + 1]) if d == SHORT else s.extreme
                if cfg.entry_mode == "ote":
                    # OTE model: the limit sits at the 70.5% retracement of the displacement leg,
                    # inside the 62-79% optimal trade entry zone (premium for sells, discount for buys).
                    r = leg_hi0 - leg_lo0
                    zone = (leg_lo0 + 0.62 * r, leg_lo0 + 0.79 * r) if d == SHORT else (leg_hi0 - 0.79 * r, leg_hi0 - 0.62 * r)
                    fvg, s.entry_kind = (zone[0], zone[1], idx), "ote"
                elif cfg.entry_mode == "unicorn":
                    # Unicorn model: the gap must overlap the breaker (the order block at the swing that
                    # was broken); the limit rests in the overlap.
                    brk = self._breaker(d, swing_i)
                    ov = (max(fvg[0], brk[0]), min(fvg[1], brk[1])) if fvg and brk else None
                    if not ov or ov[0] >= ov[1]:
                        self._log(close_time, "UN-01", f"{s.level.name} setup rejected: no fair value gap overlapping a breaker block")
                        continue
                    fvg, s.entry_kind = (ov[0], ov[1], fvg[2]), "unicorn"
                elif cfg.entry_mode == "bpr" or cfg.enable_bpr:
                    # IV-02 Balanced Price Range (BPR): an opposing FVG in the sweep leg overlapped by
                    # the displacement FVG that breaks back the other way; the limit rests in the overlap.
                    opp = inverted or self._inverted_gap(d, max(0, s.sweep_index - cfg.swing_lookback), idx - 1, bar.close)
                    disp = self._fvg_in_leg(d, max(0, s.sweep_index - 1), idx) if inverted else fvg
                    ov = (max(disp[0], opp[0]), min(disp[1], opp[1])) if disp and opp else None
                    if not ov or ov[0] >= ov[1]:
                        self._log(close_time, "IV-02", f"{s.level.name} setup rejected: no balanced price range (overlapping opposing fair value gaps)")
                        continue
                    fvg, s.entry_kind = (ov[0], ov[1], disp[2]), "bpr"
                elif cfg.entry_mode == "breaker":
                    # Breaker model: the limit rests at the mean threshold of the breaker block (the
                    # order block at the swing the displacement broke), with or without a gap.
                    brk = self._breaker(d, swing_i)
                    if not brk:
                        self._log(close_time, "BR-01", f"{s.level.name} setup rejected: no breaker block at the broken swing")
                        continue
                    fvg, s.entry_kind = (brk[0], brk[1], swing_i), "breaker"
                elif cfg.entry_mode == "ob":
                    fvg = None  # the order block itself is the entry (below)
                if fvg is None and (cfg.ob_entry_fallback or cfg.entry_mode == "ob"):
                    # OB-01: no gap to enter in, so use the order block: the last opposite
                    # candle before the displacement (down-close before a rally, up-close
                    # before a drop); the limit rests at its mean threshold (50%).
                    cands = [i for i in range(max(0, s.sweep_index - 2), idx) if (self._bars[i].close < self._bars[i].open) == (d == LONG)]
                    if cands:
                        ob = self._bars[cands[-1]]
                        fvg = (ob.low, ob.high, cands[-1])
                        s.entry_kind = "ob"
                if fvg is None:
                    self._log(close_time, "FV-01", f"{s.level.name} setup rejected: displacement left no fair value gap or order block")
                    continue
                lo, hi, mid = fvg
                fvg = (lo, hi)
                leg_hi = s.extreme if d == SHORT else max(b.high for b in self._bars[s.sweep_index:idx + 1])
                leg_lo = min(b.low for b in self._bars[s.sweep_index:idx + 1]) if d == SHORT else s.extreme
                eq, ce = (leg_hi + leg_lo) / 2, (lo + hi) / 2
                if cfg.fvg_in_premium_discount and ((d == SHORT and ce < eq) or (d == LONG and ce > eq)):
                    self._log(close_time, "FV-02", f"{s.level.name} setup rejected: gap is in {'discount' if d == SHORT else 'premium'}, wrong side of the leg's 50%")
                    continue
                if s.entry_kind == "ote":
                    limit = leg_lo0 + 0.705 * (leg_hi0 - leg_lo0) if d == SHORT else leg_hi0 - 0.705 * (leg_hi0 - leg_lo0)
                else:
                    limit = ce if (cfg.entry_mode == "ce" or s.entry_kind in ("ob", "breaker")) else (lo if d == SHORT else hi)
                lb_ = self.ladder_bias() if cfg.narrative_filter else None
                rng = lb_["range"] if lb_ else None
                if rng and (limit - (rng[0] + rng[1]) / 2) * d > 0:
                    # Narrative: sell in premium, buy in discount of the dealing range of the bias timeframe (yesterday's, or the 4-hour / 1-hour range).
                    self._log(close_time, "PD-01", f"{s.level.name} setup rejected: entry {self._fmt(limit)} is in {'discount' if d == SHORT else 'premium'} of {lb_['range_name']} (equilibrium {self._fmt((rng[0] + rng[1]) / 2)})")
                    continue
                if cfg.day_open_rule:
                    dopen = self._days[tday][0]
                    # ICT Power of 3: manipulation sweep occurs below open for buys, above open for sells
                    if (s.extreme - dopen) * d > 0:
                        self._log(close_time, "DO-01", f"{s.level.name} setup rejected: manipulation sweep at {self._fmt(s.extreme)} was not {'below' if d == LONG else 'above'} the day's open {self._fmt(dopen)}")
                        continue
                if cfg.macro_only and self._time_based():
                    m = to_ny(bar.time).minute
                    if not (m >= 50 or m <= 10):
                        self._log(close_time, "MC-01", f"{s.level.name} setup rejected: the shift closed at :{m:02d}, outside the algorithmic macro windows (xx:50 to xx:10 NY)")
                        continue
                if cfg.regime_filter and self._seek_destroy(idx):
                    self._log(close_time, "RG-02", f"{s.level.name} setup rejected: seek and destroy, both sides of the range are being stopped out")
                    continue
                if cfg.htf_context:
                    ip, ip_name = self._gate_range()
                    # ICT: When the Draw on Liquidity (DOL) is external (BSL/SSL) and still unmitigated ahead,
                    # expansion trades traveling toward the draw are valid.
                    draw = self.draw_pool(d, limit) if hasattr(self, "draw_pool") else None
                    if ip and (limit - (ip[0] + ip[1]) / 2) * d > 0 and not draw:
                        self._log(close_time, "HT-01", f"{s.level.name} setup rejected: entry {self._fmt(limit)} is in {'discount' if d == SHORT else 'premium'} of {ip_name} (equilibrium {self._fmt((ip[0] + ip[1]) / 2)})")
                        continue
                if cfg.htf_arrays:
                    poi = self._htf_poi(d, s.level, (lo, hi))
                    if poi is not None and poi["state"] == "none":
                        self._log(close_time, "HA-01", f"{s.level.name} setup rejected: {poi['text']}, and the raid was not on a higher-timeframe pool")
                        continue
                mo = self._midnight_open
                if cfg.midnight_open_filter and self._time_based() and mo is not None and (s.extreme - mo) * d > 0:
                    # Power of 3: a bearish day's high (the Judas swing) forms ABOVE the midnight
                    # open, a bullish day's low BELOW it. A sweep on the wrong side is not the run.
                    self._log(close_time, "KZ-01", f"{s.level.name} setup rejected: the sweep {'high' if d == SHORT else 'low'} {self._fmt(s.extreme)} is "
                              f"{'below' if d == SHORT else 'above'} the midnight open {self._fmt(mo)}")
                    continue
                stop = s.extreme + cfg.stop_buffer_pips * pip if d == SHORT else s.extreme - cfg.stop_buffer_pips * pip
                if cfg.stop_mode == "candle1" and s.entry_kind in ("fvg", "ifvg", "bpr") and mid >= 1:
                    c1 = self._bars[mid - 1]   # the first candle of the entry gap
                    stop = c1.high + cfg.stop_buffer_pips * pip if d == SHORT else c1.low - cfg.stop_buffer_pips * pip
                elif cfg.stop_mode == "zone":
                    stop = hi + cfg.stop_buffer_pips * pip if d == SHORT else lo - cfg.stop_buffer_pips * pip  # far side of the entry zone
                risk = (stop - limit) * -d
                risk_pips = risk / pip
                min_stop = min(cfg.min_stop_pips, 1.5)
                if risk <= 0 or risk_pips < min_stop or risk_pips > cfg.max_stop_pips:
                    self._log(close_time, "SL-03", f"{s.level.name} setup rejected: stop {risk_pips:.1f} pips outside [{min_stop}, {cfg.max_stop_pips}]")
                    continue
                tgt = self._target(d, limit)
                if cfg.sd_targets and (tgt is None or (tgt.price - limit) * d / risk < cfg.min_rr):
                    legr = leg_hi - leg_lo  # measured move: the displacement leg projected past its extreme (lab block "sdt" only)
                    for k in (1.0, 1.5, 2.0, 2.5, 4.0):
                        px = (leg_lo if d == SHORT else leg_hi) + d * k * legr
                        if (px - limit) * d / risk >= cfg.min_rr:
                            tgt = Level(f"-{k:g}SD", px, "high" if d == LONG else "low")
                            break
                if tgt is None:
                    self._log(close_time, "TP-01", f"{s.level.name} setup rejected: no opposing liquidity to target")
                    continue
                rr = (tgt.price - limit) * d / risk
                if rr < cfg.min_rr:
                    self._log(close_time, "TP-02", f"{s.level.name} setup rejected: {rr:.2f}R to {tgt.name} is below {cfg.min_rr}R")
                    continue
                if cfg.regime_filter:
                    obs = self._obstacles(d, limit, tgt.price, idx)
                    max_obs = max(cfg.max_obstacles, 5)
                    if obs >= max_obs:
                        self._log(close_time, "RG-01", f"{s.level.name} setup rejected: high resistance, {obs} unmitigated opposing gaps between the entry and {tgt.name}")
                        continue
                s.stage, s.bars = "limit", 0
                s.advice = self._advise(d, s.level, limit, tgt, rr, fvg, s.extreme, win, idx, (body / self._median_body()) if self._median_body() else None, tday)
                if s.reentry:   # valid ICT, but a second try on an idea that was just stopped out is never a plain PULL THE TRIGGER
                    s.advice = advisor.second_try(s.advice)
                if cfg.min_score and (s.advice["score"] < cfg.min_score or (s.advice["verdict"] == "SKIP" and s.advice["score"] < 50)):
                    self._log(close_time, "AD-01", f"{s.level.name} setup rejected: the judgement matrix says {s.advice['call']} ({s.advice['score']}/100, under {cfg.min_score})")
                    continue
                if cfg.narrative_required:
                    st_ = self.narrative()
                    s.story = f"NR-01 story {'bullish' if st_.bias > 0 else 'bearish'} ({st_.conviction}): {st_.going_to}"
                s.conditions = self.read_conditions(d, self._obstacles(d, limit, tgt.price, idx))
                s.conditions["array"] = s.entry_kind   # which PD array the entry sits in, so the lab shows which arrays pay
                poi_ = self._htf_poi(d, s.level, fvg) if cfg.htf_arrays else None
                if poi_ and poi_["state"] != "none":
                    s.conditions["poi"] = "htf-pool" if poi_["state"] == "pool" else f"{htf.TF_NAMES.get(poi_['tf'], poi_['tf'])} {poi_['kind'].lower()}"
                s.limit, s.stop, s.target, s.target_name, s.fvg, s.mss_level, s.window = limit, stop, tgt.price, tgt.name, fvg, swing, win.name
                s.mss_time, s.fvg_time = bar.time, self._bars[mid].time
                # TP-04: first partial at the displacement leg's extreme (the first short-term pool), when it lies between the
                # entry and the final target, and never beyond the first opposing array on the way (at least half an R away).
                leg_x = leg_lo if d == SHORT else leg_hi
                s.tp1 = self._first_partial(d, limit, risk, tgt.price, leg_x, idx)
                s.levels = tuple({"name": lv.name, "price": lv.price, "side": lv.side, "spent": lv.name in self._spent}
                                 for lv in self._tracker.levels())
                self._ev("staged", bar.time, level=s.level.name, dir=d, limit=limit, stop=stop, target=tgt.price,
                         target_name=tgt.name, tp1=s.tp1, fvg=fvg, fvg_time=s.fvg_time, rr=rr,
                         rr1=((s.tp1 - limit) * d / risk) if s.tp1 is not None else None)
                self._log(close_time, "EN-01", f"{'SELL' if d == SHORT else 'BUY'} LIMIT staged at {self._fmt(limit)} in {({'fvg': 'FVG', 'ob': 'order block (mean threshold)', 'ote': 'OTE zone (70.5%)', 'unicorn': 'unicorn (breaker + FVG)', 'bpr': 'balanced price range (BPR)', 'breaker': 'breaker block (mean threshold)', 'ifvg': 'inversion FVG'})[s.entry_kind]} [{self._fmt(lo)}, {self._fmt(hi)}], stop {self._fmt(stop)}, target {tgt.name} {self._fmt(tgt.price)} ({rr:.2f}R)")
                keep.append(s)
                continue

            # stage == "limit": wait for price to retrace into the gap
            hit_target = bar.low <= s.target if d == SHORT else bar.high >= s.target
            hit_stop = bar.high >= s.stop if d == SHORT else bar.low <= s.stop
            if hit_stop:
                self._log(close_time, "EN-04", f"{s.level.name} limit cancelled: price ran through the stop before filling")
                continue
            if hit_target:
                self._log(close_time, "EN-04", f"{s.level.name} limit cancelled: price reached the target before filling")
                continue
            filled = bar.high >= s.limit if d == SHORT else bar.low <= s.limit
            if filled and fired is None and self._signals_today < cfg.max_signals_per_day + self._extra_signals:
                rr = (s.target - s.limit) * d / ((s.stop - s.limit) * -d)
                reasons = (
                    f"KZ-1 inside {s.window} window",
                    *([s.story] if s.story else []),
                    *([f"KZ-01 the sweep {'high' if d == SHORT else 'low'} formed {'above' if d == SHORT else 'below'} the midnight open {self._fmt(self._midnight_open)}"]
                      if cfg.midnight_open_filter and self._time_based() and self._midnight_open is not None else []),
                    *([f"RL-05 daily bias {'bearish' if d == SHORT else 'bullish'}: {self.daily_bias()[1]}"]
                      if cfg.daily_bias_filter and self._time_based() and cfg.timeframe_minutes < 60 and self.daily_bias()[0] == d else []),
                    f"SW-01 swept {s.level.name} {self._fmt(s.level.price)} to {self._fmt(s.extreme)}",
                    (f"CI-01 change in state of delivery: closed back past the run's open at {self._fmt(s.mss_level)}" if cfg.trigger == "cisd" else
                     f"IV-01 displacement closed through the gap at {self._fmt(s.mss_level)}, inverting it" if cfg.trigger == "ifvg" else
                     f"MS-02 displacement candle closed through the {'swing low' if d == SHORT else 'swing high'} at {self._fmt(s.mss_level)}"),
                    *([f"PD-01 entry in {'premium' if d == SHORT else 'discount'} of yesterday's range, toward the daily draw"] if cfg.narrative_filter else []),
                    *(["RG-01 low resistance: fewer than " + str(cfg.max_obstacles) + " unmitigated opposing gaps to the target, no seek-and-destroy"] if cfg.regime_filter else []),
                    *([f"DO-01 {'sell above' if d == SHORT else 'buy below'} the day's open"] if cfg.day_open_rule else []),
                    *([f"HT-01 entry in {'premium' if d == SHORT else 'discount'} of {self._gate_range()[1]}; the weekly draw does not oppose it ({self.weekly_bias()[1]})"] if cfg.htf_context else []),
                    *([f"HA-01 {(self._htf_poi(d, s.level, s.fvg) or {}).get('text', '')}"] if cfg.htf_arrays and s.fvg and self._htf_poi(d, s.level, s.fvg) else []),
                    (f"FV-01 fair value gap [{self._fmt(s.fvg[0])}, {self._fmt(s.fvg[1])}] in {'premium' if d == SHORT else 'discount'} of the leg"
                     if s.entry_kind == "fvg" else
                     f"OTE optimal trade entry zone [{self._fmt(s.fvg[0])}, {self._fmt(s.fvg[1])}] (62-79% of the displacement leg), limit at 70.5%"
                     if s.entry_kind == "ote" else
                     f"UN-01 unicorn: fair value gap overlapping the breaker block [{self._fmt(s.fvg[0])}, {self._fmt(s.fvg[1])}]"
                     if s.entry_kind == "unicorn" else
                     f"IV-02 balanced price range [{self._fmt(s.fvg[0])}, {self._fmt(s.fvg[1])}] (overlapping opposing fair value gaps)"
                     if s.entry_kind == "bpr" else
                     f"BR-01 breaker block [{self._fmt(s.fvg[0])}, {self._fmt(s.fvg[1])}], limit at its mean threshold"
                     if s.entry_kind == "breaker" else
                     f"IV-02 inversion fair value gap [{self._fmt(s.fvg[0])}, {self._fmt(s.fvg[1])}], limit at its edge"
                     if s.entry_kind == "ifvg" else
                     f"OB-01 no fair value gap: order block [{self._fmt(s.fvg[0])}, {self._fmt(s.fvg[1])}] in {'premium' if d == SHORT else 'discount'}, limit at its mean threshold"),
                    f"EN-01 limit filled at {self._fmt(s.limit)} on the retrace into the gap",
                    f"SL-01 stop {self._fmt(s.stop)} beyond the sweep extreme",
                    f"TP-01 target {s.target_name} {self._fmt(s.target)}",
                    f"TP-02 {rr:.2f}R >= {cfg.min_rr}R",
                    *(["RE-02 second try: the first trade on this idea was stopped out and price ran on; a fresh shift at the same level, sized down"] if s.reentry else []),
                    *([advisor.line(s.advice)] if s.advice else []),
                    *aplus.line(s.advice),
                    *([conditions.tag(s.conditions)] if s.conditions else []),
                )
                self._signals_today += 1
                if cfg.osok:
                    iso_ = to_ny(bar.time).isocalendar()
                    self._osok_weeks.add((iso_[0], iso_[1]))
                self._log(close_time, "SIGNAL", f"{'SHORT' if d == SHORT else 'LONG'} limit filled after {s.level.name} sweep + MSS + FVG, {rr:.2f}R")
                self._ev("filled", bar.time, level=s.level.name, dir=d, price=s.limit)
                meta = {"level_price": s.level.price, "sweep_price": s.extreme, "sweep_time": s.sweep_time,
                        "mss_level": s.mss_level, "mss_time": s.mss_time, "fvg": s.fvg, "fvg_time": s.fvg_time,
                        "midnight_open": self._midnight_open if self._time_based() else None,
                        "target_name": s.target_name, "entry_mode": cfg.entry_mode, "swing_time": s.swing_time,
                        "tp1": s.tp1, "levels": list(s.levels), "entry_kind": s.entry_kind, "advice": s.advice, "conditions": s.conditions, "story": s.story}
                fired = Signal(close_time, d, s.limit, s.stop, s.target, rr, s.level, s.window, reasons,
                               entry_type="limit", meta=meta)
                if cfg.reentry:
                    self._live.append({"d": d, "stop": s.stop, "target": s.target, "level": s.level, "bar": idx})
                continue
            if hit_target:
                self._log(close_time, "EN-04", f"{s.level.name} limit cancelled: price reached the target before filling")
                continue
            if hit_stop:
                self._log(close_time, "EN-04", f"{s.level.name} limit cancelled: price ran through the stop before filling")
                continue
            if s.bars >= self._expiry_bars():
                self._log(close_time, "EN-04", f"{s.level.name} limit cancelled: not filled within {self._expiry_bars()} candles")
                continue
            keep.append(s)
        self._setups = keep

        # New sweeps (only inside the window; outside it a level taken is spent, RL-07)
        threshold = cfg.min_sweep_pips * pip
        for lv in levels:
            took_high = lv.side == "high" and bar.high >= lv.price + threshold
            took_low = lv.side == "low" and bar.low <= lv.price - threshold
            if not (took_high or took_low):
                continue
            if win is None:
                if cfg.invalidate_levels_taken_outside_killzone:
                    self._spent.add(lv.name)
                    tag = "BSL" if took_high else "SSL"
                    if tag not in self._swept_sides:
                        self._swept_sides.append(tag)
                    self._log(close_time, "RL-07", f"{lv.name} {self._fmt(lv.price)} taken outside the window; spent")
                continue
            self._spent.add(lv.name)   # run once, it is spent: a level that was rejected below is not re-tested on every later candle
            tag = "BSL" if took_high else "SSL"
            if tag not in self._swept_sides:
                self._swept_sides.append(tag)
            d = SHORT if took_high else LONG
            if cfg.enforce_liquidity_purge_order:
                other = "SSL" if d == SHORT else "BSL"
                if other in self._swept_sides and self._swept_sides.index(other) < self._swept_sides.index(tag):
                    self._log(close_time, "SW-02", f"{lv.name} sweep ignored: {other} was purged first today")
                    continue
            if cfg.narrative_required:
                story = self.narrative()
                is_session_sweep = lv.name in ("ASIA_H", "ASIA_L", "LONDON_H", "LONDON_L", "PDH", "PDL", "PWH", "PWL") or lv.name.startswith(("ASIA_", "LONDON_", "EQH", "EQL"))
                if story.stand_aside:
                    self._log(close_time, "NR-01", f"{lv.name} sweep ignored: stand aside, {story.stand_aside}")
                    continue
                if story.bias and story.bias != d:
                    self._log(close_time, "NR-01", f"{lv.name} sweep ignored: against the story ({'bullish' if story.bias > 0 else 'bearish'}: {story.came_from.split('. ')[1] if '. ' in story.came_from else story.came_from})")
                    continue
            if cfg.narrative_filter and not cfg.narrative_required:   # (the narrative gate above already reads the daily and weekly draw)
                bias, why = self.daily_bias()
                if bias != d:
                    self._log(close_time, "RL-05", f"{lv.name} sweep ignored: not toward the daily draw ({why})")
                    continue
            if cfg.osok and self._time_based():
                wd = to_ny(bar.time)
                if wd.weekday() > 2:
                    self._log(close_time, "OK-01", f"{lv.name} sweep ignored: the weekly manipulation comes Monday to Wednesday")
                    continue
                if self._week_open is not None and (ext_ := (bar.high if d == SHORT else bar.low)) is not None and (ext_ - self._week_open) * d > 0:
                    self._log(close_time, "OK-02", f"{lv.name} sweep ignored: a {'sell' if d == SHORT else 'buy'} needs the run {'above' if d == SHORT else 'below'} the weekly open {self._fmt(self._week_open)}")
                    continue
                wb, wwhy = self.weekly_bias()
                if wb and wb != d:
                    self._log(close_time, "OK-03", f"{lv.name} sweep ignored: against the weekly draw ({wwhy})")
                    continue
                if (wd.isocalendar()[0], wd.isocalendar()[1]) in self._osok_weeks:
                    self._log(close_time, "OK-04", f"{lv.name} sweep ignored: this week's one trade is already taken")
                    continue
            if cfg.weekly_profile and self._time_based():
                wd = to_ny(bar.time)
                if (wd.weekday() == 4 and wd.hour >= 12):
                    self._log(close_time, "WP-01", f"{lv.name} sweep ignored: Friday afternoon: no new trades into the weekend")
                    continue
                wb, wwhy = self.weekly_bias()
                if wb and wb != d:
                    self._log(close_time, "WP-02", f"{lv.name} sweep ignored: against the weekly draw ({wwhy})")
                    continue
            if cfg.htf_context:
                wb, wwhy = self.weekly_bias()
                if wb and wb != d:
                    self._log(close_time, "HT-02", f"{lv.name} sweep ignored: against the weekly draw ({wwhy})")
                    continue
            elif cfg.daily_bias_filter and self._time_based() and cfg.timeframe_minutes < 60:
                bias, why = self.daily_bias()
                if bias and bias != d:
                    self._log(close_time, "RL-05", f"{lv.name} sweep ignored: against the daily bias ({why})")
                    continue
            ext = bar.high if d == SHORT else bar.low
            self._log(close_time, "SW-01", f"{lv.name} {self._fmt(lv.price)} swept to {self._fmt(ext)} in {win.name}; watching for a shift in structure")
            self._ev("sweep", bar.time, level=lv.name, level_price=lv.price, side=lv.side, dir=d, extreme=ext, window=win.name)
            self._setups.append(_Setup(lv, d, ext, idx, window=win.name, sweep_time=bar.time))

        if cfg.liquidity_pools:
            self._update_pools(idx)

        self.staged = [{"direction": "long" if s.direction == LONG else "short", "limit": s.limit, "stop": s.stop,
                        "target": s.target, "level": s.level.name, "fvg": s.fvg, "bars_left": self._expiry_bars() - s.bars,
                        "midnight_open": self._midnight_open, "tp1": s.tp1, "target_name": s.target_name,
                        "mss_level": s.mss_level, "fvg_time": s.fvg_time, "staged_time": s.mss_time}
                       for s in self._setups if s.stage == "limit"]
        return fired
