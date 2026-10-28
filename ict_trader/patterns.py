"""Pattern models: Venom, the ATM method and the Market Maker models.

Each one is a self-contained story with its own trigger, entry and stop (taken from ICT's own rules in
rules/ict-models.json). They share the order handling of the opening-range strategy (a limit order that
rests, fills, is cancelled or expires) and the common parts of the engine: the nearest opposing liquidity as
the target, the adviser's score and the market conditions on every setup. A pattern that is not a complete
system on its own (ICT says Venom and the ATM method need a bias): each asks the story (the same read the spine uses) and takes nothing without a clear draw.

  venom   a candle runs a liquidity pool (fang 1), the next displaces back (fang 2); a limit at fang 2's
          close, a stop just beyond the sweep. Void if price trades back through fang 2's body midpoint.
  atm     a short-term high is run, structure breaks straight away, the broken level is retested: a limit
          at the broken swing, the stop 1-2 pips beyond the rejection high.
  mm      the market maker model: an original consolidation, a run away from it in steps, a smart money
          reversal that sweeps liquidity and shifts structure, then a low-risk entry in the first retrace;
          the stop at the reversal extreme, the first target the original consolidation's far side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import median

from .opening import FirstPresentedStrategy
from .sessions import Level
from .strategy import LONG, SHORT

CHOP_BARS, CHOP_MAX = 10, 3          # venom: skip when 3 of the last 10 candles are very large (chop)
ATM_WINDOW = 8                       # atm: the structure break must come within this many candles of the sweep
MM_BOX, MM_SPAN = 16, 72             # mm: consolidation length and how far back the whole story may start


def _swing_lows(bars, lo: int, hi: int) -> list[int]:
    return [i for i in range(max(1, lo), min(hi, len(bars) - 1)) if bars[i].low < bars[i - 1].low and bars[i].low < bars[i + 1].low]


def _swing_highs(bars, lo: int, hi: int) -> list[int]:
    return [i for i in range(max(1, lo), min(hi, len(bars) - 1)) if bars[i].high > bars[i - 1].high and bars[i].high > bars[i + 1].high]


@dataclass
class VenomStrategy(FirstPresentedStrategy):
    _seen: set = field(default_factory=set)

    def _detect(self, idx: int, close_time: datetime, pip: float) -> None:
        cfg, b = self.config, self._bars
        if idx < 30:
            return
        c1, c2 = b[idx - 1], b[idx]
        mb, mr = self._median_body(), self._median_range()
        if not mb or not mr or (idx - 1) in self._seen:
            return
        if sum(1 for x in b[-CHOP_BARS - 1:-1] if x.high - x.low > 1.5 * mr) >= CHOP_MAX:
            return                                          # chop: several very large candles already
        for d in (LONG, SHORT):
            if self._story_against(d):
                continue
            # fang 1: one candle runs a pool (a swing low for a buy) and closes the wrong way; fang 2 displaces back
            pools = _swing_lows(b, idx - 40, idx - 3) if d == LONG else _swing_highs(b, idx - 40, idx - 3)
            if not pools:
                continue
            p = pools[-1]
            pool = b[p].low if d == LONG else b[p].high
            ran = (c1.low < pool and b[idx - 2].low >= pool and c1.close < c1.open) if d == LONG else \
                  (c1.high > pool and b[idx - 2].high <= pool and c1.close > c1.open)
            back = (c2.close > c2.open and c2.close > c1.open and c2.close - c2.open >= cfg.displacement_body_mult * mb) if d == LONG else \
                   (c2.close < c2.open and c2.close < c1.open and c2.open - c2.close >= cfg.displacement_body_mult * mb)
            if not (ran and back):
                continue
            self._seen.add(idx - 1)
            entry = c2.close
            stop = (c1.low - cfg.stop_buffer_pips * pip) if d == LONG else (c1.high + cfg.stop_buffer_pips * pip)
            lvl = Level(f"POOL {self._fmt(pool)}", pool, "low" if d == LONG else "high")
            pre = (f"VN-01 fang 1: one candle ran the {'low' if d == LONG else 'high'} at {self._fmt(pool)} and closed the wrong way",
                   f"VN-02 fang 2: the next candle displaced back through the sweep candle's open at {self._fmt(c1.open)}",
                   f"RL-05 not against the daily draw ({self.daily_bias()[1]})")
            self._stage(d, lvl, entry, stop, (min(c2.open, c2.close), max(c2.open, c2.close)), idx, close_time, "venom", "venom entry",
                        pre=pre, ce_cancel=True, expiry=3, extreme=c1.low if d == LONG else c1.high, sweep_time=c1.time, swing_time=b[p].time)
            return


@dataclass
class ATMStrategy(FirstPresentedStrategy):
    _seen: set = field(default_factory=set)

    def _detect(self, idx: int, close_time: datetime, pip: float) -> None:
        cfg, b = self.config, self._bars
        if idx < 24:
            return
        bar = b[idx]
        for d in (SHORT, LONG):
            if self._story_against(d):
                continue
            # a short-term high (low) run in the last few candles, the key rejection extreme H2 (L2)
            highs = _swing_highs(b, idx - 40, idx - ATM_WINDOW) if d == SHORT else _swing_lows(b, idx - 40, idx - ATM_WINDOW)
            if not highs:
                continue
            h1 = highs[-1]
            ref = b[h1].high if d == SHORT else b[h1].low
            win = b[idx - ATM_WINDOW:idx]
            ran = [x for x in win if (x.high > ref if d == SHORT else x.low < ref)]
            if not ran:
                continue
            key = max(x.high for x in ran) if d == SHORT else min(x.low for x in ran)
            # structure break: this candle closes through the most recent opposite swing that sits before the run
            lows = _swing_lows(b, idx - 30, idx - 1) if d == SHORT else _swing_highs(b, idx - 30, idx - 1)
            lows = [i for i in lows if i < idx - ATM_WINDOW + 3 or b[i].time < ran[0].time]
            if not lows:
                continue
            L = b[lows[-1]].low if d == SHORT else b[lows[-1]].high
            broke = bar.close < L if d == SHORT else bar.close > L
            before = b[idx - 1].close >= L if d == SHORT else b[idx - 1].close <= L
            if not (broke and before) or key in self._seen:
                continue
            self._seen.add(key)
            entry = L                                           # the retest of the broken level
            stop = key + cfg.stop_buffer_pips * pip if d == SHORT else key - cfg.stop_buffer_pips * pip
            two = len([x for x in ran if (x.high > ref if d == SHORT else x.low < ref)]) >= 2
            lvl = Level(f"ATM {self._fmt(ref)}", ref, "high" if d == SHORT else "low")
            pre = (f"AT-01 a short-term {'high' if d == SHORT else 'low'} at {self._fmt(ref)} was run{' in two pushes' if two else ''}, rejection at {self._fmt(key)}",
                   f"AT-02 structure broke straight away: a close {'below' if d == SHORT else 'above'} {self._fmt(L)}",
                   f"RL-05 not against the daily draw ({self.daily_bias()[1]})")
            self._stage(d, lvl, entry, stop, (min(entry, bar.close), max(entry, bar.close)), idx, close_time, "atm", "ATM retest",
                        pre=pre, ce_cancel=False, expiry=12, extreme=key, sweep_time=ran[0].time, swing_time=b[lows[-1]].time)
            return


@dataclass
class MarketMakerStrategy(FirstPresentedStrategy):
    _seen: set = field(default_factory=set)

    def _detect(self, idx: int, close_time: datetime, pip: float) -> None:
        cfg, b = self.config, self._bars
        if idx < MM_SPAN:
            return
        mb, mr = self._median_body(), self._median_range()
        if not mb or not mr:
            return
        bar = b[idx]
        body = abs(bar.close - bar.open)
        for d in (SHORT, LONG):                               # SHORT = the market maker SELL model (a buy-side curve, then the reversal)
            if self._story_against(d):
                continue
            box = b[idx - MM_SPAN: idx - MM_SPAN + MM_BOX]
            bh, bl = max(x.high for x in box), min(x.low for x in box)
            if bh - bl > 4.0 * mr:
                continue                                      # not a consolidation
            run = b[idx - MM_SPAN + MM_BOX: idx]
            if d == SHORT:
                r_i = max(range(len(run)), key=lambda i: run[i].high)
                ext = run[r_i].high
                ran_out = ext - bh >= 1.0 * (bh - bl)
                steps = len(_swing_highs(run, 1, len(run) - 1)) >= 2   # at least two pushes: the curve has steps
            else:
                r_i = min(range(len(run)), key=lambda i: run[i].low)
                ext = run[r_i].low
                ran_out = bl - ext >= 1.0 * (bh - bl)
                steps = len(_swing_lows(run, 1, len(run) - 1)) >= 2
            r_abs = idx - MM_SPAN + MM_BOX + r_i
            if not (ran_out and steps) or idx - r_abs > 14 or ext in self._seen:
                continue                                      # the reversal must come soon after the extreme
            swing = self._last_swing(d, r_abs, max(0, r_abs - 20))   # the last higher low (sell) / lower high (buy) before the extreme
            if swing is None:
                continue
            level_px, _ = swing
            shifted = bar.close < level_px if d == SHORT else bar.close > level_px
            if not (shifted and body >= cfg.displacement_body_mult * mb):
                continue
            gap = self._fvg_in_leg(d, r_abs, idx)
            if gap is None:
                continue
            lo, hi, _mid = gap
            self._seen.add(ext)
            entry = (lo + hi) / 2                              # the low-risk entry: the first retrace into the gap
            stop = ext + cfg.stop_buffer_pips * pip if d == SHORT else ext - cfg.stop_buffer_pips * pip
            oc = Level(f"OC {'low' if d == SHORT else 'high'}", bl if d == SHORT else bh, "low" if d == SHORT else "high")
            pre = (f"MM-01 original consolidation {self._fmt(bl)} to {self._fmt(bh)}, then a run {'up' if d == SHORT else 'down'} in steps to {self._fmt(ext)}",
                   f"MM-02 smart money reversal: the {'high' if d == SHORT else 'low'} at {self._fmt(ext)} was taken, then a displacement closed {'below' if d == SHORT else 'above'} {self._fmt(level_px)}",
                   f"RL-05 not against the daily draw ({self.daily_bias()[1]})")
            self._stage(d, Level(f"MM {self._fmt(ext)}", ext, "high" if d == SHORT else "low"), entry, stop, (lo, hi), idx, close_time, "mm",
                        "market maker low-risk entry", pre=pre, ce_cancel=False, expiry=10, extreme=ext, sweep_time=b[r_abs].time, target=oc)
            return
