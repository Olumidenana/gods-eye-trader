"""python -m ict_trader.edge --selftest | --csv FILE | --symbol frxEURUSD [frxGBPUSD ...]  -> a verdict on whether ICT's questions carry an edge."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime

from . import synthetic, validate
from .candidates import Candidate, extract


def _save(path: str, cands: list[Candidate]) -> None:
    with open(path, "w") as fh:
        json.dump([{**asdict(c), "time": c.time.isoformat()} for c in cands], fh)


def _load(path: str) -> list[Candidate]:
    with open(path) as fh:
        return [Candidate(**{**d, "time": datetime.fromisoformat(d["time"])}) for d in json.load(fh)]


def selftest(perms: int) -> int:
    """The validator on tables with a known answer: it must find a planted edge and decline pure noise."""
    ok = True
    for label, edge, seeds in (("no edge", 0.0, (1, 2, 3)), ("planted edge", 0.8, (1, 2, 3))):
        for s in seeds:
            r = validate.validate(synthetic.table(n=1800, edge=edge, seed=s), perms=perms)
            good = r.verdict == "EDGE" if edge else r.verdict != "EDGE"
            ok = ok and good
            print(f"{label:13s} seed {s}: {r.verdict:16s} mean {r.mean_r:+.3f}R over {r.trades} trades  {'ok' if good else 'WRONG'}")
    print("the validator finds a planted edge and declines noise" if ok else "the validator is miscalibrated: do not trust its verdicts")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ict_trader.edge", description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="run the validator on fake data with a planted edge and on pure noise")
    ap.add_argument("--csv", nargs="+", help="candle files (columns time,open,high,low,close)")
    ap.add_argument("--symbol", nargs="+", help="Deriv symbols to download (frxEURUSD, R_75, ...)")
    ap.add_argument("--load", help="candidates saved earlier with --save (skips the slow extraction)")
    ap.add_argument("--save", help="write the extracted candidates here")
    ap.add_argument("--cls", default="forex", help="market class: forex, synthetic, index, metal, crypto (default forex)")
    ap.add_argument("--tf", type=int, default=5, help="candle size in minutes (default 5)")
    ap.add_argument("--days", type=int, default=365, help="history to download for --symbol (default 365)")
    ap.add_argument("--spread-pips", type=float, default=None, help="spread to charge (default: the engine's estimate for the market)")
    ap.add_argument("--perms", type=int, default=0, help="shuffled-outcome runs for the permutation test (each costs one walk-forward; 20 to 100)")
    ap.add_argument("--margin", type=float, default=0.03, help="the lower-bound expectancy (R) a candidate must clear to be taken")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest(a.perms)
    cands: list[Candidate] = []
    if a.load:
        cands = _load(a.load)
    elif a.csv:
        from ..data import load_bars_csv
        for path in a.csv:
            got = extract(load_bars_csv(path), path, a.cls, a.tf, a.spread_pips)
            print(f"{path}: {len(got)} candidates", file=sys.stderr)
            cands += got
    elif a.symbol:
        from ..research import SYNTHETICS, fetch_deriv_history
        for sym in a.symbol:
            bars = fetch_deriv_history(sym, a.tf, a.days, deadline_s=300)
            got = extract(bars, sym, "synthetic" if sym in SYNTHETICS else a.cls, a.tf, a.spread_pips)
            print(f"{sym}: {len(bars)} candles, {len(got)} candidates", file=sys.stderr)
            cands += got
    else:
        ap.print_help()
        return 2
    if a.save:
        _save(a.save, cands)
    print(validate.text(validate.validate(cands, perms=a.perms, margin=a.margin)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
