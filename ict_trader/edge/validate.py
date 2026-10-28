"""Honest validation: train on the past, bet on the next month, and ask whether chance could have done as well."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .candidates import Candidate
from .model import MIN_TRAIN, Decision, EdgeModel, NotEnoughData
from .stats import cluster_bootstrap_mean

MIN_TRADES = 100   # fewer out-of-sample trades than this and nothing is concluded


@dataclass
class Report:
    candidates: int
    trades: int
    abstained: float            # share of out-of-sample candidates the model declined
    mean_r: float               # mean result per trade, in R, after costs
    ci_low: float               # 95% cluster-bootstrap interval of the mean
    ci_high: float
    total_r: float
    hit_rate: float             # realised
    mean_p: float               # what the model predicted for the same trades
    perm_p: float | None        # the share of shuffled-outcome runs that did at least as well (None: not run)
    verdict: str
    message: str
    folds: int = 0
    coefficients: dict = field(default_factory=dict)


def walk_forward(cands: list[Candidate], min_train: int = MIN_TRAIN * 2, margin: float = 0.03, **model_kw) -> tuple[list[tuple[Candidate, Decision | None]], int]:
    """For each month with enough history before it: fit on everything earlier, decide on that month. Returns (candidate, decision) pairs."""
    ordered = sorted(cands, key=lambda c: c.time)
    months = sorted({c.month for c in ordered})
    out: list[tuple[Candidate, Decision | None]] = []
    folds = 0
    for m in months:
        train = [c for c in ordered if c.month < m]
        test = [c for c in ordered if c.month == m]
        if len(train) < min_train or not test:
            continue
        try:
            model = EdgeModel(**model_kw).fit(train)
        except NotEnoughData:
            continue
        folds += 1
        out.extend((c, model.decide(c, margin)) for c in test)
    return out, folds


def _score(pairs: list[tuple[Candidate, Decision | None]]) -> tuple[list[float], list[str], list[float], list[int]]:
    rs, days, ps, hits = [], [], [], []
    for c, d in pairs:
        if d is not None:
            rs.append(c.result(d.target))
            days.append(c.day)
            ps.append(d.p)
            hits.append(c.hit(d.target))
    return rs, days, ps, hits


def validate(cands: list[Candidate], perms: int = 0, seed: int = 5, margin: float = 0.03, min_trades: int = MIN_TRADES, **model_kw) -> Report:
    pairs, folds = walk_forward(cands, margin=margin, **model_kw)
    rs, days, ps, hits = _score(pairs)
    n_out = len(pairs)
    mean, lo, hi = cluster_bootstrap_mean(rs, days)
    total = sum(rs)
    perm_p = None
    if perms and rs:
        rnd = random.Random(seed)
        beat = 0
        for _ in range(perms):
            mf = [(c.mfe_r, c.stopped) for c in cands]
            rnd.shuffle(mf)
            shuffled = [Candidate(c.market, c.time, c.direction, c.entry, c.stop, c.risk, c.planned_rr, c.cost_r, m[0], m[1], c.x) for c, m in zip(cands, mf)]
            sp, _ = walk_forward(shuffled, margin=margin, **model_kw)
            if sum(_score(sp)[0]) >= total:
                beat += 1
        perm_p = (1 + beat) / (1 + perms)
    if len(rs) < min_trades:
        verdict, msg = "NOT ENOUGH DATA", f"{len(rs)} out-of-sample trades (at least {min_trades} are needed to conclude anything): do not trade this."
    elif lo > 0 and (perm_p is None or perm_p <= 0.05):
        verdict, msg = "EDGE", f"mean {mean:+.3f}R per trade after costs, 95% interval [{lo:+.3f}, {hi:+.3f}]" + (f", chance beat it in {perm_p:.0%} of shuffled runs" if perm_p is not None else "")
    elif mean > 0:
        verdict, msg = "UNPROVEN", f"mean {mean:+.3f}R but the interval [{lo:+.3f}, {hi:+.3f}] includes zero" + (f" and chance beat it in {perm_p:.0%} of shuffled runs" if perm_p is not None else "") + ": do not trade it yet."
    else:
        verdict, msg = "NO EDGE", f"mean {mean:+.3f}R per trade after costs: no measurable edge here."
    coefs = {}
    try:
        coefs = EdgeModel(**model_kw).fit(sorted(cands, key=lambda c: c.time)).coefficients() if len(cands) >= MIN_TRAIN else {}
    except NotEnoughData:
        pass
    return Report(len(cands), len(rs), 1 - len(rs) / n_out if n_out else 1.0, mean, lo, hi, total, (sum(hits) / len(hits)) if hits else 0.0,
                  (sum(ps) / len(ps)) if ps else 0.0, perm_p, verdict, msg, folds, coefs)


def text(r: Report) -> str:
    lines = [f"{r.verdict}: {r.message}",
             f"  candidates {r.candidates}, out-of-sample trades {r.trades} over {r.folds} monthly folds, abstained on {r.abstained:.0%} of the rest",
             f"  realised hit rate {r.hit_rate:.1%} against the model's own {r.mean_p:.1%}; total {r.total_r:+.1f}R"]
    if r.coefficients:
        lines.append("  what the data say about each ICT question (lift on the log-odds of reaching a 2R target; 0 = no effect):")
        for name, by in r.coefficients.items():
            v = by.get(2.0, 0.0)
            if abs(v) > 0.02:
                lines.append(f"    {name:14s} {v:+.2f}")
    return "\n".join(lines)
