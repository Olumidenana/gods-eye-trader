"""The chance of reaching a target before the stop, as a function of ICT's questions, pulled towards "no edge"."""

from __future__ import annotations

from dataclasses import dataclass

from . import features
from .candidates import TARGETS, Candidate
from .stats import fit_logistic, logit, predict_interval

MIN_TRAIN = 120   # fewer candidates than this and the model refuses to fit (it would only describe the sample)


@dataclass
class Decision:
    target: float   # the target (in R) with the best lower-bound expectancy
    p: float        # the model's chance of reaching it
    p_low: float    # a conservative (90% one-sided) lower bound on that chance
    ev: float       # expectancy per trade in R, after costs, at p
    ev_low: float   # ... at the lower bound: what the model is willing to bet on


class NotEnoughData(Exception):
    pass


class EdgeModel:
    """For each target T in the grid: logit p = logit(1 / (1 + T)) + b0 + x . beta.

    The first term is the chance on a random walk, so with beta = 0 the model says "no edge" and the expectancy is minus the costs. A ridge
    penalty keeps beta near zero unless the data insist, and each coefficient may only take the sign ICT's teaching gives its feature.
    """

    def __init__(self, targets: tuple[float, ...] = TARGETS, ridge: float = 8.0, z: float = 1.28) -> None:
        self.targets, self.ridge, self.z = tuple(targets), ridge, z
        self.fits: dict[float, tuple[list[float], list[list[float]]]] = {}
        self.n = 0

    def fit(self, cands: list[Candidate]) -> "EdgeModel":
        if len(cands) < MIN_TRAIN:
            raise NotEnoughData(f"{len(cands)} candidates: at least {MIN_TRAIN} are needed to fit")
        rows = [c.x for c in cands]
        for t in self.targets:
            self.fits[t] = fit_logistic(rows, [c.hit(t) for c in cands], offset=logit(1.0 / (1.0 + t)), ridge=self.ridge, signs=features.SIGNS)
        self.n = len(cands)
        return self

    def curve(self, c: Candidate) -> list[tuple[float, float, float, float, float]]:
        """(target, p, p lower bound, ev, ev at the lower bound) for each target; p never rises with the target."""
        out, p_cap, lo_cap = [], 1.0, 1.0
        for t in self.targets:
            coef, cov = self.fits[t]
            p, lo = predict_interval(coef, cov, c.x, offset=logit(1.0 / (1.0 + t)), z=self.z)
            p_cap, lo_cap = min(p_cap, p), min(lo_cap, lo)
            out.append((t, p_cap, lo_cap, p_cap * t - (1 - p_cap) - c.cost_r, lo_cap * t - (1 - lo_cap) - c.cost_r))
        return out

    def decide(self, c: Candidate, margin: float = 0.03) -> Decision | None:
        """The best target for this candidate, or None (abstain) when even the conservative expectancy does not clear `margin` R."""
        best = max(self.curve(c), key=lambda r: r[4])
        return Decision(best[0], best[1], best[2], best[3], best[4]) if best[4] > margin else None

    def coefficients(self) -> dict[str, dict[float, float]]:
        """The fitted lift of each feature, per target (0 = held at zero or no effect)."""
        out: dict[str, dict[float, float]] = {"(intercept)": {}}
        for j, name in enumerate(features.NAMES):
            out[name] = {}
        for t, (coef, _) in self.fits.items():
            out["(intercept)"][t] = round(coef[0], 3)
            for j, name in enumerate(features.NAMES):
                out[name][t] = round(coef[j + 1], 3)
        return out
