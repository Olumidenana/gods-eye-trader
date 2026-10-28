"""Small numerical tools in plain Python (the engine's core has no third-party dependencies)."""

from __future__ import annotations

import math
import random
from typing import Sequence


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve a x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("singular matrix")
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            if f:
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def inverse(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    cols = [solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def fit_logistic(rows: Sequence[Sequence[float]], y: Sequence[int], offset: float | Sequence[float] = 0.0, ridge: float = 5.0,
                 signs: Sequence[int] | None = None, iters: int = 60) -> tuple[list[float], list[list[float]]]:
    """Ridge logistic regression: logit p = offset + b0 + x . beta, by Newton's method.

    `ridge` shrinks every coefficient except the intercept towards zero (towards "the features say nothing"). `signs[j]` is +1 (the
    coefficient may not be negative), -1 (may not be positive) or 0 (free): a coefficient that would break its sign is held at zero and the
    rest are refitted. Returns ([b0, beta...], covariance of those coefficients (the Laplace approximation, for an interval on p)).
    """
    n, k = len(rows), (len(rows[0]) if rows else 0)
    off = [offset] * n if isinstance(offset, (int, float)) else list(offset)
    signs = list(signs) if signs is not None else [0] * k
    active = [True] * k
    coef = [0.0] * (k + 1)

    nz = [[(j + 1, v) for j, v in enumerate(r) if v] for r in rows]   # the non-zero features of each row (most are 0 or 1)

    def newton(active_idx: list[int]) -> list[float]:
        on = {j + 1 for j in active_idx}
        idx = [0] + [j + 1 for j in active_idx]
        pos = {c: a for a, c in enumerate(idx)}
        m = len(idx)
        w = [0.0] * (k + 1)
        for _ in range(iters):
            g = [0.0] * m
            h = [[0.0] * m for _ in range(m)]
            for i in range(n):
                xs = [(0, 1.0)] + [(pos[c], v) for c, v in nz[i] if c in on]
                z = off[i] + sum(w[idx[a]] * v for a, v in xs)
                p = sigmoid(z)
                r, wt = y[i] - p, max(p * (1 - p), 1e-6)
                for a, va in xs:
                    g[a] += r * va
                    hr = h[a]
                    wa = wt * va
                    for b2, vb in xs:
                        hr[b2] += wa * vb
            for a in range(m):
                if a:
                    g[a] -= ridge * w[idx[a]]
                    h[a][a] += ridge
                else:
                    h[a][a] += 1e-9
            step = solve(h, g)
            for a, c in enumerate(idx):
                w[c] += step[a]
            if max(abs(s) for s in step) < 1e-6:
                break
        return w

    while True:
        coef = newton([j for j in range(k) if active[j]])
        bad = [j for j in range(k) if active[j] and signs[j] and coef[j + 1] * signs[j] < -1e-12]
        if not bad:
            break
        for j in bad:
            active[j], coef[j + 1] = False, 0.0
    # covariance of the active coefficients (the Hessian at the solution), padded with zeros for the held ones
    idx = [0] + [j + 1 for j in range(k) if active[j]]
    on = set(idx[1:])
    pos = {c: a for a, c in enumerate(idx)}
    h = [[0.0] * len(idx) for _ in idx]
    for i in range(n):
        xs = [(0, 1.0)] + [(pos[c], v) for c, v in nz[i] if c in on]
        z = off[i] + coef[0] + sum(coef[j + 1] * rows[i][j] for j in range(k))
        p = sigmoid(z)
        wt = max(p * (1 - p), 1e-6)
        for a, va in xs:
            wa = wt * va
            for b2, vb in xs:
                h[a][b2] += wa * vb
    for a in range(1, len(idx)):
        h[a][a] += ridge
    h[0][0] += 1e-9
    inv = inverse(h)
    cov = [[0.0] * (k + 1) for _ in range(k + 1)]
    for a, ca in enumerate(idx):
        for b2, cb in enumerate(idx):
            cov[ca][cb] = inv[a][b2]
    return coef, cov


def predict_interval(coef: Sequence[float], cov: Sequence[Sequence[float]], x: Sequence[float], offset: float = 0.0, z: float = 1.28) -> tuple[float, float]:
    """(p, the lower bound of p) for one row: the lower bound is the logit minus z standard errors (z = 1.28 is one side of 80%, i.e. 90% one-sided)."""
    xv = [1.0] + list(x)
    m = offset + sum(c * v for c, v in zip(coef, xv))
    var = sum(xv[i] * cov[i][j] * xv[j] for i in range(len(xv)) for j in range(len(xv)))
    return sigmoid(m), sigmoid(m - z * math.sqrt(max(var, 0.0)))


def cluster_bootstrap_mean(values: Sequence[float], clusters: Sequence[object], reps: int = 1500, seed: int = 11, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lower, upper): trades on the same day are not independent, so whole days are resampled, not single trades."""
    if not values:
        return 0.0, 0.0, 0.0
    by: dict[object, list[float]] = {}
    for v, c in zip(values, clusters):
        by.setdefault(c, []).append(v)
    groups = list(by.values())
    rnd = random.Random(seed)
    means = []
    for _ in range(reps):
        pick = [groups[rnd.randrange(len(groups))] for _ in groups]
        tot = sum(sum(g) for g in pick)
        cnt = sum(len(g) for g in pick)
        means.append(tot / cnt if cnt else 0.0)
    means.sort()
    lo, hi = means[int((1 - level) / 2 * reps)], means[min(reps - 1, int((1 + level) / 2 * reps))]
    return sum(values) / len(values), lo, hi
