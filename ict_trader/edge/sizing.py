"""How much to risk: a fraction of Kelly, from the LOWER bound of the chance, capped."""

from __future__ import annotations


def kelly_fraction(p_low: float, target: float, cost_r: float = 0.0) -> float:
    """Full Kelly for a trade that wins `target` R with chance `p_low` and loses 1 R, after costs. Never negative."""
    edge = p_low * target - (1.0 - p_low) - cost_r
    return max(0.0, edge / target)


def risk_fraction(p_low: float, target: float, cost_r: float = 0.0, kelly_share: float = 0.25, cap: float = 0.01) -> float:
    """The share of the account to risk on the trade: a quarter of Kelly at the conservative chance, never above `cap` (1% by default)."""
    return min(cap, kelly_share * kelly_fraction(p_low, target, cost_r))
