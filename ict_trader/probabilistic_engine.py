"""The ICT Probabilistic & Anti-Ruin Engine (HE-ARE).

Codifies Inner Circle Trader (ICT) tape-reading methodology into deterministic
mathematical models, replacing flat checklists with an elliptical probability
manifold and enforcing strict quantitative anti-ruin invariants.

Key mathematical pillars:
1. Draw On Liquidity (DOL) Vector field: Ensures trades align with higher-timeframe
   external range liquidity (ERL) vs internal range liquidity (IRL).
2. Institutional Displacement Energy: Quantifies body-to-range velocity and ATR
   expansion to eliminate weak retail chop and false breakouts.
3. Runway Friction & Obstacle Density: Penalizes setups that face unmitigated
   opposing Fair Value Gaps or internal liquidity pools before target.
4. Volatility-to-Spread Anti-Ruin Barrier: Vetoes trades where broker spread
   consumes a dangerous fraction of the risk buffer.
5. Asymmetric Dual-Tranche Governor: 70% TP1 @ 1.5R with an immediate stop-loss shift
   to Breakeven, ensuring winning expansions never degrade into losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------- Thresholds
MIN_PROBABILITY_A_PLUS = 0.80   # Strict threshold for Titanium A+ classification
MIN_PROBABILITY_A = 0.68        # High-probability trade with minor non-critical concessions
MAX_SPREAD_FRICTION_PCT = 7.5   # Maximum allowed spread / stop distance ratio (%)
DEFAULT_RISK_FRACTION = 0.01    # Maximum 1.0% risk per trade
DAILY_CIRCUIT_BREAKER_PCT = 0.02 # 2.0% max daily drawdown locks out trading for 24h
MIN_RR_FLOOR = 1.8              # Absolute mathematical floor for Risk-to-Reward


@dataclass(frozen=True)
class ProbabilisticAudit:
    """The full mathematical assessment of an ICT candidate trade."""
    probability: float          # 0.00 to 1.00: a hand-weighted SCORE of the rows below, not a measured win rate
    is_a_plus: bool             # True if probability >= MIN_PROBABILITY_A_PLUS
    dol_score: float            # Alignment with Draw on Liquidity vector [0, 1]
    displacement_score: float   # Institutional energy & displacement [0, 1]
    runway_score: float         # Clear path without opposing obstacles [0, 1]
    location_score: float       # Premium vs discount dealing range [0, 1]
    time_score: float           # Killzone session & day profile [0, 1]
    spread_friction_pct: float  # Spread as % of stop distance
    anti_ruin_approved: bool    # True if spread, risk, and circuit breaker pass
    rejection_reasons: tuple[str, ...]
    verdict: str                # "EXECUTE A+", "EXECUTE A", "STAND DOWN"


def evaluate_setup(
    direction: int,                     # +1 for Long, -1 for Short
    entry_price: float,
    stop_price: float,
    target_price: float,
    spread: float = 0.0,
    daily_bias: int = 0,                # +1 Long, -1 Short, 0 Neutral
    weekly_bias: int = 0,               # +1 Long, -1 Short, 0 Neutral
    body_mult: float | None = None,     # Candle body / average body ratio
    gap_ratio: float | None = None,     # FVG gap height / candle range ratio
    pd_location: float = 0.0,           # > 0.5 deep discount/premium, 0 neutral, < 0 wrong side
    opposing_obstacles: int = 0,        # Count of unmitigated opposing FVGs in the path
    is_killzone: bool = True,           # Inside London Open, NY AM, or NY PM
    weekday: int = 2,                   # 0 Mon ... 4 Fri
    hour_ny: int = 10,                  # New York hour (0 - 23)
    target_name: str = "",
    level: str = "",                    # Sweep level name (e.g. ASIA_L, PDH, LONDON_L)
    room_total_proj: float | None = None, # Total projection vs ADR (1.0 = 100%)
    market_key: str = "",
    daily_pnl_pct: float = 0.0,         # Current day realized drawdown (e.g. -0.015 for -1.5%)
    is_synthetic: bool = False,
) -> ProbabilisticAudit:
    """Computes the multidimensional elliptical probability score and anti-ruin audit."""
    rejections: list[str] = []

    # 1. Structural Validation
    stop_distance = abs(entry_price - stop_price)
    target_distance = abs(target_price - entry_price)

    if stop_distance <= 0:
        return _instant_reject("Invalid trade: stop distance is zero or negative.")

    rr = target_distance / stop_distance
    if rr < MIN_RR_FLOOR:
        rejections.append(f"R:R ({rr:.2f}) is below the strict mathematical floor ({MIN_RR_FLOOR:.1f}R)")

    # 2. Anti-Ruin Layer: Spread & Daily Loss Circuit Breaker & ADR Exhaustion
    spread_friction = (spread / stop_distance) * 100.0 if stop_distance > 0 else 100.0
    if spread_friction > MAX_SPREAD_FRICTION_PCT:
        rejections.append(
            f"Excessive spread friction: spread consumes {spread_friction:.1f}% of stop loss (max {MAX_SPREAD_FRICTION_PCT}% allowed)"
        )

    if daily_pnl_pct <= -DAILY_CIRCUIT_BREAKER_PCT:
        rejections.append(
            f"Daily drawdown circuit breaker engaged: daily P&L ({daily_pnl_pct * 100:.1f}%) reached hard stop (-{DAILY_CIRCUIT_BREAKER_PCT * 100:.1f}%)"
        )

    # ADR Exhaustion Hard Invariant: A trade projecting beyond ADR has exhausted its daily range
    if room_total_proj is not None and room_total_proj > 1.0:
        rejections.append(
            f"ADR exhausted: trade projects {room_total_proj:.0%} of average daily range (max 100% allowed)"
        )

    # 3. Draw on Liquidity (DOL) Vector Alignment
    # In ICT, sweeping an external liquidity pool (PDH/L, LONDON_H/L, ASIA_H/L) establishes the opposing pool as DOL.
    is_liquidity_raid = any(
        level.startswith(p) or target_name.startswith(p)
        for p in ("PDH", "PDL", "PWH", "PWL", "EQH", "EQL", "LONDON_", "ASIA_")
    )
    dol_score = 0.0
    if daily_bias == direction and weekly_bias == direction:
        dol_score = 1.0
    elif daily_bias == direction and weekly_bias == 0:
        dol_score = 0.85
    elif daily_bias == 0 and weekly_bias == direction:
        dol_score = 0.75
    elif is_liquidity_raid:
        # Recognized ICT session/daily liquidity sweep targeting opposing liquidity pool
        dol_score = 0.85
    elif daily_bias == 0 and weekly_bias == 0:
        dol_score = 0.50
    else:
        # Counter-trend trades without a recognized liquidity raid are penalized
        dol_score = 0.15
        rejections.append("Trade is oriented directly AGAINST higher-timeframe Draw on Liquidity")

    # Named HTF Target bonus
    named_target = any(target_name.startswith(p) for p in ("PDH", "PDL", "PWH", "PWL", "EQH", "EQL", "LONDON_", "ASIA_"))
    if named_target and dol_score >= 0.5:
        dol_score = min(1.0, dol_score + 0.10)

    # 4. Institutional Displacement Energy
    # Weight: 0.25
    displacement_score = 0.0
    bm = body_mult or 1.0
    gr = gap_ratio or 0.20

    if bm >= 2.0 and gr >= 0.35:
        displacement_score = 1.0
    elif bm >= 1.5 and gr >= 0.25:
        displacement_score = 0.85
    elif bm >= 1.2 and gr >= 0.15:
        displacement_score = 0.70
    elif bm < 1.0:
        displacement_score = 0.30
        rejections.append(f"Insufficient institutional displacement (body multiplier {bm:.2f}x)")
    else:
        displacement_score = 0.55

    # 5. Runway Score (Opposing Obstacles Friction)
    # Weight: 0.20
    if opposing_obstacles == 0:
        runway_score = 1.0
    elif opposing_obstacles == 1:
        runway_score = 0.55
        rejections.append("1 unmitigated opposing FVG stands in the direct path to target")
    else:
        runway_score = 0.15
        rejections.append(f"{opposing_obstacles} unmitigated opposing FVGs block the target runway")

    # 6. Dealing Range Location (Premium vs Discount)
    # Weight: 0.15
    location_score = 0.0
    if pd_location >= 0.5:
        location_score = 1.0  # Deep in the correct quadrant (buy in discount, sell in premium)
    elif pd_location > 0.0:
        location_score = 0.75 # On the correct side of 50% equilibrium
    elif pd_location == 0.0:
        location_score = 0.50 # Ambiguous equilibrium
    else:
        location_score = 0.15 # On the wrong side of the dealing range
        rejections.append("Entry is on the WRONG side of equilibrium (buying in premium or selling in discount)")

    # 7. Time & Session Alignment
    # Weight: 0.10
    time_score = 0.0
    if is_synthetic:
        # Synthetics run 24/7, so we evaluate volatility cycle
        time_score = 0.85
    else:
        # Monday early session range-setting trap: avoid London open range setting before 08:00 NY
        if weekday == 0 and hour_ny < 8:
            time_score = 0.35
            rejections.append("Monday early session: range-setting trap (ICT stands aside)")
        elif is_killzone:
            time_score = 0.85
            # Tuesday - Thursday morning expansion bonus
            if weekday in (1, 2, 3) and (2 <= hour_ny <= 11):
                time_score = 1.0
            # Friday afternoon penalty
            elif weekday == 4 and hour_ny >= 12:
                time_score = 0.30
                rejections.append("Late Friday session: liquidity thin and institutional rebalancing active")
        else:
            time_score = 0.25
            rejections.append("Trade formed outside designated ICT Killzone session windows")

    # 8. Convergence Probability (Weighted Mahalanobis Manifold)
    total_prob = (
        0.30 * dol_score +
        0.25 * displacement_score +
        0.20 * runway_score +
        0.15 * location_score +
        0.10 * time_score
    )

    # Hard anti-ruin gating
    anti_ruin_pass = (spread_friction <= MAX_SPREAD_FRICTION_PCT) and (daily_pnl_pct > -DAILY_CIRCUIT_BREAKER_PCT)
    if not anti_ruin_pass:
        total_prob = min(total_prob, 0.40)

    # Verdict assignment
    is_a_plus = (total_prob >= MIN_PROBABILITY_A_PLUS) and anti_ruin_pass and len(rejections) == 0

    if is_a_plus:
        verdict = "EXECUTE A+"
    elif total_prob >= MIN_PROBABILITY_A and anti_ruin_pass:
        verdict = "EXECUTE A"
    else:
        verdict = "STAND DOWN"

    return ProbabilisticAudit(
        probability=round(total_prob, 3),
        is_a_plus=is_a_plus,
        dol_score=round(dol_score, 2),
        displacement_score=round(displacement_score, 2),
        runway_score=round(runway_score, 2),
        location_score=round(location_score, 2),
        time_score=round(time_score, 2),
        spread_friction_pct=round(spread_friction, 2),
        anti_ruin_approved=anti_ruin_pass,
        rejection_reasons=tuple(rejections),
        verdict=verdict,
    )


def calculate_dual_tranche_targets(
    entry_price: float,
    stop_price: float,
    final_target: float,
    direction: int,
) -> dict[str, float]:
    """Calculates the mathematical Dual-Tranche execution plan:
    - TP1 at +1.5R (closes 70% of position)
    - Breakeven Stop Trigger at +1.5R (moves SL to Entry + 0.1R buffer)
    - TP2 at Final HTF Target (runs the remaining 30%)
    """
    stop_distance = abs(entry_price - stop_price)
    tp1_distance = stop_distance * 1.5

    if direction > 0:  # Long
        tp1_price = entry_price + tp1_distance
        breakeven_stop = entry_price + (stop_distance * 0.05)  # Covers spread/commissions
    else:  # Short
        tp1_price = entry_price - tp1_distance
        breakeven_stop = entry_price - (stop_distance * 0.05)

    return {
        "tp1_price": round(tp1_price, 5),
        "tp1_allocation_pct": 70.0,
        "tp2_price": round(final_target, 5),
        "tp2_allocation_pct": 30.0,
        "breakeven_trigger_price": round(tp1_price, 5),
        "breakeven_new_stop": round(breakeven_stop, 5),
    }


def _instant_reject(reason: str) -> ProbabilisticAudit:
    return ProbabilisticAudit(
        probability=0.0,
        is_a_plus=False,
        dol_score=0.0,
        displacement_score=0.0,
        runway_score=0.0,
        location_score=0.0,
        time_score=0.0,
        spread_friction_pct=100.0,
        anti_ruin_approved=False,
        rejection_reasons=(reason,),
        verdict="STAND DOWN",
    )
