"""Optimal Trade Entry (OTE) & Standard Deviation Expansion Engine.

Derives the institutional Fibonacci dealing range and algorithmic expansion targets
as taught across the ICT Core Mentorship:
  1. Dealing Range & Equilibrium (50.0%)
  2. Optimal Trade Entry (OTE) Retracement Grid:
     - 61.8% (0.618): Shallow institutional retracement
     - 70.5% (0.705): The institutional Sweet Spot
     - 79.0% (0.790): Deep discount / premium boundary
     - 100.0% (1.00): Swing anchor invalidation point (hard stop)
  3. Standard Deviation (SD) Expansion Projections:
     - -1.0 SD: Scalp target & initial partial (+50%)
     - -1.5 SD: Intermediate target
     - -2.0 SD: Institutional terminal dealing range objective
     - -2.5 SD: Extended projection for macro trends
     - -4.0 SD: Macro exhaustion target
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .strategy import LONG, SHORT


@dataclass(frozen=True)
class DealingRange:
    """Represents an impulse leg / dealing range used for OTE and projections."""
    anchor_low: float
    anchor_high: float
    direction: int  # LONG: impulse moved up (retracing down); SHORT: impulse moved down (retracing up)

    @property
    def range_span(self) -> float:
        return self.anchor_high - self.anchor_low

    @property
    def equilibrium(self) -> float:
        """50% equilibrium level dividing Premium from Discount."""
        return self.anchor_low + self.range_span * 0.5

    def is_discount(self, price: float) -> bool:
        """True if price is in discount (below equilibrium). Favorable for longs."""
        return price < self.equilibrium

    def is_premium(self, price: float) -> bool:
        """True if price is in premium (above equilibrium). Favorable for shorts."""
        return price > self.equilibrium

    def fib_level(self, ratio: float) -> float:
        """Computes the retracement price level for a given ratio (e.g. 0.618, 0.705, 0.790).

        For LONG: Retracement moves DOWN from anchor_high towards anchor_low.
                  fib(ratio) = anchor_high - ratio * range_span
        For SHORT: Retracement moves UP from anchor_low towards anchor_high.
                  fib(ratio) = anchor_low + ratio * range_span
        """
        if self.direction == LONG:
            return self.anchor_high - ratio * self.range_span
        return self.anchor_low + ratio * self.range_span

    @property
    def ote_shallow(self) -> float:
        """61.8% retracement level."""
        return self.fib_level(0.618)

    @property
    def ote_sweet_spot(self) -> float:
        """70.5% retracement sweet spot."""
        return self.fib_level(0.705)

    @property
    def ote_deep(self) -> float:
        """79.0% retracement level."""
        return self.fib_level(0.790)

    @property
    def invalidation(self) -> float:
        """100% swing anchor level where the setup is invalidated."""
        return self.anchor_low if self.direction == LONG else self.anchor_high

    def is_in_ote(self, price: float, tolerance: float = 0.0) -> bool:
        """True if price is within the 61.8% - 79.0% OTE zone (with optional buffer)."""
        lower = min(self.ote_shallow, self.ote_deep) - tolerance
        upper = max(self.ote_shallow, self.ote_deep) + tolerance
        return lower <= price <= upper

    def standard_deviation_target(self, sd_multiple: float) -> float:
        """Projects standard deviation expansion targets beyond the swing termination.

        - sd_multiple = 1.0 -> -1.0 SD target
        - sd_multiple = 1.5 -> -1.5 SD target
        - sd_multiple = 2.0 -> -2.0 SD target
        - sd_multiple = 4.0 -> -4.0 SD target
        """
        if self.direction == LONG:
            # Expansion above anchor_high
            return self.anchor_high + sd_multiple * self.range_span
        # Expansion below anchor_low
        return self.anchor_low - sd_multiple * self.range_span

    def all_sd_targets(self) -> dict[str, float]:
        """Returns standard institutional expansion projection targets."""
        return {
            "-1.0_sd": self.standard_deviation_target(1.0),
            "-1.5_sd": self.standard_deviation_target(1.5),
            "-2.0_sd": self.standard_deviation_target(2.0),
            "-2.5_sd": self.standard_deviation_target(2.5),
            "-4.0_sd": self.standard_deviation_target(4.0),
        }


def find_dealing_range(
    swing_low: float,
    swing_high: float,
    direction: int,
) -> DealingRange | None:
    """Creates a DealingRange if valid swing points are supplied."""
    if swing_high <= swing_low:
        return None
    return DealingRange(anchor_low=swing_low, anchor_high=swing_high, direction=direction)
