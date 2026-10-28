"""Live results against the model's own predictions. When they drift apart the model switches itself off; it never re-tunes itself."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DriftMonitor:
    # measured by simulation (500 trades at 50%): a model that does what it says trips about 2% of the time; one that is 20 points short
    # is caught in about 50 trades, 15 points short in about 80
    slack: float = 0.15        # a shortfall of this much per trade is tolerated before it accumulates
    limit: float = 6.0         # the cumulative shortfall (in wins) that trips the alarm
    z_limit: float = -3.0      # calibration: realised wins against predicted, in standard errors (from 30 trades)
    s: float = 0.0
    n: int = 0
    wins: int = 0
    pred: float = 0.0
    var: float = 0.0
    off: bool = False
    reason: str = ""
    log: list[tuple[float, int]] = field(default_factory=list)

    def record(self, p_pred: float, hit: bool) -> bool:
        """Record one closed trade. Returns True while the model may keep trading, False once it has switched itself off."""
        self.n += 1
        self.wins += int(hit)
        self.pred += p_pred
        self.var += p_pred * (1 - p_pred)
        self.s = min(0.0, self.s + (int(hit) - p_pred) + self.slack)
        if not self.off and self.s < -self.limit:
            self.off, self.reason = True, f"results fell {abs(self.s):.1f} wins short of what the model predicted"
        if not self.off and self.n >= 30 and self.var > 0:
            z = (self.wins - self.pred) / math.sqrt(self.var)
            if z < self.z_limit:
                self.off, self.reason = True, f"{self.wins} wins against {self.pred:.1f} predicted over {self.n} trades ({z:.1f} standard errors)"
        return not self.off
