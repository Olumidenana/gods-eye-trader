"""The ICT Autonomous Self-Audit & Adaptive Self-Correction Engine.

Inbuilt diagnostic and self-healing subsystem that continuously monitors the
engine's decision stream, detects execution flaws or false-negative rejections
on real market moves, and automatically calibrates strategy configurations to
ensure high-probability institutional setups are never prematurely discarded.

Strictly bounded within Michael J. Huddleston's Inner Circle Trader (ICT)
teachings:
- Killzone session liquidity purges (Judas Swings at London & NY Open)
- Institutional Displacement & Market Structure Shifts (MSS)
- Fair Value Gap (FVG) and Optimal Trade Entry (OTE) delivery
- Standard Deviation (-1.5, -2.0, -2.5, -4.0 SD) measured targets
- Anti-ruin preservation (1.0R breakeven lock & strict risk control)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .config import Config, StrategyConfig, Window
from .sessions import Bar, to_ny

log = logging.getLogger("ict_trader.autolearn")

STATE_PATH = Path(os.environ.get("AUTOLEARN_STATE", "")) if os.environ.get("AUTOLEARN_STATE") else None   # off unless a file is named: nothing is written next to the code


def _to_iso(val: Any) -> str:
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val or datetime.now(timezone.utc).isoformat())


@dataclass
class AnomalyAudit:
    """An execution anomaly or false negative detected during post-trade audit."""
    market: str
    time: str
    rule: str
    flaw_type: str
    description: str
    proposed_remedy: str
    recovered_r: float = 0.0


@dataclass
class AdaptiveState:
    """Persistent learned adaptations and self-corrections."""
    last_updated: str = ""
    total_audits: int = 0
    anomalies_detected: int = 0
    anomalies_corrected: int = 0
    market_rules: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_anomalies: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AdaptiveState:
        return cls(
            last_updated=data.get("last_updated", ""),
            total_audits=data.get("total_audits", 0),
            anomalies_detected=data.get("anomalies_detected", 0),
            anomalies_corrected=data.get("anomalies_corrected", 0),
            market_rules=data.get("market_rules", {}),
            recent_anomalies=data.get("recent_anomalies", []),
        )


class SelfCorrectionEngine:
    """Inbuilt self-audit and automated parameter correction engine."""

    def __init__(self, state_file: Path | None = None):
        self.state_file = state_file if state_file is not None else STATE_PATH
        self.state = self._load_state()

    def _load_state(self) -> AdaptiveState:
        if self.state_file is not None and self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return AdaptiveState.from_json(json.load(f))
            except Exception as e:
                log.warning("Could not load adaptive state: %s", e)
        return AdaptiveState()

    def save_state(self) -> None:
        if self.state_file is None:
            return
        try:
            self.state.last_updated = datetime.now(timezone.utc).isoformat()
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state.to_json(), f, indent=2)
        except Exception as e:
            log.warning("Could not persist adaptive state: %s", e)

    def audit_and_correct(self, market_key: str, bars: list[Bar], decisions: list[Any],
                          signals: list[Any], cfg: Config) -> tuple[Config, list[AnomalyAudit]]:
        """Audits recent market decisions for false-negative rejections and self-heals the configuration.

        Checks for the classic execution flaws:
        1. RL-05/NR-01 rejecting a valid Killzone Judas session sweep (London Low / Asia Low).
        2. FV-02 rejecting a displacement FVG because the gap formed in the upper half of the leg.
        3. TP-01 rejecting an expansion setup because opposing intraday pools were depleted.
        4. RL-07 wiping out key liquidity references overnight before Killzones began.
        5. Missing London reference range for New York session liquidity monitoring.
        """
        anomalies: list[AnomalyAudit] = []
        st = cfg.strategy
        overrides: dict[str, Any] = {}
        now_iso = datetime.now(timezone.utc).isoformat()

        # Rule Flaw 1: RL-05 / NR-01 False Rejection on Session Judas Swings
        judas_rejections = [
            d for d in decisions
            if getattr(d, "rule", "") in ("RL-05", "NR-01")
            and any(k in getattr(d, "message", "") for k in ("LONDON_L", "LONDON_H", "ASIA_L", "ASIA_H"))
        ]
        if judas_rejections:
            anomalies.append(AnomalyAudit(
                market=market_key,
                time=_to_iso(getattr(judas_rejections[-1], "time", now_iso)),
                rule="RL-05/NR-01",
                flaw_type="Stale Daily Close Blocking Judas Sweep",
                description="Session liquidity sweep was discarded due to previous day's close. Judas swings deliberately run liquidity into discount/premium.",
                proposed_remedy="Proposed, never applied automatically: Unlocked session liquidity sweep exception for ASIA and LONDON levels; displacement MSS overrules stale close.",
            ))
            overrides["daily_bias_filter"] = False
            overrides["narrative_filter"] = False

        # Rule Flaw 2: FV-02 Premium/Discount Leg Measurement Rejecting Fast Expansion
        fvg_rejections = [
            d for d in decisions
            if getattr(d, "rule", "") == "FV-02" and "wrong side of the leg's 50%" in getattr(d, "message", "")
        ]
        if fvg_rejections:
            anomalies.append(AnomalyAudit(
                market=market_key,
                time=_to_iso(getattr(fvg_rejections[-1], "time", now_iso)),
                rule="FV-02",
                flaw_type="FVG Premature Discount/Premium Disqualification",
                description="High-velocity displacement gap disqualified because its midpoint was in the upper half of the displacement candle.",
                proposed_remedy="Proposed, never applied automatically: Enabled fvg_in_premium_discount=False; displacement gap accepted as valid entry zone.",
            ))
            overrides["fvg_in_premium_discount"] = False

        # Rule Flaw 3: TP-01 Target Depletion during Trend Expansion
        target_rejections = [
            d for d in decisions
            if getattr(d, "rule", "") == "TP-01" and "no opposing liquidity" in getattr(d, "message", "")
        ]
        if target_rejections or not st.sd_targets:
            anomalies.append(AnomalyAudit(
                market=market_key,
                time=_to_iso(getattr(target_rejections[-1], "time", now_iso)) if target_rejections else now_iso,
                rule="TP-01",
                flaw_type="Opposing Liquidity Pool Exhaustion",
                description="Trade setup aborted because opposing intraday named pools were already purged.",
                proposed_remedy="Proposed, never applied automatically: Activated Standard Deviation (-1.5, -2.0, -2.5 SD) measured move expansion targets.",
            ))
            overrides["sd_targets"] = True

        # Rule Flaw 4: RL-07 Pre-Market Reference Level Purge
        spent_rejections = [
            d for d in decisions
            if getattr(d, "rule", "") == "RL-07" and "taken outside the window; spent" in getattr(d, "message", "")
        ]
        if spent_rejections or st.invalidate_levels_taken_outside_killzone:
            anomalies.append(AnomalyAudit(
                market=market_key,
                time=_to_iso(getattr(spent_rejections[-1], "time", now_iso)) if spent_rejections else now_iso,
                rule="RL-07",
                flaw_type="Premature Overnight Level Depletion",
                description="Prior Day and Session levels erased before the London/NY killzone opened.",
                proposed_remedy="Proposed, never applied automatically: Set invalidate_levels_taken_outside_killzone=False; levels preserved for session sweeps.",
            ))
            overrides["invalidate_levels_taken_outside_killzone"] = False

        # Rule Flaw 5: Missing London Range in Multi-Session Markets
        has_ny = any("new_york" in w.name or "ny" in w.name for w in st.killzones)
        has_london_ref = any(w.name == "london" for w in st.reference_ranges)
        if has_ny and not has_london_ref:
            anomalies.append(AnomalyAudit(
                market=market_key,
                time=now_iso,
                rule="KZ-00",
                flaw_type="Missing London Session Range Reference",
                description="New York Open has no London High/Low reference to target or sweep.",
                proposed_remedy="Proposed, never applied automatically: Injected London reference range (02:00-05:00 NY) into strategy reference ranges.",
            ))
            from datetime import time
            overrides["reference_ranges"] = tuple(st.reference_ranges) + (Window("london", time(2, 0), time(5, 0)),)

        # Dynamic Volatility Calibration: scale buffers to current market ATR
        if bars and len(bars) >= 100:
            pip = cfg.instrument.pip_size or 0.0001
            ranges = [b.high - b.low for b in bars[-288:]]
            med_rng = median(ranges) if ranges else pip
            unit = med_rng / max(pip, 1e-12)
            if unit > 0:
                dyn_min_sweep = round(0.15 * unit, 2)
                dyn_stop_buf = round(0.25 * unit, 2)
                dyn_max_stop = round(max(st.max_stop_pips, 18.0 * unit), 1)
                overrides["min_sweep_pips"] = min(st.min_sweep_pips, dyn_min_sweep)
                overrides["stop_buffer_pips"] = max(st.stop_buffer_pips, dyn_stop_buf)
                overrides["max_stop_pips"] = dyn_max_stop

        # Report only. These "remedies" switch rules off and force measured-move targets, wider stops and no "spent" rule: tuning the engine to
        # the setups it rejected, with hindsight, which is how it came to trade random prices. They are listed as PROPOSALS for a person to
        # test on real candles (the Research lab); this method never returns a changed configuration.
        self.state.total_audits += 1
        if anomalies:
            self.state.anomalies_detected += len(anomalies)
            self.state.market_rules[market_key] = {"last_audited": now_iso, "proposed_overrides": {k: str(v) for k, v in overrides.items()}}
            for a in anomalies:
                self.state.recent_anomalies.append(asdict(a))
            self.state.recent_anomalies = self.state.recent_anomalies[-50:]
            self.save_state()
        return cfg, anomalies

    def get_audit_summary(self) -> dict[str, Any]:
        """Returns a high-level summary of the autonomous self-healing engine."""
        return {
            "status": "report-only",
            "last_updated": self.state.last_updated,
            "total_audits": self.state.total_audits,
            "anomalies_detected": self.state.anomalies_detected,
            "anomalies_corrected": 0,   # nothing is corrected automatically: see audit_and_correct
            "audited_markets": list(self.state.market_rules.keys()),
            "recent_findings": self.state.recent_anomalies[-10:],
        }


# Global singleton instance for transparent background execution
_ENGINE = SelfCorrectionEngine()


def heal_config(cfg: Config, market_key: str, bars: list[Bar] | None = None,
                decisions: list[Any] | None = None, signals: list[Any] | None = None) -> Config:
    """Entry point: inspects the configuration, audits past decisions, and returns the healed Config."""
    # Report only. This used to return a config with the filters switched off (the daily bias, the narrative, the premium / discount rule of
    # the gap, the "level spent outside the window" rule) and measured-move targets switched on, whenever a rule had rejected a setup: tuning
    # the rules to the setups they rejected, with hindsight, so the engine took ever more trades. A rule changes only in code, with its ICT
    # source and a measured result (see docs/ENGINE.md), never at run time.
    _ENGINE.audit_and_correct(
        market_key=market_key,
        bars=bars or [],
        decisions=decisions or [],
        signals=signals or [],
        cfg=cfg,
    )
    return cfg


def get_engine_status() -> dict[str, Any]:
    """Retrieves live self-correction diagnostics and audit metrics."""
    return _ENGINE.get_audit_summary()
