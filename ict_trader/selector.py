"""The model chooser: which of ICT's models fits this market right now.

A trader does not run every model at once. He reads the tape (trending or choppy, expanding, pulling back or consolidating, quiet or fast, the
draw clear or not, the first half hour of an open or the dead hours) and reaches for the model that
suits it. This does the same: the conditions (conditions.py) score every available model from priors
taken from ICT's teaching, and the Research lab's own results (results by condition) move those scores
once there is evidence for a model on this market. A condition where nothing fits says "stand aside",
which ICT counts as a trade too.

The priors are a starting judgement, written here so they can be read and argued with. Evidence beats
them: with enough trades in a condition (30 or more), a model's measured expectancy there outweighs
the prior. Evidence lives in rules/model-fitness.json (exported from the Research lab).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# (dimension, value) -> {model: how much that condition favours it, +/-}
PRIORS: dict[tuple[str, str], dict[str, float]] = {
    ("phase", "open"): {"fpfvg": 1.0, "gaps": 0.4, "silver_bullet": 0.3, "judas_swing": 0.3, "vacuum": 0.6},
    ("phase", "killzone"): {"ict2022": 0.3, "ote": 0.2, "unicorn": 0.2, "institutional": 0.2},
    ("phase", "off"): {"fpfvg": -1.0, "silver_bullet": -1.0, "judas_swing": -1.0, "gaps": -0.4},
    ("trend", "trending"): {"ict2022": 0.4, "ote": 0.5, "ob": 0.3, "weekly": 0.4, "sdt": 0.5, "institutional": 0.3, "sweep": -0.4,
                            "propulsion": 0.3, "rejection": 0.2, "vimb": 0.2},
    ("trend", "ranging"): {"sweep": 0.8, "clear": 0.5, "quality": 0.3, "ict2022": -0.2, "ote": -0.3, "sdt": -0.3,
                           "mitigation": 0.3, "rejection": 0.3, "suspension": 0.3},
    ("vol", "high"): {"clear": 0.3, "quality": 0.3, "full": 0.2, "sdt": 0.2},
    ("vol", "low"): {"sweep": 0.2, "fpfvg": -0.2, "ote": -0.2},
    ("draw", "up"): {"ict2022": 0.6, "full": 0.4, "weekly": 0.3, "judas_swing": 0.4},
    ("draw", "down"): {"ict2022": 0.6, "full": 0.4, "weekly": 0.3, "judas_swing": 0.4},
    # with no daily draw the ICT 2022 family has no story and stands aside (narrative.py): rank it down, the pattern models and Turtle Soup are not gated
    ("draw", "none"): {"ict2022": -0.8, "ote": -0.8, "unicorn": -0.8, "judas_swing": -0.8, "silver_bullet": -0.6, "full": -0.8, "weekly": -0.3, "sweep": 0.2},
    ("path", "seek"): {"clear": 0.8, "full": 0.3},
    # what the market is DOING (conditions.label_state): an expansion is not faded, a retracement is entered at the array it pulls back into,
    # and a consolidation is traded at its extremes after the false break (the raid and the shift), never in the middle (50/50)
    ("state", "expansion"): {"sdt": 0.5, "propulsion": 0.4, "stack_trend": 0.5, "ote": 0.3, "weekly": 0.3, "ict2022": 0.2, "sweep": -0.6, "judas_swing": -0.3},
    ("state", "retracement"): {"ote": 0.6, "unicorn": 0.4, "ob": 0.4, "breaker": 0.3, "mitigation": 0.3, "stack_trend": 0.4, "ict2022": 0.3, "fpfvg": 0.2, "sweep": -0.2},
    ("state", "consolidation"): {"sweep": 0.8, "judas_swing": 0.5, "silver_bullet": 0.2, "rejection": 0.3, "suspension": 0.3, "ict2022": -0.2, "ote": -0.4, "sdt": -0.4, "weekly": -0.2},
}
STAND_ASIDE_WHEN = {("path", "seek"): "both sides of the range are being stopped out (seek and destroy)"}
MIN_TRADES = 30       # a model's results in a condition need this many trades to count as evidence
EVIDENCE_FILE = Path(__file__).resolve().parents[1] / "rules" / "model-fitness.json"
_cache: dict[str, Any] = {}


def load_evidence() -> dict[str, Any]:
    """{symbol: {model: {dimension: {value: {trades, expectancy_r}}}}} from the Research lab (empty until exported)."""
    if "e" not in _cache:
        try:
            _cache["e"] = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8")).get("evidence", {}) if EVIDENCE_FILE.exists() else {}
        except Exception:
            _cache["e"] = {}
    return _cache["e"]


def _norm(sym: str) -> str:
    for p in ("frx", "cry"):
        if sym.startswith(p):
            return sym[len(p):]
    return sym


def recommend(conds: dict[str, Any], available: set[str] | list[str], key: str | None = None,
              evidence: dict[str, Any] | None = None, label=lambda m: m, top: int = 4) -> dict[str, Any]:
    """Rank the available models for these conditions. Returns {ranked, stand_aside, reason, conditions}."""
    ev = (evidence if evidence is not None else load_evidence()).get(_norm(key or ""), {})
    scores: dict[str, float] = {m: 0.0 for m in available}
    why: dict[str, list[str]] = {m: [] for m in available}
    aside = next((msg for (d, v), msg in STAND_ASIDE_WHEN.items() if conds.get(d) == v), None)
    for dim, val in ((d, conds.get(d)) for d in ("phase", "trend", "state", "vol", "draw", "path")):
        if not val:
            continue
        for m, w in PRIORS.get((dim, val), {}).items():
            if m in scores:
                scores[m] += w
                if abs(w) >= 0.3:
                    why[m].append(f"{val} {dim}: {w:+.1f}")
        for m in scores:  # what the lab measured for this model in this condition
            cell = ((ev.get(m) or {}).get(dim) or {}).get(val)
            if cell and cell.get("trades", 0) >= MIN_TRADES:
                bump = max(-1.5, min(1.5, cell["expectancy_r"] * 3)) * cell["trades"] / (cell["trades"] + MIN_TRADES)
                scores[m] += bump
                why[m].append(f"measured {cell['expectancy_r']:+.2f}R over {cell['trades']} trades in {val} {dim}: {bump:+.1f}")
    ranked = sorted(scores, key=lambda m: -scores[m])[:top]
    return {"conditions": {k: v for k, v in conds.items() if v},
            "stand_aside": bool(aside), "reason": (f"Stand aside: {aside}." if aside else None),
            "ranked": [{"model": m, "label": label(m), "score": round(scores[m], 2), "why": why[m][:4]} for m in ranked],
            "evidence": bool(ev)}
