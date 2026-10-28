"""ICT's questions as numbers. Each feature has the direction ICT's teaching gives it, and the model may not learn the opposite: a feature
that "works backwards" in a sample is noise, not a discovery (the model holds its coefficient at zero instead)."""

from __future__ import annotations

from typing import Any

# (name, sign the coefficient may take: +1 raises the chance of reaching the target, -1 lowers it, 0 free)
FEATURES: tuple[tuple[str, int], ...] = (
    ("bias_with", 1),        # the bias of the rung above the chart agrees with the trade (the draw on liquidity)
    ("bias_against", -1),    # ... opposes it
    ("wbias_with", 1),       # the rung above that agrees
    ("wbias_against", -1),
    ("pd_depth", 1),         # how deep in discount (a buy) or premium (a sell) of the dealing range: 0 to 1
    ("pd_wrong", -1),        # on the wrong side of it
    ("pd20_right", 1),       # on the right side of the 20-day range
    ("disp", 1),             # displacement: the shift candle's body against the usual body
    ("gap", 1),              # the gap it leaves, against a usual candle
    ("obstacles", -1),       # opposing arrays between the entry and the target (0 to 1)
    ("seek_destroy", -1),    # both sides of the range being stopped out
    ("raid_external", 1),    # the raid took a prior day / week extreme, an IPDA range, an opening gap
    ("raid_equal", 1),       # ... equal highs or lows
    ("raid_session", 0),     # ... a session range (free: ICT does not rank it)
    ("poi_array", 1),        # the entry sits at an unmitigated higher-timeframe array
    ("poi_pool", 1),         # ... or the raid was on a higher-timeframe pool
    ("htf_trend", 1),        # the 1h / 4h structure with the trade (-1 to 1)
    ("room_over", -1),       # the day's range already spent, plus what the target needs, beyond one average day
    ("in_window", 1),        # inside a killzone (always 1 on a synthetic index, so it carries no information there)
)
NAMES = tuple(n for n, _ in FEATURES)
SIGNS = tuple(s for _, s in FEATURES)
EXTERNAL = ("PD", "PW", "IPDA", "NDOG", "NWOG", "ORG", "CBDR")


def vector(c: dict[str, Any], hour_ny: int | None = None, clock: bool = True) -> list[float]:
    """The feature row from the advisor's context for one order (the `c` the engine keeps in a setup's advice). `hour_ny` and `clock`: the
    killzone is read from the New York hour of the order on a market with a clock (London 02 to 05, New York 07 to 10), never on a synthetic index."""
    d = c.get("dir", 1)
    lvl = str(c.get("level") or "")
    pd, pd20 = c.get("pd") or 0.0, c.get("pd20")
    poi = (c.get("poi") or {}).get("state")
    tr = c.get("htf_trend") or {}
    adr = c.get("adr")
    room = 0.0
    if adr and adr[0] and c.get("entry_price") is not None and c.get("target_price") is not None:
        room = max(0.0, min(2.0, (adr[1] + abs(c["target_price"] - c["entry_price"])) / adr[0] - 1.0))
    b, w = c.get("bias", 0), c.get("wbias", 0)
    return [
        1.0 if b == d else 0.0, 1.0 if b == -d else 0.0, 1.0 if w == d else 0.0, 1.0 if w == -d else 0.0,
        max(0.0, min(1.0, pd)), max(0.0, min(1.0, -pd)), 0.0 if pd20 is None else (1.0 if pd20 > 0 else -1.0 if pd20 < 0 else 0.0),
        min(c.get("body_mult") or 0.0, 4.0) / 4.0, min(c.get("gap_ratio") or 0.0, 1.0),
        min(c.get("obstacles", 0), 3) / 3.0, 1.0 if c.get("sd") else 0.0,
        1.0 if lvl.startswith(EXTERNAL) else 0.0, 1.0 if lvl.startswith("EQ") else 0.0, 1.0 if (lvl.endswith(("_H", "_L")) and not lvl.startswith(EXTERNAL)) else 0.0,
        1.0 if poi == "array" else 0.0, 1.0 if poi == "pool" else 0.0,
        (sum(tr.values()) / len(tr)) if tr else 0.0, room, 1.0 if (clock and hour_ny is not None and (2 <= hour_ny < 5 or 7 <= hour_ny < 10)) else 0.0,
    ]
