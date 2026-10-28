"""Zero-Lag Live Institutional Radar Scanner.

Runs the single-pass GodsEyeFluxEngine across our curated non-synthetic universe
(Forex, Metals, Commodities, Indices, Stocks) in parallel using cached delta bars.
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timezone
from typing import Any

from .engine import GodsEyeFluxEngine
from .feed import Bar, prefetch_universe, to_ny
from .universe import ASSET_CLASSES, BY_KEY, UNIVERSE, MarketSpec

DEFAULT_LIVE_KEYS = (
    "XAUUSD", "XAGUSD",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD",
)


def scan_live_radar(
    tf_minutes: int = 5,
    keys: tuple[str, ...] | list[str] | None = None,
    days: int = 12,
) -> dict[str, Any]:
    """Scans all requested institutional markets in parallel and returns the unified
    God's Eye × ICT Flux Radar payload.
    """
    t0 = _time.monotonic()
    target_keys = tuple(keys) if keys else DEFAULT_LIVE_KEYS
    # Ensure all SMT partners are included in the prefetch batch
    all_fetch_keys = list(target_keys)
    for k in target_keys:
        spec = BY_KEY.get(k)
        if spec and spec.smt_partner and spec.smt_partner not in all_fetch_keys:
            all_fetch_keys.append(spec.smt_partner)

    prefetched = prefetch_universe(all_fetch_keys, tf_minutes=tf_minutes, days=days)
    now_utc = datetime.now(timezone.utc)

    markets_out: list[dict[str, Any]] = []
    stage3_intercepts: list[dict[str, Any]] = []
    stage2_inflections: list[dict[str, Any]] = []
    recent_ledger: list[dict[str, Any]] = []

    for k in target_keys:
        spec = BY_KEY.get(k)
        if spec is None:
            continue
        fetched = prefetched.get(k)
        if not fetched:
            markets_out.append({
                "symbol": spec.key,
                "label": spec.label,
                "asset_class": spec.asset_class,
                "ok": False,
                "error": "Feed temporarily unavailable",
            })
            continue

        bars, source_label = fetched
        partner_bars = prefetched.get(spec.smt_partner, (None, ""))[0] if spec.smt_partner else None

        try:
            eng = GodsEyeFluxEngine(spec=spec, tf_minutes=tf_minutes, mode="ict")
            signals, trades, radar = eng.run(bars, partner_bars=partner_bars)
            dp = spec.decimals
            chart_bars = [
                [int(b.time.timestamp()), round(b.open, dp), round(b.high, dp), round(b.low, dp), round(b.close, dp)]
                for b in bars[-140:]
            ]
            m_dict = {
                "symbol": spec.key,
                "label": spec.label,
                "asset_class": spec.asset_class,
                "deriv_symbol": spec.deriv_symbol,
                "yahoo_symbol": spec.yahoo_symbol,
                "decimals": dp,
                "smt_partner": spec.smt_partner,
                "ok": True,
                "source": source_label,
                "last_price": radar.last_price,
                "last_time": radar.last_time.isoformat(),
                "last_time_ny": radar.last_time_ny,
                "bias": "bullish" if radar.bias > 0 else ("bearish" if radar.bias < 0 else "neutral"),
                "bias_label": radar.bias_label,
                "regime": radar.regime,
                "dealing_range_pct": radar.dealing_range_pct,
                "location_label": radar.location_label,
                "adr_used_pct": radar.adr_used_pct,
                "active_stage": radar.active_stage,
                "stage_code": radar.stage_code,
                "stage_summary": radar.stage_summary,
                "nearest_bsl": {"name": radar.nearest_bsl.name, "price": round(radar.nearest_bsl.price, dp), "tier": radar.nearest_bsl.tier} if radar.nearest_bsl else None,
                "nearest_ssl": {"name": radar.nearest_ssl.name, "price": round(radar.nearest_ssl.price, dp), "tier": radar.nearest_ssl.tier} if radar.nearest_ssl else None,
                "smt_active": radar.smt_active,
                "smt_summary": radar.smt_summary,
                "active_killzone": radar.active_killzone,
                "is_macro": radar.is_macro,
                "staged_intercept": radar.staged_intercept,
                "recent_signals": radar.recent_signals,
                "recent_trades": radar.recent_trades,
                "candles": chart_bars,
            }
            markets_out.append(m_dict)

            if radar.stage_code == 3 and radar.staged_intercept:
                stage3_intercepts.append(radar.staged_intercept)
            elif radar.stage_code == 2:
                stage2_inflections.append(m_dict)

            for tr in radar.recent_trades[:4]:
                recent_ledger.append(tr)
        except Exception as e:
            markets_out.append({
                "symbol": spec.key,
                "label": spec.label,
                "asset_class": spec.asset_class,
                "ok": False,
                "error": str(e),
            })

    recent_ledger.sort(key=lambda x: x["entry_time"], reverse=True)
    closed_recent = [t for t in recent_ledger if t["status"] != "open"]
    wins_recent = [t for t in closed_recent if t["r"] > 0.10]
    net_r_recent = round(sum(t["r"] for t in closed_recent), 2)
    win_rate_recent = round((len(wins_recent) / len(closed_recent)) * 100.0, 1) if closed_recent else 0.0

    elapsed_ms = round((_time.monotonic() - t0) * 1000.0, 1)
    return {
        "generated_at": now_utc.isoformat(),
        "generated_ny": to_ny(now_utc).strftime("%a %d %b %H:%M:%S NY"),
        "scan_latency_ms": elapsed_ms,
        "timeframe_minutes": tf_minutes,
        "asset_classes": ASSET_CLASSES,
        "universe": [
            {"key": m.key, "label": m.label, "asset_class": m.asset_class, "deriv_symbol": m.deriv_symbol, "smt_partner": m.smt_partner}
            for m in UNIVERSE
        ],
        "summary": {
            "scanned": sum(1 for m in markets_out if m.get("ok")),
            "stage3_intercepts": len(stage3_intercepts),
            "stage2_inflections": len(stage2_inflections),
            "smt_divergences": sum(1 for m in markets_out if m.get("smt_active")),
            "recent_trades": len(closed_recent),
            "recent_win_rate": win_rate_recent,
            "recent_net_r": net_r_recent,
        },
        "stage3_queue": stage3_intercepts,
        "markets": markets_out,
        "ledger": recent_ledger[:30],
    }
