"""SaaS-Grade Luxury Interactive HTML Performance Dashboard & Report.

Engineered with a modern fintech aesthetic (dark glassmorphism, responsive
neon metrics, high-precision SVG equity curves, and visual algorithmic confluences).
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

from .metrics import Trade
from .strategy import LONG, Decision


def _equity_svg(curve: list[tuple[datetime, float]], width: int = 1000, height: int = 260) -> str:
    if len(curve) < 2:
        return "<div class='empty-curve'>Awaiting closed bar executions for equity rendering.</div>"
    step = max(1, len(curve) // 1500)
    pts = curve[::step] + [curve[-1]]
    lo, hi = min(e for _, e in pts), max(e for _, e in pts)
    span = hi - lo or 1.0
    points_polyline = []
    points_area = []

    for i, (_, e) in enumerate(pts):
        x = i / (len(pts) - 1) * width
        y = height - (e - lo) / span * (height - 50) - 25
        points_polyline.append(f"{x:.1f},{y:.1f}")
        points_area.append(f"{x:.1f},{y:.1f}")

    area_coords = f"0,{height} " + " ".join(points_area) + f" {width},{height}"
    poly_coords = " ".join(points_polyline)

    return f"""
    <div class="chart-wrapper">
      <svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" aria-label="Algorithmic Equity Curve">
        <defs>
          <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#00F0FF" stop-opacity="0.32"/>
            <stop offset="65%" stop-color="#00F0FF" stop-opacity="0.08"/>
            <stop offset="100%" stop-color="#00F0FF" stop-opacity="0.0"/>
          </linearGradient>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
          </filter>
        </defs>
        <line x1="0" y1="{height/2:.1f}" x2="{width}" y2="{height/2:.1f}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4"/>
        <polygon fill="url(#equityGrad)" points="{area_coords}" />
        <polyline fill="none" stroke="#00F0FF" stroke-width="2.5" points="{poly_coords}" filter="url(#glow)"/>
      </svg>
      <div class="axis">
        <span>{pts[0][0]:%b %d, %Y}</span>
        <span class="range-badge">LOW ${lo:,.2f} &nbsp;•&nbsp; HIGH ${hi:,.2f}</span>
        <span>{pts[-1][0]:%b %d, %Y}</span>
      </div>
    </div>
    """


def render_report(
    title: str,
    stats: dict[str, Any],
    trades: list[Trade],
    curve: list[tuple[datetime, float]],
    decisions: list[Decision],
) -> str:
    e = html.escape
    win_rate = stats.get("win_rate_pct", 0.0)
    profit_factor = stats.get("profit_factor", 0.0)
    expectancy = stats.get("expectancy_r", 0.0)
    return_pct = stats.get("return_pct", 0.0)
    max_dd = stats.get("max_drawdown_pct", 0.0)
    trade_count = stats.get("trades", 0)

    trades_html = []
    for t in trades:
        is_long = t.direction == LONG
        side_cls = "badge-long" if is_long else "badge-short"
        side_str = "LONG / BUY" if is_long else "SHORT / SELL"
        pnl_cls = "pos" if t.pnl >= 0 else "neg"
        reasons_list = "".join(f"<div class='rule-tag'>• {e(r)}</div>" for r in t.reasons)
        trades_html.append(f"""
        <tr>
          <td class="mono">{t.entry_time:%Y-%m-%d %H:%M}</td>
          <td><span class="badge {side_cls}">{side_str}</span></td>
          <td class="mono num">{t.entry_price:.5f}</td>
          <td class="mono num">{t.exit_price:.5f}</td>
          <td><span class="exit-badge">{e(t.exit_reason)}</span></td>
          <td class="mono num {pnl_cls}"><b>{t.r_multiple:+.2f}R</b> <small>(${t.pnl:+,.2f})</small></td>
          <td><div class="rules-container">{reasons_list}</div></td>
        </tr>
        """)
    trades_table = "".join(trades_html) if trades_html else "<tr><td colspan='7' class='muted center'>No executed positions in backtest window.</td></tr>"

    decisions_html = []
    for d in decisions[-300:]:
        rule_type = "rule-entry" if "SIGNAL" in d.rule or "ENTRY" in d.rule else "rule-neutral"
        if "REJECT" in d.rule or "EXPIRE" in d.rule or "SW-02" in d.rule or "FV-" in d.rule:
            rule_type = "rule-filter"
        decisions_html.append(f"""
        <tr>
          <td class="mono muted">{d.time:%H:%M:%S UTC}</td>
          <td><span class="badge {rule_type}">{e(d.rule)}</span></td>
          <td class="decision-msg">{e(d.message)}</td>
        </tr>
        """)
    decisions_table = "".join(decisions_html)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(title)} • ICT SaaS Analytics</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #07090E;
      --card: rgba(15, 23, 42, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #00F0FF;
      --accent-glow: rgba(0, 240, 255, 0.25);
      --emerald: #10B981;
      --rose: #F43F5E;
      --gold: #F59E0B;
      --text: #F8FAFC;
      --text-muted: #94A3B8;
      --font-sans: 'Inter', -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(0, 240, 255, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(99, 102, 241, 0.08) 0px, transparent 50%);
      color: var(--text);
      font-family: var(--font-sans);
      min-height: 100vh;
      padding: 32px 20px;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      flex-wrap: wrap;
      gap: 16px;
    }}
    .brand {{ display: flex; align-items: center; gap: 12px; }}
    .brand-icon {{
      width: 36px; height: 36px;
      background: linear-gradient(135deg, var(--accent), #6366F1);
      border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; color: #000; font-size: 18px;
      box-shadow: 0 0 16px var(--accent-glow);
    }}
    .brand-text h1 {{ font-size: 20px; font-weight: 700; letter-spacing: -0.02em; }}
    .brand-text p {{ font-size: 13px; color: var(--text-muted); }}
    .session-pill {{
      display: inline-flex; align-items: center; gap: 8px;
      background: rgba(16, 185, 129, 0.12);
      color: var(--emerald);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 6px 14px; border-radius: 20px;
      font-size: 12px; font-weight: 600;
    }}
    .pulse-dot {{
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--emerald);
      box-shadow: 0 0 8px var(--emerald);
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ transform: scale(1); opacity: 1; }}
      50% {{ transform: scale(1.4); opacity: 0.5; }}
    }}
    .grid-kpi {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
      transition: transform 0.2s, border-color 0.2s;
    }}
    .kpi-card:hover {{
      transform: translateY(-2px);
      border-color: rgba(0, 240, 255, 0.3);
    }}
    .kpi-label {{
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}
    .kpi-value {{
      font-family: var(--font-mono);
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .card {{
      background: var(--card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 24px;
      margin-bottom: 28px;
    }}
    .card-header {{
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 18px;
    }}
    .card-title {{ font-size: 16px; font-weight: 600; display: flex; align-items: center; gap: 8px; }}
    .chart-wrapper svg {{ width: 100%; height: 260px; overflow: visible; }}
    .axis {{
      display: flex; justify-content: space-between; align-items: center;
      margin-top: 14px; font-size: 12px; color: var(--text-muted);
      font-family: var(--font-mono);
    }}
    .range-badge {{
      background: rgba(255, 255, 255, 0.06);
      padding: 4px 10px; border-radius: 6px;
      color: var(--accent); font-weight: 600;
    }}
    .table-container {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
    th {{
      color: var(--text-muted);
      font-weight: 500;
      padding: 12px 14px;
      border-bottom: 1px solid var(--card-border);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
    }}
    td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      vertical-align: middle;
    }}
    tbody tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
    .mono {{ font-family: var(--font-mono); }}
    .num {{ text-align: right; }}
    .pos {{ color: var(--emerald); font-weight: 600; }}
    .neg {{ color: var(--rose); font-weight: 600; }}
    .badge {{
      display: inline-block; padding: 3px 8px; border-radius: 6px;
      font-size: 11px; font-weight: 600; text-transform: uppercase;
    }}
    .badge-long {{ background: rgba(16, 185, 129, 0.15); color: var(--emerald); border: 1px solid rgba(16, 185, 129, 0.3); }}
    .badge-short {{ background: rgba(244, 63, 94, 0.15); color: var(--rose); border: 1px solid rgba(244, 63, 94, 0.3); }}
    .exit-badge {{ background: rgba(255, 255, 255, 0.06); padding: 2px 6px; border-radius: 4px; font-size: 11px; color: var(--text-muted); }}
    .rules-container {{ display: flex; flex-direction: column; gap: 4px; font-size: 11px; }}
    .rule-tag {{ color: var(--text-muted); }}
    .rule-entry {{ background: rgba(0, 240, 255, 0.12); color: var(--accent); }}
    .rule-filter {{ background: rgba(245, 158, 11, 0.12); color: var(--gold); }}
    .rule-neutral {{ background: rgba(255, 255, 255, 0.06); color: var(--text-muted); }}
    .decision-msg {{ font-size: 12px; color: #E2E8F0; }}
    .empty-curve {{ padding: 60px 0; text-align: center; color: var(--text-muted); font-size: 14px; }}
    .footer {{
      margin-top: 40px; text-align: center; font-size: 12px; color: var(--text-muted);
      border-top: 1px solid var(--card-border); padding-top: 20px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="brand">
        <div class="brand-icon">◈</div>
        <div class="brand-text">
          <h1>{e(title)}</h1>
          <p>ICT Time & Price Algorithmic Core • SaaS Engine Edition</p>
        </div>
      </div>
      <div class="session-pill">
        <div class="pulse-dot"></div>
        <span>AUTONOMOUS ENGINE ACTIVE</span>
      </div>
    </div>

    <div class="grid-kpi">
      <div class="kpi-card">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value" style="color: var(--emerald);">{win_rate:.1f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Profit Factor</div>
        <div class="kpi-value" style="color: var(--accent);">{profit_factor:.2f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Net Return</div>
        <div class="kpi-value { 'pos' if return_pct >= 0 else 'neg' }">{return_pct:+.2f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Expectancy (R)</div>
        <div class="kpi-value">{expectancy:+.2f}R</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value" style="color: var(--rose);">{max_dd:.2f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Total Trades</div>
        <div class="kpi-value">{trade_count}</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">📈 Institutional Equity Delivery Curve</div>
      </div>
      {_equity_svg(curve)}
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">⚡ Trade Execution Journal</div>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Time (UTC)</th>
              <th>Direction</th>
              <th class="num">Entry Fill</th>
              <th class="num">Exit Fill</th>
              <th>Exit Trigger</th>
              <th class="num">Realized R</th>
              <th>Algorithmic Confluences</th>
            </tr>
          </thead>
          <tbody>
            {trades_table}
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">🛡️ Decision & Filter Audit Stream (Latest 300)</div>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Rule ID</th>
              <th>System Verification Decision</th>
            </tr>
          </thead>
          <tbody>
            {decisions_table}
          </tbody>
        </table>
      </div>
    </div>

    <div class="footer">
      Powered by ICT Trader Algorithmic Engine • Strict Deterministic State Machine • Zero AI in Critical Execution Loop
    </div>
  </div>
</body>
</html>
"""
