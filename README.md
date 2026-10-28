# God's Eye Flux Trader (Autonomous Institutional Engine)

A completely standalone, zero-dependency institutional trading platform combining the authentic **ICT Flux** interface with the **God's Eye Flux** single-pass quantitative engine.

---

## ⚡ Key Highlights

* **Zero Environment Variables**: Operates completely out of the box with zero required API keys, zero authentication gates, and zero external payment or notification tokens.
* **Pure Public WebSocket Feed**: Connects directly to Deriv's public WebSocket gateway (`wss://api.derivws.com/trading/v1/options/ws/public`) for real-time candlestick feeds and streaming ticks.
* **Strictly 8 Real Institutional Markets**:
  * **Precious Metals**: Gold (`XAUUSD`), Silver (`XAGUSD`)
  * **Forex Majors**: `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`, `NZDUSD`, `USDCAD`
  * *Zero synthetics, zero volatility indices, zero stock indices.*
* **TradingView Canvas Interaction**:
  * **65-Bar Blank Room**: Deep visual projection room to the right (`rightOffset: 65`).
  * **Infinite Scrolling**: Pan left into deep historical price action and right into future space.
  * **Vertical Price Scale Dragging**: Click and drag the price axis up/down to stretch and compress candle heights dynamically.
  * **Clean Institutional Markups**: 50% Equilibrium (`EQ 50%`), Fair Value Gaps with Consequent Encroachment (`CE 50%`), Order Blocks (`+OB`/`-OB`), and Liquidity sweeps (`BSL`/`SSL`).
* **Genuine God's Eye Institutional Engine (`flux_engine`)**:
  * **Change in State of Delivery (`CISD`)**: Waits for candle-body close confirmation before triggering entries.
  * **Intermarket SMT Triad Confirmation**: Live cross-checks between sister markets (**Gold `XAUUSD` ↔ Silver `XAGUSD`** and **`EURUSD` ↔ `GBPUSD`**) at liquidity raids.
  * **Dual-Tranche Governor**: $+0.78R$ early breakeven lock, $+1.25R$ 70% partial banking, and $+3.50R+$ macro runner.
  * **Bayesian EV Scoring**: Empirical expected value ($EV_R$) and confidence scoring `[0–99%]`.

---

## 🚀 Quick Start (Local Run)

1. **Install minimal dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the local server**:
   ```bash
   python run_local.py
   ```

3. **Open in browser**:
   Navigate to [http://127.0.0.1:8090](http://127.0.0.1:8090).

---

## 📦 Directory Structure

```text
gods-eye-trader/
├── index.html                 # Full authentic Trade Desk, Act Now cards & TradingView chart
├── run_local.py               # Standalone local HTTP server with /api/scan & /api/research
├── lightweight-charts.js      # Self-hosted TradingView lightweight chart engine
├── flux_engine/               # Pure God's Eye single-pass institutional engine
│   ├── engine.py              # CISD, SMT Triad, and Dual-Tranche Governor
│   ├── feed.py                # Direct Deriv public WebSocket streamer
│   ├── research.py            # Quantitative walk-forward validator
│   ├── scanner.py             # Live institutional radar scanner
│   └── universe.py            # Strictly Gold/Silver and Forex Majors
├── ict_trader/                # Supplementary backtesting & PD array math
├── api/                       # Optional Vercel serverless function handlers
├── requirements.txt           # Minimal runtime dependencies
└── vercel.json                # Optional 1-click Vercel deployment configuration
```
