
# 🇮🇳 India AI Trader

AI-assisted trading system for Indian stock markets (NSE/BSE) powered by LangGraph multi-agent orchestration. Paper trading is the default; research integrations do not imply live-trading readiness.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   SCHEDULER                       │
│          (APScheduler — IST market hours)         │
├──────────────────────────────────────────────────┤
│                 LANGGRAPH PIPELINE                │
│                                                   │
│  MarketData ──┬── Technical ──┐                   │
│               ├── Fundamental ─┤                   │
│               ├── Sentiment ───┼── RiskManager     │
│               ├── Options ─────┤    ↓              │
│               ├── Macro ───────┤  PortfolioManager │
│               └── IndiaSpecific┘    ↓              │
│                                 OrderExecutor      │
├──────────────────────────────────────────────────┤
│  BROKER: Angel One SmartAPI; optional Kite adapter│
│  DATA:   yfinance, NewsAPI, NSE feeds             │
│  RISK:   stop-risk sizing, VaR, net R:R, ATR stops│
│  DB:     SQLite (trades, signals, PnL)            │
│  UI:     FastAPI + React dashboard                │
│  ALERTS: Telegram bot notifications               │
└──────────────────────────────────────────────────┘
```

## Features

- **9 Specialised AI Agents** — Market Data, Technical, Fundamental, Sentiment, Options, Macro, India-Specific, Risk Manager, Portfolio Manager
- **One governed strategy plus disabled sleeves** — `CompositeTrend` (intraday, NSE cash) is the only strategy that trades. ORB, Supertrend+ADX, Mean Reversion, Momentum Breakout, VCP/NR7, BTST strength and ETF rotation are registered sleeves that ship **disabled** behind their own flags; derivatives are out of scope
- **India Equity Costs** — estimated Angel One brokerage (0.1%, ₹5 minimum, ₹20 cap), STT, NSE charges, GST, and product-specific stamp duty
- **Net Paper P&L** — portfolio, daily history, and equity curve subtract estimated charges for every executed paper order; verify estimates against broker contract notes
- **Paper Trading Default** — Must complete 30 days paper trading before going live
- **Risk Hard Limits** — 0.5% planned equity risk per entry, 10% per-position cap, 50% gross-exposure cap, maximum 3 new entries per day, 2:1 minimum net reward/risk, cost-efficiency gate, 2% daily loss cap, and VIX halt at 28
- **Full Backtester** — With Indian cost model, Sharpe/Sortino/Calmar using 6.5% risk-free rate
- **Real-time Dashboard** — Dark terminal theme, equity curve, agent status, signal feed
- **Telegram Alerts** — Trade executions, daily PnL, risk alerts
- **Provider-neutral LLM factory** — Separate quick/deep tiers for supported APIs and local Ollama; provider integrations are optional
- **Execution safety** — Live entries require explicit acknowledgement and, in the graph path, exact-setup Time Machine evidence; ambiguous submissions are never automatically retried on another broker

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Node.js 18+ (for dashboard frontend)
- Angel One account with SmartAPI access
- Anthropic API key

### 2. Install

```bash
cd india-ai-trader

# Python dependencies
pip install -e .

# Dashboard frontend (optional)
cd dashboard/frontend
npm install
cd ../..
```

### 3. Configure

```bash
# Copy env template and fill in your credentials
cp .env.example .env
```

Edit `.env` with:
- `ANGEL_API_KEY` — Your Angel One API key
- `ANGEL_CLIENT_ID` — Your client ID
- `ANGEL_MPIN` — Your MPIN
- `ANGEL_TOTP_SECRET` — TOTP secret for 2FA
- `ANTHROPIC_API_KEY` — Your Anthropic API key
- `TELEGRAM_BOT_TOKEN` — (optional) Telegram bot token
- `TELEGRAM_CHAT_ID` — (optional) Your chat ID

The existing dashboard Claude/Economy selector is separate from the provider factory. To use the factory, set `TRADEMAX_LLM_PROVIDER` or the per-tier `TRADEMAX_QUICK_THINK_PROVIDER` / `TRADEMAX_DEEP_THINK_PROVIDER`, configure the matching key, and install the optional integrations with `poetry install --extras llm-providers`. Set a provider explicitly to avoid automatic credential-based selection. Supported identifiers are listed in `.env.example`; default model IDs are in `config/llm_factory.py`.

Zerodha Kite Connect is optional. Its trading API requires an access token obtained through the official interactive OAuth flow; a TOTP secret alone cannot create a session. Store the resulting daily token in `ZERODHA_ACCESS_TOKEN` and enable the optional adapter with `poetry install --extras zerodha`. The adapter rate-limits order writes and does not retry an ambiguous submission.

### 4. Run

```bash
# Controlled scheduler startup (paper mode, entries paused, circuit breaker halted)
python main.py --start-paused

# Single scan cycle (for testing)
python main.py --scan

# Run backtest
python main.py --backtest

# Walk-forward evaluation of a strategy under a named exit policy (read-only research; appends to evaluation_runs)
python main.py --evaluate CompositeTrend --policy daily_atr_baseline_v1 --from 2026-08-24 --to 2026-11-13

# Dashboard only
python main.py --dashboard

# Fit and validate the HMM regime model (stop the scheduler first)
python main.py --train-hmm --hmm-start 2017-10-01

# Reconcile the paper ledger, fill audit, closed lots and positions
python main.py --reconcile-paper
```

### 5. Dashboard

- Build: `cd dashboard/frontend && npm run build`
- Open the combined dashboard/API at `http://127.0.0.1:8000` after starting the scheduler.
- Optional frontend development server: `cd dashboard/frontend && npm run dev` → `http://127.0.0.1:3000`.

Entry pause and circuit-breaker state survive restarts. First startup defaults
to paused; use the local console to review, clear the halt while staying paused,
then resume paper entries separately. The console also tests Telegram delivery,
starts market data without orders, and shows tick freshness and upcoming jobs.
See [the operations runbook](docs/OPERATIONS.md) before recovery or resumption.

## Dashboard Model Modes

The **Model mode** sliding control in **System controls** selects one of two profiles:

| Mode | Analysis and research | Debate synthesis and final decisions |
| --- | --- | --- |
| Claude (initial default) | `claude-haiku-4-5-20251001` | `claude-opus-4-6` |
| Economy | `deepseek-flash` | `deepseek-v4-pro` |

Set `ANTHROPIC_API_KEY` and `DEEPSEEK_API_KEY` in your local `.env`, then restart the backend once to load credentials. Each profile is disabled in the UI until its server-side key is configured; this does not verify the key's validity or provider balance. Never enter keys into the dashboard or commit them. Model names can be overridden with the settings in `.env.example`.

After setup, switching needs no restart. SQLite remembers the selected mode. Each scan or research job captures its profile once; work already running finishes with its original provider, while subsequent jobs use the new selection. The dashboard shows **Next scan** when an older-profile job remains active. No automatic fallback from Economy to Claude occurs. Existing rule-based fallbacks remain available and failed/missing LLM calls mark the run degraded. Paper/live settings, risk limits, stop monitoring and scan frequency do not change.

**LLM cost** shows current-month estimates separately for Claude and Economy, with no spending cap. The month uses Asia/Kolkata time. Only requests made through this application after this feature was installed are counted. Tokens are taken from provider responses, including cached and thinking tokens. Unknown model prices and failures without usage are marked incomplete. USD prices are versioned estimates; DeepSeek peak rates conservatively ignore Chinese holiday discounts. INR uses `LLM_USD_INR_RATE` (default 90), not a live FX quote. Taxes and currency fees are excluded. Provider invoices remain authoritative. Requests use a configurable timeout and no automatic retries.

The dashboard is local-only. Build the frontend and use `http://127.0.0.1:8000`
for both UI and API. Start without scheduled trading using
`python3 main.py --dashboard`; use `python3 main.py --start-paused` for controlled
scheduled paper operation. Vite on port 3000 is only needed for frontend development.
Remote dashboard access is not supported by this configuration.

Switching functionality is tested offline; Economy's trading accuracy is not established by those tests. Evaluate both profiles in paper mode before relying on changed model decisions with real funds.

## Live-trading boundaries

The additional broker, strategy, point-in-time, audit-ledger, and Monte Carlo components are not a production certification. Dual-broker routing fails over only after a definitive broker rejection; timeouts and connection failures are treated as uncertain and require order reconciliation. The shared Monte Carlo gate is available, but scheduled live execution still needs verified historical evidence, fill reconciliation, position reconciliation, and testing of scheduled square-off behavior. Keep the system in paper mode until the readiness gate in [docs/OPERATIONS.md](./docs/OPERATIONS.md) is completed and approved.

## Project Structure

```
india-ai-trader/
├── main.py                    # Entry point
├── config/
│   ├── settings.py            # All configuration (Pydantic BaseSettings)
│   └── instruments.py         # Symbol → token cache
├── database/
│   ├── models.py              # SQLAlchemy models (Trade, Signal, Portfolio, etc.)
│   └── db.py                  # Database session management
├── broker/
│   ├── angel_client.py        # SmartAPI wrapper with TOTP
│   ├── order_manager.py       # Paper/live order execution
│   ├── portfolio_tracker.py   # Position & PnL tracking
│   └── websocket_feed.py      # Real-time tick data
├── data/
│   ├── instrument_master.py   # NSE instrument download
│   ├── market_data.py         # OHLCV fetching (Angel + yfinance fallback)
│   ├── indicators.py          # Technical indicators (pandas-ta)
│   ├── news_fetcher.py        # News aggregation
│   └── options_data.py        # Options chain analysis
├── agents/
│   ├── base_agent.py          # TradingState TypedDict + BaseAgent ABC
│   ├── market_data_agent.py   # OHLCV + live quotes
│   ├── technical_agent.py     # RSI, MACD, Supertrend, etc.
│   ├── fundamental_agent.py   # PE, ROCE, promoter holding
│   ├── sentiment_agent.py     # News sentiment via Claude
│   ├── options_agent.py       # PCR, max pain, unusual OI
│   ├── macro_agent.py         # Global cues, FII/DII, VIX
│   ├── india_specific_agent.py # F&O ban, bulk deals, circuits
│   ├── risk_manager_agent.py  # Hard limits, Kelly, VaR
│   ├── portfolio_manager_agent.py # Final AI decisions
│   └── graph.py               # LangGraph StateGraph wiring
├── strategies/
│   ├── nifty_options_seller.py # OTM options selling
│   ├── supertrend_adx.py      # Trend-following
│   ├── momentum_breakout.py   # 52-week high breakout
│   └── mean_reversion.py      # Bollinger Band reversion
├── risk/
│   ├── risk_engine.py         # Pre-trade risk checks
│   ├── position_sizer.py      # Kelly + VaR sizing
│   └── stop_loss_manager.py   # ATR trailing stops
├── notifications/
│   └── telegram_notifier.py   # Trade & PnL alerts
├── scheduler/
│   └── market_scheduler.py    # IST market-hours scheduler
├── backtester/
│   ├── engine.py              # Event-driven backtester
│   ├── metrics.py             # Sharpe, Sortino, Calmar, etc.
│   └── report_generator.py    # JSON + text reports
├── dashboard/
│   ├── backend/
│   │   └── app.py             # FastAPI server with SSE
│   └── frontend/
│       ├── src/App.jsx         # React dashboard (dark theme)
│       ├── package.json
│       └── vite.config.js
├── tests/
│   ├── test_settings.py
│   ├── test_indicators.py
│   ├── test_risk_engine.py
│   ├── test_backtester.py
│   └── test_agents.py
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Trading Schedule (IST)

| Time  | Activity                              |
|-------|---------------------------------------|
| 08:00 | Pre-market — instrument master        |
| 08:55 | Broker authentication and reconciliation |
| 09:00 | Connect WebSocket feed                |
| 09:30 | First scan (after 15-min quiet window)|
| 10:00–14:30 | Regime-dependent scans         |
| 14:30 | Last scheduled scan                   |
| 14:45 | Stop new intraday entries             |
| Every 5 min | Stop-loss / target checks       |
| 15:15 | Intraday square-off                   |
| 15:45 | Post-market — settle and reconcile    |
| 16:00 | Telegram daily summary                |
| 18:00 | Cleanup & logout                      |

## Risk Controls

- **Daily Loss Limit**: 2% of day-start equity, including realised and unrealised P&L → block new entries while exits continue
- **Portfolio Drawdown Pause**: Persistent 10% peak-to-trough equity halt; operator review required before reset
- **Universe**: Dated official Nifty 50 membership; unavailable or stale membership blocks scans
- **Max Positions**: 8 simultaneous
- **Per-Trade Capital**: Max 10%
- **VIX Halt**: No new trades if India VIX > 28
- **Regime Gate**: No new entries unless a fitted HMM gives a fresh, non-panic classification (exits are never blocked)
- **Paper Reconciliation**: Scans and resumes are blocked while the paper ledger fails reconciliation
- **Realistic Paper Fills**: Fills need a live reference price; LIMIT orders fill only when marketable and never beyond the limit
- **No-Trade Windows**: First/last 15 min of market
- **F&O Ban Filter**: Auto-skip banned stocks
- **Risk-Budget Sizing**: Defaults to 0.5% planned risk per entry, 10% per-symbol capital and 50% total gross exposure, with fee and slippage checks
- **Daily Entry Cap**: Defaults to three new entries per trading day
- **VaR Constraint**: Max 2% portfolio VaR per position

## Running Tests

Run the suite from an isolated copy so it can never touch the live database, lock file or notifiers:

```bash
# from the repository root (the folder that contains india-ai-trader/ and sol_tools/)
sol_tools/isolated_test.sh -q
```

The test configuration also redirects the database, process lock, control state and credentials, but running `pytest` inside a live checkout is still discouraged.

## Operations and architecture

The paper-trading runbook (including a beginner start/stop/recover guide), MVD scope, safety controls, architecture decisions, and
future live-pilot gate are maintained in [docs/OPERATIONS.md](docs/OPERATIONS.md). Readiness checklists:

- [docs/READINESS_INDIA.md](docs/READINESS_INDIA.md) — pass/fail criteria for the supervised paper session.
- [docs/READINESS_MARKETS.md](docs/READINESS_MARKETS.md) — per market and sleeve: evidence present and missing (all `blocked` today).
- [docs/DERIVATIVES_FEASIBILITY.md](docs/DERIVATIVES_FEASIBILITY.md) — why derivatives stay out of scope.

**No returns are promised or implied.** Paper results are simulated; the only quantitative evaluation so far was unfavourable and inconclusive.

## ⚠️ Disclaimer

This software is for educational purposes. Trading involves significant risk of financial loss. Paper trade for at least 30 days before considering live trading. The authors are not responsible for any losses incurred.
