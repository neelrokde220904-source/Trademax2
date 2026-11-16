# Readiness — markets and strategy sleeves (T16)

**Prepared:** 15 November 2026. **Nothing is enabled by this document.** Every optional sleeve ships disabled in the
example configuration. The operator has enabled the US paper adapter and local crypto simulator in the local `.env`;
the scheduled strategies below run only in paper/simulation mode. The bot remains paper-only. No result here is a claim
of profitability.

Status vocabulary:

- `blocked` — required evidence or capability is missing; do not enable.
- `experimental` — an unvalidated strategy can run in paper/simulation mode; do not treat results as evidence of edge.
- `ready_for_operator_decision` — the engineering and evidence listed are in place, and the remaining decision is a
  human one (risk appetite, budget). It still does not mean "profitable".

## Summary

| Market / sleeve | Flag | Status | Main reason |
|---|---|---|---|
| India core — CompositeTrend INTRADAY (NSE cash) | always on | `ready_for_operator_decision` (continue paper) | Plumbing verified by tests and journeys; economic evidence is inconclusive (see below) |
| ORB intraday | `STRATEGY_ORB_INTRADAY_ENABLED` | `blocked` | No out-of-sample evidence run for this strategy |
| Supertrend + ADX intraday | `STRATEGY_SUPERTREND_ADX_ENABLED` | `blocked` | Same |
| Mean reversion (swing) | `STRATEGY_MEAN_REVERSION_ENABLED` | `blocked` | Same; needs daily history and multi-day paper data |
| Momentum breakout (swing) | `STRATEGY_MOMENTUM_BREAKOUT_ENABLED` | `blocked` | Same |
| VCP / NR7 (swing) | `STRATEGY_VCP_NR7_SWING_ENABLED` | `blocked` | Same |
| BTST strength | `STRATEGY_BTST_STRENGTH_ENABLED` | `blocked` | Same; plus corporate-action data is not ingested |
| ETF rotation (positional) | `STRATEGY_ETF_ROTATION_ENABLED` | `blocked` | Same; ETFs are not in the watchlist, so no data loads |
| US equities, Alpaca paper | `FOREIGN_US_PAPER_ENABLED` | `experimental` | Scheduled 20/50 trend strategy is implemented; no recorded paper fills, and 2027 calendar is absent |
| Crypto spot simulator | `CRYPTO_SIM_ENABLED` | `experimental` | Scheduled simulator is implemented; no multi-day evidence, Binance public feed must be reachable |
| Derivatives (index/stock F&O) | none | `blocked` (out of scope) | See `docs/DERIVATIVES_FEASIBILITY.md` |

## India core — CompositeTrend, INTRADAY, NSE cash

**Evidence present**

- Entry and exit rules covered by regression tests; the unchanged risk limits (0.5% planned risk, 10% per position, 50%
  gross, 2% daily loss, 10% drawdown, minimum net 2:1 reward/risk) are enforced and tested.
- Journeys 1–8 of plan section 9 pass with mocks and replay (`tests/test_critical_journeys.py`).
- A live paper process has run with protective exits, a reconciliation check and a dead-man's switch.
- T10 evaluation (14 Nov 2026, free yfinance 15-minute data, 20 Nifty 50 symbols, 2026-08-24 to 2026-11-13, paper
  engine with the Indian cost model): the baseline policy produced 227 replay trades at ₹100,000 with a **negative**
  net expectancy of about −₹40 per trade (95% bootstrap interval about −₹56 to −₹23), and 0 trades at ₹10,000; the
  2x/4x policy produced 0 trades at both sizes because a gross 2:1 target cannot clear the net 2:1 gate after costs.

**Evidence missing**

- Several weeks of supervised market-hours paper results with the AI review layer active (the replay has none).
- An out-of-sample result on more data than the free source provides (about 60 sessions of 15-minute bars).
- A trustworthy economic result. The replay above is unfavourable, but it covers one regime, has overlapping trades,
  uses 15-minute bars from a free source and omits the AI review layer that filters live candidates, so its interval
  understates uncertainty. It is a warning, not a verdict, and **no rule change follows from it**; equally, it gives no
  support for any claim of profit.
- At least 300 settled paper outcomes for the signal gate (T22) and AI-conviction calibration.
- The Monday 16 Nov supervised-session checklist (`docs/READINESS_INDIA.md`) has not been run.

**Operator decision available:** continue paper-only observation with the current rules and judge the live paper
results (with the review layer) against the replay. Do not change sizing, exits or strategy on the strength of the T10
numbers, and do not read this status as an endorsement of the strategy.

## Indian sleeves (all disabled)

Common state for every sleeve below:

- **Present:** registered contract (horizon, venue, exit policy, review contract), shared exit math, adapter code,
  a router that refuses disabled sleeves, a capability entry, preflight warning when enabled, and unit tests.
- **Missing:** a T10 walk-forward result for the strategy under its own policy, multi-day paper trades, and settled
  outcomes. Free data limits any evaluation to ~60 sessions of 15-minute bars and ~2 years of daily bars.

| Sleeve | Horizon / policy | Additional missing evidence |
|---|---|---|
| ORB intraday | INTRADAY / `intraday_atr_2x_4x_v1` | The 2x/4x policy has produced no trades after costs for CompositeTrend; the same cost hurdle needs re-testing for ORB |
| Supertrend + ADX | INTRADAY / `intraday_atr_2x_4x_v1` | As above |
| Mean reversion | SWING / `daily_atr22_2x_4x_v1` | Overnight gap and settlement behaviour in paper; delivery cash settles T+1 |
| Momentum breakout | SWING / `daily_atr22_2x_4x_v1` | As above |
| VCP / NR7 | SWING / `daily_atr22_2x_4x_v1` | As above |
| BTST strength | BTST / `daily_atr_baseline_v1` | Delivery-risk rule tested in the paper model only; no corporate-action ingestion; no auction/short-delivery data |
| ETF rotation | POSITIONAL / `chandelier_atr22_k3_v1` | ETF symbols (for example NIFTYBEES, BANKBEES, GOLDBEES) must be added to the watchlist; ETF liquidity and tracking checks are not evidenced |

## US equities — Alpaca paper (experimental)

- **Present:** paper-only adapter (refuses live endpoints and missing keys), LIMIT-only orders, NYSE/Nasdaq 2026 calendar
  with early closes and DST handling, a 15-minute scheduled scan, separate USD book, no Indian square-off, and paper
  account checks. The fixed 20/50 moving-average strategy is long-only; entries require a fresh IEX quote, spreads no
  wider than 0.2%, and at least 50 completed 15-minute bars. A 2% stop, 5% target, trend exit and 15-minute pre-close
  flatten manage bot-tagged positions. New entries obey shared portfolio pause/halt controls and the 10% per-position,
  0.5% planned-risk and 50% gross exposure limits.
- **Still missing:** several weeks of recorded paper fills and out-of-sample evaluation; a published 2027 NYSE calendar;
  review of LRS/FEMA and tax treatment. USD/INR conversion uses the explicitly configured static
  `FOREIGN_USD_INR_RATE`, not a live quote. The free IEX feed is single-venue, not the consolidated tape.
- **Status:** `experimental`; running the scheduler is not evidence that the strategy is profitable.

## Crypto — local spot simulator (experimental)

- **Present:** spot-only simulator with filters, fees, slippage, independent daily halt, portfolio caps, FIFO gains and
  a 30% VDA tax estimate with 1% TDS (ADR-0027). A 15-minute job uses completed 1-hour candles and fresh public prices
  for BTCUSDT, ETHUSDT and SOLUSDT. The fixed 20/50 trend is long-only; the 2% stop, 5% target and trend exit never
  short or use leverage. Simulated state is persisted in `database/crypto_sim_state.json`; public-feed failure blocks
  new entries rather than trading on fixture prices. The daily loss cap is the lower of the crypto-specific threshold
  and the portfolio daily-loss limit.
- **Still missing:** multi-day paper evidence, a dynamic USDT/INR source, and legal/tax review. The static labelled FX
  rate and VDA tax output are estimates only; no exchange account or real order endpoint exists.
- **Status:** `experimental`; no claim of profitability or tax advice.

## External actions required (cannot be produced by code)

1. Run the Monday 16 Nov supervised session and sign off `docs/READINESS_INDIA.md`.
2. Provide Alpaca paper keys if US paper evidence is wanted.
3. Add the official 2027 NSE (and NYSE) holiday calendars when published; preflight warns from December.
4. Decide on a corporate-action data source.
5. Review the separate research-spending proposal in plan section 10 before purchasing data or model usage.
6. Accumulate several weeks of paper outcomes before reconsidering any `blocked` item.
