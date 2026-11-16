# India AI Trader — Operational Guide and Living Architecture Record

**Status:** Minimum viable deployment (MVD), paper trading by default; not certified for live trading
**Owner:** Trading-system operator  
**Last reviewed:** 2026-11-14 (hardening phases 0–8)
**Change policy:** Update this document and add an ADR whenever execution, risk limits, data sources, or a broker integration changes.

## 1. What this system does

India AI Trader is a scheduled, **short-horizon NSE equity paper trader**. It scans a configurable list of liquid Indian equities during NSE cash-market hours, builds technical/fundamental/sentiment/macro inputs, applies portfolio risk checks, and records paper fills in SQLite. It can use Angel One for authenticated data and live execution, with yfinance as an OHLCV fallback.

The currently operational MVD trades the technical-agent pipeline. It is not an HFT system, does not promise profitability, and should not be promoted to live trading merely because a scan completes.

### Implemented decision model

1. Fetch 15-minute OHLCV, live quotes, India VIX, and portfolio state.
2. Screen the dated Nifty 50 candidate list (only when a fresh official membership snapshot is available) for usable OHLCV, average prior-session turnover of at least ₹10 crore, a fresh (≤30-second) two-sided quote with spread no wider than 0.3%, relative volume, trend strength, and adequate range. A confirmed 20-bar breakout requires volume on its completed 15-minute candle to be at least 1.5× the average of matching time-of-day candles from at least five prior sessions. This is separate from the session-to-date volume filter. The screener rejects candidates whose last eight completed 15-minute closes stay within 0.5% of session VWAP. Missing depth, stale quotes, or screening errors reject the candidate; there is no silent fallback to custom symbols.
   The session-volume filter also rejects a candidate if completed bars for the current session are missing or invalid, or if no usable matching historical baseline exists. All screener indicators use completed bars only, including in research replays.
3. Capture one shared news snapshot, then run the analyst fan-out: technical, fundamental, sentiment, options, macro, India-market, and geopolitical context.
4. Apply regime classification, bull/bear synthesis, risk assessment, allocation/hedging, and compliance context.
5. Require a complete Bull/Bear/Bull-Bear synthesis review and a successful Portfolio Manager model response. Missing, invalid, or direction-conflicting review blocks the entry; there is no rule-based Portfolio Manager fallback.
6. Route Indian orders through Angel One's order manager. In paper mode, simulated fills include configured slippage and are written to SQLite.
7. Monitor registered positions every five minutes and on fresh feed ticks, and send stop, target, and time exits through the same order manager (see ADR-0020).

### Strategy modules present in the repository

| Module | Intended style | Status in scheduled MVD |
| --- | --- | --- |
| `SupertrendADX` | 15-minute intraday trend following | Sleeve behind `STRATEGY_SUPERTREND_ADX_ENABLED` (default off); no out-of-sample evidence yet |
| `ORBIntraday` | Five-minute opening-range breakout | Sleeve behind `STRATEGY_ORB_INTRADAY_ENABLED` (default off); requires completed bars, 1.5× same-time volume, VWAP alignment, no extended VWAP chop, and RSI confirmation |
| `MomentumBreakout` | Daily 52-week-high swing momentum | Sleeve behind `STRATEGY_MOMENTUM_BREAKOUT_ENABLED` (default off) |
| `VCPNR7Swing` | Daily volatility-contraction breakout | Sleeve behind `STRATEGY_VCP_NR7_SWING_ENABLED` (default off); failed the initial cost-adjusted walk-forward promotion gate |
| `MeanReversion` | Daily Bollinger/RSI swing mean reversion | Sleeve behind `STRATEGY_MEAN_REVERSION_ENABLED` (default off) |
| `BTSTStrength` | Buy-today-sell-tomorrow strength | Sleeve behind `STRATEGY_BTST_STRENGTH_ENABLED` (default off) |
| `ETFRotation` | Monthly positional ETF rotation | Sleeve behind `STRATEGY_ETF_ROTATION_ENABLED` (default off); ETFs must be in the watchlist |
| `NiftyOptionsSeller` | Short OTM index-option premium/theta strategy | Research only; not enabled for autonomous execution |

This distinction is deliberate: the Indian MVD has one governed execution path (`CompositeTrend`). Every other Indian strategy is a registered, **disabled** sleeve (ADR-0023, ADR-0025): it can only create entries after its own flag is turned on, and `docs/READINESS_MARKETS.md` lists the evidence each one still lacks. The separate US and crypto jobs use experimental paper/simulation-only strategies as described in ADR-0027. `NiftyOptionsSeller` and all derivatives are out of scope (`docs/DERIVATIVES_FEASIBILITY.md`).

## 2. Features and safeguards

- Paper trading is the default (`PAPER_TRADING=true`). Paper fills use 0.05% configured slippage and persist to `india_trader.db`.
- A **fail-closed live-trading guard** rejects `PAPER_TRADING=false` unless `LIVE_TRADING_ACKNOWLEDGEMENT=I_ACCEPT_REAL_MONEY_RISK` is explicitly supplied.
- New entries require a passed quantitative screen, an aligned deterministic technical signal, complete AI review, risk approval, and a LIMIT order; model-authored MARKET entries are rejected. The router checks that screening evidence names the entry candidate; missing or contradictory screening evidence blocks entries, not exits. It refreshes all open-position marks before sizing and uses that fresh marked equity, cash, exposure and stop risk rather than scan-time valuation; successful fills in the same scan are projected into each subsequent candidate's book. The router also enforces a 2% 95th-percentile historical one-day portfolio VaR ceiling using aligned 20-session returns and a 25% gross sector-exposure cap (half the 50% gross portfolio ceiling). Sector classifications must come from timestamped Yahoo Finance fundamentals no older than seven days; missing/stale classifications, including legacy held positions without sector metadata, block new entries. The router pins entry price/stop/target/holding period to the technical/risk result, then sizes from a 0.5% equity stop-risk budget, maximum 10% position notional, maximum 50% gross exposure, and available cash (no leverage). Paper limit orders can remain unfilled when their prices are not marketable.
- Portfolio-manager prompts include valid JSON with each risk-approved candidate's available analyst signals, debate verdict, risk and allocation, rather than cutting serialized evidence mid-field. If the complete context is invalid or exceeds 32,000 bytes, entries are blocked instead of sending a truncated review. This does not yet establish timestamped provenance or prove every upstream analyst supplied valid evidence.
- Bull/bear prompts likewise use the collected candidate reports and complete prior arguments, with valid JSON and a 16,000-byte limit for each evidence or argument block. Oversized or invalid debate context prevents a complete review for that candidate instead of silently cutting off the evidence.
- The scheduled AI-assisted path requires a schema-valid sentiment response for each entry candidate at debate, portfolio review and routing. Missing news or model failures are reported as unavailable, not fabricated as `HOLD`; a genuine valid `HOLD` is permitted. Each response is bound to the scan ID, candidate, snapshot timestamp and a digest of the reviewed news and its provenance; the debate and portfolio decision also carry a digest of the sentiment they reviewed. At each gate, the snapshot must be at most 30 minutes old, and at least one attributable article must have a known publication time within the preceding three days and a successful source outcome. General market headlines may satisfy this check, but they are **not** proof of company-specific relevance. Missing, stale, future-dated, untraceable or changed evidence blocks the entry, not protective exits. Isolated rules-only research can proceed without news and records sentiment as unavailable. These checks do not certify every upstream analyst or establish that a sentiment confidence is a calibrated probability.
- The fundamental agent reports missing or invalid metrics, unavailable models, malformed output and provider failures instead of substituting a rule-based recommendation. A rules-only fundamental signal is available solely in isolated research. The scheduled `CompositeTrend` paper path has a versioned required-agent/input contract: technical and sentiment evidence, successful bull/bear synthesis and portfolio review, verified recent 15-minute history, attributable current news, passed quantitative screening (including its fresh-quote check), risk approval, and a valid portfolio snapshot. Fundamental, options, macro, India-specific and geopolitical reports are declared optional; their absence alone does not block this strategy. Unknown strategies have no entry contract and fail closed. These rules do not certify point-in-time freshness of optional fundamentals or apply to exits.
- Entries also require at least 2:1 net reward/risk after estimated fees and slippage, with execution friction no more than 25% of target profit. `MAX_DAILY_ENTRIES` is configurable (default 3; check the local `.env`/dashboard for the active cap), with an 8-position limit. The router rechecks the projected count after each filled entry within a scan; exits remain available. This in-process check does not coordinate live pending orders or other processes.
- The normal paper order path additionally permits at most `MAX_DAILY_ENTRIES_PER_SYMBOL` filled entries in a symbol per IST day (default 1), and requires `SYMBOL_REENTRY_COOLDOWN_MINUTES` after its most recent exit (default 30; 0 disables the cooldown). The daily caps, cooldown, fill and audit are checked or written inside one SQLite write transaction, including across cooperating paper-order processes; FIFO replay also counts legacy closed entries and exits if their closed-lot records are missing. Exits are exempt from entry caps. This does not repair an inconsistent legacy ledger or replace reconciliation, and does not prove live broker pending orders are counted or cancelled.
- The daily loss halt is 2% of recorded day-start equity, including realised and marked unrealised P&L. It blocks new entries while open-position exit monitoring continues. A persistent 10% peak-to-trough portfolio drawdown latch also blocks entries; it can only be reset in paper mode after pausing entries and explicitly confirming operator review.
- A failed portfolio sync or a missing fresh WebSocket price for any open position blocks new entries. Marks are checked again before each routed entry; last-known marks may remain visible for monitoring but are not accepted as fresh risk evidence. Exit routing remains available. This is an in-process check, not an atomic cross-process or live-broker order guarantee.
- The risk manager estimates 95% one-day historical VaR from 20 close-to-close returns, not 15-minute returns. It compares direction-adjusted correlations on the same sessions, including earlier fills in a scan. Twenty observations are a small sample, not a guarantee of future risk or loss. Missing sessions, held-symbol history, invalid bars or correlation variance block new entries rather than supplying a fabricated value.
- The risk manager and order router apply the HMM's maximum position-notional percentage as an additional cap; it cannot increase the existing per-position limit. Missing or invalid HMM caps block new entries. The currently scheduled `CompositeTrend` strategy is explicitly classified as `trend_following` and is eligible only when the HMM permits that category. A required or invalid hedge directive blocks entries because an automatic hedge is not yet verified. These rules can result in no trades, especially in mean-reverting or euphoric regimes; exits remain available. Isolated research books now use the captured HMM strategy, hedge and size directives at entry, but do not reproduce all production risk and AI controls.
- `MAX_OPEN_STOP_RISK_PCT` defaults to 2% of current equity and can only be set lower. It is an additional modelled open-stop budget, separate from the day-start-equity daily-loss halt and the 0.5% per-entry risk limit. Risk review and the router count held positions from fresh marks to their active ratcheted stops, including assumed adverse exit slippage and costs. New entries include estimated entry/exit slippage and costs, and earlier same-scan fills reduce the remaining budget. A missing, widened, breached, pending-exit, identity-mismatched or mixed-lot stop blocks new entries, not exits. Changed position identities also block routing. Isolated research does not yet reproduce this control. The budget is an estimate, not a loss guarantee: gaps, stale data, unfilled exits, live pending orders and cross-process races are not bounded by it.
- The risk history calendar is limited to the configured 2026 NSE holiday list. It includes the 1 February Budget session and the 8 November Muhurat session; the latter's five 15-minute candle timestamps and final close were checked against Angel One data for three liquid stocks. If a provider omits any expected session or bar, entries are blocked rather than assuming it was a holiday. Other calendar years block entries until verified coverage is added. These blocks do not stop exits.
- The isolated 15-minute replay uses the verified regular 1 February Budget Sunday grid and rejects unexpected in-session bars on closed days. Replay refuses unverified calendar years and windows crossing the 8 November Muhurat session: its special-session order and square-off policy has not yet been implemented. It must not be interpreted as a complete exchange calendar or as a production trading gate.
- Angel historical-candle requests from the scan and historical index loader use a conservative shared local scheduler. Cooperating processes using the same `database/candle_rate_limit.state` file serialize requests and wait at least one second after the previous call completes, including failed attempts; this spaces client request starts but does not guarantee provider arrival spacing. A persisted in-flight marker makes the next process wait a full second after an interrupted request, and invalid state blocks requests explicitly. Do not delete, replace, or manually edit the state file while any process may be using it. Processes on other machines, older code, independent applications, and clients using another state path are not coordinated; this is not an account-wide rate limit. [Angel's official rate-limit documentation](https://smartapi.angelone.in/docs/RateLimit) lists candle limits of 2 requests/second, 80/minute and 5,000/hour; limits can change, and provider or network failures can still cause scans to abstain.
- Newly fetched symbol history carries provider, requested symbol/segment/interval, retrieval time, returned row count, fallback reason and a checksum of the captured DataFrame through technical reviews and new research captures. The scheduled AI-assisted `CompositeTrend` entry requires matching NSE 15-minute source metadata and a frame checksum, rechecked at debate, portfolio review and routing. Retrieval must be no more than 30 minutes old and not future-dated; missing, changed, stale or invalid evidence blocks the entry, not protective exits. Angel's adjustment and corporate-action status are explicitly **unverified**. Yahoo is requested unadjusted and accepted only when its returned action fields report zero, which does not prove that the action feed is complete. A recent retrieval does not prove the provider's bars are fresh, and neither the description nor checksum independently verifies the instrument, adjustment, provider, or historical completeness. Older research captures may not contain these fields; isolated replay remains separate.
- The yfinance `.NS` historical fallback is restricted to requests labelled NSE. If another exchange or segment cannot be fetched from the broker, the fallback returns unavailable data and logs the mismatch instead of substituting an NSE-listed symbol. It explicitly disables automatic price adjustment, requests corporate-action fields, and rejects data if those fields are missing, invalid, or report a dividend or split within the returned history. This may block a candidate for the entire lookback. The fallback does not independently verify instrument identity or guarantee the provider's corporate-action data is complete; split-adjustment and source provenance are not yet fully verified.
- VIX above 22 reduces size; VIX above 28 blocks new entries.
- The opening and closing 15-minute windows reject new entries. Intraday positions are checked for time exit at 15:15 IST; an intraday position carried from an earlier date is due from the next 09:15 IST session open.
- Stops and targets are ATR-derived; the stop manager persists entry ATR, extrema, trailing-stop ratchets, and unresolved exit IDs across restart.
- Scheduled protective paper exits use tick-aligned LIMIT orders inside a 0.3% fresh-price adverse collar (`PROTECTIVE_EXIT_COLLAR_PCT`), not a stale position mark or a MARKET order. An unfilled exit remains monitored; a price collar cannot guarantee a fill or cap gap losses. Exit intent is persisted before submission. Unknown acceptance prevents automatic retries and halts new entries until reconciliation; rejected/canceled partial orders first synchronize remaining inventory. The 15:15 time exit remains due later in the afternoon.
- Stop and target triggers need a position mark at most 60 seconds old (no more than five seconds in the future). A stale or missing mark never fires a stop or moves a trailing ratchet; the symbol is listed under unverified marks on the dashboard's Open-position protection card, while time exits are still evaluated. Paper exit intents are settled from the paper ledger; live intents with unknown acceptance still halt for manual reconciliation. Strategy exits are deferred while a protective exit is pending, and new live entries are blocked (ADR-0020).
- The scheduled `CompositeTrend` technical signal uses completed 15-minute bars and only allows long entries with RSI 40–70 (short entries 30–60). An oversold RSI by itself no longer adds a bullish vote. Breakout candles also need 1.5× same-time historical candle volume; ORB research signals apply separate five-minute, same-time volume and VWAP/RSI gates.
- Completed bars are sorted by timestamp; identical duplicates are removed and conflicting duplicates raise an explicit data error. VWAP is computed in timestamp order and remains anchored to the full session when checking the last eight candles for chop.
- Daily ATR requires at least 15 complete regular sessions with all 25 expected 15-minute bars and valid OHLC prices. Missing bars in a later prior session block ATR; the implementation does not yet verify that a whole trading day is absent from the source.
- Angel feed ticks are usable only when both the exchange event time and local receipt time are within 30 seconds, with at most two seconds of future clock skew. Stale or timestamp-less packets do not mark positions, drive tick subscribers, or fill paper orders. A connected socket can therefore be stale, especially outside market hours.
- The liquidity spread check independently rechecks both quote timestamps at decision time; saved research quotes without exchange-time provenance cannot pass the spread filter.
- Session-to-date relative volume compares matching completed 15-minute slots from the regular open. Gaps or invalid volume in the current window make it unavailable; historical sessions missing any matching slot are excluded rather than shifted forward.
- The broker instrument master is validated and atomically replaced; a failed refresh may use a validated cache no older than `INSTRUMENT_CACHE_MAX_AGE_HOURS` (default and maximum 168). A missing, corrupt, incomplete, or expired cache is unavailable. Token lookups fail closed if the master changes while loading, and detect replacement even if its timestamp and size are unchanged.
- Options contracts derive CE/PE from the broker's symbol when its optional `optiontype` field is absent. The context returns metadata for the nearest unexpired expiry, but no bulk contract quote requests: OI, volume, and option prices are unavailable, so PCR, max pain, and the disabled options-selling strategy abstain until a verified provider is integrated.
- Candidate liquidity requires at least 10 complete prior sessions, each with 20 or more valid 15-minute bars. Each new order is also capped at 1% of the symbol's estimated average daily turnover. Angel One depth prices are converted from paise to rupees before spread evaluation. Missing/stale top-of-book prices fail closed, including when REST LTP is available without depth. These filters are not a guarantee of actual executable depth or fill quality.
- A dead-man's switch halts scans after repeated critical component failures.
- The scheduler runs a pre-market data/authentication job, intraday scans, five-minute exit checks, post-market snapshots, and optional Telegram notifications.
- The legacy XGBoost signal gate and strategy selector are not wired into the scheduled execution path. The legacy gate is not calibrated and must not be enabled without the planned out-of-sample calibration and safety review.
- Backtests accept the fetcher's `datetime` column, generate signals from completed bars, fill at the next bar's open, and model stop gaps conservatively. Walk-forward output remains research evidence, not strategy approval.

### Safeguards added in the November 2026 hardening (ADR-0022 to ADR-0027)

- **One position per symbol.** A second entry on a held symbol is refused at the router and again in the paper ledger; mixed legacy lots can only be exited whole.
- **Settled cash.** Delivery sale proceeds are unavailable for sizing until the next trading day (T+1); intraday round trips net immediately. The portfolio snapshot shows settled and unsettled cash.
- **Calendars.** NSE holidays come from `data/calendars/nse_equity_holidays.json`; an unknown calendar year blocks entries (never exits). F&O expiry days are read from the contract master. US and crypto use their own calendars and are never closed by the Indian 15:15 square-off.
- **Exit policies.** Every new position records its exit policy and ATR timeframe; trailing stops only tighten, and a stop that is restored after a restart is never loosened.
- **Cost and rate controls.** Paper partial fills and bounded protective-exit re-pricing (at most three collar steps) are modelled; the order rate limiter refuses entries over the window but never drops exits; a halt cancels resting entries.
- **Budgets and deadlines.** `LLM_DAILY_BUDGET_INR` stops *new paid analysis* (never exits) and unknown pricing counts as exceeded; a scan that runs longer than `SCAN_ENTRY_DEADLINE_SECONDS` can no longer place entries.
- **Alerts and secrets.** Critical alerts are stored before delivery and retried; the dashboard shows undelivered ones. Logs pass through a redaction filter (API keys, tokens, client codes) and the broker SDK's file logging is silenced.
- **Dashboard transparency.** The overview shows the next scan trigger and whether it will run; Portfolio rows show each candidate's review contract; Operations shows the capability inventory, sleeve flags, LLM budget and skipped-scan reasons.


## 3. Repository structure

| Path | Responsibility |
| --- | --- |
| `main.py` | CLI entry point, scheduler, scan, backtest, dashboard, preflight |
| `config/` | Validated environment configuration and instrument lookup |
| `agents/` | LangGraph state and analysis/decision agents |
| `data/` | OHLCV, indicators, news, options, and market-event sources |
| `risk/` | Entry risk controls, sizing, exit manager, regime classifier, circuit breaker |
| `broker/` | Angel One client, paper/live order manager, portfolio state, router |
| `scheduler/` | IST market-hour orchestration and exit lifecycle |
| `backtester/` | Historical simulation, metrics, optimization, reports |
| `research/` | Paper-only captured inputs, model comparisons, and forecast outcomes |
| `operations/` | Non-mutating operational readiness checks |
| `database/` | SQLite models and session lifecycle |
| `docs/` | This living operating and architecture record |

## 4. Runbook

### Beginner runbook: start, watch, pause, stop, recover

Everything here is **paper trading**. Commands run from `india-ai-trader/`; the dashboard is at <http://127.0.0.1:8000>.

**Start (normal).**

1. `python3 main.py --preflight` — read-only; fix every error it prints (warnings are advice). It needs no broker call.
2. `caffeinate -dims python3 main.py --start-paused` keeps the Mac awake and starts with entries **paused** and the dead-man's switch halted, so you can review before anything trades.
3. Open the dashboard. Check `Mode = PAPER`, the build digest, reconciliation `pass`, broker authenticated, and the feed state. Then use *Reset halt* and *Resume entries* (both ask you to confirm that you reviewed).

**Watch.** Read the dashboard top to bottom: the status bar (mode, paused/halted, feed, scan), Overview ("Next trigger" and "Will it run?" say why a scan will or will not happen), Portfolio (positions, stops, and each decision's review contract), and Operations (reconciliation, protection coverage, capabilities, LLM budget, undelivered alerts, skipped scans). A skipped scan is normal on a holiday, after a halt, or when the LLM budget is exhausted; the reason is shown.

**Pause.** *Pause entries* stops new scans from entering. Protective exits, square-off and reconciliation keep running. Pausing is always safe.

**Stop the process.** Pause first; confirm `/api/operations/reconciliation` shows no pending exits. Find the PID with `lsof database/trading_service.lock` (or `ps`), then `kill -TERM <pid>`; wait until the lock has no holder. Never use `pkill` or `killall`, and never stop during market hours with an open position unless you intend to rely on the next start to restore protection (it does, from the ledger and persisted stops).

**Recover after a crash, reboot or halt.**

1. Start with `--start-paused`. The ledger and stops are restored; a position without a valid stop is reported.
2. Read reconciliation. If it says `fail`, resuming is refused until the cause is fixed.
3. A **circuit breaker (DMS) halt** names the failing component (feed, broker, LLM, graph, protective checks). Fix or wait for it, then *Reset halt* and *Resume*.
4. A **daily-loss or drawdown halt** is a latch: it persists across restarts until reviewed. Read the reason, then reset only if you understand it.
5. Missing quotes or a stale feed at the open are expected on weekends; on a trading day, use *Reconnect feed*.

**Budget.** `LLM_DAILY_BUDGET_INR` (default in `.env.example`) caps paid analysis per day. When reached, scans are skipped with a reason; exits are unaffected. Pricing that is unknown counts as exceeded. Research spending beyond this (plan section 10, about ₹1,000/month) needs your explicit approval.

**Foreign-market paper and simulation.** If `FOREIGN_US_PAPER_ENABLED=true`, a separate 15-minute schedule can trade only through Alpaca's paper endpoint during the known regular US session. Its experimental long-only 20/50 trend strategy requires fresh IEX data, uses LIMIT orders, a 2% stop / 5% target and flattens bot-managed positions before the US close. Stops are polled, not broker-hosted, so a price gap, API failure or process outage can delay an exit. If `CRYPTO_SIM_ENABLED=true`, a separate 15-minute schedule simulates a long-only 20/50 trend on supported USDT spot pairs; it requires live public candles/prices and never uses fixtures to enter. Both share the India pause, circuit breaker, daily-loss/drawdown and portfolio exposure caps. Existing positions are monitored for exits before new-entry gates. These strategies are experimental, use static FX conversion for risk reporting, and have no demonstrated edge. The operations API reports their status and latest scan reason.

**What the system does not do.** It does not place real orders or derivatives and does not predict or promise returns (see section 7). The US and crypto integrations remain paper/simulation only.

### Initial setup

```bash
cd /Users/neelrokde/Desktop/trading\ agent2/india-ai-trader
poetry install
cp .env.example .env
python3 main.py --preflight
```

Run the test suite only from an isolated copy (`sol_tools/isolated_test.sh` at the repository root). Running `pytest` directly in the live folder was once observed to write test rows into the live database; the test configuration now redirects every database, lock file and notifier, but the isolated runner also proves the copy matches the project fingerprint.

Use a broker account only for data/session access while evaluating paper trades. Do not put secrets in source control; `.env` is ignored.

### Operating commands

```bash
# Safe, no broker calls and no writes
python3 main.py --preflight

# One paper-trading pipeline cycle
python3 main.py --scan

# Scheduled paper operation plus dashboard
python3 main.py

# Controlled recovery: durable pause and safety halt before scheduler startup
python3 main.py --start-paused

# Fit, validate and save the HMM regime model (stop the scheduler first)
python3 main.py --train-hmm --hmm-start 2017-10-01

# Cross-check the paper ledger, fill audits, closed lots and positions (exit 1 on failure)
python3 main.py --reconcile-paper

# Daily MomentumBreakout walk-forward research (does not enable the strategy)
python3 main.py --backtest

# Inspect point-in-time captures and hypothetical next-session outcomes
python3 main.py --research-snapshots
python3 main.py --forecasts

# Retry settlement after completed NSE closes are available
python3 main.py --settle-forecasts

# Explicitly compare 2–10 complete captured scans; model API charges apply
python3 main.py --paper-compare SNAPSHOT_ID_1 SNAPSHOT_ID_2 --confirm-research-cost

# Rules-only baseline 1x/2.5x vs candidate 2x/4x daily ATR exits:
# no model charges, no broker orders, no automatic policy changes
python3 main.py --atr-exit-compare SNAPSHOT_ID_1 SNAPSHOT_ID_2
```

Before every session, run preflight, confirm the system is still in paper mode, review logs, and confirm the instrument master downloaded successfully. During a session, monitor the dashboard/Telegram and stop the process if data quality is suspect. After the session, review the SQLite trade ledger, daily PnL snapshot, rejected signals, and any dead-man's-switch events.

The scheduler and one-shot scan commands use an OS process lock, so a second trading scan process is rejected while another holds the lock. The Operations page reports the process ID and a source fingerprint for Python and frontend source files; it does not certify dependency versions, external data, or strategy performance. A controlled restart still requires checking paper-ledger reconciliation and open-position protection first.

Nifty 50 membership is fetched from the official `www.niftyindices.com` CSV host,
with redirect handling and a browser-compatible user agent. It must contain
exactly 50 unique symbols before a dated cache is written. Membership refresh
does not supply historical constituent lists or certify instrument-token coverage.

### Paper research and forecast tracking

- `--atr-exit-compare` gives both isolated books the same captured technical candidates and compares versioned daily ATR(14) policies. Both use a 2x Chandelier trail with a monotonic ratchet. Initial levels are anchored to simulated fills; ATR is frozen from prior sessions at signal time, and a completed bar's excursion only changes the next bar's stop. Size and net reward/risk gates remain unchanged, so a gross 2:1 candidate may have no eligible trades after costs. Results are persisted and marked **inconclusive, not for promotion**. This rules-only diagnostic omits production AI review and does not certify full portfolio/queue/latency parity; it cannot activate 2x/4x in the scheduled bot.
- Each paper scan captures the market, shared news, regime, and the actual
  fundamental, options, macro, and India-specific inputs used by the analysts.
  RSS is fetched once per scan; symbol-specific sources are combined with it.
  Articles have an observation time and known/unknown publication-time status;
  future-dated articles are excluded. Source failures and unconfigured or empty
  sources are reported rather than silently treated as fresh news. Scheduled
  executions register their protective stops before research capture begins.
- Capture is prospective: **old scans cannot be recreated from today's data**.
  A missing, invalid, or future-dated input marks the snapshot incomplete and
  prevents replay and scoring of directional forecasts from that capture.
  Research tables are added by normal database initialization. Compressed
  snapshots accumulate in the local database; monitor its size and arrange
  retention/backups appropriate for your operation. Account positions, cash,
  and broker balances are not copied into research snapshots.
- The dashboard's **Research** page and the commands above expose the same
  results. Select two to ten complete snapshots to compare Claude, Economy,
  and a genuinely rule-based baseline across the analyst pipeline. Claude and
  Economy must both be configured; the comparison does not change the dashboard
  model selection or silently substitute another provider. Each variant starts
  with an independent virtual portfolio and **never calls the broker order
  router**. Model runs can be costly and must be explicitly confirmed.
- Simulation uses the captured 15-minute bars, the next bar's open, configured
  slippage, Indian transaction costs, deterministic signal/risk gates, the
  position sizer, daily limits, and a 15:15 intraday square-off when its bar
  is available. If a next bar is missing, the order is not filled; if a required
  square-off or intermediate bar for an open position is missing, the comparison
  fails. Bars arriving after an earlier snapshot are not replayed retroactively.
  Positions open at the last
  snapshot remain marked to captured prices rather than being fictitiously
  closed. This is not a full reproduction of broker fills, operational pause
  controls, real execution latency, or live trading; current risk settings are
  applied to the comparison. Models may already know events after a historical
  capture through their training, even though supplied inputs are time-bounded.
- The technical agent's entry levels use daily ATR(14), built from completed
  prior sessions of the 15-minute bars: stop `TECHNICAL_STOP_DAILY_ATR_MULTIPLIER`
  (default 1.0) and target `TECHNICAL_TARGET_DAILY_ATR_MULTIPLIER` (default 2.5)
  daily ATRs away. 15-minute ATR targets (about 0.6%) could not clear intraday
  costs or the 2:1 net reward/risk gate. Fewer than 15 completed sessions means
  no levels, and the risk manager rejects the entry. A missing or malformed bar
  in an otherwise captured prior session also makes ATR unavailable. With daily ATR near 2%, the
  stop is roughly 2% and the target roughly 5% away, so many intraday trades will exit at the 15:15
  square-off rather than at target. An empty comparison is still not proof that
  the models agree.
- The quantitative screener runs before the LLM analysts. It compares today's
  volume over completed 15-minute bars with prior sessions over the same number
  of bars (`SCREENER_MIN_RELATIVE_VOLUME`, default 1.0) and requires 15-minute
  ATR of at least `SCREENER_MIN_ATR_PCT` (default 0.2%) of price.
- Forecasts are hypothetical directional predictions, not executed-trade P&L.
  The original confidence is shown as reported, not as a calibrated probability.
  `HOLD`/`AVOID` are recorded but unscored; no neutral range is assumed. The
  first *following* NSE trading session's close is used, not the current
  session's close. The 2026 forecast calendar includes Budget and Muhurat
  special sessions, and unverified years stay pending. Settlement runs at
  16:15 IST on weekdays and can be retried manually. It requires a valid Nifty
  daily close and one exact-session symbol close, plus nonnegative, finite
  dividend and split fields. Missing, invalid, duplicate, or detected-action
  data stays pending rather than substituting a later close. Daily settlement
  requests unadjusted Yahoo Finance prices; provider-supplied zero action fields
  do not prove complete corporate-action coverage. Provider revisions, other
  actions, and reference-price quality can still affect results. A new
  directional forecast needs either a quote with current receipt and exchange
  timestamps (at most 30 seconds old and not future-dated), or a same-session
  completed 15-minute bar whose close is at most 30 minutes old. Untimestamped
  options spot values and unverified or closed sessions do not supply a
  reference; those forecasts are marked `missing_reference` rather than scored.
  Previously captured snapshots are revalidated when replayed; pending forecasts
  have their snapshot, observation time and reference revalidated before
  settlement. Previously settled legacy outcomes remain recorded but are
  excluded from current directional-accuracy totals and labelled unverified.
  A zero price change counts as not directionally correct.

### Recovery from the first paper session

- The graph checkpoint serializer supports the ordinary numeric, datetime and
  string/object OHLCV data used by the market-data pipeline, including
  timezone-aware candles. Unsupported DataFrame types fail explicitly instead
  of using pickle or silently changing data. Each run has its own checkpoint
  thread; a new scan does not automatically resume a prior failed run.
- Angel WebSocket reconnects are supervised by one worker and subscriptions
  are restored by mode. The application overrides the SDK connection method
  to require TLS certificate and hostname verification using the certifi CA
  bundle, without an insecure fallback. This also handles Python installations
  with no configured system CA file. A connected socket is not proof of usable market data: check tick
  age and count during market hours. Freshness requires both the exchange event
  and receipt timestamps within 30 seconds, not merely a connected socket.
- A DeepSeek timeout is recorded as a failed, degraded attempt, without
  automatically retrying or changing providers. `LLM_TIMEOUT_SECONDS` can be
  adjusted in `.env` (greater than 0, at most 180 seconds), but increasing it
  may delay a scan and does not guarantee success. Review provider status,
  logs and latency before choosing a value.
- The scheduler entry pause and dead-man halt/counters now persist in
  `database/operational_state.sqlite3`. Missing entry state defaults to paused;
  corrupt state prevents startup. Do not delete this file to clear a halt.
  Graph order routing checks these controls again before new entries, so a
  pause during analysis also blocks subsequent entry submission. Position exits
  remain permitted. Persistence errors fail closed and require operator recovery.
- For a controlled restart, verify paper mode, pause entries, inspect active
  analysis and positions, and stop only the identified scheduler process.
  Restart with `python3 main.py --start-paused`. This sets the durable pause
  and halt before starting scheduler jobs. Confirm `/api/health` reports
  `paper_trading=true`, `scheduler_running=true`, `trading_paused=true` and
  `dead_mans_switch.halted=true`. Startup never automatically clears the halt.
  Review first, clear the halt while staying paused, and resume separately.
  In paper mode, startup rebuilds positions and stops from the persisted
  ledger, then reconciles them. A missing or wrong-side stop, or a failed
  reconciliation, keeps entries paused. Paper trailing-stop ratchets persist
  and restore subject to entry identity and protection checks. Live
  positions are not reconstructed; do not restart a live session with open
  positions without a separate broker reconciliation and exit-monitoring plan.
  On shutdown, the dashboard allows five seconds for connections to finish
  before cancelling remaining requests, including open status streams. This
  does not make restarting during an active order safe.

### Local operator dashboard

Build the existing UI once (and after frontend changes):

```bash
cd dashboard/frontend
npm run build
```

The scheduler serves the built UI and API together at `http://127.0.0.1:8000`
(or the configured dashboard port), bound to loopback. A separate Vite server
is not required. `--dashboard` serves the UI without running trading jobs.
`/api/health` remains the JSON health endpoint; `/` serves the operator console.

The console uses a light "soft UI" layout with five tabs (the URL hash, e.g.
`#market`, selects a tab; a menu replaces the tabs on narrow screens):

| Tab | What it answers |
|---|---|
| **Overview** | "Can the bot open a new paper trade right now?" — a plain-English answer, a checks ring, the blocking readiness checklist, pause/halt/recovery controls, the last scan explained (trade ideas, fills, blocks and their reasons), upcoming jobs and a glossary. Known AI-review failures can be shown even when no trade idea was produced; issue counts are not trade counts. |
| **Market** | HMM regime with confidence/panic gauges and what it means for sizing, today-vs-tomorrow regime odds, Nifty 50 + India VIX history shaded by regime, today's movers, breadth, and a watchlist table with day range, intraday sparkline and tick age. |
| **Portfolio** | Value/P&L/cash/positions, equity curve, cash-vs-positions donut, daily P&L bars, risk-limit meters, open positions, every portfolio-manager decision with its outcome (traded / blocked by a rule / failed / analysis only), and the execution ledger. |
| **Operations** | Feed health and tests, Telegram test, analysis-only scan, ledger reconciliation (with a fill-outcome chart), AI agents, the full job schedule and known limits. |
| **AI model** | Claude/Economy mode switch, models in use, degraded status and estimated monthly cost by mode. |

The status-stream connection (the dot in the top bar) is deliberately distinct
from the broker feed. Missing or failed data is not evidence that the system is
ready. "Degraded" on the AI tab means the last run had a failed AI call *or any*
pipeline error — including orders stopped by a safety rule such as the
naked-short block — so check the last-scan reasons before assuming a provider
fault. Charts honour the operating system's reduce-motion setting.

Read-only endpoints behind the newer panels: `GET /api/market/watchlist`
(latest tick, change, day range, tick age and an in-memory one-price-per-minute
sparkline since the backend started), `GET /api/signals` (now persisted per scan
decision with `outcome`, `block_reason`, `quantity`, `price` and
`analysis_only`), `hmm.history`/`hmm.latest.regime_probs`/`transition_probs`/
`directive` in `GET /api/operations`, and `market_session` in `GET /api/health`.
A backend started before these changes returns 404 for them; the console then
shows "available after the next controlled restart" instead of failing, and it
estimates the market session from the browser clock (holidays not checked).
Decisions from scans before that restart were only summarised in logs, so the
Portfolio decision list starts empty. Frontend-only rebuilds take effect on
browser reload without restarting the scheduler. Malformed portfolio decision
responses, non-object decisions, and decisions missing a symbol or action are
reported as errors and rejected without inventing order details. Other decisions
still pass through the normal order and risk validation.
All control mutations require loopback, an allowlisted local Origin, JSON and
`X-Dashboard-Control: 1`. Recovery controls are paper-only and require an
explicit operator-review acknowledgement. Clearing the halt never resumes
entries. The emergency halt pauses entries and latches the circuit breaker.

Use **Connect data only** while paused to authenticate and connect the Angel
feed without placing orders or calling an LLM. The instrument cache must already
contain watchlist tokens. Wait for fresh ticks; review recovery after an actual
disconnect. Do not mistake mock reconnect tests for real-market evidence.
**Test reconnect** deliberately closes the current transport while paper
entries are paused. Confirm attempts increase, all subscriptions return, and
fresh tick count resumes increasing. Tick getters reject disconnected or
older-than-30-second cached quotes instead of returning stale prices.
Use **Send test message** to verify Telegram delivery without disclosing
the configured token or recipient in the API.
**Analyse one stock (no orders)** runs the full graph on the first token-resolved
watchlist symbol with `execute_orders=False`, while entries remain paused and the
halt remains latched. This uses the selected paid model profile; it is a
connectivity/checkpoint check, not a supervised paper-fill evaluation. Overlapping
analysis/scan requests are rejected. The console records completion/degradation
and error counts but does not publish private prompts or reasoning traces.

NSE equity watchlist symbols such as `RELIANCE` resolve to the explicit
`RELIANCE-EQ` instrument-master entry. Missing symbols are listed in diagnostics;
the application does not guess replacements or silently select another series.
Exchange-confirmed renames live in `config/symbol_changes.py` and are applied
to the watchlist and instrument lookups. `TATAMOTORS` became `TMPV` (NSE
circular CML70875, effective 2025-10-24; token `3456`, `TMPV-EQ`). The demerged
commercial-vehicle company `TMCV` is a different security and is not an alias;
it is not on the watchlist.

India VIX resolves from the instrument-master `India VIX` entry (default token
`99926017`), not the previously hardcoded unrelated index. If Angel One cannot
provide a valid same-day value, the system tries Yahoo Finance `^INDIAVIX`;
the fallback must be from the same day and no more than 20 minutes old.
Missing or invalid data from both sources blocks new risk approvals. The graph
adapters use the existing queued-event,
depth and HRP APIs. An absent or invalid HMM is explicitly reported as
`UNAVAILABLE`/degraded, not a learned low-volatility classification, and it
blocks new entries (see the regime gate below). Untrained RL
does not prevent the HRP module loading; its per-asset cap retains cash rather
than renormalizing above the cap. These corrections are not certification of
the optional research components or permission to enable live execution.

### HMM regime gate

The regime model is a 4-state Gaussian HMM over daily Nifty 50 realised
volatility (20 sessions), 20-session momentum and India VIX. It is stored as
JSON at `models/hmm_regime_classifier.json` (override with `HMM_MODEL_PATH`).
Pickle files are refused, and the loader validates the feature list, state
labels, matrix shapes, positive-definite covariances and that transition/start
probabilities are finite and normalised; anything else leaves the model unavailable.

Train while the trading process is stopped (the running process loads the
model only at startup):

```bash
python3 main.py --train-hmm > hmm_report.json   # default start 2017-10-01
```

Training uses completed Angel One daily candles up to yesterday. The default
start is 2017-10-01 because Angel's Nifty 50 daily history is missing
2017-08-25..2017-09-12, and the fetch refuses any gap longer than 10 days rather
than fitting across it. Daily requests use whole-day bounds, retry only on
Angel's rate-limit response (training waits up to 15/30/60/120/180 s), and raise
on any other error instead of returning partial history. A model is saved only
if validation passes:

- the `SYSTEMIC_PANIC` state has the highest mean VIX, and that mean is at least 22;
- at least half of the 12–31 March 2020 sessions decode as `SYSTEMIC_PANIC`;
- the saved JSON reloads and reproduces the trained model's latest prediction.

States are labelled by stress (standardised realised volatility plus VIX):
lowest is `LOW_VOL_TREND`, highest `SYSTEMIC_PANIC`, second-highest
`HIGH_VOL_MEAN_REVERT`; a calmer middle state is `BULL_EUPHORIA` only when its
mean momentum is positive, otherwise it is also `HIGH_VOL_MEAN_REVERT`.

New entries are blocked when the model is missing or invalid, when inference
data cannot be fetched, when the latest completed session is more than 7 days
old (`degraded`), or when the regime is `SYSTEMIC_PANIC`. Next-day panic
probability above 30% halves the position-size cap. Exits, stops and
square-off are never blocked by the regime. A model older than
`HMM_RETRAIN_DAYS` (default 90) is reported as stale on the dashboard; retrain
then, but staleness alone does not block entries. Use **Classify now (no
orders)** on the dashboard, or wait for the next scan, to see the current regime
while entries are paused. The classification is cached per IST date; failures
are not cached.

### Realistic paper fills and reconciliation

Paper orders fill only against a WebSocket reference with both exchange and
local receipt times no older than 30 seconds. Untimestamped broker REST LTPs
do not establish freshness. Without a verified reference, an entry is
rejected (`REJECTED_NO_REFERENCE`); brokerless paper entries therefore never
fill. A LIMIT order fills only if the reference is marketable (BUY: reference ≤
limit, SELL: reference ≥ limit) and never at a price worse than its limit;
MARKET orders fill at the reference adjusted by `SLIPPAGE_PCT`. Scheduled exits
and graph-routed exits do not fall back to requested or stale prices. An exit
without a fresh price remains open and requires monitoring and intervention;
this is not a guarantee of liquidation. Reconciliation still flags historical
`requested_price_reference` fills. Every attempt, filled or not, is written
to the `paper_fill_audits` table with the requested and reference prices,
reference source and age, slippage and outcome.

The normal Angel paper route uses a SQLite `BEGIN IMMEDIATE` transaction to
serialize cooperating paper-order writers across processes, rechecking daily
entry caps, cooldown and exit-lot identity before committing a fill and its
audit. A stale, oversized, wrong-side or mixed-horizon exit is recorded as
`REJECTED_POSITION_CHANGED` instead of opening an unintended opposite position.
Other database backends fail explicitly for paper orders because this
serialization has not been implemented for them. This does not coordinate
aggregate portfolio risk, broker submissions or external writers that bypass the
paper-order route; mixed-horizon protection and live order races remain
unverified.

Executed paper orders are replayed FIFO (`broker/paper_ledger.py`). The same
replay records closed lots, rebuilds tracker positions, cash and realised P&L
after restarts, and drives reconciliation, so the three cannot disagree.

`python3 main.py --reconcile-paper`, `GET /api/operations/reconciliation` and the
dashboard **Paper reconciliation** card run the same checks:

| Severity | Codes |
|---|---|
| fail | `closed_lots_mismatch`, `missing_fill_audit`, `audit_outcome_mismatch`, `trade_price_mismatch`, `fill_price_mismatch` (audit vs. simulator), `intraday_carried_overnight`, `intraday_open_after_squareoff` (after 15:20 IST), `invalid_ledger_stop`, `tracker_position_mismatch`, `missing_stop`, `stop_side_mismatch` |
| warn | `requested_price_reference`, `stale_reference` (>30 s), `pending_paper_order`, `orphan_stop` |

A failure pauses entries before every scheduled paper scan, blocks the
dashboard **Resume** control, and is notified after the post-market run. Fix the
cause in the ledger rather than deleting audit rows; warnings require review but
do not block. The paper intraday square-off flattens every open INTRADAY ledger
position (not only stop-manager time exits) and pauses entries if any exit fails.
Reconciliation proves internal consistency of simulated fills; it does not prove
that a real broker would have filled at those prices.

### Telegram delivery setup

A successful bot or chat lookup does not prove message delivery. Send `/start`
to your own bot, then use your **personal chat ID** (or the ID of a group in
which the bot can post), not the bot's user ID. To inspect recent chat IDs
locally without exposing the token, run the following from the project directory
after sending the message:

```bash
python3 - <<'PY'
import asyncio
from telegram import Bot
from config.settings import settings

async def main():
    async with Bot(settings.telegram_bot_token) as bot:
        for update in await bot.get_updates():
            if update.effective_chat:
                print(update.effective_chat.type, update.effective_chat.id)

asyncio.run(main())
PY
```

Set `TELEGRAM_CHAT_ID` privately in `.env`; do not share the ID or token in
chat. The command needs unconsumed updates and does not work while a webhook is
configured. A controlled restart is needed to load the new value, followed by
an actual test message to verify delivery. Do not rely on Telegram alerts until
that test succeeds.

### Readiness gate for a future live pilot

1. Obtain a representative out-of-sample backtest, including Indian charges and slippage, and verify the benchmark and data timestamps.
2. Complete at least 30 market days of paper trading with reconciled fills and no unresolved circuit-breaker events.
3. Exercise order rejection, timeout/uncertain submission, restart recovery, stop exits, scheduled square-off, and post-market settlement using isolated test accounts or mocks.
4. Perform a manual review of risk limits, broker order types, token mapping, holiday calendar, static-IP allowlisting, and independent position reconciliation.
5. Create an ADR approving the exact strategy, universe, order type, size, and rollback procedure.
6. Set the explicit live acknowledgement only for a limited pilot. Remove it to return to fail-closed paper mode.

## 5. New changes in this MVD

- Fixed `.env` parsing for the documented comma-separated `GLOBAL_WATCHLIST`; this was preventing settings—and therefore the application—from starting.
- Added `--preflight`, a read-only readiness check which validates capital/risk settings, execution mode, and cache availability before the session.
- Added explicit acknowledgement required for live execution; simply changing `PAPER_TRADING` cannot silently enable real-money orders.
- Carried fill price, stop loss, target, holding period, and strategy through order results so the scheduled stop manager has the information it needs.
- Repaired scheduled exits: it now calls the synchronous order API with a resolved instrument token rather than awaiting an incompatible dictionary call.
- Made paper mode broker-optional: absent Angel One credentials no longer trigger repeated login attempts for quotes, WebSocket setup, or simulated fills; public OHLCV fallback remains available.
- Added regression coverage for preflight and watchlist parsing.

## 5.1 Provider, broker, and research additions

- The provider-neutral factory supports per-tier quick/deep provider selection. It is opt-in through `TRADEMAX_LLM_PROVIDER` or tier-specific settings; the existing dashboard Claude/Economy selector remains a separate setting. Install optional integrations with `poetry install --extras llm-providers`. No provider is switched automatically after an API error.
- Kite Connect authentication uses an access token from the official interactive OAuth flow. TOTP does not mint that token. Configure `ZERODHA_API_KEY` and the current `ZERODHA_ACCESS_TOKEN`; the optional adapter is installed with `poetry install --extras zerodha`.
- Dual-broker failover occurs only after a definitive rejection. A timeout, disconnect, or ambiguous response is not retried on the secondary broker because the first order may have been accepted; reconcile the order before acting.
- The live 15:15 square-off serializes with stop checks and checks the broker order book before submitting flatten orders. If a position has an unresolved order, square-off fails closed and triggers the Dead Man's Switch; manual broker reconciliation is required.
- The new `FastExecutionLoop`, Monte Carlo validator, point-in-time guard, hash ledger, accuracy scorer, chart formations, and strategy modules are available as components, not as proof that the scheduled MVD consumes validated data or has passed live qualification. The graph's live entry gate requires exact-setup history and five-minute bars. Paper execution does not currently enforce that Time Machine context.
- The Zerodha adapter rate-limits all order-write calls to at most eight requests per second. It does not provide an automated OAuth browser flow.
- Run `python3 main.py --preflight` before each session. Preflight can validate local configuration, but cannot independently verify broker-side static-IP allowlisting.

## 6. Architecture Decision Records (ADRs)

### ADR-0001 — Paper-first execution (accepted)

**Decision:** The default and MVD execution mode is paper trading. Live execution requires a separate literal acknowledgement in configuration.  
**Why:** Order bugs, data gaps, and model instability can create irreversible losses. A safer default makes an accidental mode change non-operational.  
**Consequences:** Real-money use has an explicit operational step; paper fills are simulations and do not establish broker fill quality.

### ADR-0002 — One governed MVD strategy path (accepted)

**Decision:** The scheduled execution path uses the multi-agent technical decision pipeline. Research strategy classes are not automatically merged into it.  
**Why:** Connecting four incompatible holding periods and instrument types without portfolio-level validation would make results uninterpretable and risk controls incomplete.  
**Consequences:** Options selling, breakout, and mean-reversion modules remain research/backtest work until individually approved.

### ADR-0003 — Fail closed on configuration/data ambiguity (accepted)

**Decision:** Invalid settings prevent startup; missing instrument tokens prevent an exit/order rather than guessing.  
**Why:** A bad configuration or unresolvable instrument identity must not create an unintended trade.  
**Consequences:** Operators must remediate configuration/cache errors before a session. Preflight surfaces these conditions early.

### ADR-0004 - Runtime LLM Profiles (accepted)

**Decision:** Retain the configured Claude Haiku/Opus profile and add a dashboard-selectable DeepSeek Flash/Pro Economy profile. Default to Claude; persist mode and revision in `llm_mode_setting`. Do not impose a hard spending limit or automatically change providers on failures.

**Execution:** `config/llm_runtime.py` captures an immutable profile using a task-local context at each scan/research boundary. LangGraph worker threads inherit the context; nested operations reuse it. `agents/llm_client.py` resolves fast/decision roles, validates final response text and records usage. The scan state carries a nonsecret profile/run identifier for inspection. Runtime settings are never mutated mid-job. `llm_runs` tracks active/degraded jobs with heartbeats; stale records are shown as interrupted rather than perpetually running. Existing trade tables are unchanged.

**Control API:** `GET/PUT /api/llm-mode` and `GET /api/llm-usage`. Updates require the current `expected_revision`; conflicts return 409. Missing provider credentials also return 409. Updates require a loopback connection, allowlisted local Origin, JSON content type and `X-Dashboard-Control: 1`; Host validation prevents DNS-rebinding hosts. Backend and Vite bind to loopback. The default frontend origins use ports 3000, 3001 or 5173. Model IDs and credentials cannot be supplied through the mutation API. Selection is committed before acknowledgement; after a timeout, reload canonical state before retrying.

**Setup and recovery:** Add `ANTHROPIC_API_KEY` and `DEEPSEEK_API_KEY` in `.env` and restart to load key changes. `LLM_FAST_MODEL` / `LLM_DECISION_MODEL` preserve Claude overrides; `ECONOMY_FAST_MODEL` / `ECONOMY_DECISION_MODEL` configure DeepSeek. Flash uses non-thinking mode; Pro uses high-effort thinking with `ECONOMY_DECISION_MAX_TOKENS` bounding total completion tokens. `LLM_TIMEOUT_SECONDS` defaults to 60, with automatic retries disabled. Provider errors never switch to the other provider; existing deterministic fallback behavior is retained and marked degraded. A selected profile is not proof that every agent successfully received an LLM answer. A lost or corrupt selection database must be repaired rather than silently selecting Economy.

**Accounting:** `llm_usage` stores one record per attempt before a request. Successful responses supply token counts and request IDs; unsuccessful/unreported attempts remain incomplete, not zero-cost. No prompts, keys or private reasoning content are stored in this ledger. Reasoning tokens are a subset of completion tokens, not an additional charge. Estimates cover current-month application usage only and use versioned rates checked on 2026-09-28; unknown custom-model rates remain unknown. DeepSeek peak-window rates are conservative upper estimates during Chinese holidays. INR is a display conversion using `LLM_USD_INR_RATE`. There is no automatic pause at INR500 or any other threshold. Reconcile with provider billing for actual fees/tax/discounts.

**Validation:** Offline tests cover both providers' request shapes, switching in both directions, real LangGraph context propagation, persistence/revisions, untrusted mutations, malformed/truncated answers, cache/reasoning accounting, cancellation and IST month boundaries. Browser verification covers desktop/mobile layout, mouse/keyboard switching, refresh and rejected updates using mocked mode responses. No paid model calls or broker execution were used for these checks. Model quality, latency under real API load and trading results still require the existing paper-first evaluation gate.

### ADR-0005 — No failover after ambiguous order submission (accepted)

**Decision:** A secondary broker may be used only after the primary reports a definitive rejection. A timeout or transport failure is treated as an uncertain submission and requires reconciliation.

**Why:** The first broker may have accepted the order even when the client did not receive its response. Retrying on another broker could create duplicate exposure.

**Consequences:** Operators must reconcile ambiguous orders before submitting replacements. This behavior applies to the dual-broker router; broker-specific operational procedures remain necessary.

### ADR-0006 - Durable protective exits and isolated ATR evaluation (accepted)

**Decision:** Persist protective exit intent before sending, prohibit resubmission while acceptance is unknown, and use a fresh-reference, tick-aligned LIMIT collar for scheduled paper stop/target/time exits. Normalize Angel SDK string order IDs and distinguish explicit rejection from transport uncertainty. Share the monotonic Chandelier helper with opt-in ATR research policies; retain the scheduled 1x/2.5x entry baseline.

**Validation:** Offline regressions cover unknown acceptance and restart, terminal partial fills, stale-reference rejection, price-collar bounds, previous-lot ratchet exclusion, next-bar-only trailing, and one daily cap across intraday bars. The rules-only ATR comparison cannot call model providers or the broker order router. Full live fill lifecycle, tick-driven monitoring, mixed-lot/horizon protection, and prospective promotion evidence remain incomplete.

**Recovery:** If protection contains `UNRESOLVED:<id>`, keep entries halted, inspect the broker order/trade books and paper ledger, and reconcile the exact remaining inventory. Only replace it with a verified broker order ID or clear it after confirming no accepted order remains. Resetting the entry circuit breaker alone does not authorize resubmission. Never remove an unknown intent merely to make the bot trade.

### ADR-0007 - Conservative aggregate stop-risk budget (accepted)

**Decision:** Add a maximum modelled aggregate open-stop risk of 2% of current equity, reducible by configuration. Keep the existing per-entry and daily-loss limits unchanged. Review and routing reject ambiguous protection; exits remain exempt.

**Why:** Several individually small entries can together exceed a prudent planned portfolio loss. The historical single-symbol position summary cannot safely represent different lots' protective stops.

**Consequences:** Held positions require fresh marks and a matching single-lot active stop; missing evidence prevents new entries. Same-scan paper fills consume an in-process projected budget. This control does not validate live pending-order coordination, simultaneous processes, execution through gaps, or research parity, and it does not guarantee a maximum daily loss.

### ADR-0008 - Transactional paper-order serialization (accepted)

**Decision:** Serialize the paper fill decision, entry-cap and cooldown checks, exit-lot check, ledger fill and audit with one SQLite write transaction.

**Why:** An in-process lock cannot prevent two paper-order processes from simultaneously passing a daily cap or closing the same lot.

**Consequences:** Competing paper-order writers using this route wait for the transaction; lock failures surface as errors. Paper orders on non-SQLite databases fail explicitly. This is not a substitute for cross-process portfolio-risk reservations, live broker reconciliation, or protection against direct ledger modifications.

### ADR-0009 - Captured HMM directives in isolated research (accepted)

**Decision:** Research replay applies the captured strategy eligibility, hedge directive and position cap to new entries. Risk and HMM blocks do not prevent simulated exits.

**Why:** Research fills that ignore the production regime's position and strategy restrictions cannot be compared meaningfully with the scheduled policy.

**Consequences:** Older captures without valid directives abstain rather than inventing them. Results from before and after this change are not directly comparable. This adds only HMM entry parity; aggregate stop risk, AI evidence, pending-order behavior and execution latency still differ from the production pipeline.

### ADR-0010 - Fail closed on unsupported research sessions (accepted)

**Decision:** Use the verified 2026 regular-session grid in replay, including Budget Sunday, and reject unexpected in-session bars, other calendar years and windows crossing Muhurat.

**Why:** Treating an exceptional open session as closed can conceal missed bars, stops or square-offs.

**Consequences:** Muhurat comparisons remain unavailable until their distinct execution and square-off rules are verified and implemented. Normal scheduled execution is unchanged.

### ADR-0011 - No fundamental-model fallback in the AI path (accepted)

**Decision:** Require usable numeric fundamental inputs and a schema-valid model response for an emitted AI fundamental signal; reserve rule-based synthesis for isolated rules research.

**Why:** A failed or invalid model response must not be presented as a successful AI analyst's recommendation.

**Consequences:** Fundamental failures are explicit and produce no signal. This does not establish source freshness or the complete strategy-specific required-agent contract.

### ADR-0012 - Forecast calendar and settlement evidence (accepted)

**Decision:** New research snapshots require usable captured fundamental metrics. Forecast settlement uses the verified 2026 session exceptions, rejects unverified calendar years and requires unique, valid index and symbol bars with explicit valid split and dividend values.

**Why:** Missing or malformed inputs, exceptional sessions and ambiguous daily bars can otherwise produce misleadingly scored forecasts.

**Consequences:** Affected forecasts stay pending with a reason. Replay and pending settlement now revalidate older captures, but already settled legacy outcomes remain historical and are excluded from current accuracy. This does not prove Yahoo's corporate-action feed is complete. Settlement remains hypothetical research, not trade P&L or calibrated performance evidence.

### ADR-0013 - Recheck debate conviction at entry boundaries (accepted)

**Decision:** The portfolio manager and entry router independently require a complete, aligned debate with a finite conviction score of at least 70 and nonempty reasoning. The debate binds the complete technical signal, and the portfolio decision binds both the technical signal and complete debate; the router rechecks their digests. Protective exits remain exempt.

**Why:** The synthesizer normally changes a below-threshold verdict to HOLD, but later entry boundaries must not trust an inconsistent or malformed verdict.

**Consequences:** Invalid or changed evidence abstains with an explicit reason. This binds the technical output actually reviewed, not its raw market-data provenance or every optional analyst input. The score is an uncalibrated model opinion, not a probability of success or proof of trading edge.

### ADR-0014 - Point-in-time forecast reference (accepted)

**Decision:** New directional forecasts use only current, timestamped quotes or recent completed bars from a verified open session. Untimestamped option spots, stale data and forming bars cannot become settlement references.

**Why:** Scoring a forecast from a stale or not-yet-completed price can falsely improve or degrade apparent model performance.

**Consequences:** Some options and macro forecasts have no scoreable reference until an appropriately timestamped index quote or bar is captured. Pending historical forecasts are rechecked and unsafe ones become ineligible; already settled legacy outcomes are excluded from current accuracy rather than silently rewritten. This does not certify the provider's source and corporate-action completeness.

### ADR-0015 - CompositeTrend review contract and candidate identity (accepted)

**Decision:** Version the scheduled CompositeTrend paper contract with required technical, sentiment, bull, bear, synthesis and portfolio reviews plus verified recent 15-minute history, attributable current news, quantitative-screen eligibility, risk approval and a valid portfolio snapshot. Fundamental, options, macro, India-specific and geopolitical analysis are optional for this contract. The router records each required agent/input result for the scan outcome and blocks entries when a declared requirement is missing or invalid. Unknown strategies have no scheduled review contract and abstain. A scan-, symbol-, contract- and technical-signal-bound candidate ID is checked through debate, portfolio review and entry routing.

**Why:** An entry should not be attributed to another candidate or silently use an undeclared strategy review path.

**Consequences:** Optional analyst output is not proof of point-in-time freshness, and the contract does not yet cover other strategies, full asset/horizon identity, or every optional input's provenance. The contract applies only to new entries; exits do not require it.

### ADR-0016 - Persist portfolio review evidence before routing (accepted)

**Decision:** The debate uses a single bounded analyst context, checks it for changes between model calls, and carries both arguments and a context digest into the portfolio review. Before returning an AI-approved entry, atomically store its candidate ID, accepted decision, exact bounded debate context and exact bounded portfolio-model context in the local `candidate_reviews` table with SHA-256 integrity fields. The debate context is retained in the recorded decision, not duplicated in the portfolio-model prompt. Refuse entries if context changes during model calls, the write fails, or an existing candidate ID conflicts. The router rechecks the committed record, decision and current bounded context before routing; missing, changed or unreadable records block entries.

**Why:** An in-memory review alone cannot reliably explain which candidate and evidence the portfolio model saw after a restart.

**Consequences:** Records can include local portfolio state and analyst outputs; protect and retain the database accordingly. Raw news articles and complete prompt templates are not stored in these records. Larger reviews may exceed context limits and block entries rather than truncate evidence. A recorded review does not prove that an order was submitted or filled, or that its optional data was current. The checksums detect accidental mismatches, not deliberate tampering by someone able to alter the database. No broker order is submitted by the persistence operation; exits do not depend on these review records.

### ADR-0017 - Reject paper entries against a changed ledger book (accepted)

**Decision:** A paper entry must carry the position quantities, holding periods and entry-lot IDs used by the router's risk check. Inside the same SQLite write transaction as the fill, compare that identity with the replayed paper ledger. Reject and audit a missing, invalid or changed book. Add each confirmed paper fill's trade ID to the expected book before evaluating another entry in the same scan. INR paper decisions use this shared paper ledger regardless of the configured live-broker preference.

**Why:** Two cooperating processes must not both fill entries using the same pre-entry portfolio snapshot.

**Consequences:** This is an optimistic ledger-identity check, not an atomic valuation of market marks, stop ratchets or equity, and not a reservation for live or pending broker orders. Callers must still perform the full risk check; a matching identity does not prove risk approval. The check does not apply to exits, which retain their separate lot-identity validation. A rejected stale-book attempt is recorded as `REJECTED_STALE_BOOK` in the paper fill audit.

### ADR-0018 - Coordinate local historical-candle calls (accepted)

**Decision:** Cooperating processes using the same state file and one-second policy serialize historical-candle calls with an OS file lock. They durably mark a call in flight before invoking the provider and record completion afterward. The next call waits one second after completion, or one full second after obtaining the lock if the preceding process left an in-flight marker. Invalid state and storage failures stop the call explicitly.

**Why:** An in-process limiter could not prevent overlapping candle requests from separate local processes, and writing a start timestamp before a slow disk flush alone would not ensure spacing between actual client calls.

**Consequences:** This is more conservative than one second between starts and may slow scans, especially when requests are slow. File locks coordinate only processes on the same host sharing the same file and policy; unrelated clients and remote hosts remain outside the limit. Do not remove or replace the state file while processes are running. Provider arrival timing is not guaranteed. The full isolated regression suite and local multiprocessing/crash tests cover the coordination; real market-session operation remains to be observed.

### ADR-0019 - Carry observed historical-data source details (accepted)

**Decision:** Record a bounded source description for each symbol-history fetch, including explicit fallback causes, unavailable results and unknown adjustment or corporate-action status. Add a checksum of the captured frame when data is available. Carry the description separately from the DataFrame so it survives graph checkpoints and new research snapshots. The scheduled `CompositeTrend` review requires matching NSE 15-minute source metadata and a frame checksum, retrieved within 30 minutes and not in the future; candidate identity rechecks them at each review and routing stage. Missing or invalid evidence blocks entries, while exits remain independent.

**Why:** Operators and future research need to distinguish broker history from a fallback without claiming that unknown source properties were verified.

**Consequences:** The checksum detects changes to the supported serialized frame, not deliberate tampering with both frame and metadata, provider authenticity, point-in-time accuracy, adjustment correctness or row completeness. Old research snapshots remain readable and isolated replay does not claim this live-entry gate. Recognized review and router failures appear in the scan summary as historical-data evidence issues without exposing raw errors. More stringent source/adjustment, provider-bar freshness and instrument-identity validation remains outstanding.

### ADR-0020 - Tick-triggered protective exits and fail-closed live entries (accepted)

**Decision:** Subscribe held positions first on the Angel feed, ahead of the watchlist, within the 50-token limit. Report any held position left without a feed subscription on the dashboard and through Telegram. A fresh feed tick that crosses a registered stop or target schedules an exit on the event loop. Under the shared exit lock, the exit is rechecked against a fresh market reference before the existing collared exit path submits it. Duplicate ticks coalesce, and each symbol waits five seconds after an attempt. Trailing ratchets and time exits remain on the five-minute check. That check needs a mark at most 60 seconds old before ratcheting or firing a stop or target, still evaluates time exits for stale marks, and makes an intraday position carried from an earlier date due from the next 09:15 IST open. Paper exit intents are resolved from the identity-bound paper ledger: a flat position closes the intent, and remaining original entry lots with the same side and holding period resume protection. Missing identity, new/replacement lots, a changed holding period, a side mismatch or an unreadable ledger keep the intent pending and halt entries for operator review. Recovery must not bind the old stop to a newly opened lot in the same symbol. Strategy exits wait while a protective exit is pending. Repeated protective-check failures (three by default) trigger the Dead Man's Switch through the `protective_exits` component. New live entries are refused while `LIVE_ORDER_LIFECYCLE_VALIDATED` is `False`; live exits remain available.

**Why:** A five-minute polling interval could leave a crossed stop unattended for minutes. A stale mark could fire a stop spuriously or move a trailing ratchet on bad data. An uncertain paper exit could block protection indefinitely. Live fill confirmation, partial-fill handling and exit reconciliation are not yet validated end to end.

**Consequences:** A tick trigger cannot guarantee a fill or cap gap losses, and the feed is usable only when fresh, so a stale or disconnected socket falls back to the five-minute check. Held positions without an instrument token or beyond the token limit, and brokerless paper mode (no feed), have only the scheduled check; neither path can fill without a fresh price. In paper mode, a persistent submission failure retries once per check only after the original entry identity is verified, and keeps entries halted. Legacy intents without entry IDs require operator reconciliation rather than guessing an identity. Live `UNRESOLVED` intents still require manual reconciliation (see ADR-0006). Tick-level trailing (T07) and the validated live order lifecycle remain outstanding; enabling live entries requires that validation and a reviewed code change. Behaviour during a live market session remains to be observed.

**Validation (14 November 2026):** Re-ran the credential-free isolated suite: **666 passed**, including **144 focused exit/recovery regressions**. The replacement-lot regression failed before the identity guard and passed afterward. [Paper-ledger tests](../tests/test_paper_integrity.py) recreate tracker and stop-manager objects from SQLite at interruption boundaries before a fill, after a partial close, and after a full close; they verify exactly the original quantity closes, the ratchet survives, reconciliation passes and the halt remains active. [Protective-exit tests](../tests/test_protective_exit_lifecycle.py) also cover feed-thread dispatch, coalescing/backoff, fresh-price rechecks, overdue time exits, identity/horizon mismatch, live-entry refusal, subscription priority and failure thresholds. [Operator API tests](../tests/test_operational_recovery.py) caught a separate manual-feed path that omitted held positions; scheduled and dashboard connections now share held-first selection, watchlist-unavailable fallback, gap alerts and explicit connection errors. These are offline simulations, not real broker partial-fill or process-crash tests. The dashboard production build passes (with the existing large-chunk warning); relevant Pylance diagnostics have no errors. The Python suite reports two broker-SDK TLS deprecation warnings. Run the tests only in a source copy without `.env`, production databases or credentials; do not run fault injection against the running account.

### ADR-0021 — Operator-reviewed cash-flow statement imports (accepted)

**Decision:** Record external cash transfers only from an operator-reviewed, normalized Angel One statement import; do not infer transfers from RMS/margin balances. The local dashboard import is paper-only, requires entries paused and no scan in progress, and accepts only completed deposits/withdrawals with unique transaction IDs and timezone-qualified timestamps at or after the persisted paper cash-flow baseline. The import is content-digested and atomic; replaying a statement is idempotent.
**Why:** The existing broker adapter exposes funds/margins, not a complete cash-transfer ledger. A changing available-margin value cannot distinguish deposits or withdrawals from fills, settlement, or margin movements.
**Consequences:** Operators must download an official broker statement and normalize it to `transaction_id,timestamp,direction,amount,status`; transfers before the persisted paper baseline are rejected rather than added to initial capital. New rows remain entry-blocking until a fresh portfolio sync acknowledges the ledger generation. The importer does not clear daily-loss or drawdown halts, does not ingest orders/fees, and is not a live-account reconciliation path. T05 stays in progress until tested against a real official statement and the remaining cross-process/live-account controls are satisfied.

### ADR-0022 — Phase 1 safety layer: one position identity, halt latching, LLM budget, scan deadline (accepted)

**Decision:**
1. A symbol may hold one open position identity at a time. The router blocks an entry when the projected book already has the symbol (`existing_position`), and `_paper_trade` re-checks the ledger inside `BEGIN IMMEDIATE` (fill audit `EXISTING_POSITION`). Legacy mixed-horizon lots are summarised conservatively (tightest stop, nearest target, earliest time exit, `mixed_lots=True`) and may only be flattened whole.
2. `baseline_unavailable` blocks entries while it lasts but is never persisted as a latch; a real 2% loss still latches and persists. Stored `baseline_unavailable` records are ignored.
3. A paid scan is skipped when the daily-loss halt is active or when today's LLM spend reaches `LLM_DAILY_BUDGET_INR` (or cannot be measured). Exits are separate jobs and are unaffected. Every skip is recorded with a reason in the `scan_status` control record, which survives a restart.
4. `SCAN_ENTRY_DEADLINE_SECONDS` (900) blocks entries, never exits, in a scan that ran too long (`scan_deadline`).
5. Paper entries are IOC-style: a LIMIT entry that is not marketable is audited and cancelled, not left resting. Resting entry orders are out of scope while live entries are blocked (ADR-0020). `PAPER_PARTIAL_FILL_ENABLED` (off) models partial entry fills from displayed top-of-book depth; exits are never partially filled by the model.
6. An unfilled protective exit re-prices with the collar widened by one step of `PROTECTIVE_EXIT_COLLAR_PCT` per attempt, at most three steps (0.9%), then raises one deduplicated CRITICAL alert and keeps monitoring. It never uses MARKET and never a stale price.
7. `operations/order_rate_limiter.py` limits submissions to 5/s and 100/min. An entry over the limit is refused; an exit waits at most one second and is then sent with a warning.
8. Latching a daily-loss or drawdown halt cancels resting entry orders (never exits).
9. NSE holidays come from `data/calendars/nse_equity_holidays.json`; `NSE_HOLIDAYS` is an additive override. Unknown years block entries only.
10. F&O expiry is read from the contract master, not a fixed weekday (`None` when the master is unavailable). The geopolitical analysis runs after macro and reads `macro_signals["data"]`; the delta-hedger log reads `delta`.
11. Alerts are stored in `AlertOutbox` before delivery, retried every 60 s with exponential backoff (10 attempts), and de-duplicated for 30 minutes; a `<key>:recovered` alert re-arms `<key>`.

**Why:** Observed on 14 Nov 2026: a transient `baseline_unavailable` latched for the whole day; the hard-coded Thursday expiry was wrong (NIFTY weeklies expire on other weekdays after exchange changes and holidays); the fallback geopolitical rules silently read nothing; a 10.7-minute scan could place entries on stale prices; and a Telegram outage lost alerts.

**Consequences:** A day with a one-off sync failure no longer needs an operator reset, but entries wait until the baseline is valid. A same-symbol add-on is impossible by design. Live partial fills and resting entries remain unvalidated. The calendar file must be extended every year.

**Rollback:** Each rule is independent code; revert the router or order-manager check to restore the previous behaviour. Settings default to their previous effective behaviour except the new budget, deadline and rate limits.

### ADR-0023 — Typed candidate contract and horizon/product separation (accepted)

**Decision:** Entries are described by a frozen `TradeCandidate` (`strategies/candidate.py`) with an `AssetId` (venue, segment, currency, symbol), a `Horizon` (`INTRADAY | SWING | BTST | POSITIONAL`) and the broker `Product` derived from it (`INTRADAY` → `INTRADAY`; the others → `DELIVERY`). `validate()` fails closed on wrong-side stop or target, prices that are not multiples of the tick (exact `Decimal` arithmetic), a quantity that is not a lot multiple, an expired candidate, a missing ATR value or timeframe, an unknown exit policy, and a product that does not match the horizon. `strategies/contracts.py` is now a registry (`STRATEGY_CONTRACTS`) of horizons, venues, review contract, enabling setting and default exit policy; `CompositeTrend` keeps `review-v2`, INTRADAY and NSE cash, always on. An entry needs an enabled registered contract and a horizon that contract lists; the legacy `DELIVERY` value is no longer a valid horizon for new entries and is read as `SWING` or as a product only when reading old rows. The router rounds an entry to the nearest tick and the stop/target one tick at most, toward the entry, so risk is never understated. `Trade` stores `horizon`, `strategy_version`, `exit_policy`, `atr_timeframe` and `book`; `ClosedTrade` gains `venue`, `currency`, `exit_policy` and `book`. All additive columns are applied to an existing SQLite database by `database.db.SQLITE_ADDED_COLUMNS`.

**Why:** A single `INTRADAY`/`DELIVERY` field conflated how long a strategy holds with how the broker carries the order. Without separate fields, a swing position could be flattened at 15:15 or an overnight horizon could not be represented at all.

**Consequences:** 15:15 square-off and the time exit act on `INTRADAY` only. Mixed legacy lots can still be exited (whole position, product-compatible). A candidate without a positive ATR is refused. Exits never consult the registry, so an unknown or retired strategy cannot trap an open position.

**Rollback:** Revert the router contract lookup and `place_order(horizon=...)`; the new columns are nullable and can stay.

### ADR-0024 — Shared exit policies, horizon time exits and paper settlement (accepted)

**Decision:**
1. `risk/exit_policy.py` owns every level calculation. New research-only policies: `intraday_atr_2x_4x_v1` (ATR(14) on completed 15-minute bars, `INTRADAY_15M_14`), `daily_atr22_2x_4x_v1` and `chandelier_atr22_k3_v1` (k=3 trail on ATR(22) from entry). The live default stays `daily_atr_baseline_v1` (1x/2.5x) until T10 shows out-of-sample improvement. `atr_before()` uses only bars completed before the entry (no look-ahead). `structure_stop()` returns a tick-aligned stop beyond a structural level, or `None` (skip the candidate) when its distance is outside 0.5–3 ATR.
2. The backtester and the live `StopLossManager` use the same `stop_target_hit`, `gap_fill_price` (a bar opening through a level fills at the open) and `ratcheted_stop`. Each position stores its `exit_policy` and `atr_timeframe`; `register_trade` raises when either is missing for a new trade, restored positions may lack them (labelled `UNSPECIFIED_LEGACY`), and a restored stop is never loosened.
3. `time_exit_due()`: INTRADAY at 15:15 (or the next open when carried); BTST 30 minutes after the next session opens; SWING and POSITIONAL after `SWING_MAX_HOLDING_DAYS` (10) / `POSITIONAL_MAX_HOLDING_DAYS` (60) calendar days; a failed thesis (< 0.5R after 4 completed bars) when measured. Legacy `DELIVERY` rows keep their old rules (no forced exit). An unknown holiday calendar never defers an exit.
4. Paper cash is settled by horizon. INTRADAY nets the same day; proceeds of a delivery sale are unsettled until the next NSE session (`CashLedgerEntry`, backfilled for older trades). Sizing uses `PortfolioTracker.available_capital` = settled cash less open-order reservations; `snapshot()` reports `settled_cash` and `unsettled_cash`.
5. The previous close's marks of carried positions are saved with the daily snapshot (`closing_marks` control record) and reused after a restart (`mark_source="previous_close"`, not a fresh mark), so a carried gain is not counted as today's gain. If a carried position has no recorded closing mark the baseline is unavailable (entries wait, non-latching).
6. A BTST exit before the modelled delivery (`BTST_DELIVERY_MINUTES_AFTER_OPEN`, 15) is refused with the audit outcome `SHORT_DELIVERY_RISK`; the position stays protected and the next protective pass retries. Other horizons are not blocked by this rule.
7. `corporate_actions` is a blocking check only: an unprocessed action with an ex-date on or before today blocks new entries in that symbol (an unreadable table also blocks entries); exits proceed with a warning. Rows are imported by an operator with a source; there is **no automatic official ingestion**.
8. `ClosedTrade` records `venue`, `currency`, `exit_policy` and `book`.

**Why:** Fixed daily exits ignored holding period and timeframe; the simulator and the live manager each had their own level math; and a swing sale's proceeds could fund an entry the same day.

**Consequences:** Nothing changes for CompositeTrend entries except an explicit policy name and timeframe on each position. T07 policies other than the baseline are not used live. Paper settlement is a model, not a broker statement. The corporate-action data gap is a user action item.

**Rollback:** Revert the manager and ledger changes; the added tables and columns are additive.

### ADR-0025 — Disabled strategy sleeves behind one adapter interface (accepted)

**Decision:** Seven strategy sleeves are implemented but ship **disabled**, each behind its own flag (default `False`): `STRATEGY_ORB_INTRADAY_ENABLED`, `STRATEGY_SUPERTREND_ADX_ENABLED`, `STRATEGY_MEAN_REVERSION_ENABLED`, `STRATEGY_MOMENTUM_BREAKOUT_ENABLED`, `STRATEGY_VCP_NR7_SWING_ENABLED`, `STRATEGY_BTST_STRENGTH_ENABLED`, `STRATEGY_ETF_ROTATION_ENABLED`. `strategies/adapters.py` gives them one interface, `generate(symbol, bars_by_tf, as_of) -> TradeCandidate | None`, that normalises inputs to completed bars only and replaces each strategy's fixed stop/target with its registered exit policy (ORB uses a structure stop and is skipped when the structure is out of range). `strategies/btst_strength.py` (late-session strength, entry 15:00–15:20, next-session exit) and `strategies/etf_rotation.py` (strongest trending ETF or cash) are new. Each sleeve is registered in `STRATEGY_CONTRACTS` with an AI-required review contract identical to CompositeTrend's (same agents and inputs, its own strategy name). The graph node `strategy_candidates` follows `quant_screener`, calls `select_strategies` on the median ADX/Choppiness of the qualified symbols, India VIX, the HMM regime and the daily-loss ratio, and runs only enabled sleeves. Its candidates are overlaid by the technical agent onto symbols the core strategy left on HOLD, keeping the symbol's verified 15-minute history provenance, so the same AI review, risk approval, router gates and exits apply. Each sleeve trades in its own paper book (`Trade.book`); sizing and exposure caps stay portfolio-wide. The HMM strategy gate now maps a strategy to its `hmm_tag` through the registry and rejects disabled sleeves. `nifty_options_seller` (derivatives out of scope) and `pairs_trading` (needs short inventory) are not wrapped; `MeanReversion` is long-only for the same reason.

**Why:** The strategy interfaces were inconsistent (`scan(market_data)`, `scan(symbol, bars)`, free functions) and one used an unvalidated horizon. A single contract lets every sleeve face the same gates and be evaluated the same way (T10).

**Consequences:** With every flag off the node returns an empty update and the technical overlay never runs, so CompositeTrend behaviour is unchanged (checked by test). Enabling a sleeve is an operator decision that needs T10 evidence (see `docs/READINESS_MARKETS.md`). ETF rotation also needs the ETF symbols in the watchlist for data. The alpha miner, RL and the legacy signal gate remain off.

**Rollback:** Leave the flags off, or revert the node edge in `build_trading_graph`.

### ADR-0026 — Evaluation harness, calibrated signal gate and half-Kelly (accepted)

**Decision:**
1. `research/evaluation.py` (`run_walk_forward_evaluation`, CLI `main.py --evaluate STRATEGY --policy NAME --from YYYY-MM-DD --to YYYY-MM-DD`) evaluates a strategy with a named exit policy over consecutive out-of-sample folds, dropping purge bars at the start and embargo bars at the end of each window. It reuses `BacktestEngine` (shared exit policies, gap fills, the Indian cost model, the live sizer and, for INTRADAY strategies, the live entry window and 15:15 flatten) and reads only cached or free data (`research/data_cache`, then yfinance). Every configuration run is **appended** to `evaluation_runs`; rows are never deleted. Reports give net expectancy (₹ and R), profit factor, max drawdown, exposure, turnover, MAE/MFE, the stop/target/time-exit mix, the fill rate and a deterministic 95% bootstrap interval of expectancy. The outcome is `inconclusive` when there are fewer than 100 trades or the interval includes zero. The CLI runs ₹10,000 and ₹100,000 as separate rows and writes JSON under `research/reports/`. Nothing in it promotes, enables or reconfigures a strategy.
2. `models/signal_gate/gate.py` is rewritten: label = net-of-cost R outcome from `ClosedTrade` under the policy that ran; chronological 60/20/20 train/calibration/test splits with a purge gap; logistic regression with Platt scaling (isotonic only with at least 500 calibration rows); Brier score, log loss, reliability bins and ECE; a JSON artefact with a SHA-256 checksum (the pickle is gone); threshold `p* = (L + C) / (W + L) + 0.02`. `should_execute` returns `(False, reason)` when the model is missing, has a bad checksum or mismatched features, raises, or was trained on fewer than 300 samples. `SIGNAL_GATE_ENABLED` (default `false`) is the only switch; AI conviction can be checked with the same maths on settled `ResearchForecast` rows (`conviction_report`).
3. `PositionSizer.size` accepts an optional calibrated win probability and payoff ratio. The quantity is `min(existing, half-Kelly)`; `f <= 0` blocks with `kelly_non_positive`. The router passes them only when the gate is enabled **and** the stored artefact reports `calibration_passed`.

**Evidence (14 Nov 2026):** the free source gives about 60 sessions of 15-minute bars. CompositeTrend with the baseline policy over 2026-08-24 to 2026-11-13 on 20 Nifty 50 symbols produced 227 replay trades at ₹100,000 (net expectancy about −₹40 per trade, 95% bootstrap interval about −₹56 to −₹23, classified `negative`) and none at ₹10,000; the 2x/4x policy produced none at either size because a gross 2:1 target cannot clear the unchanged net 2:1 reward/risk gate after costs. These are one regime of overlapping trades without the AI review layer, so the bootstrap interval understates uncertainty: **no conclusion and no change follows**. The gate report says "insufficient samples" because there are far fewer than 300 settled paper outcomes.

**Consequences:** T22/T23 stay off until several hundred settled outcomes exist (plan section 9). Any strategy or policy promotion needs out-of-sample evidence from this harness on more data than the free source supplies.

**Rollback:** Leave `SIGNAL_GATE_ENABLED` false (the default); the harness only reads data and appends rows.

### ADR-0027 — US paper adapter, crypto simulator and derivatives decision (accepted)

**Decision:**
1. **US equities (T11)** run only through `broker/alpaca_paper_adapter.py`, behind `FOREIGN_US_PAPER_ENABLED` (default `false`). The adapter refuses non-paper endpoints and missing keys. Its optional schedule uses the configured `GLOBAL_WATCHLIST`, a 15-minute 20/50 moving-average trend strategy, fresh IEX quotes (single venue, not the consolidated tape), LIMIT-only orders, a 2% stop, 5% target and pre-close flatten. It requires 50 completed 15-minute bars, known US calendar data and regular session hours. New entries use shared pause/halt gates and the portfolio caps; performance is unvalidated. With the flag off the router keeps returning `PAPER_UNSUPPORTED`.
2. **Crypto (T12)** is a local spot simulator (`broker/crypto_sim.py`, `CRYPTO_SIM_ENABLED`, default `false`): no exchange account, no keys, no leverage, no futures, no shorting. A separate 15-minute schedule uses completed 1-hour candles and fresh public Binance prices for BTCUSDT, ETHUSDT and SOLUSDT; missing market data blocks entries (fixture prices are not used by the scheduler). The long-only 20/50 trend uses 2% stops, 5% targets and trend exits. The simulator applies symbol tick/step/min-notional filters, a 0.1% taker fee, 5 bps slippage, a 24x7 UTC clock reported in IST, persisted simulated state, and its own daily-loss halt (`CRYPTO_HALT_DAILY_LOSS_PCT`) constrained by the portfolio daily-loss cap. Entries also obey the shared portfolio-wide 10% per-position and 50% gross caps. INR values use a static, labelled rate (`CRYPTO_USDT_INR_RATE`). `vda_tax_estimate` shows 30% on each positive gain with no loss set-off and 1% TDS as withholding; it is an estimate, not tax advice.
3. **Derivatives (T13)** stay out of scope: `docs/DERIVATIVES_FEASIBILITY.md` shows that no listed index spread fits the unchanged 0.5% planned-risk rule at ₹10,000 or ₹100,000, and that the system has no OI provider, derivatives cost model or assignment handling.

**Evidence and limits:** the NYSE 2026 calendar was read from nyse.com on 14 Nov 2026; 2027 is not published yet, so US entries are refused from 1 Jan 2027 until it is added. The scheduler and its risk gates are tested in isolation; several weeks of paper results and out-of-sample evidence are still absent. These strategies are experimental and make no profitability claim.

**Rollback:** leave both flags false (default). The adapter and simulator are not imported by the core India path.

### New ADR Template

Copy this section for every material change:

```text
### ADR-NNNN — Title (proposed | accepted | superseded)
Decision:
Context and evidence:
Alternatives considered:
Risk and rollback:
Validation / success metrics:
Owner and review date:
```

## 7. Known limitations and roadmap

- The dashboard is an operator console: it provides health, portfolio, P&L, positions, signals, execution history, and pause/resume controls. It intentionally cannot enable live mode; that remains an explicit configuration-and-review procedure.
- Agent cards show pipeline readiness (or paused state), not fabricated per-agent execution telemetry. Persisted per-node timing and failure telemetry is the next observability increment.
- Paper fills, closed lots, positions and stops are reconciled automatically; a supervised multi-day paper evaluation is still required, and live broker order/fill reconciliation is not yet automated. New live entries are therefore blocked in the order router (ADR-0020).
- Trailing-stop ratchets and pending exit intent persist in the operational control database. New paper registrations include entry trade IDs to prevent applying a previous lot's ratchet to a new lot; legacy records without identity and mixed-lot/horizon protection still need reconciliation.
- The intelligent-router (non-`angel_only`) paper path returns `PAPER_UNSUPPORTED`; paper fills use the Angel path only. The one exception is US orders through the Alpaca **paper** adapter when `FOREIGN_US_PAPER_ENABLED=true` (ADR-0027).
- NSE equity holidays are read from `data/calendars/nse_equity_holidays.json` (2026 only: NSE circular NSE/CMTR/71775 plus the 15 Jan 2026 election closure, cross-checked against Angel daily candles). A year absent from the file blocks new entries and never blocks exits. The official 2027 NSE circular could not be retrieved (the NSE holiday API returns 2026 only), so **2027 must be added from the official circular before 1 January 2027**.
- LLM output is schema-parsed but should be constrained with a formal structured-output model and independent order validator.
- The L2, HMM, MARL, Kafka, global-broker, and episodic-memory components are optional/experimental integrations; treat them as non-essential until monitored and tested in the deployed environment.
- The strategy modules require walk-forward validation, transaction-cost stress tests, and portfolio-level conflict rules before production wiring.
- The broker router, strategy selector, FastExecutionLoop, ledger, and newly added alpha modules are not all wired into the scheduled MVD. Passing component tests is not equivalent to order lifecycle, data-quality, or live-readiness validation.
- T05 remains in progress because broker RMS/margin APIs do not provide a complete cash-transfer ledger. In paper mode, an operator may import a normalized Angel One statement from Operations while entries are paused; accepted CSV rows must use `transaction_id,timestamp,direction,amount,status`, include only SUCCESS/COMPLETED deposits or withdrawals, use timezone-qualified timestamps, and postdate the persisted paper cash-flow baseline. The importer stores a SHA-256 statement digest and idempotent transaction references, then leaves entries blocked until fresh portfolio marks reconcile the ledger generation. It does not import trade activity, verify settlement or broker cash balances, clear any halt, or enable live imports. Transfers absent from the uploaded official statement remain unknown. Paper entry fills are serialized against expected ledger lot identity; this does not reserve live buying power or establish a real-broker portfolio snapshot.

### No promised returns

Nothing in this repository, its dashboard or its reports promises, forecasts or implies profit. Paper results come from simulated fills against live reference prices with a slippage model; they omit real-world queue position, partial liquidity and behaviour under stress. The only quantitative strategy evaluation to date (T10, ADR-0026) was **unfavourable and inconclusive** for the baseline and produced no trades for the 2x/4x policy. Published evidence also argues for caution: SEBI found that about 91–93% of individual equity F&O traders lost money (`docs/DERIVATIVES_FEASIBILITY.md`). A small account (₹10,000) also faces fixed costs and lot-size limits that a larger one does not.

### Unsupported features (what is intentionally not available)

- Live trading: guarded by an explicit acknowledgement, and live **entries** stay blocked until a live order lifecycle is validated (ADR-0020).
- Derivatives, margin, leverage and short-selling outside the intraday paper rules.
- Automatic corporate-action ingestion (a manual table blocks entries when an action is unprocessed).
- Any automatic promotion of a strategy, model or exit policy.
- US trading other than Alpaca paper; crypto other than the local simulator.
- Tax advice. The crypto tax view is an estimate only.

### Capability inventory (generated from `operations/capabilities.py`)

The dashboard (Operations → Capabilities) shows the same list. Status meanings: `active` output can change an entry decision or an exit; `advisory` produces information that reviews or logs may read and cannot place or block an order alone; `experimental` means an unvalidated strategy can place paper/simulated orders only and must never be treated as evidence for live trading; `disabled` is not wired in or is switched off by default.

| Capability | Status | Module |
| --- | --- | --- |
| `market_data` | active | `agents/market_data_agent.py` |
| `kafka_events` | experimental | `data/event_driven_kafka_stream.py` |
| `l2_order_flow` | experimental | `data/l2_order_flow_analyzer.py` |
| `hmm_regime` | advisory | `agents/graph.py, risk/hmm_regime_classifier.py` |
| `quant_screener` | active | `agents/quantitative_screener.py` |
| `strategy_candidates` | disabled | `agents/strategy_candidates.py` |
| `shared_news` | advisory | `agents/graph.py` |
| `technical` | active | `agents/technical_agent.py` |
| `fundamental` | advisory | `agents/fundamental_agent.py` |
| `sentiment` | active | `agents/sentiment_agent.py` |
| `options` | advisory | `agents/options_agent.py` |
| `macro` | advisory | `agents/macro_agent.py` |
| `india_specific` | advisory | `agents/india_specific_agent.py` |
| `geopolitical_nlp` | advisory | `agents/geopolitical_nlp_agent.py` |
| `bull_bear_debate` | active | `agents/bull_bear_synthesis_agent.py` |
| `risk_manager` | active | `agents/risk_manager_agent.py` |
| `marl_optimizer` | experimental | `risk/marl_portfolio_optimizer.py` |
| `delta_hedger` | experimental | `risk/ai_options_delta_hedger.py` |
| `compliance_check` | advisory | `config/fema_lrs_controller.py` |
| `portfolio_manager` | active | `agents/portfolio_manager_agent.py` |
| `order_router` | active | `agents/graph.py` |
| `episodic_reflection` | advisory | `agents/episodic_reflection_agent.py` |
| `signal_gate` | disabled | `models/signal_gate/gate.py` |
| `strategy_selector` | disabled | `models/signal_gate/strategy_selector.py` |
| `alpha_miner` | disabled | `agents/evolutionary_alpha_miner.py` |
| `stop_loss_manager` | active | `risk/stop_loss_manager.py` |
| `dead_mans_switch` | active | `risk/dead_mans_switch.py` |
| `alpaca_us_paper` | disabled by default; experimental when enabled | `broker/alpaca_paper_adapter.py` |
| `crypto_sim` | disabled by default; experimental when enabled | `broker/crypto_sim.py` |

### Open external items (code cannot produce these)

Multi-day market-hours paper evidence; at least 300 settled outcomes for the signal gate and AI-conviction calibration; out-of-sample T10 evidence on more than the free ~60 sessions of intraday data; Alpaca paper keys; official 2027 NSE and NYSE holiday calendars; a corporate-action data source; approval of the ₹1,000/month research budget.

## 8. Self-documenting code conventions

- Public modules, classes, and functions must state purpose, inputs, outputs, failure mode, and side effects in docstrings.
- Execution functions must state whether they can place an order or only simulate one.
- Risk checks must return a human-readable rejection reason and log it.
- Configuration additions require `.env.example`, a preflight check where applicable, and a corresponding test.
- Any real-money, sizing, data-source, or schedule change requires an ADR update above.
