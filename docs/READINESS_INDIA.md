# Readiness — India core (supervised paper session, Monday 16 November 2026)

**Purpose.** A checklist of pass/fail criteria for the first supervised market-hours session of the paper trader.
Passing it shows that the *plumbing* behaves; it says nothing about profitability, and one session is not evidence of an
edge. The system is **paper only**; this document is not permission to trade real money.

**Operator:** watch from 08:55 IST to 16:00 IST. All dashboard calls below are read-only `GET`s against
`http://127.0.0.1:8000`. Do not stop or restart the bot during market hours.

**If any criterion marked ✋ fails:** pause entries from the dashboard (Controls → Pause), leave the protective jobs
running, note the evidence, and do not resume until the cause is understood. Exits are never paused.

## Before the open (08:30–09:15)

| # | Check | How to read it | Pass |
|---|---|---|---|
| 1 | Running build is the tested build | `/api/health` → `runtime_build.digest` | equals the fingerprint in the deployment note |
| 2 | Paper mode | `/api/health` → `mode`, `paper_trading` | `PAPER`, `true` |
| 3 | Calendar knows today | `python3 main.py --preflight` | no "No NSE holiday calendar" error |
| 4 | Entries state is intended | `/api/health` → `trading_paused`, `dead_mans_switch` | unpaused, DMS not halted |
| 5 | Reconciliation | `/api/operations/reconciliation` → `status` | `pass` (no `fail` issues) |
| 6 | Broker authenticated | `/api/health` → `broker_authenticated` | `true` after the 08:55 job |
| 7 | Every sleeve is off | `/api/operations` → `sleeves` | all `disabled` |
| 8 | LLM budget available | `/api/operations` → `llm_budget` | spent < daily budget; pricing known |

## During the session

| # | Criterion | Exact measurement | Pass |
|---|---|---|---|
| 9 ✋ | **Feed fresh ≤ 30 s** | `/api/operations` → `feed.tick_age_seconds` and `feed.exchange_tick_age_seconds`, polled every minute from 09:20 | both ≤ 30 and `feed.fresh` is `true` in every sample between 09:20 and 15:15; any gap is explained by the DMS/alert log |
| 10 ✋ | **A scan completes under the deadline** | `/api/health` → `last_scan`: `finished_at − started_at` for each scan with `status` of `completed`/`degraded` | below `SCAN_ENTRY_DEADLINE_SECONDS` (default 900); no `scan_deadline` in `scan_status.last_skip` for a scan that should have run |
| 11 ✋ | **Every decision has a review record** | `/api/signals` → each row's `review_contract` | present on every row that shows a decision; `contract_version` is `review-v2` and `passed` is `true` for every candidate that reached an order (a `passed: false` row must show a `failed_requirements` list and no order); `/api/trades` has no entry without a matching signal |
| 12 ✋ | **A stop fires within 1 s of the tick that crossed it** | For each stop/target exit: `/api/operations` → `protection.positions[].last_exit_attempt` (`source: "tick"`), and the matching `paper_fill_audits` row (`action` opposite to the position, exit fill): `reference_age_seconds` | `reference_age_seconds ≤ 1.0` and the `[PAPER]` fill line in `logs/` is timestamped ≤ 1 s after the tick time. If no stop is hit during the day this is **not evaluated**; record "no sample" |
| 13 | New-entry cut-offs | Orders in `/api/trades` | no entry after the entry deadline and no entry while paused/halted |
| 14 | Every open position is protected | `/api/operations` → `protection.positions[]` | `tracked: true`, a stop present, `mark_fresh: true` |
| 15 | No duplicate exits | `/api/trades` | at most one exit order per lot |
| 16 | Alerts reach a human | `/api/operations` → `undelivered_critical_alerts` | `0` at the end of the day (otherwise Telegram is not delivering) |
| 17 | Skipped scans are explained | `/api/health` → `scan_status.last_skip` | every skipped scan carries a code and reason |

## Close (15:15–16:00)

| # | Criterion | Exact measurement | Pass |
|---|---|---|---|
| 18 ✋ | **Square-off at 15:15 leaves the intraday book flat** | `/api/portfolio` positions after 15:20; `/api/operations/reconciliation` → no `intraday_open_after_squareoff` | zero intraday positions after 15:20 IST. If the 15:15 attempt failed, `squareoff_retry` runs every minute to 15:30 |
| 19 ✋ | **EOD reconciliation PASS** | `/api/operations/reconciliation` after 15:45 | `status: pass`, 0 `fail` issues |
| 20 | Daily snapshot recorded | `/api/pnl/daily` | a row for the day; closing marks stored |
| 21 | Spend within budget | `/api/llm-usage` | below `LLM_DAILY_BUDGET_INR` |

## What to write down (for the next decision)

- Count of scans, orders, rejected entries and the top `risk_block_reasons` / `review_block_reasons`.
- Any DMS halt or feed gap with its time and cause.
- Whether criterion 12 had a sample.
- Anything that looked wrong on screen, even if the table says pass.

## What this session cannot show

- Profitability or an edge: paper fills are simulated against live reference prices with a slippage model, and one
  session is one regime.
- Behaviour of a live broker order lifecycle: live entries stay blocked (ADR-0020).
- Corporate-action handling: there is no ingestion source; a missing record blocks entries only when one is present.
- The 2027 holiday calendar: it is not available yet; the preflight warns from December.

## Sign-off

| Result | Date | Operator |
|---|---|---|
| Criteria 9–12, 18, 19 passed (✋) | | |
| Notes | | |
