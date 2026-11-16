import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  Activity, AlertTriangle, Bot, CalendarClock, ClipboardCheck, ListChecks, PlugZap, RadioTower, RefreshCw, ScanSearch, Send, ShieldAlert, ShieldCheck, Workflow,
} from "lucide-react";
import { agentLabel, ANIMATE, axisProps, CHART, istDateTime, istTime, isScanJob, JOB_INFO, precise, relative, scanStatusLabel, tooltipProps } from "./lib";
import { Button, Card, Chip, DetailList, Empty, Hint, Meter } from "./ui";

const FILL_OUTCOMES = {
  FILLED: ["Filled", CHART.teal],
  NOT_MARKETABLE: ["Limit not reached", CHART.warn],
  INVALID_PRICE: ["Invalid price", CHART.bad],
  REJECTED_NO_REFERENCE: ["No live reference price", CHART.bad],
  REJECTED_POSITION_CHANGED: ["Position changed before exit", CHART.bad],
  REJECTED_ENTRY_LIMIT: ["Entry-frequency limit", CHART.bad],
};

const AGENT_INFO = {
  MarketDataAgent: "Collects prices, candles and indicators",
  TechnicalAgent: "Reads chart patterns and momentum",
  FundamentalAgent: "Checks financials and valuation",
  SentimentAgent: "Scores news and exchange announcements",
  OptionsAgent: "Reads option-chain positioning",
  MacroAgent: "Global cues, FII/DII flows, currency",
  IndiaSpecificAgent: "Results season, policy and events",
  RiskManagerAgent: "Sizes trades and enforces limits",
  PortfolioManagerAgent: "Makes the final buy / sell / hold call",
};

function tickTone(feed) {
  if (!feed?.connected) return "bad";
  return feed.fresh ? "ok" : "warn";
}

function FeedCard({ feed, instruments, state, busy, unavailable, mutate }) {
  const tone = unavailable ? "warn" : tickTone(feed);
  const age = feed?.tick_age_seconds;
  return <Card title="Live price feed" icon={RadioTower} tone={tone} subtitle="Angel One WebSocket — prices used for paper fills"
    action={<Chip tone={tone}>{unavailable ? "Unconfirmed" : !feed?.connected ? "Disconnected" : feed.fresh ? "Fresh" : "Stale"}</Chip>}>
    <div className="pulse">
      <div className={`pulse-value tone-${tone}`}>
        <strong>{age == null ? "—" : age < 10 ? age.toFixed(1) : Math.round(age)}</strong><small>seconds since last price</small>
      </div>
      <Meter label="Stocks ticking" value={feed?.tokens_seen || 0} max={feed?.subscribed_tokens || 1}
        detail={`${feed?.tokens_seen ?? "—"} / ${feed?.subscribed_tokens ?? "—"}`} tone={feed?.tokens_seen === feed?.subscribed_tokens ? "ok" : "warn"} />
    </div>
    <DetailList items={[
      ["Last price received", `${istDateTime(feed?.last_tick_at)} IST`],
      ["Last exchange timestamp", `${istDateTime(feed?.last_exchange_at)} IST`],
      ["Accepted prices", feed?.tick_count != null ? feed.tick_count.toLocaleString("en-IN") : "—"],
      ["Rejected stale / invalid packets", feed?.rejected_ticks != null ? feed.rejected_ticks.toLocaleString("en-IN") : "—"],
      ["Security", feed?.tls_verified ? "Encrypted, certificate verified" : "Unconfirmed"],
      ["Connection attempts", feed?.connection_attempts ?? "—"],
      feed?.last_error_type && ["Last error", feed.last_error_type],
    ]} />
    {instruments?.unresolved?.length > 0 && <p className="alert-inline tone-warn" role="status"><AlertTriangle size={16} aria-hidden="true" />
      Not tradable until resolved: {instruments.unresolved.join(", ")}</p>}
    <div className="control-group">
      <Button icon={PlugZap} disabled={busy || !state?.paper_trading || !state?.trading_paused || unavailable}
        onClick={() => mutate("/operations/feed-connect")}>Connect data only</Button>
      <Button icon={RefreshCw} disabled={busy || !state?.paper_trading || !state?.trading_paused || unavailable || !feed?.connected}
        onClick={() => mutate("/operations/feed-reconnect")}>Test reconnect</Button>
    </div>
    <Hint>Buttons work only while paused. No orders or AI calls. “Fresh” requires both receipt time and exchange time within 30 seconds; stale prices cannot fill paper orders.</Hint>
  </Card>;
}

function TelegramCard({ telegram, state, busy, unavailable, mutate }) {
  const delivered = telegram?.delivery_status === "delivered";
  return <Card title="Telegram alerts" icon={Send} tone={delivered ? "ok" : "warn"} subtitle="Trade, risk and daily-summary messages"
    action={<Chip tone={delivered ? "ok" : "warn"}>{delivered ? "Delivered" : "Needs a test"}</Chip>}>
    <DetailList items={[
      ["Configured", telegram ? telegram.configured ? "Yes (kept private)" : "Missing" : "Unknown"],
      ["Last result", telegram?.delivery_status ? telegram.delivery_status.replaceAll("_", " ") : "Unknown"],
      ["Last delivered", `${istDateTime(telegram?.last_delivery_at)} IST`],
    ]} />
    <div className="control-group">
      <Button icon={Send} disabled={busy || !state?.paper_trading || !telegram?.configured || unavailable}
        onClick={() => mutate("/operations/telegram-test")}>Send test message</Button>
    </div>
    <Hint>The chat must be your personal chat or a group the bot can post to — never the bot itself. Change it privately in <code>.env</code>, then do a controlled restart.</Hint>
  </Card>;
}

function ScanCard({ scan, state, busy, unavailable, mutate }) {
  const last = scan?.last;
  const tone = scan?.in_progress ? "accent" : last?.status === "success" ? "ok" : last ? "warn" : "neutral";
  return <Card title="Scan supervision" icon={ScanSearch} tone={tone} subtitle="The AI pipeline run that looks for trades"
    action={<Chip tone={tone}>{scan?.in_progress ? "Running" : state?.trading_paused ? "Paused" : "Idle"}</Chip>}>
    <DetailList items={[
      ["Last scan", last ? scanStatusLabel(last) : "Not run in this process"],
      ["Finished", `${istDateTime(last?.finished_at)} IST`],
      ["Orders / problems", last ? `${last.orders ?? 0} / ${last.errors ?? 0}` : "—"],
      ["India VIX at analysis", last?.india_vix > 0 ? last.india_vix.toFixed(2) : "Unavailable"],
      ["Pause & halt survive restart", state?.controls_persistent ? "Yes" : "Unconfirmed"],
    ]} />
    <div className="control-group">
      <Button icon={Activity} disabled={busy || unavailable || !state?.paper_trading || !state?.trading_paused || scan?.in_progress}
        onClick={() => mutate("/operations/analysis")}>Analyse one stock (no orders)</Button>
    </div>
    <Hint>Needs entries paused. Uses live market data and the selected AI model (costs apply); order placement is disabled for this run.</Hint>
  </Card>;
}

function ScanSkipCard({ scan }) {
  const skip = scan?.last_skip;
  return <Card title="Why scans were skipped" icon={ScanSearch} tone={skip ? "warn" : "neutral"}
    subtitle="Every skipped scan is recorded with its reason and survives a restart">
    <DetailList items={[
      ["Will the next scan run?", scan?.will_run == null ? "Unconfirmed" : scan.will_run ? "Yes" : "No"],
      ["Reason", scan?.next_eligible_reason || "—"],
      ["Next trigger", scan?.next_trigger ? `${istDateTime(scan.next_trigger)} IST` : "—"],
      ["Last skip", skip ? `${skip.reason.replaceAll("_", " ")} at ${istDateTime(skip.at)} IST` : "None recorded"],
      ["Last successful analysis", scan?.last_successful_analysis?.finished_at ? `${istDateTime(scan.last_successful_analysis.finished_at)} IST` : "None yet"],
    ]} />
  </Card>;
}

function ReconciliationCard({ summary, unavailable }) {
  const [report, setReport] = useState(null), [running, setRunning] = useState(false), [error, setError] = useState("");
  const shown = report || summary;
  const run = async () => {
    setRunning(true); setError("");
    try {
      const r = await fetch("/api/operations/reconciliation", { signal: AbortSignal.timeout(20000) });
      const body = await r.json();
      if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Reconciliation unavailable.");
      setReport(body);
    } catch (err) { setError(err.message); } finally { setRunning(false); }
  };
  const status = shown?.status, s = shown?.summary;
  const tone = status === "pass" ? "ok" : status === "fail" ? "bad" : status ? "warn" : "neutral";
  const outcomes = Object.entries(s?.fill_outcomes || {}).map(([key, value]) => ({
    key, label: FILL_OUTCOMES[key]?.[0] || key.replaceAll("_", " ").toLowerCase(), value, color: FILL_OUTCOMES[key]?.[1] || CHART.muted,
  }));
  const issues = report?.issues || [];
  return <Card title="Paper ledger check" icon={ClipboardCheck} tone={tone}
    subtitle="Cross-checks every simulated fill, position, stop-loss and P&L figure"
    action={<Chip tone={tone}>{status ? status.toUpperCase() : "Not run"}</Chip>}>
    <div className="figure-row">
      <div><strong>{s?.executed_orders ?? "—"}</strong><small>orders filled</small></div>
      <div><strong>{s?.pending_orders ?? "—"}</strong><small>pending</small></div>
      <div><strong>{s?.open_positions ?? "—"}</strong><small>open positions</small></div>
      <div><strong>{s?.closed_lots ?? "—"}</strong><small>closed trades</small></div>
    </div>
    {outcomes.length ? <div className="chart" style={{ height: Math.max(120, outcomes.length * 44) }}>
      <ResponsiveContainer>
        <BarChart data={outcomes} layout="vertical" margin={{ left: 4, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} stroke={CHART.grid} />
          <XAxis type="number" allowDecimals={false} {...axisProps} />
          <YAxis type="category" dataKey="label" width={150} {...axisProps} />
          <Tooltip {...tooltipProps} formatter={value => [`${value} attempts`, "Fill attempts"]} />
          <Bar isAnimationActive={ANIMATE} dataKey="value" radius={8} barSize={16}>{outcomes.map(row => <Cell key={row.key} fill={row.color} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
    </div> : <Empty icon={ClipboardCheck} title="No fill attempts yet">A chart of filled vs. rejected paper orders appears after the first order.</Empty>}
    <DetailList items={[
      ["Last checked", `${istDateTime(shown?.checked_at)} IST`],
      ["Realised paper P&L", s ? precise(s.realised_pnl) : "—"],
      ["Average slippage", s?.avg_slippage_bps != null ? `${s.avg_slippage_bps} bps vs. live price` : "—"],
      ["Issues", report ? issues.length : shown?.issue_count ?? "—"],
    ]} />
    {issues.length > 0 && <ul className="issues">{issues.slice(0, 8).map((issue, index) => <li key={index}>
      <Chip tone={issue.severity === "fail" ? "bad" : "warn"}>{issue.severity.toUpperCase()}</Chip>
      <span><b>{issue.code.replaceAll("_", " ")}{issue.symbol ? ` · ${issue.symbol}` : ""}</b> {issue.detail}</span>
    </li>)}</ul>}
    {error && <p className="alert-inline tone-bad" role="alert"><AlertTriangle size={16} aria-hidden="true" />{error}</p>}
    <div className="control-group">
      <Button icon={ClipboardCheck} disabled={running || unavailable} onClick={run}>{running ? "Checking…" : "Run check now"}</Button>
    </div>
    <Hint>Runs automatically after the close. A FAIL pauses new entries and blocks resume until reviewed.</Hint>
  </Card>;
}

function CashFlowStatementCard({ state, scan, busy, unavailable, mutate }) {
  const [file, setFile] = useState(null);
  const [reviewed, setReviewed] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const selectFile = event => {
    const selected = event.target.files?.[0] || null;
    setResult(null);
    setError(selected && selected.size > 1_000_000 ? "Choose a CSV no larger than 1 MB." : "");
    setFile(selected && selected.size <= 1_000_000 ? selected : null);
    setReviewed(false);
  };
  const importStatement = async () => {
    if (!file || !reviewed) return;
    setError("");
    try {
      const response = await mutate("/operations/cash-flows/angelone-statement", {
        csv_text: await file.text(),
        reviewed: true,
      });
      if (response) {
        setResult(response);
        setReviewed(false);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not read the selected statement.");
    }
  };

  return <Card title="Broker cash-flow statement" icon={ClipboardCheck} tone="warn"
    subtitle="Import completed Angel One deposits and withdrawals">
    <label className="cash-flow-file">
      <span>Normalized CSV statement</span>
      <input type="file" accept=".csv,text/csv" onChange={selectFile} disabled={busy} />
    </label>
    <label className="review">
      <input type="checkbox" checked={reviewed} onChange={event => setReviewed(event.target.checked)}
        disabled={busy || !file} />
      <span className="check-box" aria-hidden="true"><ClipboardCheck size={16} /></span>
      <span>I downloaded this statement from the official broker portal and included only completed cash transfers since the paper cash-flow baseline.</span>
    </label>
    <div className="control-group">
      <Button icon={ClipboardCheck}
        disabled={busy || unavailable || !state?.paper_trading || !state?.trading_paused
          || scan?.in_progress || !file || !reviewed || Boolean(error)}
        onClick={importStatement}>Import statement</Button>
    </div>
    {error && <p className="alert-inline tone-bad" role="alert">{error}</p>}
    {result && <p className="alert-inline tone-warn" role="status">
      {result.imported} new, {result.duplicates} already-recorded transfer(s).
      {result.reconciliation_pending ? " New entries remain blocked pending a fresh portfolio sync." : " No new reconciliation is pending."}
    </p>}
    <Hint>Use columns transaction_id,timestamp,direction,amount,status; timestamps need an explicit timezone. Only SUCCESS or COMPLETED DEPOSIT/WITHDRAWAL rows since the paper cash-flow baseline are accepted. Entries stay blocked until reconciled.</Hint>
  </Card>;
}

function AgentsCard({ agents }) {
  const rows = agents.data || [];
  return <Card title="AI pipeline" icon={Workflow} subtitle="The specialist agents each scan runs through, in order">
    {rows.length ? <ol className="pipeline">
      {rows.map(agent => <li key={agent.name}>
        <span className="pipeline-step" aria-hidden="true"><Bot size={16} /></span>
        <div><strong>{agentLabel(agent.name)}</strong><p>{AGENT_INFO[agent.name] || "Pipeline stage"}</p></div>
        <Chip tone={agent.status === "ready" ? "ok" : agent.status === "error" ? "bad" : "accent"}>{agent.status}</Chip>
      </li>)}
    </ol> : <Empty icon={Workflow} title={agents.error ? "Agent list unavailable" : "Loading agents"} />}
    <Hint>“Ready” means the agent is loaded in this process — it is not a record of past runs. Scan results are on the Overview page.</Hint>
  </Card>;
}

function FullSchedule({ jobs, now }) {
  const rows = (jobs || []).slice().sort((a, b) => (a.next_run || "~").localeCompare(b.next_run || "~"));
  return <Card title="Full automatic schedule" icon={CalendarClock} subtitle="Every scheduler job and when it next runs (IST)">
    {rows.length ? <div className="table-wrap"><table>
      <thead><tr><th>Job</th><th>What it does</th><th className="num">Next run</th></tr></thead>
      <tbody>{rows.map(job => <tr key={job.id} className={isScanJob(job) ? "is-scan" : ""}>
        <td><b>{job.name}</b></td><td>{JOB_INFO[job.id] || "Scheduled job"}</td>
        <td className="num">{job.next_run ? <>{istTime(job.next_run)}<small>{relative(job.next_run, now)}</small></> : "Paused"}</td>
      </tr>)}</tbody>
    </table></div> : <Empty icon={CalendarClock} title="No jobs reported" />}
  </Card>;
}

function Limitations({ items }) {
  return <Card title="Known limits" icon={ShieldAlert} tone="warn" subtitle="This is a paper-evaluation system, not a live-trading certificate">
    <ul className="limits">{(items || ["Waiting for operational diagnostics."]).map(item => <li key={item}>{item}</li>)}</ul>
  </Card>;
}

const PENDING_EXIT = { unresolved_intent: "Unresolved — awaiting ledger/broker check", submitted_order: "Submitted — awaiting fill" };

function protectionTone(protection) {
  if (!protection) return "neutral";
  const rows = protection.positions || [];
  if (protection.last_check?.error || protection.unsubscribed_positions?.length
    || rows.some(row => !row.tracked || row.pending_exit)) return "bad";
  if (rows.some(row => !row.mark_fresh) || (rows.length && !protection.tick_triggers_active)) return "warn";
  return "ok";
}

function ProtectionCard({ protection, unavailable }) {
  const rows = protection?.positions || [];
  const check = protection?.last_check;
  const tone = unavailable ? "warn" : protectionTone(protection);
  const label = unavailable || !protection ? "Unconfirmed" : tone === "ok" ? "Covered" : tone === "bad" ? "Needs review" : "Degraded";
  return <Card title="Open-position protection" icon={ShieldCheck} tone={tone}
    subtitle="Stop/target triggers, price freshness and pending exits for each open position"
    action={<Chip tone={tone}>{label}</Chip>}>
    {rows.length ? <div className="table-wrap"><table>
      <thead><tr><th>Position</th><th className="num">Stop / target</th><th className="num">Mark</th><th>Exit state</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.symbol}>
        <td><b>{row.symbol}</b><small>{row.quantity} · {row.holding_period || "—"}</small></td>
        <td className="num">{row.tracked ? `${precise(row.stop_loss)} / ${precise(row.target)}` : "Untracked"}</td>
        <td className="num">{precise(row.mark_price)}<small>{row.mark_fresh ? `fresh · ${Math.max(0, row.mark_age_seconds)}s` : row.mark_age_seconds == null ? "no market mark" : `stale · ${Math.round(row.mark_age_seconds)}s`}</small></td>
        <td>{row.pending_exit ? PENDING_EXIT[row.pending_exit] || row.pending_exit
          : row.last_exit_attempt ? `${row.last_exit_attempt.outcome.replaceAll("_", " ").toLowerCase()} (${row.last_exit_attempt.source})` : "Monitoring"}</td>
      </tr>)}</tbody>
    </table></div> : <Empty icon={ShieldCheck} title="No open positions">Protection details appear here while a position is open.</Empty>}
    <DetailList items={[
      ["Tick-driven stop/target", protection ? protection.tick_triggers_active ? "Active" : "Off — 5-minute check only" : "Unconfirmed"],
      ["Last 5-minute check", check?.at ? `${istDateTime(check.at)} IST` : "Not run yet"],
      check?.error && ["Last check error", check.error],
      check?.unverified_marks?.length > 0 && ["Stop/target unverified (stale mark)", check.unverified_marks.join(", ")],
      protection?.unsubscribed_positions?.length > 0 && ["No live feed for", protection.unsubscribed_positions.join(", ")],
    ]} />
    <Hint>A stale mark never triggers a stop; the exit is re-priced from a fresh tick. Intraday time exits still run without a fresh mark.</Hint>
  </Card>;
}

const CAPABILITY_TONE = { active: "ok", advisory: "accent", experimental: "warn", disabled: "neutral" };

function CapabilitiesCard({ capabilities, sleeves }) {
  const rows = capabilities || [];
  return <Card title="What is actually running" icon={ListChecks} className="span-full"
    subtitle="Every pipeline node and module, labelled by what it can do to an order">
    {rows.length ? <div className="table-wrap"><table>
      <thead><tr><th>Component</th><th>Status</th><th>Notes</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.name}>
        <td><b>{row.name.replaceAll("_", " ")}</b><small>{row.module}</small></td>
        <td><Chip tone={CAPABILITY_TONE[row.status] || "neutral"}>{row.status}</Chip></td>
        <td>{row.notes}</td>
      </tr>)}</tbody>
    </table></div> : <Empty icon={ListChecks} title="Inventory unavailable" />}
    {sleeves?.length ? <DetailList items={sleeves.map(sleeve => [
      `${sleeve.name} (${sleeve.horizons.join("/")}, ${sleeve.book} book)`, sleeve.enabled ? "ENABLED" : "Disabled",
    ])} /> : null}
    <Hint>Optional strategy sleeves ship disabled. Foreign-market strategies are experimental and may place paper/simulated orders when their flags are enabled; this does not demonstrate profitability.</Hint>
  </Card>;
}

function BudgetAlertsCard({ budget, alerts }) {
  const spend = budget?.spend_inr, limit = budget?.budget_inr;
  const tone = budget?.exceeded ? "bad" : spend != null && limit && spend / limit > 0.75 ? "warn" : "ok";
  return <Card title="AI spend and undelivered alerts" icon={Send} tone={alerts?.length ? "bad" : tone}
    subtitle="Daily model budget and critical alerts Telegram has not delivered">
    <DetailList items={[
      ["Today’s model spend", spend == null ? "Unconfirmed" : `₹${spend.toFixed(2)} of ₹${limit}`],
      ["Budget state", budget?.exceeded ? (budget.unknown_pricing_calls ? "Spend cannot be measured — paid scans skipped" : "Reached — paid scans skipped until tomorrow") : "Within budget"],
      ["Undelivered critical alerts", alerts?.length ?? 0],
    ]} />
    {alerts?.length ? <ul className="limits">{alerts.map(alert =>
      <li key={alert.id}>{istDateTime(alert.created_at)} IST — {alert.text} ({alert.attempts} attempts{alert.last_error ? `, ${alert.last_error}` : ""})</li>)}</ul> : null}
    <Hint>Exits and stop-losses never depend on this budget. Alerts are stored first and retried every minute.</Hint>
  </Card>;
}

function ForeignMarketsCard({ markets }) {
  const us = markets?.us, crypto = markets?.crypto;
  const summarize = (last) => {
    const symbols = Object.entries(last?.symbols || {}).map(([symbol, result]) =>
      `${symbol}: ${result.status}${result.reason ? ` (${result.reason})` : ""}`);
    return symbols.join("; ") || last?.reason || "—";
  };
  const rows = [
    ["Schedule", markets?.running ? "Running" : "Not running"],
    ["US paper", us?.enabled ? "Enabled — Alpaca paper only" : "Disabled"],
    ["US symbols", us?.watchlist?.join(", ") || "—"],
    ["US next run", us?.next_run ? `${istDateTime(us.next_run)} IST` : "—"],
    ["US latest result", summarize(us?.last_run)],
    ["Crypto simulation", crypto?.enabled ? "Enabled — simulated only" : "Disabled"],
    ["Crypto pairs", crypto?.watchlist?.join(", ") || "—"],
    ["Crypto next run", crypto?.next_run ? `${istDateTime(crypto.next_run)} IST` : "—"],
    ["Crypto latest result", summarize(crypto?.last_run)],
  ];
  return <Card title="US and crypto schedules" icon={CalendarClock}
    tone={markets?.running ? "warn" : "neutral"}
    subtitle="Independent 15-minute jobs; 20/50 trend, long-only, paper/simulation only">
    <DetailList items={rows} />
    <Hint>US uses Alpaca IEX quotes during known regular US sessions. Crypto needs fresh public candles and prices; it will not enter using fixture data. Strategy results are experimental, not evidence of profitability. Both obey the global entry pause and risk halts.</Hint>
  </Card>;
}

function RiskCard({ risk, build, processId }) {
  const drawdownHalted = risk?.portfolio_drawdown_halted;
  const dailyHalted = risk?.daily_loss_halted;
  const flowPending = risk?.cash_flow_reconciliation_pending;
  const tone = drawdownHalted || dailyHalted || flowPending ? "bad" : risk?.daily_baseline_valid ? "ok" : "warn";
  return <Card title="Portfolio safety and runtime" icon={ShieldAlert} tone={tone}
    subtitle="Persistent equity limits and the source version loaded by this process">
    <DetailList items={[
      ["Daily net loss", risk ? `${(risk.daily_loss_pct * 100).toFixed(2)}% / ${(risk.daily_loss_limit_pct * 100).toFixed(0)}%` : "Unconfirmed"],
      ["Daily baseline", risk ? risk.daily_baseline_valid ? "Valid" : "Unknown — entries fail closed" : "Unconfirmed"],
      ["Daily entry cap", risk?.max_daily_entries ?? "Unconfirmed"],
      ["Per-symbol daily cap (paper)", risk?.max_daily_entries_per_symbol ?? "Unconfirmed"],
      ["Re-entry cooldown (paper)", risk?.symbol_reentry_cooldown_minutes == null ? "Unconfirmed" : `${risk.symbol_reentry_cooldown_minutes} minutes`],
      ["Portfolio drawdown", risk ? `${(risk.portfolio_drawdown_pct * 100).toFixed(2)}% / ${(risk.portfolio_drawdown_limit_pct * 100).toFixed(0)}%` : "Unconfirmed"],
      ["Drawdown latch", risk ? drawdownHalted ? "Active — reset requires review" : "Clear" : "Unconfirmed"],
      ["Cash-flow reconciliation", risk ? flowPending ? "Pending — entries blocked" : "Clear" : "Unconfirmed"],
      ["Process ID", processId ?? "Unconfirmed"],
      ["Source fingerprint", build?.digest ? `${build.digest.slice(0, 16)}…` : "Unconfirmed"],
    ]} />
    <Hint>The fingerprint identifies source files loaded when this dashboard process started; it does not certify dependencies or profitability.</Hint>
  </Card>;
}

export default function Operations({ ops, opsUnavailable, state, busy, mutate, agents, now }) {
  return <div className="view">
    <div className="view-head"><h1>Operations</h1><p>Health of the price feed, alerts, scans and the paper ledger — with safe, no-order test buttons.</p></div>
    <div className="grid grid-3">
      <FeedCard feed={ops?.feed} instruments={ops?.instruments} state={state} busy={busy} unavailable={opsUnavailable} mutate={mutate} />
      <TelegramCard telegram={ops?.telegram} state={state} busy={busy} unavailable={opsUnavailable} mutate={mutate} />
      <ScanCard scan={ops?.scan} state={state} busy={busy} unavailable={opsUnavailable} mutate={mutate} />
    </div>
    <div className="grid grid-3">
      <ReconciliationCard summary={ops?.reconciliation} unavailable={opsUnavailable} />
      <CashFlowStatementCard state={state} scan={ops?.scan} busy={busy}
        unavailable={opsUnavailable} mutate={mutate} />
      <AgentsCard agents={agents} />
    </div>
    <div className="grid grid-2">
      <FullSchedule jobs={ops?.jobs} now={now} />
      <Limitations items={ops?.limitations} />
    </div>
    <div className="grid grid-2">
      <ProtectionCard protection={ops?.protection} unavailable={opsUnavailable} />
      <RiskCard risk={ops?.portfolio_risk} build={state?.runtime_build} processId={state?.process_id} />
    </div>
    <div className="grid grid-2">
      <ForeignMarketsCard markets={ops?.foreign_markets} />
      <BudgetAlertsCard budget={ops?.llm_budget} alerts={ops?.undelivered_critical_alerts} />
    </div>
    <div className="grid grid-2">
      <ScanSkipCard scan={ops?.scan} />
    </div>
    <CapabilitiesCard capabilities={ops?.capabilities} sleeves={ops?.sleeves} />
  </div>;
}
