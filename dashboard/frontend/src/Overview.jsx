import {
  Activity, ArrowRight, Ban, BookOpen, Briefcase, CalendarClock, CircleCheck, CircleHelp, Clock, Gauge as GaugeIcon,
  IndianRupee, ListChecks, OctagonX, PauseCircle, PlayCircle, Radio, RotateCcw, ScanSearch, Send,
  ShieldAlert, ShieldCheck, TrendingDown, TrendingUp, TriangleAlert, Wallet, Waves,
} from "lucide-react";
import {
  isScanJob, istDateTime, istTime, JOB_INFO, money, regimeInfo, relative, scanStatusLabel, signed,
} from "./lib";
import { Button, Card, Chip, Empty, Hint, Ring, Stat, TONE_COLOR, Well } from "./ui";
import { MarketWorkspaceGrid, StrategyWorkspaceGrid } from "./Workspaces";

export function nextScanJob(jobs) {
  return (jobs || []).filter(isScanJob).filter(job => job.next_run)
    .sort((a, b) => a.next_run.localeCompare(b.next_run))[0];
}

export function readinessChecks({ state, ops, session, opsUnavailable }) {
  const feed = ops?.feed, hmm = ops?.hmm, latest = hmm?.latest, recon = ops?.reconciliation;
  const halted = state?.dead_mans_switch?.halted;
  const portfolioRisk = state?.portfolio_risk;
  const regime = regimeInfo(latest?.regime);
  const regimeAllows = hmm?.fitted && latest && !latest.degraded && latest.regime !== "SYSTEMIC_PANIC"
    && latest.directive?.allow_new_positions !== false;
  const unknown = !state || opsUnavailable;
  return [
    {
      id: "paper", blocking: true, why: "the system is not in paper mode", icon: ShieldCheck, label: "Simulated money only",
      tone: !state ? "neutral" : state.paper_trading ? "ok" : "bad",
      status: !state ? "Checking" : state.paper_trading ? "Paper mode" : "LIVE mode",
      explain: "Orders are simulated against live prices. No real money can move from this dashboard.",
    },
    {
      id: "entries", blocking: true, why: "new entries are paused", icon: state?.trading_paused ? PauseCircle : PlayCircle,
      label: "New entries enabled",
      tone: !state ? "neutral" : state.trading_paused ? "warn" : "ok",
      status: !state ? "Checking" : state.trading_paused ? "Paused" : "Enabled",
      explain: state?.trading_paused
        ? "Paused by you or by a safety check. Stop-losses and exits keep running."
        : "Scheduled scans may open new simulated trades.",
    },
    {
      id: "breaker", blocking: true, why: "the circuit breaker is halted", icon: halted ? ShieldAlert : ShieldCheck, label: "Circuit breaker clear",
      tone: !state ? "neutral" : halted ? "bad" : "ok",
      status: !state ? "Checking" : halted ? "Halted" : "Clear",
      explain: halted
        ? `Tripped: ${state.dead_mans_switch.halt_reason || "manual halt"}. Review, then clear it in Safety controls.`
        : "Trips automatically after repeated failures and blocks new trades until reviewed.",
    },
    {
      id: "drawdown", blocking: true, why: "the 10% portfolio drawdown pause is active",
      icon: portfolioRisk?.portfolio_drawdown_halted ? ShieldAlert : ShieldCheck,
      label: "Portfolio drawdown limit",
      tone: !portfolioRisk ? "neutral" : portfolioRisk.portfolio_drawdown_halted ? "bad" : "ok",
      status: !portfolioRisk ? "Unconfirmed"
        : `${(portfolioRisk.portfolio_drawdown_pct * 100).toFixed(2)}% / ${(portfolioRisk.portfolio_drawdown_limit_pct * 100).toFixed(0)}%`,
      explain: portfolioRisk?.portfolio_drawdown_halted
        ? "New entries are blocked until the drawdown latch is reset after review."
        : "A persistent peak-to-trough equity limit blocks new entries.",
    },
    {
      id: "session", blocking: true, why: "outside the 09:30–14:45 entry window", icon: Clock, label: "Inside the entry window",
      tone: session.entry_window_open ? "ok" : "neutral",
      status: session.entry_window_open ? "09:30–14:45 IST" : session.label,
      explain: "New intraday trades are only opened between 09:30 and 14:45 IST on trading days.",
    },
    {
      id: "feed", blocking: true, why: "live prices are not fresh", icon: Radio, label: "Live prices flowing",
      tone: unknown ? "neutral" : feed?.fresh ? "ok" : feed?.connected ? "warn" : session.is_open ? "bad" : "neutral",
      status: unknown ? "Unconfirmed" : feed?.fresh ? `Tick ${feed.tick_age_seconds ?? 0}s ago`
        : feed?.connected ? "Connected, stale" : "Disconnected",
      explain: "Paper fills use the latest Angel One tick; without fresh ticks no trade can be filled.",
    },
    {
      id: "regime", blocking: true, why: "the market-regime gate is not satisfied", icon: Waves, label: "Market regime allows entries",
      tone: unknown ? "neutral" : !hmm?.fitted ? "bad" : !latest ? "warn" : regimeAllows ? "ok"
        : latest.regime === "SYSTEMIC_PANIC" ? "bad" : "warn",
      status: unknown ? "Unconfirmed" : !hmm?.fitted ? "Model not fitted" : !latest ? "Not classified yet"
        : latest.degraded ? "Data too old" : regime.name,
      explain: regimeAllows ? regime.effect : "Entries need a fresh, non-panic regime classification.",
    },
    {
      id: "ledger", blocking: true, why: "the paper ledger is not reconciled", icon: ListChecks, label: "Paper ledger reconciled",
      tone: unknown ? "neutral" : recon?.status === "pass" ? "ok" : recon?.status === "fail" ? "bad" : "warn",
      status: unknown ? "Unconfirmed" : recon?.status === "pass" ? `Passed ${istTime(recon.checked_at)}`
        : recon?.status === "fail" ? "Failed" : "Not run yet",
      explain: "Every fill, position and P&L figure is cross-checked before each scan; a failure pauses entries.",
    },
    {
      id: "broker", blocking: false, icon: Activity, label: "Broker session active",
      tone: !state ? "neutral" : state.broker_authenticated ? "ok" : "warn",
      status: !state ? "Checking" : state.broker_authenticated ? "Angel One logged in"
        : state.broker_configured ? "Not logged in" : "Not configured",
      explain: "Needed for market data and history. Paper orders never reach the broker.",
    },
    {
      id: "telegram", blocking: false, icon: Send, label: "Telegram alerts working",
      tone: unknown ? "neutral" : ops?.telegram?.delivery_status === "delivered" ? "ok" : "warn",
      status: unknown ? "Unconfirmed" : ops?.telegram?.delivery_status === "delivered"
        ? `Delivered ${istTime(ops.telegram.last_delivery_at)}` : "Needs a test message",
      explain: "Trade, error and daily-summary alerts are sent to your private chat.",
    },
  ];
}

function CommandHero({ state, ops, onNavigate, now }) {
  const foreign = ops?.foreign_markets;
  const scans = [
    ...(ops?.jobs || []).filter(job => isScanJob(job) && job.next_run),
    ...(foreign?.us?.next_run ? [{ name: "US paper scan", next_run: foreign.us.next_run }] : []),
    ...(foreign?.crypto?.next_run ? [{ name: "Crypto simulation scan", next_run: foreign.crypto.next_run }] : []),
  ].sort((a, b) => a.next_run.localeCompare(b.next_run));
  const next = scans[0];
  const halted = state?.dead_mans_switch?.halted;
  const paused = state?.trading_paused || state?.portfolio_risk?.portfolio_drawdown_halted
    || state?.portfolio_risk?.daily_loss_halted;
  const tone = !state ? "neutral" : !state.paper_trading || halted ? "bad" : paused ? "warn" : "ok";
  const entries = !state ? "Checking" : !state.paper_trading ? "Not paper"
    : halted ? "Safety halt" : paused ? "Paused" : "Enabled";
  return <section className={`card command-hero hero-${tone}`} aria-live="polite">
    <div className="command-copy">
      <span className="eyebrow">Multi-market paper trading</span>
      <h1>Trading command centre</h1>
      <p>India, US paper and crypto each have independent sessions and scan status. Forex is not connected.</p>
      <button type="button" className="link command-link" onClick={() => onNavigate?.("workspaces")}>
        Explore markets and strategies <ArrowRight size={16} aria-hidden="true" />
      </button>
    </div>
    <dl className="command-status">
      <div><dt>Mode</dt><dd>{state ? state.paper_trading ? "Paper only" : "Not paper" : "Unconfirmed"}</dd></div>
      <div><dt>New entries</dt><dd><Chip tone={tone}>{entries}</Chip></dd></div>
      <div><dt>Next scheduled scan</dt><dd>{next
        ? <><strong>{next.name}</strong><span>{istDateTime(next.next_run)} IST · {relative(next.next_run, now)}</span></>
        : "Schedule unavailable"}</dd></div>
      <div><dt>Workspaces</dt><dd>India · US · Crypto · Forex status</dd></div>
    </dl>
  </section>;
}

function heroStatus(state, session, nextScan, now) {
  if (!state) return { tone: "neutral", title: "Connecting to the trading engine…",
    body: "Waiting for the first status update from the local backend." };
  if (!state.paper_trading) return { tone: "bad", title: "Live mode detected",
    body: "This dashboard is for paper evaluation only. Stop the process and review the configuration." };
  if (state.dead_mans_switch?.halted) return { tone: "bad", title: "Trading is halted",
    body: `The circuit breaker stopped new trades (${state.dead_mans_switch.halt_reason || "manual halt"}). Stop-losses and exits still run. Review, clear the halt, then resume when ready.` };
  if (state.portfolio_risk?.portfolio_drawdown_halted) return { tone: "bad", title: "Portfolio risk pause",
    body: "The persistent portfolio drawdown limit is active. New entries are blocked; open-position exits remain enabled. Review the equity history before resetting this latch." };
  if (state.trading_paused) return { tone: "warn", title: "Paper trading is paused",
    body: "The bot keeps protecting open positions (stop-losses and square-off) but will not open new trades until you resume." };
  const next = nextScan ? ` The next scan runs at ${istTime(nextScan.next_run)} IST (${relative(nextScan.next_run, now)}).` : "";
  if (!session.is_open) return { tone: "ok", title: "Paper trading is on — market closed",
    body: `New entries are enabled and will be considered when the market reopens.${next}` };
  if (!session.entry_window_open) return { tone: "ok", title: "Paper trading is on — outside the entry window",
    body: `New intraday trades are only opened between 09:30 and 14:45 IST. Open positions are still managed and intraday positions are squared off at 15:15.${next}` };
  return { tone: "ok", title: "Paper trading is active",
    body: `The bot scans the watchlist on a schedule and may open simulated trades. No real money is used.${next}` };
}

export function Hero({ state, session, checks, nextScan, regime, now, scanStatus }) {
  const status = heroStatus(state, session, nextScan, now);
  const passing = checks.filter(check => check.tone === "ok").length;
  const blocker = checks.find(check => check.blocking && check.tone !== "ok");
  const HeroIcon = status.tone === "bad" ? OctagonX : status.tone === "warn" ? PauseCircle : ShieldCheck;
  return <section className={`card hero hero-${status.tone}`} aria-live="polite">
    <div className="hero-copy">
      <div className="hero-kicker"><Well icon={HeroIcon} tone={status.tone} size="lg" className="float" />
        <span className="eyebrow">Right now</span></div>
      <h1>{status.title}</h1>
      <p>{status.body}</p>
      <div className="hero-answer">
        <span>Can the bot open a new paper trade right now?</span>
        <Chip tone={blocker ? "warn" : "ok"}>{!state ? "Checking" : blocker ? `Not now — ${blocker.why}` : "Yes"}</Chip>
      </div>
      <dl className="hero-facts">
        <div><dt>Market</dt><dd>{session.label}</dd></div>
        <div><dt>Next trigger</dt><dd>{nextScan ? `${istTime(nextScan.next_run)} · ${relative(nextScan.next_run, now)}` : "—"}</dd></div>
        <div><dt>Will it run?</dt><dd>{scanStatus?.will_run == null ? "—"
          : `${scanStatus.will_run ? "Yes" : "No"} — ${scanStatus.next_eligible_reason || "no reason reported"}`}</dd></div>
        <div><dt>Market mood</dt><dd style={{ color: "inherit" }}><i className="dot" style={{ background: regime.color }} />{regime.name}</dd></div>
      </dl>
    </div>
    <div className="hero-ring">
      <Ring value={passing} max={checks.length} tone={blocker ? "warn" : "ok"} caption="checks OK" />
      <a className="link" href="#checklist">See checklist</a>
    </div>
  </section>;
}

export function KpiRow({ portfolio, config }) {
  const daily = portfolio?.daily_pnl;
  const invested = portfolio ? Math.max(0, (portfolio.total_value || 0) - (portfolio.available_capital || 0)) : null;
  return <div className="grid grid-4">
    <Stat icon={Wallet} label="Portfolio value" value={money(portfolio?.total_value)}
      detail={config?.initial_capital ? `Started with ${money(config.initial_capital)}` : "Paper capital"} />
    <Stat icon={daily < 0 ? TrendingDown : TrendingUp} label="Today’s P&L" value={signed(daily)}
      tone={daily < 0 ? "bad" : daily > 0 ? "ok" : "accent"} valueTone={daily < 0 ? "bad" : daily > 0 ? "ok" : undefined}
      detail={portfolio ? `${money(portfolio.gross_daily_pnl)} gross − ${money(portfolio.transaction_costs)} estimated charges` : "Realised + unrealised − estimated charges"} />
    <Stat icon={IndianRupee} label="Cash available" value={money(portfolio?.available_capital)} tone="ok"
      detail={invested != null ? `${money(invested)} invested` : "—"} />
    <Stat icon={Briefcase} label="Open positions" value={portfolio?.position_count ?? "—"} tone="warn"
      detail={config?.max_positions ? `Limit ${config.max_positions} at a time` : "—"} />
  </div>;
}

export function Checklist({ checks }) {
  return <Card title="India · can CompositeTrend open?" icon={ListChecks}
    subtitle="NSE entry readiness only. Protective exits never wait for these checks."
    className="span-2" >
    <ul className="checklist" id="checklist">
      {checks.map(check => <li key={check.id}>
        <Well icon={check.icon} tone={check.tone} size="sm" />
        <div>
          <strong>{check.label}{!check.blocking && <em> · informational</em>}</strong>
          <p>{check.explain}</p>
        </div>
        <Chip tone={check.tone}>{check.status}</Chip>
      </li>)}
    </ul>
  </Card>;
}

function resumeBlocker(state, healthError, reviewed) {
  if (healthError || !state) return "Waiting for a confirmed backend status.";
  if (!state.paper_trading) return "Resuming is only possible in paper mode.";
  if (!state.trading_paused) return "Entries are already enabled.";
  if (state.dead_mans_switch?.halted) return "Clear the circuit breaker first (entries stay paused).";
  if (state.portfolio_risk?.portfolio_drawdown_halted) return "Reset the portfolio drawdown latch first (entries stay paused).";
  if (state.scan_in_progress) return "Wait for the running scan to finish.";
  if (!state.scheduler_running) return "The scheduler is not running.";
  if (!reviewed) return "Tick the review box to enable recovery actions.";
  return "";
}

export function SafetyControls({ state, healthError, busy, reviewed, setReviewed, control }) {
  const halted = state?.dead_mans_switch?.halted;
  const drawdownHalted = state?.portfolio_risk?.portfolio_drawdown_halted;
  const blocker = resumeBlocker(state, healthError, reviewed);
  return <Card title="Safety controls" icon={ShieldCheck} tone={halted || drawdownHalted ? "bad" : "accent"}
    subtitle="Stop new trades instantly. Recovery needs a review.">
    <div className="control-group">
      <Button icon={PauseCircle} disabled={busy || !state || healthError || state.trading_paused}
        onClick={() => control("pause")}>Pause new entries</Button>
      <Button variant="danger" icon={OctagonX} disabled={busy || !state || healthError}
        onClick={() => control("halt")}>Emergency halt</Button>
    </div>
    <p className="control-note">Pause stops new trades only. Halt also trips the circuit breaker. Stop-losses and exits keep running in both cases.</p>
    <label className="review">
      <input type="checkbox" checked={reviewed} onChange={event => setReviewed(event.target.checked)} />
      <span className="check-box" aria-hidden="true"><CircleCheck size={16} /></span>
      <span>I reviewed the data, risk limits and the paper ledger.</span>
    </label>
    <div className="control-group">
      <Button icon={RotateCcw}
        disabled={busy || healthError || !reviewed || !state?.paper_trading || !state?.trading_paused || !state?.dead_mans_switch?.halted || state?.scan_in_progress}
        onClick={() => control("reset_halt")}>Clear halt, stay paused</Button>
      <Button icon={RotateCcw}
        disabled={busy || healthError || !reviewed || !state?.paper_trading || !state?.trading_paused || !drawdownHalted || state?.scan_in_progress}
        onClick={() => control("reset_drawdown")}>Reset drawdown, stay paused</Button>
      <Button variant="primary" icon={PlayCircle}
        disabled={busy || healthError || !reviewed || !state?.paper_trading || !state?.scheduler_running || !state?.trading_paused || state?.dead_mans_switch?.halted || state?.scan_in_progress}
        onClick={() => control("resume")}>Resume paper entries</Button>
    </div>
    {blocker && <p className="control-note" role="status"><Ban size={14} aria-hidden="true" /> Resume: {blocker}</p>}
    <Hint>Pause and halt survive restarts. Clearing a halt never resumes trading on its own.</Hint>
  </Card>;
}

const OUTCOME_SEGMENTS = [
  ["executed", "Traded", "ok"], ["blocked", "Blocked by a safety rule", "warn"],
  ["failed", "Could not complete", "bad"], ["other", "No result recorded", "neutral"],
];

export function scanOutcomeCounts(scan) {
  if (!scan || scan.decisions == null) return null;
  const counts = { executed: scan.executed || 0, blocked: scan.blocked || 0, failed: scan.failed || 0 };
  counts.other = Math.max(0, scan.decisions - counts.executed - counts.blocked - counts.failed);
  return counts;
}

export function LastScan({ scan, inProgress, now }) {
  const counts = scanOutcomeCounts(scan);
  const total = counts ? Object.values(counts).reduce((sum, value) => sum + value, 0) : 0;
  const tone = !scan ? "neutral" : scan.status === "failed" ? "bad" : scan.status === "degraded" ? "warn" : "ok";
  const statusText = inProgress ? "Running now" : scanStatusLabel(scan);
  return <Card title="India scan, explained" icon={ScanSearch} tone={tone}
    subtitle={scan ? `Finished ${istTime(scan.finished_at)} IST · ${relative(scan.finished_at, now)}` : "No scan has run in this process yet."}
    action={<Chip tone={inProgress ? "accent" : tone}>{statusText}</Chip>}>
    {!scan ? <Empty icon={ScanSearch} title="Nothing to explain yet">The first scheduled scan of the day runs at 09:30 IST.</Empty> : <>
      <div className="scan-numbers">
        <div><strong>{scan.decisions ?? "—"}</strong><span>trade ideas</span></div>
        <div><strong>{scan.orders ?? 0}</strong><span>paper orders filled</span></div>
        <div><strong>{scan.errors ?? 0}</strong><span>reported issues</span></div>
      </div>
      {counts && total > 0 ? <>
        <div className="stackbar" role="img" aria-label={OUTCOME_SEGMENTS.map(([key, label]) => `${label}: ${counts[key]}`).join(", ")}>
          {OUTCOME_SEGMENTS.filter(([key]) => counts[key]).map(([key, , tone]) =>
            <span key={key} style={{ flexGrow: counts[key], background: TONE_COLOR[tone] }} />)}
        </div>
        <ul className="legend">
          {OUTCOME_SEGMENTS.filter(([key]) => counts[key]).map(([key, label, tone]) =>
            <li key={key}><i style={{ background: TONE_COLOR[tone] }} />{label}<b>{counts[key]}</b></li>)}
        </ul>
        {Object.keys(scan.block_reasons || {}).length > 0 && <ul className="reasons">
          {Object.entries(scan.block_reasons).map(([reason, count]) =>
            <li key={reason}><TriangleAlert size={15} aria-hidden="true" />{reason}<b>×{count}</b></li>)}
        </ul>}
      </> : scan.errors > 0 && <Hint>Reported issues include safety checks and service failures. A scan can report issues without producing a trade idea.</Hint>}
      {Object.keys(scan.risk_block_reasons || {}).length > 0 && <>
        <p>Risk checks before trade decisions:</p>
        <ul className="reasons">
          {Object.entries(scan.risk_block_reasons).map(([reason, count]) =>
            <li key={reason}><TriangleAlert size={15} aria-hidden="true" />{reason}<b>×{count}</b></li>)}
        </ul>
      </>}
      {Object.keys(scan.review_block_reasons || {}).length > 0 && <>
        <p>AI review issues before trade decisions:</p>
        <ul className="reasons">
          {Object.entries(scan.review_block_reasons).map(([reason, count]) =>
            <li key={reason}><TriangleAlert size={15} aria-hidden="true" />{reason}<b>×{count}</b></li>)}
        </ul>
      </>}
      <div className="scan-context">
        <span><GaugeIcon size={15} aria-hidden="true" /> India VIX {scan.india_vix > 0 ? scan.india_vix.toFixed(2) : "—"}</span>
        <span><Waves size={15} aria-hidden="true" /> {regimeInfo(scan.hmm_regime).name}</span>
      </div>
    </>}
  </Card>;
}

export function Schedule({ jobs, now, limit = 6 }) {
  const upcoming = (jobs || []).filter(job => job.next_run && job.id !== "check_stops")
    .sort((a, b) => a.next_run.localeCompare(b.next_run)).slice(0, limit);
  return <Card title="India schedule" icon={CalendarClock} subtitle="Upcoming India jobs (IST)">
    {upcoming.length ? <ol className="timeline">
      {upcoming.map((job, index) => <li key={job.id} className={`${isScanJob(job) ? "is-scan" : ""} ${index === 0 ? "is-next" : ""}`}>
        <time dateTime={job.next_run}>{istTime(job.next_run)}</time>
        <div><strong>{job.name.replace(/\s\d{2}:\d{2}(-\d{2}:\d{2})?$/, "")}</strong><p>{JOB_INFO[job.id] || "Scheduled job"}</p></div>
        <small>{relative(job.next_run, now)}</small>
      </li>)}
    </ol> : <Empty icon={CalendarClock} title="No jobs scheduled">The scheduler is not reporting any jobs.</Empty>}
    <Hint>Stop-loss checks also run every 5 minutes during market hours.</Hint>
  </Card>;
}

const GLOSSARY = [
  ["Paper trading", "Orders are simulated against live market prices and recorded in a local ledger. No real money moves."],
  ["Pause vs. halt", "Pause stops new trades. Halt also trips the circuit breaker, which must be reviewed and cleared before you can resume."],
  ["Circuit breaker", "Trips automatically after repeated failures (AI pipeline, broker, data feed) so a broken system can’t keep trading."],
  ["Market regime", "A statistical (HMM) model reads Nifty volatility, momentum and India VIX each day to decide trade size — or block entries in a panic."],
  ["Reconciliation", "Cross-checks every paper fill, position, stop and P&L figure. Any mismatch pauses new entries for review."],
  ["Entry window", "New intraday trades only between 09:30 and 14:45 IST; intraday positions are closed at 15:15."],
];

export function Glossary() {
  return <Card title="How to read this dashboard" icon={BookOpen} className="span-full"
    subtitle="Plain-English meanings of the terms used here.">
    <div className="glossary">
      {GLOSSARY.map(([term, meaning]) => <details key={term}>
        <summary><CircleHelp size={16} aria-hidden="true" />{term}</summary>
        <p>{meaning}</p>
      </details>)}
    </div>
  </Card>;
}

export default function Overview({ state, ops, opsUnavailable, healthError, portfolio, config, session, now,
  busy, reviewed, setReviewed, control, onNavigate }) {
  const checks = readinessChecks({ state, ops, session, opsUnavailable });
  return <div className="view">
    <CommandHero state={state} ops={ops} onNavigate={onNavigate} now={now} />
    <section className="workspace-section">
      <div className="workspace-section-head">
        <div><h2>Markets</h2><p>Equal-sized workspaces; each market keeps its own session, data and scan status.</p></div>
        <button type="button" className="link" onClick={() => onNavigate?.("workspaces")}>All workspaces <ArrowRight size={16} aria-hidden="true" /></button>
      </div>
      <MarketWorkspaceGrid state={state} ops={ops} portfolio={portfolio} session={session} now={now} />
    </section>
    <section className="workspace-section">
      <div className="workspace-section-head">
        <div><h2>Strategy lineup</h2><p>Core, disabled and experimental strategies are visible without changing their settings.</p></div>
        <button type="button" className="link" onClick={() => onNavigate?.("workspaces")}>Browse strategies <ArrowRight size={16} aria-hidden="true" /></button>
      </div>
      <StrategyWorkspaceGrid ops={ops} />
    </section>
    <section className="workspace-section">
      <div className="workspace-section-head">
        <div><h2>India paper book</h2><p>These portfolio figures are India-only; foreign paper and simulated crypto use separate books and currencies.</p></div>
      </div>
      <KpiRow portfolio={portfolio} config={config} />
    </section>
    <div className="grid grid-3">
      <Checklist checks={checks} />
      <SafetyControls state={state} healthError={healthError} busy={busy} reviewed={reviewed}
        setReviewed={setReviewed} control={control} />
    </div>
    <div className="grid grid-2">
      <LastScan scan={ops?.scan?.last || state?.last_scan || state?.scan_status?.last_scan} inProgress={ops?.scan?.in_progress || state?.scan_in_progress} now={now} />
      <Schedule jobs={ops?.jobs} now={now} />
    </div>
    <Glossary />
  </div>;
}
