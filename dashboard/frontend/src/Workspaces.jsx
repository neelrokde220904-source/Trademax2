import { useState } from "react";
import {
  Activity, Briefcase, CandlestickChart, Globe, TrendingUp,
} from "lucide-react";
import { isScanJob, istDateTime, relative } from "./lib";
import { Card, Chip, DetailList, Empty, Hint } from "./ui";

const STRATEGY_INFO = {
  CompositeTrend: {
    title: "Composite Trend",
    description: "Core NSE intraday strategy with technical signals, AI review and portfolio risk gates.",
  },
  ORBIntraday: {
    title: "Opening-range breakout",
    description: "Intraday breakout sleeve with volume, VWAP and RSI confirmation.",
  },
  SupertrendADX: {
    title: "Supertrend + ADX",
    description: "15-minute trend-following sleeve.",
  },
  MeanReversion: {
    title: "Mean reversion",
    description: "Daily Bollinger/RSI swing strategy.",
  },
  MomentumBreakout: {
    title: "Momentum breakout",
    description: "Daily 52-week-high swing strategy.",
  },
  VCPNR7Swing: {
    title: "VCP / NR7",
    description: "Daily volatility-contraction breakout sleeve.",
  },
  BTSTStrength: {
    title: "BTST strength",
    description: "Late-session strength with a planned next-session exit.",
  },
  ETFRotation: {
    title: "ETF rotation",
    description: "Monthly positional rotation across eligible ETFs.",
  },
};

const FILTERS = [
  ["ALL", "All"],
  ["INTRADAY", "Intraday"],
  ["SWING", "Swing"],
  ["BTST", "BTST"],
  ["POSITIONAL", "Positional"],
  ["FOREIGN", "Foreign markets"],
];

function isEntryPaused(state) {
  return !state || !state.paper_trading || state.trading_paused
    || state.dead_mans_switch?.halted
    || state.portfolio_risk?.daily_loss_halted
    || state.portfolio_risk?.portfolio_drawdown_halted;
}

function nextIndiaScan(jobs) {
  return (jobs || []).filter(isScanJob).filter(job => job.next_run)
    .sort((a, b) => a.next_run.localeCompare(b.next_run))[0];
}

function nextRunLabel(value, now) {
  return value ? `${istDateTime(value)} IST · ${relative(value, now)}` : "Not scheduled";
}

function foreignStatus(enabled, state, mode) {
  if (enabled !== true) return { label: enabled === false ? "Disabled" : "Unconfirmed", tone: "neutral" };
  if (!state) return { label: "Unconfirmed", tone: "neutral" };
  if (!state.paper_trading) return { label: "Not paper", tone: "bad" };
  if (isEntryPaused(state)) return { label: "Entries paused", tone: "warn" };
  return { label: `Experimental · ${mode}`, tone: "accent" };
}

function lastRunSummary(run) {
  if (!run?.at) return "No scan recorded";
  const symbolReason = Object.values(run.symbols || {})
    .map(row => row?.reason)
    .find(Boolean);
  const reason = symbolReason || run.reason;
  if (reason === "entry gates clear") return "Checks completed";
  if (!reason) return run.status?.replaceAll("_", " ") || "Result unavailable";
  return reason.replaceAll("_", " ");
}

function readablePolicy(value) {
  return value ? value.replace(/_v\d+$/, "").replaceAll("_", " ") : "See strategy contract";
}

function MarketCard({ title, icon: Icon, subtitle, status, tone, items }) {
  return <Card
    title={title}
    icon={Icon}
    subtitle={subtitle}
    tone={tone}
    action={<Chip tone={tone}>{status}</Chip>}
    className="workspace-card market-workspace-card"
  >
    <DetailList items={items} />
  </Card>;
}

export function MarketWorkspaceGrid({ state, ops, portfolio, session, now }) {
  const paused = isEntryPaused(state);
  const indiaJob = nextIndiaScan(ops?.jobs);
  const indiaTone = !state ? "neutral" : !state.paper_trading ? "bad" : paused ? "warn" : "ok";
  const indiaStatus = !state ? "Unconfirmed"
    : !state.paper_trading ? "Not paper"
      : paused ? "Entries paused" : "Paper · checks apply";
  const us = ops?.foreign_markets?.us;
  const crypto = ops?.foreign_markets?.crypto;
  const usStatus = foreignStatus(us?.enabled, state, "paper");
  const cryptoStatus = foreignStatus(crypto?.enabled, state, "simulated");
  const forexItems = [
    ["Connection", "Not configured"],
    ["Trading", "No currency-pair orders are supported"],
    ["Next step", "A broker, market-data source and strategy are required"],
  ];

  return <div className="grid grid-4 workspace-grid">
    <MarketCard
      title="India equities"
      icon={CandlestickChart}
      subtitle="NSE cash · CompositeTrend"
      status={indiaStatus}
      tone={indiaTone}
      items={[
        ["Session", session?.label || "Unconfirmed"],
        ["Price feed", ops?.feed?.fresh ? "Fresh"
          : ops?.feed?.connected ? "Connected · stale" : "Disconnected"],
        ["Open positions", portfolio?.position_count ?? "Unavailable"],
        ["Next scan", indiaJob ? nextRunLabel(indiaJob.next_run, now) : "Not scheduled"],
      ]}
    />
    <MarketCard
      title="US equities"
      icon={Briefcase}
      subtitle="NYSE / Nasdaq · Alpaca paper"
      status={usStatus.label}
      tone={usStatus.tone}
      items={[
        ["Execution", "Alpaca paper only · LIMIT orders"],
        ["Strategy", "Long-only 20/50 trend"],
        ["Watchlist", us?.watchlist?.join(", ") || "Unavailable"],
        ["Last scan", us?.last_run?.at
          ? `${lastRunSummary(us.last_run)} · ${istDateTime(us.last_run.at)}`
          : "No scan recorded"],
        ["Next scan", nextRunLabel(us?.next_run, now)],
      ]}
    />
    <MarketCard
      title="Crypto"
      icon={Activity}
      subtitle="24/7 spot pairs · local simulation"
      status={cryptoStatus.label}
      tone={cryptoStatus.tone}
      items={[
        ["Execution", "Local simulator · no exchange orders"],
        ["Strategy", "Long-only 20/50 trend"],
        ["Watchlist", crypto?.watchlist?.join(", ") || "Unavailable"],
        ["Last scan", crypto?.last_run?.at
          ? `${lastRunSummary(crypto.last_run)} · ${istDateTime(crypto.last_run.at)}`
          : "No scan recorded"],
        ["Next scan", nextRunLabel(crypto?.next_run, now)],
      ]}
    />
    <MarketCard
      title="Forex"
      icon={Globe}
      subtitle="Currency pairs"
      status="Not connected"
      tone="neutral"
      items={forexItems}
    />
  </div>;
}

function strategyRows(ops) {
  const indiaRows = (ops?.sleeves || []).map(strategy => {
    const info = STRATEGY_INFO[strategy.name] || {};
    const category = strategy.horizons?.[0] || "OTHER";
    const horizon = ({
      INTRADAY: "Intraday",
      SWING: "Swing",
      BTST: "BTST",
      POSITIONAL: "Positional",
    })[category] || category;
    const core = strategy.name === "CompositeTrend";
    return {
      id: strategy.name,
      title: info.title || strategy.name,
      market: "India equities",
      horizon,
      category,
      description: info.description || "Registered strategy sleeve.",
      status: core ? "Core · active" : strategy.enabled ? "Enabled · experimental" : "Disabled",
      tone: core ? "ok" : strategy.enabled ? "warn" : "neutral",
      execution: "NSE paper",
      control: core ? "Always enabled" : strategy.flag || "Feature flag",
      exitPolicy: strategy.exit_policy,
    };
  });
  const us = ops?.foreign_markets?.us;
  const crypto = ops?.foreign_markets?.crypto;
  const usEnabled = us?.enabled === true;
  const cryptoEnabled = crypto?.enabled === true;
  return [
    ...indiaRows,
    {
      id: "us-paper-trend",
      title: "US 20/50 trend",
      market: "US equities",
      horizon: "US session",
      category: "FOREIGN",
      description: "Long-only trend strategy using 15-minute IEX bars and fresh quotes.",
      status: !us ? "Unconfirmed" : usEnabled ? "Experimental · enabled" : "Disabled",
      tone: !us ? "neutral" : usEnabled ? "warn" : "neutral",
      execution: "Alpaca paper",
      control: "FOREIGN_US_PAPER_ENABLED",
    },
    {
      id: "crypto-spot-trend",
      title: "Crypto 20/50 trend",
      market: "Crypto",
      horizon: "24/7 spot",
      category: "FOREIGN",
      description: "Long-only simulator using completed 1-hour candles and public prices.",
      status: !crypto ? "Unconfirmed" : cryptoEnabled ? "Experimental · enabled" : "Disabled",
      tone: !crypto ? "neutral" : cryptoEnabled ? "warn" : "neutral",
      execution: "Local simulation",
      control: "CRYPTO_SIM_ENABLED",
    },
  ];
}

function StrategyCard({ strategy }) {
  return <Card
    title={strategy.title}
    icon={TrendingUp}
    subtitle={strategy.description}
    tone={strategy.tone}
    action={<Chip tone={strategy.tone}>{strategy.status}</Chip>}
    className="workspace-card strategy-workspace-card"
  >
    <DetailList items={[
      ["Market", strategy.market],
      ["Horizon", strategy.horizon],
      ["Exit policy", readablePolicy(strategy.exitPolicy)],
      ["Execution", strategy.execution],
      ["Control", strategy.control],
    ]} />
  </Card>;
}

export function StrategyWorkspaceGrid({ ops }) {
  const [filter, setFilter] = useState("ALL");
  const rows = strategyRows(ops);
  const visible = filter === "ALL" ? rows : rows.filter(row => row.category === filter);
  return <div className="strategy-workspaces">
    <div className="workspace-filter" role="group" aria-label="Filter strategies by horizon or market">
      {FILTERS.map(([value, label]) => <button
        key={value}
        type="button"
        className={`workspace-filter-btn ${filter === value ? "is-selected" : ""}`}
        aria-pressed={filter === value}
        onClick={() => setFilter(value)}
      >{label}</button>)}
    </div>
    {visible.length
      ? <div className="grid grid-4 workspace-grid">{visible.map(strategy =>
        <StrategyCard key={strategy.id} strategy={strategy} />)}</div>
      : <Empty icon={TrendingUp} title="Strategy catalogue unavailable">
        Waiting for the backend strategy contracts.
      </Empty>}
    <Hint>
      Card size shows visibility, not allocation. Disabled and experimental strategies remain off or unvalidated; this view does not change trading settings.
    </Hint>
  </div>;
}

export default function Workspaces({ state, ops, portfolio, session, now }) {
  const [active, setActive] = useState("markets");
  return <div className="view">
    <div className="view-head">
      <h1>Markets &amp; strategies</h1>
      <p>Separate workspaces, equal visibility. Status, schedules and safeguards remain market-specific.</p>
    </div>
    <div className="workspace-tabs" role="group" aria-label="Choose workspace view">
      <button type="button" className={active === "markets" ? "is-selected" : ""}
        aria-pressed={active === "markets"} onClick={() => setActive("markets")}>
        <Globe size={17} aria-hidden="true" /> Markets
      </button>
      <button type="button" className={active === "strategies" ? "is-selected" : ""}
        aria-pressed={active === "strategies"} onClick={() => setActive("strategies")}>
        <TrendingUp size={17} aria-hidden="true" /> Strategies
      </button>
    </div>
    {active === "markets"
      ? <MarketWorkspaceGrid state={state} ops={ops} portfolio={portfolio} session={session} now={now} />
      : <StrategyWorkspaceGrid ops={ops} />}
    {active === "markets" && <Hint>
      Forex is shown for completeness but is not connected. US and crypto workspaces are experimental; an enabled schedule does not mean a trade was placed.
    </Hint>}
  </div>;
}
