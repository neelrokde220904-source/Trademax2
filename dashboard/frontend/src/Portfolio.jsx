import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { BarChart3, Briefcase, Lightbulb, LineChart, PieChart as PieIcon, Receipt, Scale } from "lucide-react";
import { ANIMATE, axisProps, CHART, istDateTime, istDay, money, pct, precise, signed, tooltipProps } from "./lib";
import { KpiRow } from "./Overview";
import { Card, Chip, Empty, Hint, Meter } from "./ui";

const OUTCOMES = {
  executed: ["Traded", "ok", CHART.teal],
  blocked: ["Blocked by a rule", "warn", CHART.warn],
  failed: ["Could not complete", "bad", CHART.bad],
  analysis_only: ["Analysis only", "accent", CHART.accent],
  not_executed: ["No result recorded", "neutral", "#A3B1C6"],
};
const outcomeOf = row => OUTCOMES[row.outcome] || OUTCOMES.not_executed;

function EquityCurve({ data, initial }) {
  const rows = data || [];
  return <Card title="Portfolio value over time" icon={LineChart} className="span-2"
    subtitle="One point per trading day, recorded after the close">
    {rows.length ? <div className="chart" style={{ height: 280 }}>
      <ResponsiveContainer>
        <AreaChart data={rows} margin={{ left: 4, right: 8, top: 8, bottom: 0 }}>
          <defs>
            <linearGradient id="equityFill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor={CHART.accent} stopOpacity={0.35} />
              <stop offset="1" stopColor={CHART.accent} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke={CHART.grid} />
          <XAxis dataKey="date" tickFormatter={istDay} minTickGap={24} {...axisProps} />
          <YAxis width={72} domain={["auto", "auto"]} tickFormatter={money} {...axisProps} />
          {initial && <ReferenceLine y={initial} stroke={CHART.muted} strokeDasharray="4 4" label={{ value: "Start", fill: CHART.muted, fontSize: 12, position: "insideTopLeft" }} />}
          <Tooltip {...tooltipProps} labelFormatter={istDay} formatter={value => [money(value), "Portfolio value"]} />
          <Area isAnimationActive={ANIMATE} dataKey="value" type="monotone" stroke={CHART.accent} strokeWidth={2.5} fill="url(#equityFill)" />
        </AreaChart>
      </ResponsiveContainer>
    </div> : <Empty icon={LineChart} title="No history yet">The curve starts after the first market-close snapshot (16:00 IST daily summary).</Empty>}
  </Card>;
}

function Allocation({ portfolio }) {
  const positions = portfolio?.positions || [];
  const slices = positions.map((row, index) => ({
    name: row.symbol, value: Math.abs((row.quantity || 0) * (row.ltp || row.avg_price || 0)),
    color: [CHART.accent, CHART.teal, CHART.accentLight, CHART.warn, "#4FD1C5", "#B794F4", "#F6AD55", "#63B3ED"][index % 8],
  })).filter(row => row.value > 0);
  const cash = portfolio?.available_capital || 0;
  const data = [...slices, ...(cash > 0 ? [{ name: "Cash", value: cash, color: "#C3CBD6" }] : [])];
  const total = data.reduce((sum, row) => sum + row.value, 0);
  return <Card title="Where the money is" icon={PieIcon} subtitle="Cash vs. open positions at the latest price">
    {total > 0 ? <>
      <div className="chart donut" style={{ height: 200 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie isAnimationActive={ANIMATE} data={data} dataKey="value" nameKey="name" innerRadius={60} outerRadius={88} paddingAngle={slices.length ? 3 : 0} stroke="none">
              {data.map(row => <Cell key={row.name} fill={row.color} />)}
            </Pie>
            <Tooltip {...tooltipProps} formatter={(value, name) => [`${money(value)} · ${pct(value / total)}`, name]} />
          </PieChart>
        </ResponsiveContainer>
        <div className="donut-center"><strong>{pct(cash / total, 0)}</strong><small>in cash</small></div>
      </div>
      <ul className="legend vertical">{data.map(row => <li key={row.name}><i style={{ background: row.color }} />{row.name}<b>{money(row.value)}</b></li>)}</ul>
    </> : <Empty icon={PieIcon} title="No capital data">Waiting for the portfolio tracker.</Empty>}
  </Card>;
}

function DailyPnl({ data }) {
  const rows = data || [];
  const wins = rows.reduce((sum, row) => sum + (row.wins || 0), 0), losses = rows.reduce((sum, row) => sum + (row.losses || 0), 0);
  return <Card title="Profit & loss by day" icon={BarChart3}
    subtitle={rows.length ? `${rows.length} days · ${wins} winning and ${losses} losing trades` : "Green = profit, red = loss"}>
    {rows.length ? <div className="chart" style={{ height: 240 }}>
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ left: 4, right: 8, top: 8, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke={CHART.grid} />
          <XAxis dataKey="date" tickFormatter={istDay} minTickGap={16} {...axisProps} />
          <YAxis width={64} tickFormatter={money} {...axisProps} />
          <ReferenceLine y={0} stroke={CHART.muted} />
          <Tooltip {...tooltipProps} labelFormatter={istDay} formatter={value => [signed(value, precise), "P&L"]} />
          <Bar isAnimationActive={ANIMATE} dataKey="pnl" radius={[8, 8, 8, 8]} maxBarSize={28}>
            {rows.map(row => <Cell key={row.date} fill={row.pnl >= 0 ? CHART.teal : CHART.bad} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div> : <Empty icon={BarChart3} title="No closed days yet">Daily results are recorded after each market close.</Empty>}
  </Card>;
}

function RiskLimits({ portfolio, config }) {
  const total = portfolio?.total_value || config?.initial_capital || 0;
  const invested = portfolio ? Math.max(0, total - (portfolio.available_capital || 0)) : 0;
  const lossLimit = config ? total * config.max_daily_loss_pct : null;
  const todayLoss = Math.max(0, -(portfolio?.daily_pnl || 0));
  return <Card title="Risk limits" icon={Scale} subtitle="How much room the bot has left today">
    <div className="meters">
      <Meter label="Open positions" value={portfolio?.position_count || 0} max={config?.max_positions || 1}
        detail={`${portfolio?.position_count ?? "—"} of ${config?.max_positions ?? "—"}`}
        tone={(portfolio?.position_count || 0) >= (config?.max_positions || Infinity) ? "bad" : "accent"} />
      <Meter label="Capital in use" value={invested} max={total || 1} detail={`${money(invested)} of ${money(total)}`} tone="ok" />
      <Meter label="Today’s loss vs. daily limit" value={todayLoss} max={lossLimit || 1}
        detail={lossLimit ? `${money(todayLoss)} of ${money(lossLimit)}` : "—"}
        tone={lossLimit && todayLoss / lossLimit > 0.75 ? "bad" : todayLoss > 0 ? "warn" : "ok"} />
    </div>
    <Hint>{config ? `Each trade can use at most ${pct(config.max_capital_per_trade_pct, 0)} of capital (${money(total * config.max_capital_per_trade_pct)}); the market regime may lower this further. Trading stops for the day at a ${pct(config.max_daily_loss_pct, 0)} loss.` : "Risk configuration unavailable."}</Hint>
  </Card>;
}

function Positions({ rows }) {
  return <Card title="Open positions" icon={Briefcase} className="span-full" subtitle="Live P&L on simulated holdings">
    {rows.length ? <div className="table-wrap"><table>
      <thead><tr><th>Stock</th><th className="num">Qty</th><th className="num">Avg. cost</th><th className="num">Last price</th><th className="num">P&L</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.symbol}>
        <td><b>{row.symbol}</b><small>{row.product}</small></td>
        <td className="num">{row.quantity}</td><td className="num">{precise(row.avg_price)}</td><td className="num">{precise(row.ltp)}</td>
        <td className={`num ${row.pnl >= 0 ? "ink-ok" : "ink-bad"}`}>{signed(row.pnl, precise)}</td>
      </tr>)}</tbody>
    </table></div> : <Empty icon={Briefcase} title="No open positions">The bot is fully in cash. Positions appear here as soon as a paper order fills.</Empty>}
  </Card>;
}

function ReviewContract({ report }) {
  const agents = Object.entries(report.required_agents || {});
  const inputs = Object.entries(report.required_inputs || {});
  const failed = report.failed_requirements || [];
  return <details className="review-contract">
    <summary>AI review contract {report.contract_version ? `(${report.contract_version})` : ""}: {report.passed ? "all requirements passed" : `${failed.length} failed`}</summary>
    <ul>
      {agents.map(([name, ok]) => <li key={`a-${name}`} className={ok ? "ink-ok" : "ink-bad"}>{ok ? "✓" : "✗"} review: {name.replaceAll("_", " ")}</li>)}
      {inputs.map(([name, ok]) => <li key={`i-${name}`} className={ok ? "ink-ok" : "ink-bad"}>{ok ? "✓" : "✗"} input: {name.replaceAll("_", " ")}</li>)}
    </ul>
  </details>;
}

function Decisions({ signals }) {
  const rows = signals.data || [];
  const counts = Object.entries(OUTCOMES).map(([key, [label, , color]]) => ({
    key, label, color, value: rows.filter(row => (OUTCOMES[row.outcome] ? row.outcome : "not_executed") === key).length,
  })).filter(row => row.value);
  return <Card title="What the AI decided" icon={Lightbulb} className="span-full"
    subtitle="Every trade idea from the portfolio manager and what happened to it">
    {rows.length ? <div className="decisions-layout">
      <div>
        <div className="chart donut" style={{ height: 190 }}>
          <ResponsiveContainer>
            <PieChart>
              <Pie isAnimationActive={ANIMATE} data={counts} dataKey="value" nameKey="label" innerRadius={56} outerRadius={82} paddingAngle={3} stroke="none">
                {counts.map(row => <Cell key={row.key} fill={row.color} />)}
              </Pie>
              <Tooltip {...tooltipProps} formatter={(value, name) => [`${value} ideas`, name]} />
            </PieChart>
          </ResponsiveContainer>
          <div className="donut-center"><strong>{rows.length}</strong><small>ideas</small></div>
        </div>
        <ul className="legend vertical">{counts.map(row => <li key={row.key}><i style={{ background: row.color }} />{row.label}<b>{row.value}</b></li>)}</ul>
      </div>
      <ul className="decision-list">
        {rows.slice(0, 12).map(row => {
          const [label, tone] = outcomeOf(row);
          return <li key={row.id}>
            <span className={`side side-${row.signal?.includes("SELL") ? "sell" : row.signal?.includes("BUY") ? "buy" : "hold"}`}>{row.signal}</span>
            <div>
              <strong>{row.symbol}{row.quantity ? <small> · {row.quantity} @ {precise(row.price)}</small> : null}</strong>
              <p>{row.block_reason || row.reasoning || "No rationale recorded."}</p>
              {row.review_contract ? <ReviewContract report={row.review_contract} /> : null}
              <time>{istDateTime(row.created_at)}{row.confidence != null ? ` · confidence ${Math.round(row.confidence)}%` : ""}</time>
            </div>
            <Chip tone={tone}>{label}</Chip>
          </li>;
        })}
      </ul>
    </div> : <Empty icon={Lightbulb} title={signals.error ? "Decisions unavailable" : "No decisions recorded yet"}>
      Trade ideas and their outcomes are saved from the next controlled restart onward. Earlier scans were only summarised in logs.
    </Empty>}
  </Card>;
}

function Ledger({ rows }) {
  return <Card title="Execution ledger" icon={Receipt} className="span-full" subtitle="Every simulated order the bot placed">
    {rows.length ? <div className="table-wrap"><table>
      <thead><tr><th>Time (IST)</th><th>Stock</th><th>Side</th><th className="num">Price</th><th className="num">Qty</th><th>Strategy</th><th>Status</th><th className="num">P&L</th></tr></thead>
      <tbody>{rows.slice(0, 25).map(row => <tr key={row.id}>
        <td>{istDateTime(row.created_at)}</td><td><b>{row.symbol}</b></td>
        <td><span className={`side side-${row.action === "BUY" ? "buy" : "sell"}`}>{row.action}</span></td>
        <td className="num">{precise(row.price)}</td><td className="num">{row.quantity}</td>
        <td>{row.strategy || "Pipeline"}</td><td><Chip tone="neutral" dot={false}>{row.status}</Chip></td>
        <td className={`num ${(row.pnl || 0) >= 0 ? "ink-ok" : "ink-bad"}`}>{row.pnl == null ? "—" : signed(row.pnl, precise)}</td>
      </tr>)}</tbody>
    </table></div> : <Empty icon={Receipt} title="No paper orders yet">Filled paper orders are listed here with their price, status and realised P&L.</Empty>}
  </Card>;
}

export default function Portfolio({ portfolio, config, equity, pnl, trades, signals }) {
  return <div className="view">
    <div className="view-head"><h1>Portfolio</h1><p>Simulated capital, positions and every decision the bot made.</p></div>
    <KpiRow portfolio={portfolio} config={config} />
    <div className="grid grid-3">
      <EquityCurve data={equity} initial={config?.initial_capital} />
      <Allocation portfolio={portfolio} />
    </div>
    <div className="grid grid-2">
      <DailyPnl data={pnl} />
      <RiskLimits portfolio={portfolio} config={config} />
    </div>
    <div className="grid grid-1">
      <Positions rows={portfolio?.positions || []} />
      <Decisions signals={signals} />
      <Ledger rows={trades || []} />
    </div>
  </div>;
}
