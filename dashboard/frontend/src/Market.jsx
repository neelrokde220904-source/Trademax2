import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, Pie, PieChart, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { BarChart3, Brain, CandlestickChart, Hourglass, PieChart as PieIcon, RefreshCw, Sparkles, Waves } from "lucide-react";
import {
  ANIMATE, axisProps, CHART, istDateTime, istDay, istTime, pct, precise, REGIMES, regimeInfo, signedPct, tooltipProps,
} from "./lib";
import { Button, Card, Chip, DetailList, Empty, Gauge, Hint, RangeBar, Sparkline } from "./ui";

const RESTART_NOTE = "This panel needs the updated backend. It will appear after the next controlled restart (no data is lost).";

function panicTone(risk) {
  return risk == null ? "neutral" : risk < 0.1 ? "ok" : risk < 0.3 ? "warn" : "bad";
}

export function RegimeCard({ hmm, busy, unavailable, mutate }) {
  const latest = hmm?.latest, fitted = hmm?.fitted, info = regimeInfo(latest?.regime);
  const directive = latest?.directive;
  const tone = !fitted || latest?.degraded ? "warn" : info.tone;
  return <Card title="Market mood (regime)" icon={Waves} tone={tone}
    subtitle="A statistical model reads daily Nifty volatility, momentum and India VIX."
    action={<Chip tone={tone}>{!hmm ? "Unknown" : !fitted ? "Model not fitted" : latest ? info.name : "Not classified yet"}</Chip>}>
    <div className="regime-summary">
      <span className="regime-swatch" style={{ background: info.color }} aria-hidden="true" />
      <div>
        <strong>{latest ? info.name : fitted ? "Waiting for the first classification" : "Entries blocked until the model is fitted"}</strong>
        <p>{info.summary}</p>
      </div>
    </div>
    <div className="gauges">
      <Gauge value={latest?.confidence} tone="accent" label={pct(latest?.confidence)} caption="How sure the model is" />
      <Gauge value={latest?.panic_risk} tone={panicTone(latest?.panic_risk)} label={pct(latest?.panic_risk)}
        caption="Chance of panic tomorrow" />
    </div>
    <div className="effect">
      <span className="eyebrow">What it means for trading</span>
      <p>{directive?.description || info.effect}</p>
      {directive && <div className="chips">
        <Chip tone={directive.allow_new_positions ? "ok" : "bad"}>{directive.allow_new_positions ? "New entries allowed" : "New entries blocked"}</Chip>
        {directive.max_position_size_pct != null && <Chip tone="accent" dot={false}>Max {pct(directive.max_position_size_pct, 0)} per trade</Chip>}
        {directive.hedge_required && <Chip tone="warn">Hedge required</Chip>}
        {(directive.allowed_strategies || []).map(name => <Chip key={name} tone="neutral" dot={false}>{name.replaceAll("_", " ")}</Chip>)}
      </div>}
      {latest?.panic_risk >= 0.3 && <Hint>Panic risk above 30% halves position sizes.</Hint>}
    </div>
    <DetailList items={[
      ["Classified from", latest?.as_of ? `Daily data up to ${latest.as_of}` : "—"],
      ["Model trained", fitted ? `${istDateTime(hmm.trained_at)} · ${hmm.n_observations} trading days (${hmm.data_start} → ${hmm.data_end})` : hmm?.load_error || "—"],
      ["Model age", fitted ? hmm.model_stale ? "Stale — retrain recommended" : "Fresh (within retrain window)" : "—"],
    ]} />
    <div className="control-group">
      <Button icon={RefreshCw} disabled={busy || unavailable || !fitted} onClick={() => mutate("/operations/regime")}>Classify now (no orders)</Button>
    </div>
    <Hint>Exits are never blocked by the regime. Retrain with <code>python3 main.py --train-hmm</code> while the process is stopped.</Hint>
  </Card>;
}

export function RegimeProbabilities({ latest }) {
  const names = Object.keys(REGIMES);
  const data = latest?.regime_probs ? names.map(name => ({
    name: REGIMES[name].name, today: latest.regime_probs[name] || 0,
    tomorrow: latest.transition_probs?.[name] || 0, color: REGIMES[name].color,
  })) : [];
  return <Card title="Regime odds" icon={BarChart3} subtitle="Probability of each market mood today and tomorrow">
    {data.length ? <>
      <div className="chart" style={{ height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }} barGap={4}>
            <CartesianGrid horizontal={false} stroke={CHART.grid} />
            <XAxis type="number" domain={[0, 1]} tickFormatter={value => `${Math.round(value * 100)}%`} {...axisProps} />
            <YAxis type="category" dataKey="name" width={150} {...axisProps} />
            <Tooltip {...tooltipProps} formatter={(value, key) => [pct(value), key === "today" ? "Today" : "Tomorrow"]} />
            <Bar isAnimationActive={ANIMATE} dataKey="today" radius={[0, 8, 8, 0]} barSize={12}>
              {data.map(row => <Cell key={row.name} fill={row.color} />)}
            </Bar>
            <Bar isAnimationActive={ANIMATE} dataKey="tomorrow" radius={[0, 8, 8, 0]} barSize={12} fill={CHART.accentLight} fillOpacity={0.55} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ul className="legend">
        <li><i style={{ background: `linear-gradient(90deg, ${data.map(row => row.color).join(", ")})` }} />Today (coloured by mood)</li>
        <li><i style={{ background: CHART.accentLight, opacity: 0.6 }} />Tomorrow’s outlook</li>
      </ul>
      <RegimeTakeaways latest={latest} />
    </> : <Empty icon={Hourglass} title="Odds not available yet">{latest ? RESTART_NOTE : "Shown after the regime is classified."}</Empty>}
  </Card>;
}

function RegimeTakeaways({ latest }) {
  const tomorrow = Object.entries(latest.transition_probs || {}).sort((a, b) => b[1] - a[1])[0];
  const stay = latest.transition_probs?.[latest.regime];
  return <DetailList items={[
    tomorrow && ["Most likely tomorrow", `${regimeInfo(tomorrow[0]).name} · ${pct(tomorrow[1], 0)}`],
    Number.isFinite(stay) && ["Chance the mood changes by tomorrow", pct(1 - stay, 0)],
  ]} />;
}

function regimeSegments(history) {
  const segments = [];
  for (const row of history) {
    const last = segments.at(-1);
    if (last && last.regime === row.regime) last.end = row.date;
    else segments.push({ regime: row.regime, start: row.date, end: row.date });
  }
  return segments;
}

export function RegimeHistory({ history }) {
  const rows = history || [];
  const segments = regimeSegments(rows);
  const days = Object.keys(REGIMES).map(name => ({
    name, label: REGIMES[name].name, color: REGIMES[name].color, value: rows.filter(row => row.regime === name).length,
  })).filter(row => row.value);
  return <Card title="Nifty 50, India VIX and market mood" icon={CandlestickChart} className="span-full"
    subtitle={rows.length ? `Last ${rows.length} trading days · background colour = regime the model assigned to each day` : "Daily history used by the regime model"}>
    {rows.length ? <div className="history-layout">
      <div className="chart" style={{ height: 320 }}>
        <ResponsiveContainer>
          <ComposedChart data={rows} margin={{ left: 4, right: 4, top: 8, bottom: 0 }}>
            {segments.map(segment => <ReferenceArea key={segment.start} yAxisId="nifty" x1={segment.start} x2={segment.end}
              fill={regimeInfo(segment.regime).color} fillOpacity={0.14} strokeOpacity={0} ifOverflow="extendDomain" />)}
            <CartesianGrid vertical={false} stroke={CHART.grid} />
            <XAxis dataKey="date" tickFormatter={istDay} minTickGap={28} {...axisProps} />
            <YAxis yAxisId="nifty" domain={["auto", "auto"]} width={64}
              tickFormatter={value => value.toLocaleString("en-IN", { maximumFractionDigits: 0 })} {...axisProps} />
            <YAxis yAxisId="vix" orientation="right" domain={[0, "auto"]} width={36} {...axisProps} />
            <Tooltip {...tooltipProps} labelFormatter={value => istDay(value)}
              formatter={(value, key, item) => key === "nifty_close"
                ? [value.toLocaleString("en-IN"), "Nifty 50"]
                : [value.toFixed(2), `India VIX · ${regimeInfo(item.payload.regime).name}`]} />
            <Line isAnimationActive={ANIMATE} yAxisId="nifty" dataKey="nifty_close" stroke={CHART.accent} strokeWidth={2.5} dot={false} />
            <Line isAnimationActive={ANIMATE} yAxisId="vix" dataKey="india_vix" stroke={CHART.warn} strokeWidth={1.8} strokeDasharray="5 4" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="history-side">
        <div className="chart donut" style={{ height: 180 }}>
          <ResponsiveContainer>
            <PieChart>
              <Pie isAnimationActive={ANIMATE} data={days} dataKey="value" nameKey="label" innerRadius={52} outerRadius={78} paddingAngle={3} stroke="none">
                {days.map(row => <Cell key={row.name} fill={row.color} />)}
              </Pie>
              <Tooltip {...tooltipProps} formatter={(value, name) => [`${value} days`, name]} />
            </PieChart>
          </ResponsiveContainer>
          <div className="donut-center"><strong>{rows.length}</strong><small>days</small></div>
        </div>
        <ul className="legend vertical">
          <li><i style={{ background: CHART.accent }} />Nifty 50 close (left axis)</li>
          <li><i className="dashed" style={{ borderColor: CHART.warn }} />India VIX (right axis)</li>
          {days.map(row => <li key={row.name}><i style={{ background: row.color }} />{row.label}<b>{row.value} d</b></li>)}
        </ul>
      </div>
    </div> : <Empty icon={Hourglass} title="History not available yet">{RESTART_NOTE} It is built from the same daily data the model already downloads — no extra broker calls.</Empty>}
  </Card>;
}

function MoversChart({ rows }) {
  const data = rows.filter(row => Number.isFinite(row.change_pct))
    .sort((a, b) => b.change_pct - a.change_pct)
    .map(row => ({ symbol: row.symbol, change: row.change_pct }));
  if (!data.length) return <Empty icon={Hourglass} title="No prices yet">Prices appear once the live feed delivers ticks.</Empty>;
  return <div className="chart" style={{ height: Math.max(260, data.length * 24) }}>
    <ResponsiveContainer>
      <BarChart data={data} layout="vertical" margin={{ left: 4, right: 16, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke={CHART.grid} />
        <XAxis type="number" tickFormatter={value => `${value}%`} {...axisProps} />
        <YAxis type="category" dataKey="symbol" width={96} interval={0} {...axisProps} />
        <ReferenceLine x={0} stroke={CHART.muted} />
        <Tooltip {...tooltipProps} formatter={value => [signedPct(value), "Change vs previous close"]} />
        <Bar isAnimationActive={ANIMATE} dataKey="change" radius={6} barSize={14}>
          {data.map(row => <Cell key={row.symbol} fill={row.change >= 0 ? CHART.teal : CHART.bad} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  </div>;
}

function Breadth({ rows }) {
  const priced = rows.filter(row => Number.isFinite(row.change_pct));
  const data = [
    { name: "Rising", value: priced.filter(row => row.change_pct > 0).length, color: CHART.teal },
    { name: "Falling", value: priced.filter(row => row.change_pct < 0).length, color: CHART.bad },
    { name: "Flat", value: priced.filter(row => row.change_pct === 0).length, color: "#A3B1C6" },
  ].filter(row => row.value);
  const average = priced.length ? priced.reduce((sum, row) => sum + row.change_pct, 0) / priced.length : null;
  if (!priced.length) return null;
  const ranked = [...priced].sort((a, b) => b.change_pct - a.change_pct);
  return <div className="breadth">
    <div className="chart donut" style={{ height: 200 }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie isAnimationActive={ANIMATE} data={data} dataKey="value" nameKey="name" innerRadius={58} outerRadius={86} paddingAngle={3} stroke="none">
            {data.map(row => <Cell key={row.name} fill={row.color} />)}
          </Pie>
          <Tooltip {...tooltipProps} formatter={(value, name) => [`${value} stocks`, name]} />
        </PieChart>
      </ResponsiveContainer>
      <div className="donut-center"><strong className={average >= 0 ? "ink-ok" : "ink-bad"}>{signedPct(average)}</strong><small>average</small></div>
    </div>
    <ul className="legend vertical">{data.map(row => <li key={row.name}><i style={{ background: row.color }} />{row.name}<b>{row.value}</b></li>)}</ul>
    <DetailList items={[
      ["Strongest today", `${ranked[0].symbol} ${signedPct(ranked[0].change_pct)}`],
      ["Weakest today", `${ranked.at(-1).symbol} ${signedPct(ranked.at(-1).change_pct)}`],
    ]} />
  </div>;
}

export function MarketWatch({ watchlist, feed }) {
  const rows = watchlist.data?.symbols || [];
  if (watchlist.missing || (!watchlist.data && watchlist.error)) {
    return <Card title="Watchlist prices" icon={Sparkles} className="span-full"
      subtitle={feed ? `Feed ${feed.connected ? "connected" : "disconnected"} · ${feed.tokens_seen ?? 0}/${feed.subscribed_tokens ?? 0} stocks ticking` : "Live Angel One feed"}>
      <Empty icon={Hourglass} title={watchlist.missing ? "Price charts arrive after the next controlled restart" : "Prices unavailable"}>
        {watchlist.missing ? "The running backend predates the market-watch endpoint. The feed itself is healthy if Operations shows fresh ticks." : "The backend did not answer. Retrying automatically."}
      </Empty>
    </Card>;
  }
  const fresh = rows.filter(row => row.fresh).length;
  return <>
    <Card title="Today’s movers" icon={BarChart3} subtitle="Change since yesterday’s close, best to worst">
      <MoversChart rows={rows} />
    </Card>
    <Card title="Market breadth" icon={PieIcon} subtitle="How many watchlist stocks are up vs down">
      {rows.some(row => Number.isFinite(row.change_pct)) ? <Breadth rows={rows} />
        : <Empty icon={Hourglass} title="Waiting for prices">Breadth appears once the feed delivers ticks.</Empty>}
      <Hint>{fresh}/{rows.length} prices refreshed in the last 30 seconds{watchlist.data?.as_of ? ` · checked ${istTime(watchlist.data.as_of)} IST` : ""}.</Hint>
    </Card>
    <Card title="Watchlist prices" icon={Sparkles} className="span-full"
      subtitle="Latest price, change, where it sits in today’s range, and the intraday trend">
      <div className="table-wrap">
        <table>
          <thead><tr><th>Stock</th><th className="num">Last price</th><th className="num">Change</th><th>Day range</th><th>Trend today</th><th className="num">Updated</th></tr></thead>
          <tbody>
            {rows.map(row => <tr key={row.symbol}>
              <td><b>{row.symbol}</b>{!row.resolved && <small className="ink-warn">Not resolved</small>}</td>
              <td className="num">{precise(row.ltp)}</td>
              <td className={`num ${row.change_pct > 0 ? "ink-ok" : row.change_pct < 0 ? "ink-bad" : ""}`}>{signedPct(row.change_pct)}</td>
              <td><RangeBar low={row.low} high={row.high} value={row.ltp} open={row.open} />
                {Number.isFinite(row.low) && <small className="range-labels"><span>{precise(row.low)}</span><span>{precise(row.high)}</span></small>}</td>
              <td><Sparkline points={row.sparkline} /></td>
              <td className="num"><Chip tone={row.fresh ? "ok" : row.has_tick ? "warn" : "neutral"}>{row.tick_age_seconds != null ? `${Math.round(row.tick_age_seconds)}s` : "—"}</Chip></td>
            </tr>)}
          </tbody>
        </table>
      </div>
      <Hint>Trend lines start when the dashboard backend starts and keep one price per minute. Range bar: dot = last price, tick = today’s open.</Hint>
    </Card>
  </>;
}

export default function Market({ ops, opsUnavailable, watchlist, busy, mutate }) {
  const hmm = ops?.hmm;
  return <div className="view">
    <div className="view-head"><h1>Market</h1><p>What the market is doing and how the bot is adapting to it.</p></div>
    <div className="grid grid-2">
      <RegimeCard hmm={hmm} busy={busy} unavailable={opsUnavailable} mutate={mutate} />
      <RegimeProbabilities latest={hmm?.latest} />
    </div>
    <RegimeHistory history={hmm?.history} />
    <div className="grid grid-2">
      <MarketWatch watchlist={watchlist} feed={ops?.feed} />
    </div>
    {!hmm && <Empty icon={Brain} title="Regime data unavailable">Waiting for operations data from the backend.</Empty>}
  </div>;
}
