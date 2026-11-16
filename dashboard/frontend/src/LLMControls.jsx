import { useCallback, useEffect, useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { AlertTriangle, BrainCircuit, Coins, Cpu, Zap } from "lucide-react";
import { ANIMATE, CHART, istDateTime, tooltipProps } from "./lib";
import { Card, Chip, DetailList, Empty, Hint } from "./ui";

const labels = { claude: "Claude", economy: "Economy" };
const blurbs = {
  claude: "Anthropic models — highest reasoning quality, highest cost.",
  economy: "DeepSeek models — much cheaper; fast model for analysis, Pro model for decisions.",
};
const modeColors = { claude: CHART.accent, economy: CHART.teal };
const rupees = value => new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 2 }).format(value || 0);
const count = value => (value || 0).toLocaleString("en-IN");

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`/api/${path}`, { ...options, signal: controller.signal });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Model service unavailable.");
    return body;
  } finally {
    clearTimeout(timeout);
  }
}

function CostCard({ usage, usageError }) {
  const modes = Object.keys(labels).map(key => ({ key, name: labels[key], ...(usage?.modes?.[key] || {}) }));
  const slices = modes.filter(mode => mode.estimated_inr > 0).map(mode => ({ name: mode.name, value: mode.estimated_inr, color: modeColors[mode.key] }));
  return <Card title={`AI cost · ${usage?.month || "this month"}`} icon={Coins} subtitle="Estimated from token counts and published prices"
    action={usage?.incomplete || usageError ? <Chip tone="warn">{usageError ? "Stale" : "Partly unknown"}</Chip> : <Chip tone="ok">Up to date</Chip>}>
    {usage ? <>
      <div className="cost-layout">
        <div className="chart donut" style={{ height: 190 }}>
          <ResponsiveContainer>
            <PieChart>
              <Pie isAnimationActive={ANIMATE} data={slices.length ? slices : [{ name: "No spend", value: 1, color: CHART.track }]} dataKey="value" nameKey="name"
                innerRadius={58} outerRadius={84} paddingAngle={slices.length > 1 ? 3 : 0} stroke="none">
                {(slices.length ? slices : [{ name: "No spend", color: CHART.track }]).map(row => <Cell key={row.name} fill={row.color} />)}
              </Pie>
              {slices.length > 0 && <Tooltip {...tooltipProps} formatter={(value, name) => [rupees(value), name]} />}
            </PieChart>
          </ResponsiveContainer>
          <div className="donut-center"><strong>{rupees(usage.estimated_inr)}</strong><small>total</small></div>
        </div>
        <DetailList items={[
          ["In US dollars", `$${usage.estimated_usd.toFixed(4)} at ₹${usage.usd_inr_rate}/USD`],
          ["Spending cap", usage.hard_limit ? rupees(usage.hard_limit) : "None configured"],
          ["Updated", `${istDateTime(usage.updated_at)} IST`],
        ]} />
      </div>
      <div className="table-wrap"><table>
        <thead><tr><th>Mode</th><th className="num">Requests</th><th className="num">Tokens in / out</th><th className="num">Cost</th></tr></thead>
        <tbody>{modes.map(mode => <tr key={mode.key}>
          <td className="nowrap"><i className="dot" style={{ background: modeColors[mode.key] }} /><b>{mode.name}</b></td>
          <td className="num">{count(mode.requests)}{mode.unknown_requests ? <small>{mode.unknown_requests} unpriced</small> : null}</td>
          <td className="num">{count(mode.input_tokens)} / {count(mode.output_tokens)}</td>
          <td className="num">{rupees(mode.estimated_inr)}</td>
        </tr>)}</tbody>
      </table></div>
      <Hint>Estimates exclude tax and card/currency fees. {usage.incomplete ? "Some requests had no usable price, so the true cost may be slightly higher." : ""}</Hint>
    </> : <Empty icon={Coins} title={usageError ? "Usage unavailable" : "Loading usage"} />}
  </Card>;
}

export default function LLMControls({ serverState }) {
  const [selection, setSelection] = useState(null);
  const [usage, setUsage] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [offline, setOffline] = useState(false);
  const [usageError, setUsageError] = useState(false);
  const accept = useCallback(next => {
    if (!next) return;
    setSelection(previous => {
      if (previous && (next.revision < previous.revision ||
          (next.revision === previous.revision && next.observed_at < previous.observed_at))) return previous;
      return next;
    });
  }, []);
  const refresh = useCallback(async () => {
    try {
      accept(await request("llm-mode"));
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, [accept]);

  useEffect(() => { accept(serverState); }, [serverState, accept]);
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);
  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const next = await request("llm-usage");
        if (mounted) { setUsage(next); setUsageError(false); }
      } catch {
        if (mounted) setUsageError(true);
      }
    };
    load();
    const timer = setInterval(load, 15000);
    return () => { mounted = false; clearInterval(timer); };
  }, []);

  const changeMode = async mode => {
    if (busy || !selection || mode === selection.selected_mode) return;
    setBusy(true);
    setError("");
    try {
      accept(await request("llm-mode", {
        method: "PUT",
        headers: { "Content-Type": "application/json", "X-Dashboard-Control": "1" },
        body: JSON.stringify({ mode, expected_revision: selection.revision }),
      }));
      setOffline(false);
    } catch (failure) {
      setError(failure.name === "AbortError" ? "Switch timed out. Confirming saved selection." : failure.message);
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const mode = selection?.selected_mode;
  const profile = selection?.profiles?.[mode];
  const unavailable = Object.entries(selection?.profiles || {}).filter(([, value]) => !value.configured).map(([name]) => labels[name]);
  const status = busy ? "Saving…" : offline ? "Offline" : !selection ? "Loading…" : selection.pending ? "Applies next scan" : `${labels[mode]} active`;
  return <div className="view">
    <div className="view-head"><h1>AI model</h1><p>Which language models analyse stocks and make the final call — and what they cost.</p></div>
    <div className="grid grid-2">
      <Card title="Model mode" icon={BrainCircuit} subtitle="Switching takes effect from the next scan; a running scan keeps its model"
        action={<Chip tone={offline ? "bad" : selection?.pending ? "warn" : "ok"}>{status}</Chip>}>
        <div aria-busy={busy}>
          <fieldset className="segmented" disabled={busy || offline || !selection}>
            <legend className="visually-hidden">LLM model mode</legend>
            {Object.entries(labels).map(([value, label]) => {
              const option = selection?.profiles?.[value];
              return <label key={value} className={`${mode === value ? "is-selected" : ""} ${!option?.configured ? "unavailable" : ""}`}>
                <input type="radio" className="visually-hidden" name="llm-mode" value={value} checked={mode === value}
                  disabled={!option?.configured} onChange={() => changeMode(value)} />
                <span className="segment-title">{label}{!option?.configured && <small> · not configured</small>}</span>
                <span className="segment-copy">{blurbs[value]}</span>
              </label>;
            })}
          </fieldset>
          {profile && <DetailList items={[
            ["Analysis model (fast)", <span className="mono"><Zap size={14} aria-hidden="true" /> {profile.fast_model}</span>],
            ["Decision model", <span className="mono"><Cpu size={14} aria-hidden="true" /> {profile.decision_model}</span>],
          ]} />}
          <div className="chips" aria-live="polite">
            {selection?.active_runs?.map(run => <Chip key={run.id} tone={run.degraded ? "warn" : "accent"}>{labels[run.mode]} · {run.kind}{run.degraded ? " · degraded" : " · running"}</Chip>)}
            {selection?.degraded && !selection?.active_runs?.length && <Chip tone="warn">Last run degraded</Chip>}
            {unavailable.length > 0 && <Chip tone="neutral">{unavailable.join(", ")}: no API key</Chip>}
          </div>
          {error && <p className="alert-inline tone-bad" role="alert"><AlertTriangle size={16} aria-hidden="true" />{error}</p>}
          {offline && <p className="alert-inline tone-bad" role="alert"><AlertTriangle size={16} aria-hidden="true" />Model service unavailable. Selection unconfirmed.</p>}
        </div>
        <Hint>“Degraded” means the last run had a failed AI call or any pipeline error — including trades stopped by a safety rule (see Overview → Last scan). Failed AI calls are never retried and never switch provider.</Hint>
      </Card>
      <CostCard usage={usage} usageError={usageError} />
    </div>
  </div>;
}
