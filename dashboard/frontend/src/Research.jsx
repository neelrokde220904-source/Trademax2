import { useState } from "react";
import { BarChart3, CalendarClock, FlaskConical, Play } from "lucide-react";
import { API, istDateTime, money, pct, precise, signedPct, useApi } from "./lib";
import { Card, Chip, Empty, Hint } from "./ui";

export default function Research() {
  const research = useApi("/research", 15000);
  const [selected, setSelected] = useState([]);
  const [acknowledged, setAcknowledged] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const data = research.data;
  const job = data?.job || {};
  const available = (data?.snapshots || []).filter(row => row.status === "complete");
  const forecast = data?.forecast_summary;

  const toggle = id => setSelected(current => current.includes(id)
    ? current.filter(item => item !== id)
    : current.length < 10 ? [...current, id] : current);

  const post = async (path, body) => {
    setBusy(true); setError(""); setMessage("");
    try {
      const response = await fetch(`${API}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Dashboard-Control": "1" },
        body: JSON.stringify(body), signal: AbortSignal.timeout(path.endsWith("settle") ? 120000 : 30000),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Research request failed.");
      if (result.errors?.length) throw new Error(result.errors.join("; "));
      setMessage(path.endsWith("settle")
        ? `Settled ${result.settled}; ${result.pending} still pending.`
        : "Comparison started. Results will appear below when complete.");
    } catch (failure) {
      setError(failure.name === "TimeoutError" ? "Request timed out. Refresh the saved status before retrying." : failure.message);
    } finally { setBusy(false); research.refresh(); }
  };

  return <div className="view">
    <div className="view-head"><h1>Paper research</h1>
      <p>Compare isolated analyst pipelines and track hypothetical next-session forecasts. No broker orders are placed.</p></div>
    {research.error && <div className="banner tone-bad" role="alert">Research data is unavailable; the page will retry.</div>}
    {error && <div className="banner tone-bad" role="alert">{error}</div>}
    {message && <div className="banner tone-ok" role="status">{message}</div>}
    {job.status === "failed" && <div className="banner tone-bad" role="alert">Comparison failed: {job.error || "see saved result"}</div>}
    <div className="grid grid-2">
      <Card title="Compare model variants" icon={FlaskConical} subtitle="Claude, Economy and a rule-based baseline; separate simulated portfolios"
        action={<Chip tone={job.status === "running" ? "warn" : "neutral"}>{job.status === "running" ? "Running" : `${available.length} ready`}</Chip>}>
        {available.length ? <div className="research-snapshots" role="group" aria-label="Captured scans">
          {available.slice(0, 30).map(row => <label key={row.id} className="research-option">
            <input type="checkbox" checked={selected.includes(row.id)} onChange={() => toggle(row.id)}
              disabled={!selected.includes(row.id) && selected.length >= 10} />
            <span>{istDateTime(row.observed_at)} IST <small className="mono">{row.id}</small></span>
          </label>)}
        </div> : <Empty icon={FlaskConical} title="No complete captured scans">New paper scans save point-in-time inputs. At least two are required.</Empty>}
        {(data?.snapshots || []).some(row => row.status !== "complete") && <Hint>Incomplete snapshots cannot be replayed. Check the CLI snapshot list for reasons.</Hint>}
        <label className="research-option"><input type="checkbox" checked={acknowledged} onChange={event => setAcknowledged(event.target.checked)} />
          <span>I understand Claude and Economy API calls can incur additional charges.</span></label>
        <button type="button" className="btn btn-primary" onClick={() => post("/research/compare", {
          snapshot_ids: selected, acknowledge_model_cost: acknowledged,
        })} disabled={busy || job.status === "running" || !data?.paper_trading || !acknowledged || selected.length < 2}>
          <Play size={16} aria-hidden="true" /> Run comparison ({selected.length} selected)
        </button>
        <Hint>Uses saved inputs and current risk settings. A model may know events after the captured date; fills assume zero model latency.</Hint>
      </Card>
      <Card title="Forecast outcomes" icon={CalendarClock} subtitle="First completed NSE session after each forecast">
        <div className="table-wrap"><table><thead><tr><th>Agent</th><th className="num">Scored</th><th className="num">Directional accuracy</th></tr></thead>
          <tbody>{Object.entries(forecast?.by_agent || {}).map(([agent, row]) => <tr key={agent}>
            <td>{agent.replaceAll("_", " ")}</td><td className="num">{row.settled}</td>
            <td className="num">{pct(row.directional_accuracy)}</td></tr>)}</tbody></table></div>
        <p>Pending: {forecast?.counts?.pending || 0} · Unscored: {forecast?.counts?.unscored || 0} · Missing reference: {forecast?.counts?.missing_reference || 0} · Ineligible input: {forecast?.counts?.ineligible_input || 0} · Legacy scores excluded: {forecast?.counts?.legacy_settled || 0}</p>
        <button type="button" className="btn" disabled={busy || !data?.paper_trading}
          onClick={() => post("/research/settle", {})}>Check completed closes</button>
        <Hint>Accuracy excludes legacy outcomes without revalidated capture and reference evidence. HOLD and AVOID are not scored. Confidence is uncalibrated; results are not trade P&amp;L.</Hint>
      </Card>
      <Card title="Comparison results" icon={BarChart3} className="span-full" subtitle="Virtual equity includes estimated costs and may include open positions">
        {(data?.comparisons || []).length ? data.comparisons.slice(0, 5).map(result => <div key={result.id}>
          <p><b>{istDateTime(result.created_at)} IST</b> · {result.status} · {result.snapshot_ids?.length || 0} scans</p>
          {result.error && <p className="ink-bad">{result.error}</p>}
          {!!result.variants?.length && <div className="table-wrap"><table>
            <thead><tr><th>Variant</th><th className="num">Virtual equity</th><th className="num">Return</th><th className="num">Open / closed</th><th className="num">Pending</th></tr></thead>
            <tbody>{result.variants.map(row => <tr key={row.mode}><td>{row.mode}</td>
              <td className="num">{money(row.final_equity)}</td><td className="num">{signedPct(row.return_pct)}</td>
              <td className="num">{row.open_positions.length} / {row.closed_trades.length}</td><td className="num">{row.pending_orders}</td></tr>)}</tbody>
          </table></div>}
        </div>) : <Empty icon={BarChart3} title="No comparisons yet">Select at least two complete paper scans to start.</Empty>}
      </Card>
      <Card title="Recent forecasts" icon={CalendarClock} className="span-full" subtitle="Agent signals, including blocked and analysis-only decisions">
        {forecast?.forecasts?.length ? <div className="table-wrap"><table>
          <thead><tr><th>Observed</th><th>Agent / symbol</th><th>Direction</th><th className="num">Confidence</th><th className="num">Reference</th><th>Outcome</th><th>Trade context</th></tr></thead>
          <tbody>{forecast.forecasts.map(row => <tr key={`${row.run_id}-${row.agent}-${row.symbol}`}>
            <td>{istDateTime(row.observed_at)} IST</td><td>{row.agent.replaceAll("_", " ")}<small>{row.symbol}</small></td>
            <td>{row.direction}</td><td className="num">{row.confidence ?? "—"}</td><td className="num">{precise(row.reference_price)}</td>
            <td>{row.status === "settled" ? `${row.evidence_validated ? "" : "Legacy (unverified) · "}${row.direction_correct ? "Correct" : "Incorrect"} · ${signedPct(row.directional_return_pct)}` : row.status.replaceAll("_", " ")}
              <small>{row.settlement_session || row.settlement_note}</small></td>
            <td>{row.block_reason || "Decision recorded"}</td>
          </tr>)}</tbody>
        </table></div> : <Empty icon={CalendarClock} title="No forecasts recorded">Signals from future paper scans will appear here.</Empty>}
      </Card>
    </div>
  </div>;
}
