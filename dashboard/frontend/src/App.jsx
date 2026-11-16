import { useEffect, useState } from "react";
import {
  AlertTriangle, BrainCircuit, BriefcaseBusiness, CandlestickChart, CheckCircle2, FlaskConical, Globe, LayoutDashboard, Menu, ServerCog, ShieldCheck,
  TrendingUp, X,
} from "lucide-react";
import "./dashboard.css";
import { API, clientMarketSession, istDateTime, useApi, useHashView, useNow } from "./lib";
import Overview from "./Overview";
import Market from "./Market";
import Portfolio from "./Portfolio";
import Operations from "./OperationsPanel";
import LLMControls from "./LLMControls";
import Research from "./Research";
import Workspaces from "./Workspaces";

const VIEWS = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "workspaces", label: "Workspaces", icon: Globe },
  { id: "market", label: "Market data", icon: CandlestickChart },
  { id: "portfolio", label: "Portfolio", icon: BriefcaseBusiness },
  { id: "operations", label: "Operations", icon: ServerCog },
  { id: "ai", label: "AI model", icon: BrainCircuit },
  { id: "research", label: "Research", icon: FlaskConical },
];

function Banner({ tone, icon: Icon, children, role = "status" }) {
  return <div className={`banner tone-${tone}`} role={role}><Icon size={18} aria-hidden="true" /><span>{children}</span></div>;
}

export default function App() {
  const health = useApi("/health", 5000), operations = useApi("/operations", 5000);
  const portfolio = useApi("/portfolio", 5000), trades = useApi("/trades", 10000), signals = useApi("/signals", 10000);
  const equity = useApi("/equity-curve", 30000), pnl = useApi("/pnl/daily", 30000), agents = useApi("/agents/status", 15000);
  const config = useApi("/config", 60000), watchlist = useApi("/market/watchlist", 10000);
  const [live, setLive] = useState(null), [busy, setBusy] = useState(false), [connected, setConnected] = useState(false);
  const [controlError, setControlError] = useState(""), [notice, setNotice] = useState(""), [reviewed, setReviewed] = useState(false);
  const [view, go] = useHashView();
  const [menuOpen, setMenuOpen] = useState(false);
  const now = useNow(15000);

  useEffect(() => {
    const source = new EventSource(`${API}/stream`);
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = ({ data }) => {
      try { const event = JSON.parse(data); if (event.type === "system_status") setLive(event.data); } catch { /* ignore malformed event */ }
    };
    return () => source.close();
  }, []);
  useEffect(() => {
    const label = VIEWS.find(item => item.id === view)?.label;
    document.title = `${label} · India AI Trader`;
  }, [view]);
  useEffect(() => {
    if (!menuOpen) return undefined;
    const onKey = event => { if (event.key === "Escape") setMenuOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menuOpen]);
  useEffect(() => {
    if (!notice) return undefined;
    const id = setTimeout(() => setNotice(""), 5000);
    return () => clearTimeout(id);
  }, [notice]);

  const state = live && (!health.data || live.timestamp > health.data.timestamp) ? live : health.data;
  const session = state?.market_session || clientMarketSession(now);
  const ops = operations.data;
  const failed = [portfolio, trades, signals, equity, pnl, agents, health, operations, config].some(item => item.error);

  const mutate = async (path, body = {}) => {
    setBusy(true); setControlError(""); setNotice("");
    try {
      const r = await fetch(`${API}${path}`, {
        method: "POST", headers: { "Content-Type": "application/json", "X-Dashboard-Control": "1" },
        body: JSON.stringify(body), signal: AbortSignal.timeout(30000),
      });
      const result = await r.json();
      if (!r.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Operation rejected.");
      if (result.system) setLive(result.system);
      setReviewed(false);
      setNotice("Done — the server accepted the request. Status below is refreshed from the server.");
      return result;
    } catch (error) {
      setControlError(error.name === "TimeoutError" ? "Request timed out. Refreshing saved state; do not assume success." : error.message);
    } finally { setBusy(false); health.refresh(); operations.refresh(); }
  };
  const control = action => mutate("/control", { action, reviewed });
  const choose = id => { go(id); setMenuOpen(false); };
  const paper = state?.paper_trading;

  return <div className="app">
    <a className="skip-link" href="#main">Skip to content</a>
    <header className="topbar">
      <div className="topbar-inner">
        <a className="brand" href="#overview" onClick={event => { event.preventDefault(); choose("overview"); }}>
          <span className="brand-mark" aria-hidden="true"><TrendingUp size={22} strokeWidth={2.4} /></span>
          <span className="brand-text"><strong>India AI Trader</strong><small>Multi-market · paper only</small></span>
        </a>
        <nav id="main-nav" className={`nav ${menuOpen ? "is-open" : ""}`} aria-label="Dashboard sections">
          {VIEWS.map(({ id, label, icon: Icon }) => <button type="button" key={id} className={`nav-item ${view === id ? "is-active" : ""}`}
            aria-current={view === id ? "page" : undefined} onClick={() => choose(id)}>
            <Icon size={18} strokeWidth={2.2} aria-hidden="true" /><span>{label}</span>
          </button>)}
        </nav>
        <div className="topbar-status">
          <span className={`session-pill ${session.is_open ? "is-open" : ""}`} title={session.approximate ? "Estimated from your clock (holidays not checked)" : "From the server’s NSE calendar"}>
            <i aria-hidden="true" />{session.label}
          </span>
          <span className={`mode-badge ${state && !paper ? "is-live" : ""}`} title="Trading mode reported by the server">
            {health.error && !state ? "Unconfirmed" : !state ? "Checking" : paper ? "Paper" : state.mode || "Not paper"}
          </span>
          <span className={`stream-dot ${connected ? "is-on" : ""}`} title={connected ? "Receiving live status updates" : "Live stream unavailable — polling every 5 s"}>
            <i aria-hidden="true" /><span className="stream-label">{connected ? "Live" : "Polling"}</span>
          </span>
        </div>
        <button type="button" className="menu-toggle" aria-expanded={menuOpen} aria-controls="main-nav"
          aria-label={menuOpen ? "Close menu" : "Open menu"} onClick={() => setMenuOpen(open => !open)}>
          {menuOpen ? <X size={22} aria-hidden="true" /> : <Menu size={22} aria-hidden="true" />}
        </button>
      </div>
    </header>

    <main id="main" className="container" tabIndex={-1}>
      <div className="banners">
        {state && !paper && <Banner tone="bad" icon={AlertTriangle} role="alert">The server is not reporting paper mode. Stop and investigate before doing anything else.</Banner>}
        {state?.dead_mans_switch?.halted && <Banner tone="bad" icon={AlertTriangle}>
          Circuit breaker is halted: {state.dead_mans_switch.halt_reason || "no reason recorded"}. New trades are blocked until you review and clear it on the Overview page.
        </Banner>}
        {health.error && <Banner tone="bad" icon={AlertTriangle}>Cannot reach the trading server. Showing the last known values; retrying every 5 seconds.</Banner>}
        {failed && !health.error && <Banner tone="warn" icon={AlertTriangle}>Some dashboard data is unavailable. Requests retry automatically.</Banner>}
      </div>
      {view === "overview" && <Overview state={state} ops={ops} opsUnavailable={operations.error} healthError={health.error}
        portfolio={portfolio.data} config={config.data} session={session} now={now}
        busy={busy} reviewed={reviewed} setReviewed={setReviewed} control={control} onNavigate={choose} />}
      {view === "workspaces" && <Workspaces state={state} ops={ops} portfolio={portfolio.data} session={session} now={now} />}
      {view === "market" && <Market ops={ops} opsUnavailable={operations.error} watchlist={watchlist} busy={busy} mutate={mutate} />}
      {view === "portfolio" && <Portfolio portfolio={portfolio.data} config={config.data} equity={equity.data} pnl={pnl.data}
        trades={trades.data} signals={signals} />}
      {view === "operations" && <Operations ops={ops} opsUnavailable={operations.error} state={state} busy={busy} mutate={mutate}
        agents={agents} now={now} />}
      {view === "ai" && <LLMControls serverState={state?.llm} />}
      {view === "research" && <Research />}
      <footer className="footer">
        <ShieldCheck size={16} aria-hidden="true" />
        <span>Paper evaluation only — not certified for live trading, and this dashboard has no live-mode controls.
          Last server update {istDateTime(state?.timestamp)} IST.</span>
      </footer>
    </main>

    <div className="toasts" aria-live="polite">
      {busy && <div className="toast tone-accent"><span className="spinner" aria-hidden="true" />Sending request…</div>}
      {controlError && <div className="toast tone-bad" role="alert"><AlertTriangle size={18} aria-hidden="true" /><span>{controlError}</span>
        <button type="button" className="toast-close" aria-label="Dismiss" onClick={() => setControlError("")}><X size={16} /></button></div>}
      {notice && !controlError && <div className="toast tone-ok"><CheckCircle2 size={18} aria-hidden="true" /><span>{notice}</span>
        <button type="button" className="toast-close" aria-label="Dismiss" onClick={() => setNotice("")}><X size={16} /></button></div>}
    </div>
  </div>;
}
