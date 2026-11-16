import { useCallback, useEffect, useState } from "react";

export const API = "/api";

const wholeRupees = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const paise = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const isNumber = value => value != null && value !== "" && Number.isFinite(Number(value));

export const money = value => isNumber(value) ? `₹${wholeRupees.format(value)}` : "—";
export const precise = value => isNumber(value) ? `₹${paise.format(value)}` : "—";
export const signed = (value, format = money) => !isNumber(value) ? "—"
  : `${value > 0 ? "+" : value < 0 ? "−" : ""}${format(Math.abs(value))}`;
export const pct = (value, digits = 1) => isNumber(value) ? `${(value * 100).toFixed(digits)}%` : "—";
export const signedPct = (value, digits = 2) => !isNumber(value) ? "—"
  : `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}%`;

const IST = "Asia/Kolkata";
export const istTime = value => value ? new Date(value).toLocaleTimeString("en-IN", {
  timeZone: IST, hour: "2-digit", minute: "2-digit", hour12: false,
}) : "—";
export const istDateTime = value => value ? new Date(value).toLocaleString("en-IN", {
  timeZone: IST, day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false,
}) : "Not yet";
export const istDay = value => value ? new Date(value).toLocaleDateString("en-IN", {
  timeZone: IST, day: "2-digit", month: "short",
}) : "—";

export function relative(value, now = Date.now()) {
  if (!value) return "";
  const minutes = Math.round((new Date(value).getTime() - now) / 60000);
  const span = Math.abs(minutes);
  const text = span < 1 ? "now" : span < 60 ? `${span} min` : span < 1440
    ? `${Math.floor(span / 60)} h ${span % 60 ? `${span % 60} min` : ""}`.trim()
    : `${Math.round(span / 1440)} d`;
  if (text === "now") return "just now";
  return minutes >= 0 ? `in ${text}` : `${text} ago`;
}

/** Poll a JSON endpoint. A 404 means "endpoint not in the running backend yet" and slows polling. */
export function useApi(path, delay) {
  const [state, setState] = useState({ data: null, error: false, missing: false, loaded: false });
  const load = useCallback(async () => {
    try {
      const response = await fetch(`${API}${path}`, { signal: AbortSignal.timeout(10000) });
      if (response.status === 404) {
        setState(previous => ({ ...previous, missing: true, error: false, loaded: true }));
        return;
      }
      if (!response.ok) throw new Error(String(response.status));
      const data = await response.json();
      setState({ data, error: false, missing: false, loaded: true });
    } catch {
      setState(previous => ({ ...previous, error: true, loaded: true }));
    }
  }, [path]);
  useEffect(() => {
    load();
    const id = setInterval(load, state.missing ? Math.max(delay, 60000) : delay);
    return () => clearInterval(id);
  }, [load, delay, state.missing]);
  return { ...state, refresh: load };
}

export function useNow(interval = 30000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), interval);
    return () => clearInterval(id);
  }, [interval]);
  return now;
}

const VIEW_IDS = ["overview", "market", "portfolio", "operations", "ai", "research"];

export function useHashView() {
  const read = () => {
    const id = window.location.hash.replace("#", "");
    return VIEW_IDS.includes(id) ? id : "overview";
  };
  const [view, setView] = useState(read);
  useEffect(() => {
    const onChange = () => setView(read());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const go = useCallback(id => {
    if (window.location.hash !== `#${id}`) window.location.hash = id;
    setView(id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);
  return [view, go];
}

/** Browser-side fallback when the backend does not report the session (no holiday calendar). */
export function clientMarketSession(now = Date.now()) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-GB", {
    timeZone: IST, weekday: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(new Date(now)).map(part => [part.type, part.value]));
  const minutes = Number(parts.hour) * 60 + Number(parts.minute);
  let state = "closed", label = "Closed for the day";
  if (parts.weekday === "Sat" || parts.weekday === "Sun") [state, label] = ["weekend", "Closed — weekend"];
  else if (minutes < 540) [state, label] = ["pre_market", "Closed — opens 09:15 IST"];
  else if (minutes < 555) [state, label] = ["pre_open", "Pre-open auction"];
  else if (minutes < 930) [state, label] = ["open", "Market open"];
  return {
    state, label, is_open: state === "open", approximate: true,
    entry_window_open: state === "open" && minutes >= 570 && minutes < 885,
  };
}

export const REGIMES = {
  LOW_VOL_TREND: {
    name: "Calm uptrend", tone: "ok", color: "#38B2AC",
    summary: "Prices are trending with low volatility — the friendliest conditions for trend-following trades.",
    effect: "Normal position sizes (up to 12% of capital per trade).",
  },
  HIGH_VOL_MEAN_REVERT: {
    name: "Choppy, higher volatility", tone: "warn", color: "#ED8936",
    summary: "Prices swing more than usual and tend to snap back, so the bot trades smaller and prefers mean-reversion setups.",
    effect: "Reduced sizes (up to 8% of capital per trade).",
  },
  BULL_EUPHORIA: {
    name: "Strong rally / overheating", tone: "warn", color: "#8B84FF",
    summary: "A fast, broad rally that can reverse sharply, so the bot trades smaller and requires hedges.",
    effect: "Reduced sizes (up to 8%) and hedges required.",
  },
  SYSTEMIC_PANIC: {
    name: "Market panic", tone: "bad", color: "#E53E3E",
    summary: "Stress conditions similar to March 2020. All new entries are blocked; exits and stop-losses still run.",
    effect: "No new positions.",
  },
};
export const regimeInfo = regime => REGIMES[regime] || {
  name: regime ? regime.replaceAll("_", " ").toLowerCase() : "Not classified", tone: "neutral", color: "#A3B1C6",
  summary: "The regime model has not produced a usable classification, so new entries are blocked.",
  effect: "New entries blocked until a fresh classification exists.",
};

export const JOB_INFO = {
  check_stops: "Checks stop-losses on open positions",
  pause_intraday_entries: "No new intraday trades after this",
  squareoff: "Closes all intraday positions",
  post_market: "Reconciles the paper ledger",
  daily_summary: "Sends the daily summary to Telegram",
  eod_cleanup: "End-of-day housekeeping",
  overnight_monitor: "Watches delivery positions overnight",
  pre_market: "Downloads the instrument list",
  broker_auth: "Logs in to Angel One and reconciles",
  premarket_mandate: "Pre-market analysis briefing",
  ws_connect: "Connects the live price feed",
  scan_0930: "First trading scan of the day",
  adaptive_scan: "Regime-aware scans every 30–60 min",
  scan_1430: "Last trading scan of the day",
};
export const isScanJob = job => job?.id?.startsWith("scan_") || job?.id === "adaptive_scan";

export const ANIMATE = !(typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);

export const scanStatusLabel = scan => !scan ? "Not run yet" : scan.status === "failed" ? "Failed"
  : scan.analysis_only ? "Analysis only" : scan.status === "degraded" ? "Finished with blocks/errors" : "Completed";

export const agentLabel = name => (name || "").replace(/Agent$/, "").replace(/([a-z])([A-Z])/g, "$1 $2");

export const CHART = {
  accent: "#6C63FF", accentLight: "#8B84FF", teal: "#38B2AC", warn: "#ED8936", bad: "#E53E3E",
  muted: "#5A6270", grid: "rgba(163, 177, 198, 0.45)", track: "#C8D0DB",
};

export const tooltipProps = {
  contentStyle: {
    background: "#E0E5EC", border: "none", borderRadius: 16, color: "#3D4852",
    boxShadow: "5px 5px 10px rgb(163,177,198,0.6), -5px -5px 10px rgba(255,255,255,0.5)",
    fontFamily: "DM Sans, system-ui, sans-serif", fontSize: 13, padding: "10px 14px",
  },
  labelStyle: { color: "#3D4852", fontWeight: 700, marginBottom: 4 },
  itemStyle: { color: "#3D4852", padding: 0 },
  cursor: { fill: "rgba(163, 177, 198, 0.18)" },
};
export const axisProps = {
  tick: { fill: "#5A6270", fontSize: 12, fontFamily: "DM Sans, system-ui, sans-serif" },
  axisLine: false, tickLine: false,
};
