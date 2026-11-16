import { useId } from "react";
import { Inbox, Info } from "lucide-react";

export const TONE_COLOR = {
  ok: "#38B2AC", warn: "#ED8936", bad: "#E53E3E", accent: "#6C63FF", neutral: "#A3B1C6",
};

export function Card({ title, subtitle, icon, tone = "accent", action, className = "", children }) {
  const id = useId();
  return <section className={`card ${className}`} aria-labelledby={title ? id : undefined}>
    {(title || action) && <header className="card-head">
      {icon && <Well icon={icon} tone={tone} />}
      <div className="card-title">
        {title && <h2 id={id}>{title}</h2>}
        {subtitle && <p>{subtitle}</p>}
      </div>
      {action && <div className="card-action">{action}</div>}
    </header>}
    {children}
  </section>;
}

export function Well({ icon: Icon, tone = "accent", size = "md", className = "" }) {
  return <span className={`well well-${size} tone-${tone} ${className}`} aria-hidden="true">
    <Icon size={size === "lg" ? 26 : size === "sm" ? 16 : 20} strokeWidth={2.2} />
  </span>;
}

export function Chip({ tone = "neutral", children, dot = true, title }) {
  return <span className={`chip tone-${tone}`} title={title}>{dot && <i aria-hidden="true" />}{children}</span>;
}

export function Button({ variant = "neutral", icon: Icon, children, className = "", ...rest }) {
  return <button type="button" className={`btn btn-${variant} ${className}`} {...rest}>
    {Icon && <Icon size={18} strokeWidth={2.2} aria-hidden="true" />}
    <span>{children}</span>
  </button>;
}

export function Empty({ icon = Inbox, title, children, tone = "neutral" }) {
  return <div className="empty">
    <Well icon={icon} tone={tone} />
    <div>{title && <strong>{title}</strong>}{children && <p>{children}</p>}</div>
  </div>;
}

export function Hint({ children }) {
  return <p className="hint"><Info size={15} aria-hidden="true" /><span>{children}</span></p>;
}

export function DetailList({ items }) {
  return <dl className="details">
    {items.filter(Boolean).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
  </dl>;
}

/** Circular progress with nested depth: extruded disc, inset track, raised core. */
export function Ring({ value, max, size = 148, stroke = 12, tone = "accent", label, caption }) {
  const radius = (size - stroke) / 2 - 10;
  const circumference = 2 * Math.PI * radius;
  const ratio = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0;
  const gradient = useId().replace(/:/g, "");
  return <div className="ring" style={{ width: size, height: size }} role="img"
    aria-label={`${value} of ${max}${caption ? ` ${caption}` : ""}`}>
    <div className="ring-track" />
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
      <defs>
        <linearGradient id={gradient} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor={tone === "accent" ? "#8B84FF" : TONE_COLOR[tone]} />
          <stop offset="1" stopColor={tone === "accent" ? "#6C63FF" : TONE_COLOR[tone]} />
        </linearGradient>
      </defs>
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke={`url(#${gradient})`}
        strokeWidth={stroke} strokeLinecap="round" strokeDasharray={circumference}
        strokeDashoffset={circumference * (1 - ratio)} transform={`rotate(-90 ${size / 2} ${size / 2})`}
        className="ring-arc" />
    </svg>
    <div className="ring-core">
      <strong>{label ?? `${value}/${max}`}</strong>
      {caption && <small>{caption}</small>}
    </div>
  </div>;
}

/** Half-circle gauge for a 0–1 value. */
export function Gauge({ value, tone = "accent", label, caption }) {
  const ratio = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
  const width = 180, height = 104, radius = 76, cx = width / 2, cy = 92;
  const arc = end => {
    const angle = Math.PI * (1 - end);
    return `M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius * Math.cos(angle)} ${cy - radius * Math.sin(angle)}`;
  };
  return <figure className="gauge" role="img" aria-label={`${caption}: ${label}`}>
    <svg viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <path d={arc(1)} className="gauge-track" />
      {ratio > 0.002 && <path d={arc(ratio)} stroke={TONE_COLOR[tone]} className="gauge-fill" />}
    </svg>
    <figcaption><strong>{label}</strong><small>{caption}</small></figcaption>
  </figure>;
}

export function Meter({ value, max, tone = "accent", label, detail }) {
  const ratio = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0;
  return <div className="meter">
    <div className="meter-head"><span>{label}</span><b>{detail}</b></div>
    <div className="meter-track" role="progressbar" aria-label={label} aria-valuemin={0}
      aria-valuemax={max} aria-valuenow={value}>
      <span style={{ width: `${ratio * 100}%`, background: TONE_COLOR[tone] }} />
    </div>
  </div>;
}

export function Stat({ icon, label, value, detail, tone = "accent", valueTone }) {
  return <article className="card stat">
    <Well icon={icon} tone={tone} size="lg" />
    <div>
      <small>{label}</small>
      <strong className={valueTone ? `ink-${valueTone}` : ""}>{value}</strong>
      {detail && <span>{detail}</span>}
    </div>
  </article>;
}

export function Sparkline({ points, width = 112, height = 32, tone }) {
  const values = (points || []).map(point => point.p).filter(Number.isFinite);
  if (values.length < 2) return <span className="sparkline-empty">Collecting…</span>;
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const path = values.map((value, index) => {
    const x = (index / (values.length - 1)) * (width - 4) + 2;
    const y = height - 3 - ((value - min) / span) * (height - 6);
    return `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
  const color = TONE_COLOR[tone || (values.at(-1) >= values[0] ? "ok" : "bad")];
  return <svg className="sparkline" width={width} height={height} viewBox={`0 0 ${width} ${height}`}
    role="img" aria-label={`Intraday trend from ${values[0]} to ${values.at(-1)}`}>
    <path d={path} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>;
}

/** Day low–high range with markers for the open and the latest price. */
export function RangeBar({ low, high, value, open }) {
  if (![low, high, value].every(Number.isFinite) || high <= low) return <span className="muted">—</span>;
  const at = number => `${Math.min(100, Math.max(0, ((number - low) / (high - low)) * 100))}%`;
  return <div className="range" role="img"
    aria-label={`Day range ${low} to ${high}, last ${value}`}>
    <div className="range-track">
      {Number.isFinite(open) && <i className="range-open" style={{ left: at(open) }} />}
      <b className="range-value" style={{ left: at(value) }} />
    </div>
  </div>;
}
