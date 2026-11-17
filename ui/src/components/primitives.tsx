import { useEffect, useId, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { FitVerdict } from "../api";
import { Icon, type IconName } from "./Icon";
import { GridLoader, type LoaderKind } from "./GridLoader";
import { Spinner } from "./Spinner";

/* ── Button ─────────────────────────────────────────────── */
type BtnVariant = "primary" | "secondary" | "ghost" | "danger";
export function Button({
  variant = "secondary",
  size = "md",
  icon,
  busy = false,
  children,
  className = "",
  disabled,
  ...rest
}: {
  variant?: BtnVariant;
  size?: "sm" | "md";
  icon?: IconName;
  /** Shows a spinner in place of the icon and disables the button. */
  busy?: boolean;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  const px = size === "sm" ? 14 : 15;
  return (
    <button className={`btn btn-${variant} btn-${size} ${busy ? "is-busy" : ""} ${className}`} disabled={disabled || busy} aria-busy={busy || undefined} {...rest}>
      {busy ? <Spinner kind="ring" size={px} /> : icon && <Icon name={icon} size={px} />}
      {children}
    </button>
  );
}

export function IconButton({
  icon,
  label,
  size = 16,
  className = "",
  ...rest
}: { icon: IconName; label: string; size?: number } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button className={`icon-btn ${className}`} aria-label={label} title={label} {...rest}>
      <Icon name={icon} size={size} />
    </button>
  );
}

/* ── Chips ──────────────────────────────────────────────── */
export function Chip({ children, tone, icon, mono }: { children: ReactNode; tone?: "ok" | "warn" | "err"; icon?: IconName; mono?: boolean }) {
  return (
    <span className={`chip ${tone ? `chip-${tone}` : ""} ${mono ? "mono" : ""}`}>
      {icon && <Icon name={icon} size={12} />}
      {children}
    </span>
  );
}

const FIT: Record<FitVerdict, { label: string; tone?: "ok" | "warn" | "err"; mark: string }> = {
  fits_with_margin: { label: "Fits", tone: "ok", mark: "●" },
  tight: { label: "Tight", tone: "warn", mark: "◐" },
  does_not_fit: { label: "Does not fit", tone: "err", mark: "○" },
  unknown: { label: "Unknown", mark: "?" },
};

/** Fit verdict: always a word (plus a shape), never colour alone. */
export function FitChip({ fit }: { fit: FitVerdict }) {
  const f = FIT[fit];
  return (
    <span className={`chip fit ${f.tone ? `chip-${f.tone}` : ""}`} title={fit === "unknown" ? "Fit is estimated after the file is downloaded" : "Estimate from file geometry and free RAM"}>
      <span aria-hidden className="fit-mark">
        {f.mark}
      </span>
      {f.label}
    </span>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="kbd">{children}</kbd>;
}

/* ── Segmented control ──────────────────────────────────── */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  disabled,
}: {
  value: T;
  options: { value: T; label: string; icon?: IconName }[];
  onChange: (v: T) => void;
  label: string;
  disabled?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pill, setPill] = useState<{ left: number; width: number } | null>(null);
  useEffect(() => {
    const el = ref.current?.querySelector<HTMLElement>(`[data-value="${value}"]`);
    if (el) setPill({ left: el.offsetLeft, width: el.offsetWidth });
  }, [value, options.length]);
  return (
    <div className={`segmented ${disabled ? "disabled" : ""}`} role="radiogroup" aria-label={label} aria-disabled={disabled || undefined} ref={ref}>
      {pill && <span className="segmented-pill" style={{ transform: `translateX(${pill.left - 3}px)`, width: pill.width }} />}
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={o.value === value}
          data-value={o.value}
          className={o.value === value ? "on" : ""}
          disabled={disabled}
          onClick={() => onChange(o.value)}
        >
          {o.icon && <Icon name={o.icon} size={14} />}
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button role="switch" aria-checked={checked} aria-label={label} className={`switch ${checked ? "on" : ""}`} onClick={() => onChange(!checked)}>
      <span />
    </button>
  );
}

export function Progress({ value, label }: { value: number; label?: string }) {
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value)} aria-label={label}>
      <span style={{ width: `${Math.max(2, Math.min(100, value))}%` }} />
    </div>
  );
}

/* ── Shimmer text + pixel loader (after Beautiful UI's Loading State, MIT) ── */
export function Shimmer({ children }: { children: ReactNode }) {
  return <span className="shimmer">{children}</span>;
}

/** Old name for the grid loader: "drive" (writing) or "orbit" (checking). */
export function PixelLoader({ variant = "drive", round = false }: { variant?: "drive" | "orbit"; round?: boolean }) {
  return <GridLoader pattern={variant} tone={variant === "orbit" ? "teal" : "blue"} round={round} />;
}

export function useElapsed(since: number | undefined, running: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(t);
  }, [running]);
  if (!since) return "0.0s";
  const s = Math.max(0, (now - since) / 1000);
  return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${(s % 60).toFixed(0)}s`;
}

/** Loader + shimmering label + live elapsed timer. */
export function LoadingState({ label, since, kind = "loading" }: { label: string; since?: number; kind?: LoaderKind }) {
  const elapsed = useElapsed(since, true);
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <GridLoader kind={kind} />
      <Shimmer>{label}</Shimmer>
      {since !== undefined && <span className="mono muted tnum">{elapsed}</span>}
    </div>
  );
}

/* ── Glide menu: one highlight that glides between rows (after Beautiful UI) ── */
export function GlideMenu({ children, className = "" }: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<{ top: number; height: number } | null>(null);
  const [visible, setVisible] = useState(false);
  const moveTo = (t: EventTarget | null) => {
    const c = ref.current;
    if (!(t instanceof Element) || !c) return;
    const row = t.closest("[data-glide]");
    if (!(row instanceof HTMLElement) || !c.contains(row)) return;
    const cr = c.getBoundingClientRect();
    const rr = row.getBoundingClientRect();
    setBox({ top: rr.top - cr.top + c.scrollTop, height: rr.height });
    setVisible(true);
  };
  return (
    <div
      ref={ref}
      className={`glide ${className}`}
      onMouseOver={(e) => moveTo(e.target)}
      onMouseLeave={() => setVisible(false)}
      onFocusCapture={(e) => moveTo(e.target)}
      onBlurCapture={(e) => {
        if (!ref.current?.contains(e.relatedTarget as Node | null)) setVisible(false);
      }}
    >
      <span aria-hidden className="glide-hl" style={{ top: box?.top ?? 0, height: box?.height ?? 0, opacity: box && visible ? 1 : 0 }} />
      {children}
    </div>
  );
}

/* ── Charts ─────────────────────────────────────────────── */
export function Sparkline({ data, height = 64 }: { data: number[]; height?: number }) {
  const w = 300;
  if (data.length < 2) return <div className="spark-empty muted">No turns yet</div>;
  const max = Math.max(...data, 1);
  const pts = data.map((v, i) => [(i / (data.length - 1)) * w, height - 4 - (v / max) * (height - 10)]);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const area = `${line} L${w} ${height} L0 ${height} Z`;
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" role="img" aria-label="Tokens per second over recent turns">
      <path d={area} className="spark-area" />
      <path d={line} className="spark-line" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** Ring gauge 0..1. */
export function Ring({ value, size = 64, stroke = 6, children, color }: { value: number; size?: number; stroke?: number; children?: ReactNode; color?: boolean }) {
  const gid = useId();
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value));
  return (
    <span className="ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
        {color && (
          <defs>
            <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="var(--c-teal)" />
              <stop offset="60%" stopColor={v > 0.8 ? "var(--c-amber)" : "var(--c-blue)"} />
              <stop offset="100%" stopColor={v > 0.9 ? "var(--c-red)" : "var(--c-violet)"} />
            </linearGradient>
          </defs>
        )}
        <circle cx={size / 2} cy={size / 2} r={r} className="ring-track" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          className="ring-value"
          style={color ? { stroke: `url(#${gid})` } : undefined}
          strokeWidth={stroke}
          fill="none"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - v)}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      {children && <span className="ring-label">{children}</span>}
    </span>
  );
}

/* ── Dialog ─────────────────────────────────────────────── */
export function Dialog({
  open,
  title,
  children,
  onClose,
  actions,
}: {
  open: boolean;
  title: string;
  children: ReactNode;
  onClose: () => void;
  actions: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [open, onClose]);
  if (!open) return null;
  return createPortal(
    <div className="scrim" onMouseDown={onClose}>
      <div className="dialog" role="alertdialog" aria-modal="true" aria-label={title} onMouseDown={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        <div className="dialog-body">{children}</div>
        <div className="dialog-actions">{actions}</div>
      </div>
    </div>,
    document.body,
  );
}

export function SampleBadge({ children = "Sample data" }: { children?: ReactNode }) {
  return (
    <span className="chip sample" title="Preview values for layout only. Nothing here was measured.">
      <Icon name="info" size={12} />
      {children}
    </span>
  );
}

/* ── Soon pill: a feature that is shown but not wired up yet ───────────── */
export function SoonPill({ title = "Coming soon: not wired up yet" }: { title?: string }) {
  return (
    <span className="soon" title={title}>
      Soon
    </span>
  );
}

/* ── Equalizer loader: the chat-history loader (Monitor keeps the pixel grid) ── */
export function EqLoader({ label = "Generating" }: { label?: string }) {
  return <GridLoader kind="writing" size={11} label={label} />;
}

/** Closes a popover on outside click and Escape. */
export function useDismiss(open: boolean, close: () => void) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const down = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && close();
    const esc = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("mousedown", down);
    window.addEventListener("keydown", esc);
    return () => {
      window.removeEventListener("mousedown", down);
      window.removeEventListener("keydown", esc);
    };
  }, [open, close]);
  return ref;
}

/* ── Tokens-per-second chart: area line, grid, avg line ─────────────────
 * Drawn at the real pixel width (ResizeObserver), so text and strokes are
 * never stretched. One point still draws (a dot and its value).
 */
/** Top of the y axis: four equal steps of a round size (1, 2, 2.5, 5 × 10^n). */
function niceMax(v: number) {
  if (!(v > 0)) return 10;
  const raw = (v * 1.1) / 4;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) return m * p * 4;
  return 40 * p;
}

export function TpsChart({
  values,
  xs,
  xLabel,
  height = 168,
  label,
}: {
  values: number[];
  /** x position of each value (seconds, or reply index). */
  xs: number[];
  xLabel: (x: number) => string;
  height?: number;
  label: string;
}) {
  const gid = useId();
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(560);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  if (!values.length) return <div className="chart-empty muted">{label}</div>;
  const padL = 34;
  const padR = 10;
  const padT = 10;
  const padB = 22;
  const max = niceMax(Math.max(...values));
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  const sx = (x: number) => (x1 === x0 ? padL + (w - padL - padR) / 2 : padL + ((x - x0) / (x1 - x0)) * (w - padL - padR));
  const sy = (v: number) => padT + (1 - v / max) * (height - padT - padB);
  const pts = values.map((v, i) => [sx(xs[i]), sy(v)] as const);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const base = height - padB;
  const area = pts.length > 1 ? `${line} L${pts[pts.length - 1][0].toFixed(1)} ${base} L${pts[0][0].toFixed(1)} ${base} Z` : "";
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * max);
  const last = pts[pts.length - 1];
  return (
    <div className="tps-chart" ref={box}>
      <svg width={w} height={height} viewBox={`0 0 ${w} ${height}`} role="img" aria-label={label}>
        <defs>
          <linearGradient id={`${gid}-line`} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--c-teal)" />
            <stop offset="50%" stopColor="var(--c-blue)" />
            <stop offset="100%" stopColor="var(--c-violet)" />
          </linearGradient>
          <linearGradient id={`${gid}-area`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--c-blue)" stopOpacity="0.28" />
            <stop offset="100%" stopColor="var(--c-blue)" stopOpacity="0" />
          </linearGradient>
        </defs>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={padL} x2={w - padR} y1={sy(t)} y2={sy(t)} className={t === 0 ? "chart-axis" : "chart-grid"} />
            <text x={padL - 6} y={sy(t) + 3.5} className="chart-tick" textAnchor="end">
              {Number.isInteger(t) ? t : t.toFixed(1)}
            </text>
          </g>
        ))}
        {area && <path d={area} className="chart-area" style={{ fill: `url(#${gid}-area)` }} />}
        {pts.length > 1 && <path d={line} className="chart-line" style={{ stroke: `url(#${gid}-line)` }} />}
        <line x1={padL} x2={w - padR} y1={sy(avg)} y2={sy(avg)} className="chart-avg" />
        <text x={padL + 6} y={sy(avg) - 5} className="chart-tick chart-avg-label">
          avg {avg.toFixed(1)}
        </text>
        <circle cx={last[0]} cy={last[1]} r={3.5} className="chart-dot" />
        <text x={padL} y={height - 6} className="chart-tick">
          {xLabel(x0)}
        </text>
        {x1 !== x0 && (
          <text x={w - padR} y={height - 6} className="chart-tick" textAnchor="end">
            {xLabel(x1)}
          </text>
        )}
      </svg>
    </div>
  );
}
