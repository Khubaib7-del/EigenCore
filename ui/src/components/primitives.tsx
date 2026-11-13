import { useEffect, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import type { FitVerdict } from "../api";
import { Icon, type IconName } from "./Icon";

/* ── Button ─────────────────────────────────────────────── */
type BtnVariant = "primary" | "secondary" | "ghost" | "danger";
export function Button({
  variant = "secondary",
  size = "md",
  icon,
  children,
  className = "",
  ...rest
}: { variant?: BtnVariant; size?: "sm" | "md"; icon?: IconName } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button className={`btn btn-${variant} btn-${size} ${className}`} {...rest}>
      {icon && <Icon name={icon} size={size === "sm" ? 14 : 15} />}
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

const CHEVRON = Array.from({ length: 9 }, (_, i) => {
  const r = Math.floor(i / 3),
    c = i % 3;
  return (c + Math.abs(r - 1)) * 90;
});
const ORBIT_ORDER = [0, 1, 2, 5, 8, 7, 6, 3];
const ORBIT = Array.from({ length: 9 }, (_, i) => {
  const k = ORBIT_ORDER.indexOf(i);
  return k === -1 ? null : k * 110;
});

/** 3×3 pixel grid. "drive" = chevron wavefront, "orbit" = comet round the edge. */
export function PixelLoader({ variant = "drive", round = false }: { variant?: "drive" | "orbit"; round?: boolean }) {
  const delays = variant === "orbit" ? ORBIT : CHEVRON;
  const dur = variant === "orbit" ? 950 : 650;
  return (
    <span aria-hidden className="pixel-grid">
      {delays.map((d, i) => (
        <span
          key={i}
          className={round ? "round" : ""}
          style={{ opacity: d === null ? 0.08 : 0.16, animation: d === null ? "none" : `pixel-on ${dur}ms ease-in-out ${d}ms infinite` }}
        />
      ))}
    </span>
  );
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
export function LoadingState({ label, since, variant }: { label: string; since?: number; variant?: "drive" | "orbit" }) {
  const elapsed = useElapsed(since, true);
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <PixelLoader variant={variant} />
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
export function Ring({ value, size = 64, stroke = 6, children }: { value: number; size?: number; stroke?: number; children?: ReactNode }) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value));
  return (
    <span className="ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} className="ring-track" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          className="ring-value"
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
  return (
    <div className="scrim" onMouseDown={onClose}>
      <div className="dialog" role="alertdialog" aria-modal="true" aria-label={title} onMouseDown={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        <div className="dialog-body">{children}</div>
        <div className="dialog-actions">{actions}</div>
      </div>
    </div>
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
