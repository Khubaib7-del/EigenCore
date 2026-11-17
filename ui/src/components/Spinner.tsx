import { useId, type CSSProperties, type ReactNode } from "react";

/* ── Spinners: small single-element loaders ──────────────────────────────
 * The grid loader (GridLoader.tsx) stays the chat's loader. Spinners are the
 * small ones for everything around it: a busy button, a download row, the
 * update check, a running tool step, the sidebar while a chat streams and
 * the Monitor's live chip.
 *
 * Each spinner is an SVG drawn on a 24-unit box (two use HTML for real 3D
 * or stacking) and animated by CSS keyframes in styles/spinners.css. They
 * draw in currentColor; pass `tone` for an app colour token (--c-*, with a
 * dark and a light value) or `color` for any CSS colour. With reduced motion
 * the shape holds still and only its opacity breathes.
 *
 * Shapes and motion follow a reference gallery the team picked (Arc to
 * Trace). Circular dots, Classic and Classic v2 were named there but never
 * shown in motion, so those three, plus Bars and Hourglass, are our own.
 */

export type SpinnerKind =
  | "arc"
  | "atom"
  | "blocks"
  | "circular-dots"
  | "classic"
  | "classic-v2"
  | "clock"
  | "comet"
  | "compass"
  | "dual"
  | "eclipse"
  | "fill"
  | "flip"
  | "gather"
  | "leap"
  | "linear-dots"
  | "loading"
  | "loop"
  | "morph"
  | "orbit"
  | "phases"
  | "pulse"
  | "radar"
  | "ring"
  | "ripple"
  | "scan"
  | "slide"
  | "snake"
  | "split"
  | "steps"
  | "swirl"
  | "trace"
  | "bars"
  | "hourglass";

export type Tone = "violet" | "purple" | "blue" | "sky" | "cyan" | "teal" | "green" | "lime" | "amber" | "orange" | "red" | "pink" | "mono";

export interface SpinnerInfo {
  kind: SpinnerKind;
  name: string;
  /** Loop length in ms (also the CSS --sp-dur). */
  ms: number;
  /** Colour token it is shown in by default (the showcase, the app). */
  tone: Tone;
  /** Where the app uses it, or what it suits. */
  use: string;
  /** True for the ones not in the reference gallery. */
  extra?: boolean;
}

export const SPINNERS: SpinnerInfo[] = [
  { kind: "arc", name: "Arc", ms: 800, tone: "blue", use: "Generic busy, inline text" },
  { kind: "atom", name: "Atom", ms: 1600, tone: "violet", use: "Loading a model into memory" },
  { kind: "blocks", name: "Blocks", ms: 1200, tone: "cyan", use: "Indexing, building a list" },
  { kind: "circular-dots", name: "Circular dots", ms: 1000, tone: "mono", use: "Neutral wait", extra: true },
  { kind: "classic", name: "Classic", ms: 1000, tone: "mono", use: "System-style wait", extra: true },
  { kind: "classic-v2", name: "Classic v2", ms: 800, tone: "mono", use: "System-style wait, bolder", extra: true },
  { kind: "clock", name: "Clock", ms: 1200, tone: "amber", use: "Waiting on a timer, a queue" },
  { kind: "comet", name: "Comet", ms: 900, tone: "sky", use: "Checking for updates" },
  { kind: "compass", name: "Compass", ms: 700, tone: "teal", use: "Finding the way: routing a task" },
  { kind: "dual", name: "Dual", ms: 1000, tone: "purple", use: "Two things at once: syncing" },
  { kind: "eclipse", name: "Eclipse", ms: 1200, tone: "violet", use: "Switching models" },
  { kind: "fill", name: "Fill", ms: 2000, tone: "green", use: "Timed step, warming up" },
  { kind: "flip", name: "Flip", ms: 1200, tone: "orange", use: "Swapping a view, a card" },
  { kind: "gather", name: "Gather", ms: 1400, tone: "cyan", use: "Collecting results" },
  { kind: "leap", name: "Leap", ms: 1950, tone: "blue", use: "Queued, next up" },
  { kind: "linear-dots", name: "Linear dots", ms: 900, tone: "blue", use: "Sidebar: chat is writing" },
  { kind: "loading", name: "Loading", ms: 1100, tone: "teal", use: "Dot-matrix wait, splash" },
  { kind: "loop", name: "Loop", ms: 1400, tone: "pink", use: "Retrying, reconnecting" },
  { kind: "morph", name: "Morph", ms: 1400, tone: "lime", use: "Converting a file" },
  { kind: "orbit", name: "Orbit", ms: 1100, tone: "orange", use: "Tool step: running" },
  { kind: "phases", name: "Phases", ms: 2800, tone: "amber", use: "Long job, slow progress" },
  { kind: "pulse", name: "Pulse", ms: 1300, tone: "violet", use: "Sidebar: reading the prompt" },
  { kind: "radar", name: "Radar", ms: 1400, tone: "sky", use: "Tool step: searching files" },
  { kind: "ring", name: "Ring", ms: 750, tone: "mono", use: "Button busy state" },
  { kind: "ripple", name: "Ripple", ms: 1500, tone: "teal", use: "Waiting for a connection" },
  { kind: "scan", name: "Scan", ms: 2000, tone: "sky", use: "Reading a document" },
  { kind: "slide", name: "Slide", ms: 2640, tone: "purple", use: "Rearranging, sorting" },
  { kind: "snake", name: "Snake", ms: 1400, tone: "blue", use: "Indeterminate progress" },
  { kind: "split", name: "Split", ms: 1600, tone: "pink", use: "Merging, splitting" },
  { kind: "steps", name: "Steps", ms: 1600, tone: "teal", use: "Model download row" },
  { kind: "swirl", name: "Swirl", ms: 900, tone: "orange", use: "Busy, compact grid" },
  { kind: "trace", name: "Trace", ms: 1400, tone: "green", use: "Update: downloading" },
  { kind: "bars", name: "Bars", ms: 1000, tone: "pink", use: "Monitor: live", extra: true },
  { kind: "hourglass", name: "Hourglass", ms: 2400, tone: "green", use: "Update: installing", extra: true },
];

const INFO = Object.fromEntries(SPINNERS.map((s) => [s.kind, s])) as Record<SpinnerKind, SpinnerInfo>;

/* ── geometry helpers ── */
const C = 12;
const polar = (r: number, deg: number) => {
  const a = ((deg - 90) * Math.PI) / 180;
  return [C + r * Math.cos(a), C + r * Math.sin(a)] as const;
};
const sector = (r: number, from: number, to: number) => {
  const [x1, y1] = polar(r, from);
  const [x2, y2] = polar(r, to);
  return `M${C} ${C}L${x1.toFixed(2)} ${y1.toFixed(2)}A${r} ${r} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}Z`;
};
const d = (i: number, n: number, ms: number) => ({ animationDelay: `${(-(n - i) * ms) / n}ms` }) as CSSProperties;
const R = (n: number) => Array.from({ length: n }, (_, i) => i);

/** Track: the dim full shape the moving part runs on. */
const track = { opacity: 0.22 };

const GRID3 = [5, 12, 19];
const RING8 = [
  [0, 0],
  [1, 0],
  [2, 0],
  [2, 1],
  [2, 2],
  [1, 2],
  [0, 2],
  [0, 1],
] as const;
const LOOP = "M12 12C9 8.4 4 8.4 4 12s5 3.6 8 0 8-3.6 8 0-5 3.6-8 0Z";
const RADAR_TAIL = [0.9, 0.55, 0.36, 0.22, 0.12, 0.06];

function body(kind: SpinnerKind, ms: number, uid: string): ReactNode {
  switch (kind) {
    case "arc":
      return <circle className="sp-rot" cx={C} cy={C} r={9} pathLength={100} strokeDasharray="28 72" fill="none" stroke="currentColor" strokeWidth={2.4} strokeLinecap="round" />;
    case "atom":
      return (
        <>
          <circle cx={C} cy={C} r={10} fill="none" stroke="currentColor" strokeWidth={1.6} />
          {[0, 60, 120].map((a, i) => (
            <g key={a} transform={`rotate(${a} 12 12)`}>
              <ellipse className="sp-atom" style={d(i, 3, ms)} cx={C} cy={C} rx={10} ry={10} fill="none" stroke="currentColor" strokeWidth={1.1} opacity={0.75} />
            </g>
          ))}
        </>
      );
    case "blocks":
      return R(9).map((i) => {
        const r = Math.floor(i / 3);
        const c = i % 3;
        return <rect key={i} className="sp-block" style={{ animationDelay: `${((r + c) * ms) / 10 - ms}ms` }} x={GRID3[c] - 2.8} y={GRID3[r] - 2.8} width={5.6} height={5.6} rx={1.4} fill="currentColor" />;
      });
    case "circular-dots":
      return R(8).map((i) => {
        const [x, y] = polar(8, i * 45);
        return <circle key={i} className="sp-fade" style={d(i, 8, ms)} cx={x} cy={y} r={2} fill="currentColor" />;
      });
    case "classic":
      return R(12).map((i) => (
        <line key={i} className="sp-fade" style={d(i, 12, ms)} x1={C} y1={3} x2={C} y2={7.5} transform={`rotate(${i * 30} 12 12)`} stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
      ));
    case "classic-v2":
      return (
        <g className="sp-step8">
          {R(8).map((i) => (
            <line key={i} x1={C} y1={2.6} x2={C} y2={7} transform={`rotate(${i * 45} 12 12)`} stroke="currentColor" strokeWidth={3} strokeLinecap="round" opacity={0.15 + (0.85 * i) / 7} />
          ))}
        </g>
      );
    case "clock":
      return (
        <>
          <circle cx={C} cy={C} r={10} fill="currentColor" opacity={0.25} />
          <path className="sp-rot" d="M12 12Q16.2 7.4 12 2.6Q7.8 7.4 12 12Z" fill="currentColor" />
        </>
      );
    case "comet":
      return (
        <>
          <circle cx={C} cy={C} r={9} fill="none" stroke="currentColor" strokeWidth={2.2} style={track} />
          <g className="sp-rot">
            {R(8).map((i) => (
              <circle key={i} cx={C} cy={C} r={9} transform="matrix(1 0 0 -1 0 24)" pathLength={100} strokeDasharray={`${2 + i * 8} 100`} fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" opacity={i ? 0.22 : 1} />
            ))}
          </g>
        </>
      );
    case "compass":
      return (
        <g className="sp-quarter">
          {[0, 90, 180, 270].map((a) => (
            <line key={a} x1={C} y1={2.5} x2={C} y2={7} transform={`rotate(${a} 12 12)`} stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" />
          ))}
        </g>
      );
    case "dual":
      return (
        <>
          <circle className="sp-rot" cx={C} cy={C} r={9.5} pathLength={100} strokeDasharray="30 70" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" />
          <circle className="sp-rot-rev" cx={C} cy={C} r={5} pathLength={100} strokeDasharray="34 66" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" />
        </>
      );
    case "fill":
      return (
        <>
          <circle cx={C} cy={C} r={10} fill="none" stroke="currentColor" strokeWidth={1.8} opacity={0.3} />
          <circle className="sp-pie" cx={C} cy={C} r={3.75} transform="rotate(-90 12 12)" pathLength={100} fill="none" stroke="currentColor" strokeWidth={7.5} />
        </>
      );
    case "gather":
      return (
        <g className="sp-gather">
          {[
            [-1, -1],
            [1, -1],
            [-1, 1],
            [1, 1],
          ].map(([x, y], i) => (
            <rect key={i} className={`sp-g sp-g${i}`} x={C + x * 4.6 - 3.6} y={C + y * 4.6 - 3.6} width={7.2} height={7.2} rx={2} fill="currentColor" />
          ))}
        </g>
      );
    case "leap":
      return R(3).map((i) => <circle key={i} className="sp-leap" style={{ animationDelay: `${(-i * ms) / 3}ms` }} cx={5} cy={14} r={2.4} fill="currentColor" />);
    case "linear-dots":
      return R(3).map((i) => <circle key={i} className="sp-fade" style={{ animationDelay: `${(i * ms) / 5 - ms}ms` }} cx={5 + i * 7} cy={C} r={2.4} fill="currentColor" />);
    case "loading":
      return (
        <>
          {R(16).map((i) => {
            const [x, y] = polar(9, i * 22.5);
            return <rect key={i} className="sp-fade" style={d(i, 16, ms)} x={x - 0.95} y={y - 0.95} width={1.9} height={1.9} fill="currentColor" />;
          })}
          {R(10).map((i) => {
            const [x, y] = polar(6, i * 36 + 18);
            return <rect key={`in${i}`} className="sp-fade" style={d(i, 10, ms)} x={x - 0.8} y={y - 0.8} width={1.6} height={1.6} fill="currentColor" opacity={0.7} />;
          })}
        </>
      );
    case "loop":
      return (
        <>
          <path d={LOOP} fill="none" stroke="currentColor" strokeWidth={2.4} style={track} />
          <path className="sp-dash" d={LOOP} pathLength={100} strokeDasharray="26 74" fill="none" stroke="currentColor" strokeWidth={2.4} strokeLinecap="round" />
        </>
      );
    case "morph":
      return <rect className="sp-morph" x={5.5} y={5.5} width={13} height={13} rx={6.5} fill="currentColor" />;
    case "orbit":
      return (
        <>
          <circle cx={C} cy={C} r={2.6} fill="currentColor" />
          <circle className="sp-rot sp-orbit" cx={C} cy={C} r={8.5} pathLength={100} strokeDasharray="40 60" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" />
        </>
      );
    case "phases": {
      const m = `sp-moon-${uid}`;
      return (
        <>
          <defs>
            <mask id={m}>
              <path className="sp-ph-r" d="M12 2a10 10 0 0 1 0 20Z" fill="#fff" />
              <path className="sp-ph-l" d="M12 2a10 10 0 0 0 0 20Z" fill="#fff" />
              <ellipse className="sp-ph-dark" cx={C} cy={C} rx={10.2} ry={10.2} fill="#000" />
              <ellipse className="sp-ph-lit" cx={C} cy={C} rx={10} ry={10} fill="#fff" />
            </mask>
          </defs>
          <circle cx={C} cy={C} r={10} fill="currentColor" opacity={0.25} />
          <circle cx={C} cy={C} r={10} fill="currentColor" mask={`url(#${m})`} />
        </>
      );
    }
    case "pulse":
      return (
        <>
          <circle className="sp-halo" cx={C} cy={C} r={10} fill="currentColor" />
          <circle className="sp-beat" cx={C} cy={C} r={3.4} fill="currentColor" />
        </>
      );
    case "radar":
      return (
        <>
          <circle cx={C} cy={C} r={10.5} fill="currentColor" opacity={0.14} />
          <g className="sp-rot">
            {RADAR_TAIL.map((o, i) => (
              <path key={i} d={sector(10.5, -18 * (i + 1), -18 * i)} fill="currentColor" opacity={o} />
            ))}
          </g>
          <circle cx={C} cy={C} r={5.6} fill="none" stroke="currentColor" strokeWidth={1.2} opacity={0.45} />
          <circle cx={C} cy={C} r={2.4} fill="currentColor" />
        </>
      );
    case "ring":
      return (
        <>
          <circle cx={C} cy={C} r={9} fill="none" stroke="currentColor" strokeWidth={2.4} style={track} />
          <circle className="sp-rot" cx={C} cy={C} r={9} pathLength={100} strokeDasharray="25 75" fill="none" stroke="currentColor" strokeWidth={2.4} strokeLinecap="round" />
        </>
      );
    case "ripple":
      return R(3).map((i) => <circle key={i} className="sp-ripple" style={d(i, 3, ms)} cx={C} cy={C} r={10} fill="none" stroke="currentColor" strokeWidth={1.4} />);
    case "scan":
      return (
        <>
          <rect x={3} y={3} width={18} height={18} rx={4.5} fill="none" stroke="currentColor" strokeWidth={2} opacity={0.4} />
          <rect className="sp-scan" x={6.5} y={6.5} width={11} height={2.6} rx={1.3} fill="currentColor" />
        </>
      );
    case "slide":
      return R(3).map((i) => <circle key={i} className={`sp-slide sp-slide${i}`} cx={7} cy={7} r={3.6} fill="currentColor" />);
    case "snake":
      return (
        <g className="sp-rot-slow">
          <circle className="sp-snake" cx={C} cy={C} r={9} pathLength={100} fill="none" stroke="currentColor" strokeWidth={2.4} strokeLinecap="round" />
        </g>
      );
    case "split":
      return (
        <>
          <circle className="sp-split-l" cx={7.4} cy={C} r={2.9} fill="currentColor" />
          <circle className="sp-split-r" cx={16.6} cy={C} r={2.9} fill="currentColor" />
        </>
      );
    case "steps":
      return R(4).map((i) => <rect key={i} className="sp-step" style={{ animationDelay: `${(i * ms) / 8 - ms}ms` }} x={2.5 + i * 5} y={8.5} width={3.6} height={7} rx={1.1} fill="currentColor" />);
    case "swirl":
      return RING8.map(([c, r], i) => <circle key={i} className="sp-fade" style={d(i, 8, ms)} cx={GRID3[c]} cy={GRID3[r]} r={2.3} fill="currentColor" />);
    case "trace":
      return (
        <>
          <rect x={3.5} y={3.5} width={17} height={17} rx={5} fill="none" stroke="currentColor" strokeWidth={2.2} style={track} />
          <rect className="sp-dash" x={3.5} y={3.5} width={17} height={17} rx={5} pathLength={100} strokeDasharray="22 78" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" />
        </>
      );
    case "bars":
      return R(4).map((i) => <rect key={i} className="sp-bar" style={{ animationDelay: `${[-0.2, -0.65, -0.4, -0.85][i] * ms}ms` }} x={3.6 + i * 4.6} y={4} width={3} height={16} rx={1.5} fill="currentColor" />);
    case "hourglass":
      return (
        <g className="sp-hg">
          <path d="M6 3h12M6 21h12M7.5 3c0 4.6 3.6 6.2 3.6 9s-3.6 4.4-3.6 9M16.5 3c0 4.6-3.6 6.2-3.6 9s3.6 4.4 3.6 9" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" />
          <path className="sp-hg-top" d="M9 6.5h6c-.8 1.8-3 2.6-3 4.2 0-1.6-2.2-2.4-3-4.2Z" fill="currentColor" />
          <path className="sp-hg-bot" d="M8.6 19.6h6.8c-.6-2.2-2.4-3.2-3.4-4.6-1 1.4-2.8 2.4-3.4 4.6Z" fill="currentColor" />
        </g>
      );
    default:
      return null;
  }
}

/** HTML spinners: real 3D (flip) and a z-order swap (eclipse). */
function htmlBody(kind: SpinnerKind): ReactNode | null {
  if (kind === "flip") return <i className="sp-flip-tile" />;
  if (kind === "eclipse")
    return (
      <>
        <i className="sp-ecl sp-ecl-a" />
        <i className="sp-ecl sp-ecl-b" />
      </>
    );
  return null;
}

const TONES = new Set<string>(["violet", "purple", "blue", "sky", "cyan", "teal", "green", "lime", "amber", "orange", "red", "pink"]);

export function Spinner({
  kind = "ring",
  size = 16,
  tone,
  color,
  label,
  className = "",
  style,
}: {
  kind?: SpinnerKind;
  /** Pixels; 16–24 is the intended range. */
  size?: number;
  /** An app colour token (--c-*); "mono" is the ink colour. */
  tone?: Tone;
  /** Any CSS colour; wins over tone. Without either the spinner uses currentColor. */
  color?: string;
  /** Accessible name. Without one the spinner is decorative. */
  label?: string;
  className?: string;
  style?: CSSProperties;
}) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const info = INFO[kind] ?? INFO.ring;
  const c = color ?? (tone === "mono" ? "var(--ink-2)" : tone && TONES.has(tone) ? `var(--c-${tone})` : undefined);
  const html = htmlBody(info.kind);
  const css = { width: size, height: size, ["--sp-dur" as string]: `${info.ms}ms`, ...(c ? { color: c } : null), ...style } as CSSProperties;
  const a11y = label ? { role: "img" as const, "aria-label": label } : { "aria-hidden": true as const };
  if (html)
    return (
      <span className={`spinner sp-${info.kind} ${className}`} style={css} {...a11y}>
        {html}
      </span>
    );
  return (
    <svg className={`spinner sp-${info.kind} ${className}`} style={css} viewBox="0 0 24 24" width={size} height={size} focusable="false" {...a11y}>
      {body(info.kind, info.ms, uid)}
    </svg>
  );
}
