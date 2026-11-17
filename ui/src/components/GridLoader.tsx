import { useSyncExternalStore } from "react";

/* ── Grid loader: one loader for the whole app ───────────────────────────
 * A 3×3 grid of cells. Each pattern is a frame function: step → the level
 * (0..1) of each of the nine cells. Every loader on screen reads one shared
 * clock, so two loaders with the same pattern move in step. Cells fade with
 * a short CSS transition, so a pattern with few frames still reads as motion.
 *
 * Patterns are generic; "kinds" give each job in the app its own pattern and
 * colour (thinking, writing, code, tools, loading a model...). Colour comes
 * from the --gl-* tokens, which have a dark and a light value.
 */

export type GridPattern =
  | "drive"
  | "orbit"
  | "pulse"
  | "snake"
  | "rain"
  | "scan"
  | "checker"
  | "spiral"
  | "cross"
  | "diagonal"
  | "sparkle"
  | "heartbeat"
  | "leap"
  | "fill"
  | "wave"
  | "breathe"
  | "check"
  | "fail"
  | "halt";

const RC = (i: number) => [Math.floor(i / 3), i % 3] as const;
const ring = [0, 1, 2, 5, 8, 7, 6, 3];
const snakePath = [0, 1, 2, 5, 4, 3, 6, 7, 8];
const spiralPath = [0, 1, 2, 5, 8, 7, 6, 3, 4];
const twinkle = [4, 0, 8, 2, 6, 1, 7, 3, 5, 4, 2, 6, 0, 8, 5, 3, 1, 7];
const statics: Partial<Record<GridPattern, number[]>> = {
  check: [3, 7, 5, 2],
  fail: [0, 2, 4, 6, 8],
  halt: [3, 4, 5],
};

/** Frames per cycle and the level of cell i at step s. */
const PATTERNS: Record<GridPattern, { frames: number; ms: number; level: (i: number, s: number) => number }> = {
  // chevron wavefront moving right
  drive: {
    frames: 6,
    ms: 110,
    level: (i, s) => {
      const [r, c] = RC(i);
      const d = c + Math.abs(r - 1);
      return d === s % 6 ? 1 : d === (s + 5) % 6 ? 0.45 : 0;
    },
  },
  // comet round the edge, two-cell tail
  orbit: {
    frames: 8,
    ms: 95,
    level: (i, s) => {
      const k = ring.indexOf(i);
      if (k < 0) return 0;
      const b = (s - k + 8) % 8;
      return b === 0 ? 1 : b === 1 ? 0.5 : b === 2 ? 0.2 : 0;
    },
  },
  // centre, then plus, then corners: a ripple
  pulse: {
    frames: 4,
    ms: 170,
    level: (i, s) => {
      const [r, c] = RC(i);
      const d = Math.abs(r - 1) + Math.abs(c - 1);
      return d === s ? 1 : d === s - 1 ? 0.35 : 0;
    },
  },
  // snake through the rows, three-cell body
  snake: {
    frames: 9,
    ms: 100,
    level: (i, s) => {
      const k = snakePath.indexOf(i);
      const b = (s - k + 9) % 9;
      return b === 0 ? 1 : b === 1 ? 0.6 : b === 2 ? 0.25 : 0;
    },
  },
  // columns fall at different speeds, like code scrolling by
  rain: {
    frames: 6,
    ms: 110,
    level: (i, s) => {
      const [r, c] = RC(i);
      const head = (s + [0, 2, 4][c]) % 6;
      return head === r ? 1 : head === r + 1 ? 0.4 : 0;
    },
  },
  // a scanline sweeps down and back up
  scan: {
    frames: 4,
    ms: 150,
    level: (i, s) => {
      const [r] = RC(i);
      const row = [0, 1, 2, 1][s % 4];
      return r === row ? 1 : 0.08;
    },
  },
  // checkerboard flips
  checker: { frames: 2, ms: 320, level: (i, s) => ((i + s) % 2 === 0 ? 1 : 0.1) },
  // fills a spiral into the centre, then empties it
  spiral: {
    frames: 18,
    ms: 80,
    level: (i, s) => {
      const k = spiralPath.indexOf(i);
      return s < 9 ? (k <= s ? 1 : 0) : k > s - 9 ? 1 : 0;
    },
  },
  // plus morphs into an X
  cross: {
    frames: 2,
    ms: 360,
    level: (i, s) => {
      const [r, c] = RC(i);
      const plus = r === 1 || c === 1;
      const x = r === c || r + c === 2;
      return (s ? x : plus) ? 1 : 0.08;
    },
  },
  // diagonal band from top-left to bottom-right
  diagonal: {
    frames: 6,
    ms: 120,
    level: (i, s) => {
      const [r, c] = RC(i);
      const d = r + c;
      return d === s % 6 ? 1 : d === (s + 5) % 6 ? 0.4 : 0;
    },
  },
  // two cells twinkle at a time
  sparkle: { frames: twinkle.length, ms: 120, level: (i, s) => (twinkle[s] === i ? 1 : twinkle[(s + twinkle.length - 1) % twinkle.length] === i ? 0.5 : 0.08) },
  // lub-dub: two beats, then rest
  heartbeat: {
    frames: 8,
    ms: 110,
    level: (i, s) => {
      const [r, c] = RC(i);
      const core = r === 1 || c === 1;
      const on = s === 0 || s === 2;
      return on ? (core ? 1 : 0.55) : s === 1 || s === 3 ? (core ? 0.4 : 0.1) : core ? 0.2 : 0.06;
    },
  },
  // a dot leaps corner to corner through the centre
  leap: {
    frames: 8,
    ms: 120,
    level: (i, s) => {
      const path = [0, 4, 8, 4, 2, 4, 6, 4];
      return path[s] === i ? 1 : path[(s + 7) % 8] === i ? 0.35 : 0.06;
    },
  },
  // fills bottom-up, row by row, then clears (loading, downloading)
  fill: {
    frames: 5,
    ms: 220,
    level: (i, s) => {
      const [r] = RC(i);
      return s >= 1 && 3 - r <= s && s < 4 ? 1 : s === 4 ? 0.15 : 0.08;
    },
  },
  // equaliser: three columns rise and fall
  wave: {
    frames: 8,
    ms: 100,
    level: (i, s) => {
      const [r, c] = RC(i);
      const h = [1, 2, 3, 2, 1, 2, 3, 2][(s + c * 2) % 8];
      return 2 - r < h ? 1 : 0.08;
    },
  },
  // the whole grid breathes (idle, ready)
  breathe: {
    frames: 10,
    ms: 160,
    level: (i, s) => {
      const [r, c] = RC(i);
      const t = 0.35 + 0.65 * Math.abs(Math.sin((s / 10) * Math.PI));
      return r === 1 && c === 1 ? t : t * 0.55;
    },
  },
  check: { frames: 1, ms: 1000, level: (i) => (statics.check!.includes(i) ? 1 : 0.1) },
  fail: { frames: 1, ms: 1000, level: (i) => (statics.fail!.includes(i) ? 1 : 0.1) },
  halt: { frames: 1, ms: 1000, level: (i) => (statics.halt!.includes(i) ? 1 : 0.1) },
};

export const GRID_PATTERNS = Object.keys(PATTERNS) as GridPattern[];

/** What each job in the app looks like. */
export type LoaderKind =
  | "thinking"
  | "reasoning"
  | "writing"
  | "code"
  | "tool"
  | "approval"
  | "loading"
  | "download"
  | "update"
  | "search"
  | "live"
  | "idle"
  | "done"
  | "error"
  | "stopped";

export const KINDS: Record<LoaderKind, { pattern: GridPattern; tone: string; label: string }> = {
  thinking: { pattern: "pulse", tone: "violet", label: "Thinking" },
  reasoning: { pattern: "spiral", tone: "purple", label: "Reasoning" },
  writing: { pattern: "drive", tone: "blue", label: "Writing" },
  code: { pattern: "rain", tone: "cyan", label: "Writing code" },
  tool: { pattern: "snake", tone: "orange", label: "Running a tool" },
  approval: { pattern: "heartbeat", tone: "amber", label: "Waiting for approval" },
  loading: { pattern: "fill", tone: "green", label: "Loading the model" },
  download: { pattern: "wave", tone: "teal", label: "Downloading" },
  update: { pattern: "orbit", tone: "teal", label: "Checking for updates" },
  search: { pattern: "scan", tone: "sky", label: "Searching" },
  live: { pattern: "wave", tone: "pink", label: "Live" },
  idle: { pattern: "breathe", tone: "mono", label: "Ready" },
  done: { pattern: "check", tone: "green", label: "Done" },
  error: { pattern: "fail", tone: "red", label: "Error" },
  stopped: { pattern: "halt", tone: "amber", label: "Stopped" },
};

/* One shared clock for every loader on screen. */
let now = 0;
const subs = new Set<() => void>();
let timer: ReturnType<typeof setInterval> | null = null;
const reduce = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
function subscribe(fn: () => void) {
  subs.add(fn);
  if (!timer && !reduce)
    timer = setInterval(() => {
      now = performance.now();
      subs.forEach((f) => f());
    }, 40);
  return () => {
    subs.delete(fn);
    if (!subs.size && timer) {
      clearInterval(timer);
      timer = null;
    }
  };
}
const snap = () => now;

export function GridLoader({
  kind,
  pattern,
  tone,
  size = 14,
  round = false,
  label,
  className = "",
}: {
  kind?: LoaderKind;
  pattern?: GridPattern;
  tone?: string;
  size?: number;
  round?: boolean;
  /** Accessible name; defaults to the kind's label. Without one the loader is decorative. */
  label?: string;
  className?: string;
}) {
  const k = kind ? KINDS[kind] : undefined;
  const p = PATTERNS[pattern ?? k?.pattern ?? "drive"];
  const t = useSyncExternalStore(subscribe, snap, snap);
  const step = p.frames === 1 ? 0 : Math.floor(t / p.ms) % p.frames;
  const name = label ?? k?.label;
  const gap = Math.max(1, Math.round(size / 9));
  return (
    <span
      className={`grid-loader tone-${tone ?? k?.tone ?? "mono"} ${round ? "round" : ""} ${className}`}
      style={{ width: size, height: size, gap, ["--gl-ms" as string]: `${Math.min(p.ms, 200)}ms` }}
      role={name ? "img" : undefined}
      aria-label={name}
      aria-hidden={name ? undefined : true}
    >
      {Array.from({ length: 9 }, (_, i) => (
        <i key={i} style={{ opacity: 0.12 + 0.88 * p.level(i, step) }} />
      ))}
    </span>
  );
}
