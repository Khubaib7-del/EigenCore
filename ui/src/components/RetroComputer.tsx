import { useEffect, useRef, useState, type CSSProperties, type PointerEvent } from "react";
import type { FitVerdict } from "../api/types";

/*
 * The empty-chat wordmark: "EIGEN [computer] CORE". A small 3D desktop
 * computer (CRT with a deep rear housing, on a low desktop case, keyboard in
 * front) sits between the two words, standing on their baseline. Original
 * CSS 3D art, no images, no canvas or WebGL.
 *
 * The CRT is a button with a live screen. Everything on it is real state:
 *  - boot: a short typed boot line the first time it shows;
 *  - idle: the EigenCore mark and a prompt with a blinking cursor, or what
 *    the engine is doing (loading a model, no model yet);
 *  - typing: echoes the last characters in the message box;
 *  - click / Enter / Space: cycles mark → system (version, CPU, threads,
 *    model, context) → fit verdict for the active model (when known).
 * Timers run only while booting; everything else is CSS. Hovering tilts the
 * model a few degrees toward the pointer (one CSS-var write per frame, only
 * while hovered); pressing the screen pushes it in. Animations pause while
 * the window is hidden; tilt and animation are off under reduced motion.
 */

function reducedMotion() {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

const FIT_WORD: Record<Exclude<FitVerdict, "unknown">, [string, string]> = {
  fits_with_margin: ["FITS", "with margin"],
  tight: ["TIGHT", "fits, little spare"],
  does_not_fit: ["NO FIT", "too big for RAM"],
};

/* ── 3D scene ────────────────────────────────────────────────────────────
 * The computer is a CSS 3D model: boxes made of flat faces in a
 * preserve-3d scene, seen three-quarter from the front-left and a little
 * above. Units are em of .rc's font-size; origin is the middle of the desk
 * under the case, y grows down, z grows toward the viewer. Only the faces the
 * camera can see are drawn (front, left, top), so nothing is coplanar and
 * nothing z-fights; each face is shaded in CSS from one top-left light. */

type V3 = [number, number, number];
type BoxDef = { c: V3; s: V3 }; // centre, size (w, h, d)
type FaceKey = "f" | "l" | "t" | "r" | "k" | "bt";

const box = (c: V3, s: V3): BoxDef => ({ c, s });

// Desktop case, a low foot, the monitor's front shell, and the stepped CRT
// housing behind it; a keyboard slab in front.
const CASE = box([0, -0.95, -0.2], [13.6, 1.9, 7.6]);
const FOOT = box([0, -2.1, 0.4], [4.6, 0.4, 3]);
const MON = box([0, -6.9, 1.5], [12.6, 9.2, 1.8]);
const REAR1 = box([0, -6.95, -0.45], [11.2, 8.3, 2.1]);
const REAR2 = box([0, -6.8, -2.85], [8.6, 6.6, 2.7]);
const KEYS = box([0.6, -0.2, 4.75], [11, 0.4, 2.1]);

// The screen opening in the bezel, and how deep the glass sits.
const HOLE = { l: 0.72, r: 0.72, t: 0.66, b: 1.42 };
const RECESS = 0.62;
const front = MON.c[2] + MON.s[2] / 2;
const holeW = MON.s[0] - HOLE.l - HOLE.r;
const holeH = MON.s[1] - HOLE.t - HOLE.b;
const holeX = MON.c[0] - MON.s[0] / 2 + HOLE.l + holeW / 2;
const holeY = MON.c[1] - MON.s[1] / 2 + HOLE.t + holeH / 2;
const BEZEL = MON;
const GLASS = box([holeX, holeY, front - RECESS], [holeW, holeH, 0]);

// Bezel buttons and power LED: small real bumps on the chin.
const chinY = MON.c[1] + MON.s[1] / 2 - HOLE.b / 2;
const monR = MON.c[0] + MON.s[0] / 2;
const BTN1 = box([monR - 1.55, chinY, front + 0.09], [0.5, 0.42, 0.18]);
const BTN2 = box([monR - 0.95, chinY, front + 0.09], [0.5, 0.42, 0.18]);
const LED = box([monR - 2.3, chinY, front + 0.06], [0.55, 0.26, 0.12]);

function face(b: BoxDef, k: FaceKey): CSSProperties {
  const [x, y, z] = b.c;
  const [w, h, d] = b.s;
  let fw = w,
    fh = h,
    t = "";
  switch (k) {
    case "f":
      // Fronts run 0.02em past their edges so the seam with the top and side
      // faces never antialiases to the page behind.
      fw = w + 0.04;
      fh = h + 0.04;
      t = `translate3d(${x}em,${y}em,${z + d / 2}em)`;
      break;
    case "k":
      t = `translate3d(${x}em,${y}em,${z - d / 2}em) rotateY(180deg)`;
      break;
    case "l":
      fw = d;
      t = `translate3d(${x - w / 2}em,${y}em,${z}em) rotateY(-90deg)`;
      break;
    case "r":
      fw = d;
      t = `translate3d(${x + w / 2}em,${y}em,${z}em) rotateY(90deg)`;
      break;
    case "t":
      fh = d;
      t = `translate3d(${x}em,${y - h / 2}em,${z}em) rotateX(90deg)`;
      break;
    case "bt":
      fh = d;
      t = `translate3d(${x}em,${y + h / 2}em,${z}em) rotateX(-90deg)`;
      break;
  }
  return { width: `${fw}em`, height: `${fh}em`, transform: `translate(-50%,-50%) ${t}` };
}

function Box({ b, cls, faces = ["l", "t", "f"] }: { b: BoxDef; cls: string; faces?: FaceKey[] }) {
  return (
    <>
      {faces.map((k) => (
        <span key={k} className={`rc-face ${cls} rc-${k}`} style={face(b, k)} aria-hidden />
      ))}
    </>
  );
}

// Inner walls of the screen opening. Each is the matching face of the hole's
// box turned 180° so it faces into the hole. From the default camera only the
// right wall and the floor show; all four are drawn so a tilt never opens a gap.
const HOLE_BOX = box([holeX, holeY, front - (RECESS + 0.06) / 2], [holeW, holeH, RECESS + 0.06]);
// A dark plate just behind the glass so its rounded corners and the seams
// where it meets the walls never show the page through the shell.
const BACKPLATE = box([holeX, holeY, front - RECESS - 0.06], [holeW + 0.3, holeH + 0.3, 0]);
const WALLS = (["l", "r", "t", "bt"] as FaceKey[]).map((k) => {
  const st = face(HOLE_BOX, k);
  return { k, st: { ...st, transform: `${st.transform} ${k === "l" || k === "r" ? "rotateY(180deg)" : "rotateX(180deg)"}` } };
});

const SHADOW: CSSProperties = {
  width: "20em",
  height: "12em",
  transform: "translate(-50%,-50%) translate3d(0.2em,0.02em,0.6em) rotateX(90deg)",
};

const SCENE = (
  <>
    <span className="rc-face rc-floor" style={SHADOW} aria-hidden />
    <Box b={REAR2} cls="rc-shell rc-rear2" />
    <Box b={REAR1} cls="rc-shell rc-rear1" />
    <Box b={CASE} cls="rc-shell rc-case" faces={["l", "t"]} />
    <span className="rc-face rc-shell rc-case rc-f" style={face(CASE, "f")} aria-hidden>
      <span className="rc-vents" />
      <span className="rc-floppy" />
      <span className="rc-drive-led" />
    </span>
    <Box b={FOOT} cls="rc-shell rc-foot" />
    {/* The monitor's top and side stop short of the bezel's rounded corners. */}
    <Box b={box(MON.c, [MON.s[0] - 1.1, MON.s[1], MON.s[2]])} cls="rc-shell rc-mon" faces={["t"]} />
    <Box b={box(MON.c, [MON.s[0], MON.s[1] - 1.1, MON.s[2]])} cls="rc-shell rc-mon" faces={["l"]} />
    <span className="rc-face rc-backplate" style={face(BACKPLATE, "f")} aria-hidden />
    {WALLS.map(({ k, st }) => (
      <span key={k} className={`rc-face rc-wall rc-wall-${k}`} style={st} aria-hidden />
    ))}
    <Box b={BTN1} cls="rc-bump rc-btn" />
    <Box b={BTN2} cls="rc-bump rc-btn" />
    <Box b={KEYS} cls="rc-shell rc-keys" />
  </>
);

// "EC" in a chunky 6×7 pixel face, one path of unit squares.
const GLYPHS = [
  ["111111", "110000", "110000", "111110", "110000", "110000", "111111"],
  ["011111", "110000", "110000", "110000", "110000", "110000", "011111"],
];
const EC_PATH = GLYPHS.flatMap((rows, g) =>
  rows.flatMap((row, y) => [...row].map((on, x) => (on === "1" ? `M${g * 7 + x} ${y}h1v1h-1z` : ""))),
).join("");

type Page = "home" | "sys" | "fit" | "speed";
const TAB: Record<Page, string> = { home: "HOME", sys: "SYS", fit: "FIT", speed: "SPD" };

let bootedOnce = false;

function clock() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/** A phosphor sparkline of recent generation speeds (tok/s), scaled to the peak. */
function Spark({ data }: { data: number[] }) {
  const pts = data.slice(-24);
  const peak = Math.max(...pts, 1);
  const w = 48;
  const h = 14;
  const step = pts.length > 1 ? w / (pts.length - 1) : w;
  const d = pts.map((v, i) => `${i ? "L" : "M"}${(i * step).toFixed(2)} ${(h - (v / peak) * (h - 1) - 0.5).toFixed(2)}`).join("");
  return (
    <svg className="rc-spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden>
      <path d={`${d}L${w} ${h}L0 ${h}Z`} className="rc-spark-fill" />
      <path d={d} className="rc-spark-line" />
    </svg>
  );
}

export function RetroComputer({
  version,
  model,
  cpu,
  threads,
  loading,
  typed,
  tps = [],
  ram,
}: {
  version: string;
  model: { name: string; quant: string; context: number; fit: FitVerdict } | null;
  cpu: string;
  threads: string;
  loading: boolean;
  /** What is in the message box right now (echoed on the screen). */
  typed: string;
  /** Recent generation speeds in tok/s, oldest first (real samples only). */
  tps?: number[];
  /** Installed RAM, e.g. "16 GB". */
  ram?: string;
}) {
  // A short POST: each line types in, then the screen settles on HOME.
  const post = [
    `EIGENCORE BIOS v${version}`,
    `CPU  ${cpu}`,
    ram ? `MEM  ${ram} OK` : "MEM  OK",
    `LLM  ${loading ? "loading" : model ? model.name : "none"}`,
    "BOOT OK",
  ];
  const total = post.reduce((n, l) => n + l.length, 0);
  const [boot, setBoot] = useState(() => (bootedOnce || reducedMotion() ? total + 1 : 0));
  const [page, setPage] = useState<Page>("home");
  const [hidden, setHidden] = useState(typeof document !== "undefined" && document.hidden);
  const [now, setNow] = useState(clock);

  useEffect(() => {
    if (boot > total) {
      bootedOnce = true;
      return;
    }
    const t = setTimeout(() => setBoot((n) => n + 2), boot >= total - 1 ? 520 : 22);
    return () => clearTimeout(t);
  }, [boot, total]);

  useEffect(() => {
    const on = () => setHidden(document.hidden);
    document.addEventListener("visibilitychange", on);
    const t = setInterval(() => setNow(clock()), 15_000);
    return () => {
      document.removeEventListener("visibilitychange", on);
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    if (typed) setPage("home");
  }, [typed]);

  const hasFit = !!model && !loading && model.fit !== "unknown";
  const pages: Page[] = ["home", "sys", ...(hasFit ? (["fit"] as Page[]) : []), "speed"];
  const cur = pages.includes(page) ? page : "home";
  const next = () => setPage(pages[(pages.indexOf(cur) + 1) % pages.length]);

  const booting = boot <= total;
  const echo = typed.replace(/\s+/g, " ");
  const status = loading ? "LOADING" : model ? "READY" : "NO MODEL";
  const last = tps.length ? tps[tps.length - 1] : null;

  let body;
  if (booting) {
    let left = boot;
    body = (
      <div className="rc-lines rc-post">
        {post.map((l, i) => {
          const n = Math.max(0, Math.min(l.length, left));
          left -= l.length;
          return n > 0 ? (
            <div key={i} className={i === 0 || i === post.length - 1 ? "rc-hi" : ""}>
              {l.slice(0, n)}
            </div>
          ) : null;
        })}
      </div>
    );
  } else if (cur === "sys") {
    body = (
      <div className="rc-lines rc-table">
        <div><b>VER</b> {version}</div>
        <div><b>CPU</b> {cpu} · {threads.split(" ")[0]}</div>
        {ram && <div><b>RAM</b> {ram}</div>}
        <div className="rc-hi"><b>LLM</b> {loading ? "loading…" : model ? `${model.name}` : "none"}</div>
        {model && !loading && <div><b>CTX</b> {model.context.toLocaleString()} · {model.quant}</div>}
      </div>
    );
  } else if (cur === "fit" && model && model.fit !== "unknown") {
    const [word, sub] = FIT_WORD[model.fit];
    const lvl = model.fit === "fits_with_margin" ? 8 : model.fit === "tight" ? 5 : 2;
    body = (
      <div className="rc-lines rc-fit">
        <div>fit on this pc</div>
        <div className="rc-big">{word}</div>
        <div className="rc-meter" aria-hidden>
          {Array.from({ length: 10 }, (_, i) => (
            <i key={i} className={i < lvl ? "on" : ""} />
          ))}
        </div>
        <div>{sub}</div>
      </div>
    );
  } else if (cur === "speed") {
    body = tps.length ? (
      <div className="rc-lines rc-speed">
        <div className="rc-row">
          <span>tok/s</span>
          <span className="rc-big">{last!.toFixed(1)}</span>
        </div>
        <Spark data={tps} />
        <div className="rc-row">
          <span>peak {Math.max(...tps).toFixed(1)}</span>
          <span>{tps.length} runs</span>
        </div>
      </div>
    ) : (
      <div className="rc-lines rc-fit">
        <div>tok/s</div>
        <div className="rc-big">--</div>
        <div>no runs yet</div>
      </div>
    );
  } else {
    body = (
      <div className="rc-lines rc-markpage">
        <svg className="rc-mark" viewBox="0 0 13 7" aria-hidden shapeRendering="crispEdges">
          <path d={EC_PATH} />
        </svg>
        {echo ? (
          <div className="rc-prompt">
            &gt; {echo.length > 18 ? "…" + echo.slice(-17) : echo}
            <i className="rc-cursor" />
          </div>
        ) : (
          <div className="rc-prompt">
            &gt; {loading ? "loading model…" : model ? "ask me anything" : "pick a model"}
            <i className="rc-cursor" />
          </div>
        )}
      </div>
    );
  }

  const where = booting ? "booting" : { home: "home", sys: "system details", fit: "model fit", speed: "speed" }[cur];

  const rootRef = useRef<HTMLSpanElement>(null);
  const rafRef = useRef(0);
  const ptRef = useRef({ x: 0, y: 0 });
  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);
  const onMove = (e: PointerEvent<HTMLSpanElement>) => {
    if (e.pointerType === "touch" || reducedMotion()) return;
    const r = e.currentTarget.getBoundingClientRect();
    const clamp = (v: number) => Math.max(-1, Math.min(1, v));
    ptRef.current = { x: clamp(((e.clientX - r.left) / r.width) * 2 - 1), y: clamp(((e.clientY - r.top) / r.height) * 2 - 1) };
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = 0;
      const el = rootRef.current;
      if (!el) return;
      el.classList.add("tilting");
      el.style.setProperty("--tx", ptRef.current.x.toFixed(3));
      el.style.setProperty("--ty", ptRef.current.y.toFixed(3));
    });
  };
  const onLeave = () => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
    const el = rootRef.current;
    if (!el) return;
    el.classList.remove("tilting");
    el.style.removeProperty("--tx");
    el.style.removeProperty("--ty");
  };

  return (
    <span ref={rootRef} className={`rc ${hidden ? "paused" : ""}`} onPointerMove={onMove} onPointerLeave={onLeave}>
      <span className="rc-stage">
        <span className="rc-obj">
          {SCENE}
          <span className="rc-face rc-bezel" style={face(BEZEL, "f")} aria-hidden>
          </span>
          <Box b={LED} cls={`rc-bump rc-led ${loading ? "busy" : model ? "on" : ""}`} />
          <button
            type="button"
            className="rc-face rc-screen"
            style={face(GLASS, "f")}
            onClick={next}
            aria-label={`EigenCore screen, showing ${where}. Model: ${status.toLowerCase()}. Press to show the next page.`}
            title="Click to flip pages"
          >
            <span className="rc-glass mono" aria-live="off">
              <span className="rc-bar">
                <span>
                  <i className={`rc-dot ${loading ? "busy" : model ? "on" : ""}`} />
                  {booting ? "POST" : status}
                </span>
                <span>{now}</span>
              </span>
              <span className="rc-body">{body}</span>
              <span className="rc-tabs">
                {pages.map((p) => (
                  <i key={p} className={!booting && p === cur ? "on" : ""}>
                    {TAB[p]}
                  </i>
                ))}
              </span>
            </span>
            <span className="rc-scan" aria-hidden />
          </button>
        </span>
      </span>
    </span>
  );
}

/** Split a word into letter spans for the staggered entrance and the idle wave. */
function Letters({ word, from }: { word: string; from: number }) {
  return (
    <>
      {[...word].map((ch, i) => (
        <span key={i} className="wm-l" style={{ "--i": from + i } as CSSProperties}>
          {ch}
        </span>
      ))}
    </>
  );
}

/** "EIGEN [computer] CORE": the empty-chat heading. */
export function Wordmark(props: Parameters<typeof RetroComputer>[0]) {
  return (
    <div className="ec-wordmark" role="group" aria-label="EigenCore">
      <span className="wm-word" aria-hidden>
        <Letters word="EIGEN" from={0} />
      </span>
      <RetroComputer {...props} />
      <span className="wm-word" aria-hidden>
        <Letters word="CORE" from={6} />
      </span>
    </div>
  );
}
