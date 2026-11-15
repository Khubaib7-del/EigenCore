import { useEffect, useState } from "react";
import type { FitVerdict } from "../api/types";

/*
 * The empty-chat wordmark: "EIGEN [computer] CORE". A small desktop computer
 * (CRT on a base, keyboard in front) sits between the two words, standing on
 * their baseline. Original CSS art, no images.
 *
 * The CRT is a button with a live screen. Everything on it is real state:
 *  - boot: a short typed boot line the first time it shows;
 *  - idle: the EigenCore mark and a prompt with a blinking cursor, or what
 *    the engine is doing (loading a model, no model yet);
 *  - typing: echoes the last characters in the message box;
 *  - click / Enter / Space: cycles mark → system (version, CPU, threads,
 *    model, context) → fit verdict for the active model (when known).
 * Timers run only while booting; everything else is CSS. Animations pause
 * while the window is hidden and are off under prefers-reduced-motion.
 */

function reducedMotion() {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

const FIT_WORD: Record<Exclude<FitVerdict, "unknown">, [string, string]> = {
  fits_with_margin: ["FITS", "with margin"],
  tight: ["TIGHT", "fits, little spare"],
  does_not_fit: ["NO FIT", "too big for RAM"],
};

type Page = "mark" | "system" | "fit";

let bootedOnce = false;

export function RetroComputer({
  version,
  model,
  cpu,
  threads,
  loading,
  typed,
}: {
  version: string;
  model: { name: string; quant: string; context: number; fit: FitVerdict } | null;
  cpu: string;
  threads: string;
  loading: boolean;
  /** What is in the message box right now (echoed on the screen). */
  typed: string;
}) {
  const bootLine = `EIGENCORE v${version}`;
  const [boot, setBoot] = useState(() => (bootedOnce || reducedMotion() ? bootLine.length + 1 : 0));
  const [page, setPage] = useState<Page>("mark");
  const [hidden, setHidden] = useState(typeof document !== "undefined" && document.hidden);

  // Boot: type one line, then hold "OK" briefly. Runs once per app session.
  useEffect(() => {
    if (boot > bootLine.length) {
      bootedOnce = true;
      return;
    }
    const t = setTimeout(() => setBoot((n) => n + 1), boot === bootLine.length ? 420 : 38);
    return () => clearTimeout(t);
  }, [boot, bootLine.length]);

  useEffect(() => {
    const on = () => setHidden(document.hidden);
    document.addEventListener("visibilitychange", on);
    return () => document.removeEventListener("visibilitychange", on);
  }, []);

  // Typing always brings the echo back into view.
  useEffect(() => {
    if (typed) setPage("mark");
  }, [typed]);

  const hasFit = !!model && !loading && model.fit !== "unknown";
  const pages: Page[] = hasFit ? ["mark", "system", "fit"] : ["mark", "system"];
  const cur = pages.includes(page) ? page : "mark";
  const next = () => setPage(pages[(pages.indexOf(cur) + 1) % pages.length]);

  const booting = boot <= bootLine.length;
  const echo = typed.replace(/\s+/g, " ");
  const status = loading ? "loading model" : model ? "ready" : "no model";

  let body;
  if (booting) {
    body = (
      <div className="rc-lines">
        <div className="rc-hi">{bootLine.slice(0, boot)}</div>
        {boot >= bootLine.length && <div>boot ok</div>}
      </div>
    );
  } else if (cur === "system") {
    body = (
      <div className="rc-lines">
        <div className="rc-hi">v{version}</div>
        <div>{cpu}</div>
        <div>{threads}</div>
        <div className={loading || !model ? "rc-hi" : ""}>{loading ? "loading…" : model ? model.name : "no model"}</div>
        {model && !loading && <div>ctx {model.context.toLocaleString()}</div>}
      </div>
    );
  } else if (cur === "fit" && model && model.fit !== "unknown") {
    const [word, sub] = FIT_WORD[model.fit];
    body = (
      <div className="rc-lines rc-fit">
        <div>fit on this pc</div>
        <div className="rc-big">{word}</div>
        <div>{sub}</div>
      </div>
    );
  } else {
    body = (
      <div className="rc-lines rc-markpage">
        <div className="rc-mark" aria-hidden>
          EC
        </div>
        {echo ? (
          <div className="rc-prompt">
            &gt;{echo.length > 11 ? "…" + echo.slice(-10) : echo}
            <i className="rc-cursor" />
          </div>
        ) : loading || !model ? (
          <div className="rc-prompt rc-hi">{loading ? "loading…" : "no model"}</div>
        ) : (
          <div className="rc-prompt">
            &gt;<i className="rc-cursor" />
          </div>
        )}
      </div>
    );
  }

  const where = booting ? "booting" : cur === "system" ? "system details" : cur === "fit" ? "model fit" : "EigenCore mark";
  return (
    <span className={`rc ${hidden ? "paused" : ""}`}>
      <span className="rc-monitor">
        <button
          type="button"
          className="rc-screen"
          onClick={next}
          aria-label={`EigenCore screen, showing ${where}. Model: ${status}. Press to show the next page.`}
          title="Click for system details"
        >
          <span className="rc-glass mono" aria-live="off">
            {body}
          </span>
          <span className="rc-scan" aria-hidden />
        </button>
        <span className="rc-chin" aria-hidden>
          <span className="rc-dots">
            {pages.map((p) => (
              <i key={p} className={!booting && p === cur ? "on" : ""} />
            ))}
          </span>
          <span className={`rc-led ${loading ? "busy" : model ? "on" : ""}`} />
        </span>
      </span>
      <span className="rc-base" aria-hidden>
        <span className="rc-slot" />
      </span>
      <span className="rc-keys" aria-hidden />
    </span>
  );
}

/** "EIGEN [computer] CORE": the empty-chat heading. */
export function Wordmark(props: Parameters<typeof RetroComputer>[0]) {
  return (
    <div className="ec-wordmark" role="group" aria-label="EigenCore">
      <span className="wm-word" aria-hidden>
        EIGEN
      </span>
      <RetroComputer {...props} />
      <span className="wm-word" aria-hidden>
        CORE
      </span>
    </div>
  );
}
