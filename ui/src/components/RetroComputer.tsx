import { useEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent as ReactMouseEvent } from "react";

/*
 * The empty-chat centrepiece: a small beige/grey computer with a CRT that
 * boots into EigenCore. Original CSS art (no images). Interactive:
 *  - hover tilts it towards the pointer and brightens the phosphor;
 *  - clicking the screen reboots it (replays the boot text);
 *  - clicking keys types on the screen's prompt; ↵ puts that text in the composer;
 *  - the suggestions on the screen put their prompt in the composer.
 * Everything on screen is real: version, CPU, threads, the loaded model.
 * Under prefers-reduced-motion there is no tilt, no typing and no flicker.
 */

export interface RetroSuggestion {
  short: string;
  text: string;
}

const ROWS = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"];

function reducedMotion() {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function RetroComputer({
  version,
  model,
  cpu,
  threads,
  loading,
  suggestions,
  onFill,
}: {
  version: string;
  model: { name: string; quant: string; context: number } | null;
  cpu: string;
  threads: string;
  loading: boolean;
  suggestions: RetroSuggestion[];
  onFill: (text: string) => void;
}) {
  const script = useMemo(
    () => [
      `EIGENCORE v${version}  local runtime`,
      `CPU   ${cpu}`,
      `      ${threads}`,
      loading ? "MODEL loading…" : model ? `MODEL ${model.name} ${model.quant}` : "MODEL none loaded: open Models",
      model && !loading ? `CTX   ${model.context.toLocaleString()} tokens` : "",
      model && !loading ? "READY." : "WAITING FOR A MODEL.",
    ].filter(Boolean),
    [version, cpu, threads, model, loading],
  );
  const total = script.reduce((n, l) => n + l.length + 1, 0);
  const [shown, setShown] = useState(() => (reducedMotion() ? total : 0));
  const [typed, setTyped] = useState("");
  const [tilt, setTilt] = useState({ x: 0, y: 0, on: false });
  const [pressed, setPressed] = useState<string | null>(null);
  const boot = useRef(0);

  // Type the boot text out, a few characters per frame.
  useEffect(() => {
    if (reducedMotion()) {
      setShown(total);
      return;
    }
    if (shown >= total) return;
    const t = setTimeout(() => setShown((n) => Math.min(total, n + 3)), 16);
    return () => clearTimeout(t);
  }, [shown, total]);
  // Re-type when what the screen says changes (model loaded, switched...).
  useEffect(() => {
    if (!reducedMotion()) setShown((n) => Math.min(n, script[0].length + 1));
  }, [script]);

  const reboot = () => {
    boot.current += 1;
    setTyped("");
    setShown(reducedMotion() ? total : 0);
  };
  const press = (k: string) => {
    setPressed(k);
    setTimeout(() => setPressed((p) => (p === k ? null : p)), 140);
    if (k === "↵") {
      if (typed.trim()) onFill(typed.trim().toLowerCase());
      setTyped("");
    } else if (k === "⌫") setTyped((t) => t.slice(0, -1));
    else if (k === "space") setTyped((t) => (t.length < 26 ? t + " " : t));
    else setTyped((t) => (t.length < 26 ? t + k.toLowerCase() : t));
  };
  const onMove = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (reducedMotion()) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5;
    const y = (e.clientY - r.top) / r.height - 0.5;
    setTilt({ x: x * 10, y: -y * 8, on: true });
  };

  // Lay the revealed characters over the script lines.
  let left = shown;
  const lines = script.map((l) => {
    const part = l.slice(0, Math.max(0, left));
    left -= l.length + 1;
    return part;
  });
  const booted = shown >= total;
  const style = { "--rx": `${tilt.x.toFixed(2)}deg`, "--ry": `${tilt.y.toFixed(2)}deg` } as CSSProperties;

  return (
    <div
      className={`retro ${tilt.on ? "hover" : ""}`}
      style={style}
      onMouseMove={onMove}
      onMouseLeave={() => setTilt({ x: 0, y: 0, on: false })}
    >
      <div className="retro-unit">
        <div className="retro-monitor">
          <div className="retro-bezel">
            <div
              className="retro-screen"
              role="button"
              tabIndex={0}
              aria-label="EigenCore screen. Click to restart it."
              onClick={reboot}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  reboot();
                }
              }}
            >
              <div className="retro-text mono">
                {lines.map((l, i) =>
                  l ? (
                    <div key={`${boot.current}-${i}`} className={i === 0 ? "retro-title" : script[i].startsWith("READY") || script[i].startsWith("WAITING") ? "retro-ready" : ""}>
                      {l}
                    </div>
                  ) : null,
                )}
                {booted && (
                  <>
                    <div className="retro-prompt">
                      <span>&gt; {typed}</span>
                      <span className="retro-cursor" aria-hidden />
                    </div>
                    <div className="retro-sugs">
                      {suggestions.map((s, i) => (
                        <button
                          key={s.text}
                          className="retro-sug"
                          onClick={(e) => {
                            e.stopPropagation();
                            onFill(s.text);
                          }}
                          title={s.text}
                        >
                          [{i + 1}] {s.short}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
              <span className="retro-scan" aria-hidden />
              <span className="retro-glare" aria-hidden />
            </div>
          </div>
          <div className="retro-chin">
            <span className="retro-badge">EIGENCORE</span>
            <span className="grow" />
            <span className="retro-vents" aria-hidden>
              <i />
              <i />
              <i />
              <i />
            </span>
            <span className={`retro-led ${loading ? "busy" : model ? "on" : ""}`} aria-hidden />
          </div>
        </div>
        <div className="retro-neck" aria-hidden />
        <div className="retro-keyboard" role="group" aria-label="Keyboard: type on the screen, Enter puts it in the message box">
          {ROWS.map((row, r) => (
            <div key={row} className={`retro-row r${r}`}>
              {row.split("").map((k) => (
                <button key={k} className={`retro-key ${pressed === k ? "down" : ""}`} onClick={() => press(k)} tabIndex={-1}>
                  {k}
                </button>
              ))}
              {r === 1 && (
                <button className={`retro-key wide ${pressed === "↵" ? "down" : ""}`} onClick={() => press("↵")} aria-label="Enter: put the typed text in the message box">
                  ↵
                </button>
              )}
              {r === 2 && (
                <button className={`retro-key wide ${pressed === "⌫" ? "down" : ""}`} onClick={() => press("⌫")} aria-label="Backspace" tabIndex={-1}>
                  ⌫
                </button>
              )}
            </div>
          ))}
          <div className="retro-row r3">
            <button className={`retro-key space ${pressed === "space" ? "down" : ""}`} onClick={() => press("space")} aria-label="Space" tabIndex={-1} />
          </div>
        </div>
      </div>
      <div className="retro-shadow" aria-hidden />
    </div>
  );
}
