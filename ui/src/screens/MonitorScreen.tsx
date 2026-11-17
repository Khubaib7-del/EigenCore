import { useState } from "react";
import { api } from "../api";
import { Icon, type IconName } from "../components/Icon";
import { GridLoader } from "../components/GridLoader";
import { Spinner } from "../components/Spinner";
import { Chip, Ring, SampleBadge, Segmented, TpsChart } from "../components/primitives";
import { useStore, type EnginePhase } from "../state/store";

const PHASES: { id: EnginePhase; label: string }[] = [
  { id: "idle", label: "Idle" },
  { id: "loading", label: "Loading" },
  { id: "reading", label: "Reading prompt" },
  { id: "writing", label: "Writing" },
];

function Stat({ icon, label, value, unit, foot, tint }: { icon: IconName; label: string; value: string; unit?: string; foot: string; tint: string }) {
  return (
    <div className="card stat">
      <div className="stat-label">
        <span className={`stat-ico ${tint}`}>
          <Icon name={icon} size={14} />
        </span>
        {label}
      </div>
      <div className="stat-value tnum">
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </div>
      <div className="stat-foot muted">{foot}</div>
    </div>
  );
}

/**
 * Tokens per second. "This reply": live samples the backend measures every
 * ~0.5 s while a reply is written ("tps" events). "Per reply": the output
 * rate of each finished reply. Real numbers in the desktop app; the browser
 * preview labels its numbers mock or sample.
 */
function SpeedCard() {
  const { state } = useStore();
  const live = state.liveTps;
  const [view, setView] = useState<"live" | "replies">(live.length || !state.tps.length ? "live" : "replies");
  const writing = state.engine.phase === "writing" || state.engine.phase === "reading";
  const values = view === "live" ? live.map((p) => p.rate) : state.tps;
  const xs = view === "live" ? live.map((p) => p.t) : state.tps.map((_, i) => i + 1);
  const cur = values.length ? values[values.length - 1] : null;
  const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  const peak = values.length ? Math.max(...values) : null;
  const kind = state.sampleTelemetry ? "sample" : api.kind === "mock" ? "mock stream" : "measured";
  const empty =
    view === "live"
      ? writing
        ? "Measuring… the first sample arrives after half a second of writing"
        : "Send a message: speed is drawn live while the reply is written"
      : "No replies yet: each finished reply adds a point";
  const fmt = (v: number | null) => (v == null ? "—" : v.toFixed(1));
  return (
    <div className="card chart-card">
      <div className="card-head chart-head">
        <span className="chart-title">
          <Icon name="activity" size={14} />
          Tokens per second
        </span>
        <span className={`chip chart-kind ${kind === "measured" ? "chip-ok" : "sample"}`} title={kind === "measured" ? "Timed by the engine on this PC" : "Not a measurement of a model"}>
          {kind === "measured" && <Icon name="check" size={11} />}
          {kind}
        </span>
        <span className="grow" />
        <Segmented
          label="Speed view"
          value={view}
          onChange={setView}
          options={[
            { value: "live", label: "This reply" },
            { value: "replies", label: "Per reply" },
          ]}
        />
      </div>
      <div className="chart-stats">
        <div>
          <span className="muted">{view === "live" ? (writing ? "Now" : "Last sample") : "Last reply"}</span>
          <b className="mono tnum">
            {fmt(cur)}
            <small> tok/s</small>
          </b>
        </div>
        <div>
          <span className="muted">Average</span>
          <b className="mono tnum">
            {fmt(avg)}
            <small> tok/s</small>
          </b>
        </div>
        <div>
          <span className="muted">Peak</span>
          <b className="mono tnum">
            {fmt(peak)}
            <small> tok/s</small>
          </b>
        </div>
        {view === "live" && writing && (
          <span className="chart-live">
            <Spinner kind="bars" tone="pink" size={14} /> live
          </span>
        )}
      </div>
      <TpsChart
        values={values}
        xs={xs}
        xLabel={view === "live" ? (x) => `${x.toFixed(1)} s` : (x) => `#${x}`}
        label={values.length ? `Tokens per second, ${view === "live" ? "this reply" : "per reply"}` : empty}
      />
    </div>
  );
}

export function MonitorScreen() {
  const { state } = useStore();
  const phase = state.engine.phase;
  const last = state.turns[0];
  const model = state.models.find((m) => m.name === state.activeModel);
  const chat = state.chats.find((c) => c.id === state.activeChatId);
  const ctxUsed = Math.round((chat?.messages.reduce((n, m) => n + m.content.length, 0) ?? 0) / 4);
  const ctxTotal = model?.context_length ?? 0;
  const tag = state.sampleTelemetry ? "Sample · " : api.kind === "mock" ? "Mock stream · " : "";
  const desktop = api.kind === "pywebview";
  const liveNow = phase === "writing" && state.liveTps.length > 0 ? state.liveTps[state.liveTps.length - 1].rate : null;
  const speed = liveNow ?? (last ? (last.decode_tok_per_sec ?? last.tok_per_sec) : null);
  const speedFoot = liveNow != null ? `${tag}now, over the last half second` : !last ? "Shown after the first reply" : last.decode_tok_per_sec != null ? `${tag}last turn, output after first token` : `${tag}last turn, end to end`;
  const sessionTokens = state.turns.reduce((n, t) => n + t.tokens, 0);
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Monitor</h1>
          <p className="muted">Live engine state for the active model.</p>
        </div>
        {state.sampleTelemetry ? <SampleBadge>Sample telemetry · not measured</SampleBadge> : !desktop && <SampleBadge>Mock engine</SampleBadge>}
      </header>

      <div className="card engine-card">
        <div className="engine-now">
          <span className={`engine-dot lg ${phase}`} aria-hidden />
          <div>
            <div className="engine-phase">{phase === "error" ? "Error" : PHASES.find((p) => p.id === phase)?.label}</div>
            <div className="muted mono">{model ? `${model.name} · ${model.quant} · ${state.profile.recommended_threads} threads` : "No model loaded"}</div>
          </div>
          {(phase === "reading" || phase === "writing" || phase === "loading") && <GridLoader kind={phase === "loading" ? "loading" : phase === "reading" ? "thinking" : "writing"} />}
        </div>
        <ol className="phase-track" aria-label="Engine state">
          {PHASES.map((p) => (
            <li key={p.id} className={p.id === phase ? "on" : ""} aria-current={p.id === phase ? "step" : undefined}>
              {p.label}
            </li>
          ))}
        </ol>
      </div>

      <div className="stat-grid">
        <Stat tint="tint-amber" icon="bolt" label="Speed" value={speed != null ? speed.toFixed(1) : "—"} unit="tok/s" foot={speedFoot} />
        <Stat tint="tint-violet" icon="sparkle" label="Tokens" value={sessionTokens.toLocaleString()} foot={`${tag}${state.turns.length} turn${state.turns.length === 1 ? "" : "s"} this session`} />
        <Stat tint="tint-teal" icon="clock" label="First token" value={last?.first_token_s != null ? last.first_token_s.toFixed(2) : "—"} unit="s" foot={last ? `${tag}last turn` : "Shown after the first reply"} />
        <Stat
          tint="tint-pink"
          icon="memory"
          label="Engine RAM"
          value={model ? `≈ ${model.size_gb.toFixed(1)}` : "—"}
          unit="GB"
          foot="Estimate: model file, plus KV cache"
        />
      </div>

      <div className="monitor-row">
        <SpeedCard />
        <div className="card gauge-card">
          <div className="card-head">
            <span>Context</span>
            <span className="muted">approx.</span>
          </div>
          <div className="gauge-body">
            <Ring value={ctxTotal ? ctxUsed / ctxTotal : 0} size={96} stroke={8} color>
              <span className="tnum">{ctxTotal ? Math.round((ctxUsed / ctxTotal) * 100) : 0}%</span>
            </Ring>
            <div className="mono tnum gauge-text">
              {ctxUsed.toLocaleString()}
              <span className="muted"> / {ctxTotal.toLocaleString()}</span>
            </div>
          </div>
        </div>
        <div className="card cpu-card">
          <div className="card-head">
            <span>CPU</span>
          </div>
          <div className="cpu-name">{state.profile.cpu_name}</div>
          <div className="thread-cells" aria-label={`${state.profile.recommended_threads} of ${state.profile.logical_cores} threads used`}>
            {Array.from({ length: state.profile.logical_cores }, (_, i) => (
              <span key={i} className={i < state.profile.recommended_threads ? "on" : ""} />
            ))}
          </div>
          <div className="muted">
            {state.profile.recommended_threads} of {state.profile.logical_cores} threads
          </div>
        </div>
      </div>

      <div className="card table-card">
        <div className="card-head">
          <span>Recent turns</span>
        </div>
        <table className="table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Model</th>
              <th className="num">Prompt</th>
              <th className="num">Output</th>
              <th className="num">tok/s</th>
              <th className="num">First token</th>
              <th className="num">Total</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {state.turns.length === 0 && (
              <tr>
                <td colSpan={8} className="table-empty muted">
                  No turns yet. Send a message in Chat and each reply shows up here.
                </td>
              </tr>
            )}
            {state.turns.map((t) => (
              <tr key={t.id}>
                <td className="mono">{t.at}</td>
                <td className="mono">{t.model}</td>
                <td className="num mono">{t.prompt_tokens?.toLocaleString() ?? "—"}</td>
                <td className="num mono">{t.tokens.toLocaleString()}</td>
                <td className="num mono" title={t.decode_tok_per_sec != null ? `end to end ${t.tok_per_sec.toFixed(1)} tok/s` : undefined}>
                  {(t.decode_tok_per_sec ?? t.tok_per_sec).toFixed(1)}
                </td>
                <td className="num mono">{t.first_token_s != null ? `${t.first_token_s.toFixed(2)} s` : "—"}</td>
                <td className="num mono">{t.seconds.toFixed(1)} s</td>
                <td>
                  <Chip tone={t.status === "done" ? undefined : t.status === "stopped" ? "warn" : "err"}>
                    {t.status === "done" ? "Done" : t.status === "stopped" ? "Stopped" : "Error"}
                  </Chip>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
