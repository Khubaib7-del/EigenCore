import { api } from "../api";
import { Icon, type IconName } from "../components/Icon";
import { Chip, PixelLoader, Ring, SampleBadge, Sparkline } from "../components/primitives";
import { useStore, type EnginePhase } from "../state/store";

const PHASES: { id: EnginePhase; label: string }[] = [
  { id: "idle", label: "Idle" },
  { id: "loading", label: "Loading" },
  { id: "reading", label: "Reading prompt" },
  { id: "writing", label: "Writing" },
];

function Stat({ icon, label, value, unit, foot }: { icon: IconName; label: string; value: string; unit?: string; foot: string }) {
  return (
    <div className="card stat">
      <div className="stat-label">
        <Icon name={icon} size={14} />
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

export function MonitorScreen() {
  const { state } = useStore();
  const phase = state.engine.phase;
  const last = state.turns[0];
  const model = state.models.find((m) => m.name === state.activeModel);
  const chat = state.chats.find((c) => c.id === state.activeChatId);
  const ctxUsed = Math.round((chat?.messages.reduce((n, m) => n + m.content.length, 0) ?? 0) / 4);
  const ctxTotal = model?.context_length ?? 0;
  const tag = state.sampleTelemetry ? "Sample · " : api.kind === "mock" ? "Mock stream · " : "";
  const sessionTokens = state.turns.reduce((n, t) => n + t.tokens, 0);
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Monitor</h1>
          <p className="muted">Live engine state for the active model.</p>
        </div>
        {state.sampleTelemetry ? <SampleBadge>Sample telemetry · not measured</SampleBadge> : <SampleBadge>Mock engine</SampleBadge>}
      </header>

      <div className="card engine-card">
        <div className="engine-now">
          <span className={`engine-dot lg ${phase}`} aria-hidden />
          <div>
            <div className="engine-phase">{phase === "error" ? "Error" : PHASES.find((p) => p.id === phase)?.label}</div>
            <div className="muted mono">{model ? `${model.name} · ${model.quant} · ${state.profile.recommended_threads} threads` : "No model loaded"}</div>
          </div>
          {(phase === "reading" || phase === "writing" || phase === "loading") && <PixelLoader />}
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
        <Stat icon="bolt" label="Speed" value={last ? last.tok_per_sec.toFixed(1) : "—"} unit="tok/s" foot={last ? `${tag}last turn, decode` : "Shown after the first reply"} />
        <Stat icon="sparkle" label="Tokens" value={sessionTokens.toLocaleString()} foot={`${tag}${state.turns.length} turns this session`} />
        <Stat icon="clock" label="First token" value={last?.first_token_s != null ? last.first_token_s.toFixed(2) : "—"} unit="s" foot={last ? `${tag}last turn` : "Shown after the first reply"} />
        <Stat
          icon="memory"
          label="Engine RAM"
          value={model ? `≈ ${model.size_gb.toFixed(1)}` : "—"}
          unit="GB"
          foot="Estimate: model file, plus KV cache"
        />
      </div>

      <div className="monitor-row">
        <div className="card chart-card">
          <div className="card-head">
            <span>Tokens per second</span>
            <span className="muted mono">last {state.tps.length} samples</span>
          </div>
          <Sparkline data={state.tps} height={96} />
        </div>
        <div className="card gauge-card">
          <div className="card-head">
            <span>Context</span>
            <span className="muted">approx.</span>
          </div>
          <div className="gauge-body">
            <Ring value={ctxTotal ? ctxUsed / ctxTotal : 0} size={96} stroke={8}>
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
                <td className="num mono">{t.tok_per_sec.toFixed(1)}</td>
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
