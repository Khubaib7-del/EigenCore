import { api } from "../api";
import { Icon } from "../components/Icon";
import { Button, Chip } from "../components/primitives";
import { ProviderLogo } from "../components/ProviderLogo";
import { useStore } from "../state/store";

export function ProfileScreen() {
  const { state, dispatch, actions } = useStore();
  const p = state.profile;
  const usedPct = p.total_ram_gb > 0 ? ((p.total_ram_gb - p.available_ram_gb) / p.total_ram_gb) * 100 : 0;
  const desktop = api.kind === "pywebview";
  // Desktop: the largest registry model within the profiler's size limit that
  // is not on disk yet (a rule of thumb from the estimate, not a benchmark).
  const suggested = desktop
    ? state.models
        .filter((m) => !m.downloaded && m.params_b <= p.max_model_params_b && m.size_gb < p.available_ram_gb)
        .sort((a, b) => b.params_b - a.params_b)[0]
    : state.models.find((m) => m.name === "qwen2.5-3b");
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Profile</h1>
          <p className="muted">What EigenCore found on this machine, and what it can run.</p>
        </div>
        <Button icon="refresh" size="sm" onClick={actions.rescanProfile}>
          Re-scan
        </Button>
      </header>

      <div className="profile-grid">
        <div className="card hw">
          <div className="hw-label">
            <Icon name="cpu" size={14} /> Processor
          </div>
          <div className="hw-value">{p.cpu_name}</div>
          <div className="muted mono">
            {p.physical_cores} cores / {p.logical_cores} threads · {p.arch}
          </div>
          <div className="isa">
            {p.instruction_sets.length === 0 && <span className="muted">No SIMD extensions detected</span>}
            {p.instruction_sets.map((s) => (
              <span key={s} className="chip mono">
                {s}
              </span>
            ))}
          </div>
        </div>

        <div className="card hw">
          <div className="hw-label">
            <Icon name="memory" size={14} /> Memory
          </div>
          <div className="hw-value tnum">
            {p.available_ram_gb.toFixed(1)} GB <span className="muted hw-of">free of {p.total_ram_gb.toFixed(1)} GB</span>
          </div>
          <div className="ram-bar" aria-label={`${usedPct.toFixed(0)}% in use`}>
            <span style={{ width: `${usedPct}%` }} />
          </div>
          <div className="muted">Models load into RAM. Leave headroom for the KV cache.</div>
        </div>

        <div className="card hw">
          <div className="hw-label">
            <Icon name="gauge" size={14} /> Memory bandwidth
          </div>
          <div className="hw-value tnum">
            {p.estimated_bandwidth_gbps != null ? `≈ ${p.estimated_bandwidth_gbps.toFixed(0)} GB/s` : "— GB/s"}
          </div>
          <div className="muted">
            <Chip>Estimate</Chip> Decode speed on CPU is mostly limited by this. Run a benchmark to measure it.
          </div>
        </div>

        <div className="card hw">
          <div className="hw-label">
            <Icon name="monitor" size={14} /> Graphics
          </div>
          <div className="hw-value">{p.gpu_name ?? "None detected"}</div>
          <div className="muted">EigenCore runs on the CPU. No GPU is needed.</div>
        </div>
      </div>

      <div className="card reco">
        <div className="reco-head">
          <span className="reco-icon">
            <Icon name="sparkle" size={16} />
          </span>
          <div>
            <div className="reco-title">
              Up to {Number(p.max_model_params_b.toFixed(1))}B models at {p.optimal_quantization}
            </div>
            <div className="muted">
              Based on {p.available_ram_gb.toFixed(1)} GB free RAM and the profiler’s estimates. Context {p.recommended_context_length.toLocaleString()} tokens,{" "}
              {p.recommended_threads} threads.
            </div>
          </div>
          <Chip tone="warn" icon="info">
            Estimate until benchmarked
          </Chip>
        </div>
        <div className="reco-meter" aria-label="Confidence: estimate">
          <span className="on" />
          <span className="on" />
          <span />
          <span />
          <span className="muted reco-meter-label">Confidence: estimate</span>
        </div>
        {suggested && (
          <div className="reco-pick">
            <ProviderLogo provider={suggested.provider} size={16} tile />
            <div>
              <div className="mono">{suggested.name}</div>
              <div className="muted">Suggested next model. {suggested.description}</div>
            </div>
            <span className="grow" />
            <Button size="sm" onClick={() => dispatch({ type: "nav", screen: "models" })}>
              Open Models
            </Button>
          </div>
        )}
        <div className="reco-actions">
          <span className="muted mono">eigencore benchmark</span>
          <span className="grow" />
          {desktop ? (
            <Button size="sm" icon="bolt" disabled title="Coming soon in the app. Run `eigencore benchmark` in a terminal for now.">
              Run benchmark · coming soon
            </Button>
          ) : (
            <Button size="sm" variant="primary" icon="bolt">
              Run benchmark
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
