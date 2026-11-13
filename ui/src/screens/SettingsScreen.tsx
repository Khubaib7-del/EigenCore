import type { ReactNode } from "react";
import type { Settings } from "../api";
import { Icon } from "../components/Icon";
import { Button, PixelLoader, Progress, Segmented } from "../components/primitives";
import { ProviderLogo, PROVIDER_NAME } from "../components/ProviderLogo";
import { useStore } from "../state/store";

function Row({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="set-row">
      <div className="set-text">
        <div className="set-label">{label}</div>
        {hint && <div className="set-hint muted">{hint}</div>}
      </div>
      <div className="set-control">{children}</div>
    </div>
  );
}

function Updates() {
  const { state, actions } = useStore();
  const u = state.update;
  let body: ReactNode;
  switch (u.kind) {
    case "checking":
      body = (
        <span className="update-line">
          <PixelLoader variant="orbit" /> Checking GitHub Releases…
        </span>
      );
      break;
    case "available":
      body = (
        <>
          <span className="update-line">
            <Icon name="download" size={14} /> {u.tag} is available
          </span>
          <Button size="sm" variant="primary" onClick={actions.installUpdate}>
            Update to {u.tag}
          </Button>
        </>
      );
      break;
    case "downloading":
      body = (
        <div className="update-dl">
          <span className="mono tnum muted">
            Downloading {u.tag} · {Math.round(u.pct)}%
          </span>
          <Progress value={u.pct} label="Downloading update" />
        </div>
      );
      break;
    case "ready_to_install":
      body = (
        <>
          <span className="update-line">
            <Icon name="check" size={14} /> {u.tag} downloaded
          </span>
          <Button size="sm" variant="primary">
            Restart and install
          </Button>
        </>
      );
      break;
    case "up_to_date":
      body = <span className="update-line">You have the latest version</span>;
      break;
    case "error":
      body = (
        <>
          <span className="update-line err">{u.message}</span>
          <Button size="sm" onClick={actions.checkUpdate}>
            Try again
          </Button>
        </>
      );
      break;
    default:
      body = (
        <Button size="sm" icon="refresh" onClick={actions.checkUpdate}>
          Check for updates
        </Button>
      );
  }
  return (
    <Row label={`EigenCore ${state.version}`} hint="Updates come from GitHub Releases. The installer runs after you confirm.">
      <div className="update-box">{body}</div>
    </Row>
  );
}

const MARKS = ["meta", "microsoft", "qwen", "mistral", "gemma", "deepseek", "huggingface", "ai2", "tinyllama"] as const;

export function SettingsScreen() {
  const { state, dispatch } = useStore();
  const s = state.settings;
  const set = (patch: Partial<Settings>) => dispatch({ type: "settings", patch });
  const threadOpts = ["auto", "2", "4", "6", "8"] as const;
  return (
    <div className="page settings">
      <header className="page-head">
        <div>
          <h1>Settings</h1>
          <p className="muted">Changes apply the next time a model loads.</p>
        </div>
      </header>

      <section className="set-section">
        <h2>Appearance</h2>
        <Row label="Theme" hint="System follows your Windows setting.">
          <Segmented
            label="Theme"
            value={s.theme}
            onChange={(theme) => set({ theme })}
            options={[
              { value: "dark", label: "Dark", icon: "moon" },
              { value: "light", label: "Light", icon: "sun" },
              { value: "system", label: "System", icon: "laptop" },
            ]}
          />
        </Row>
      </section>

      <section className="set-section">
        <h2>Engine</h2>
        <Row label="Models folder" hint="Where GGUF files are saved.">
          <span className="path mono">{s.models_dir}</span>
          <Button size="sm" icon="folder">
            Change…
          </Button>
        </Row>
        <Row label="Threads" hint={`Auto uses ${state.profile.recommended_threads} of ${state.profile.logical_cores} on this CPU.`}>
          <Segmented
            label="Threads"
            value={String(s.threads)}
            onChange={(v) => set({ threads: v === "auto" ? "auto" : Number(v) })}
            options={threadOpts.map((t) => ({ value: t, label: t === "auto" ? "Auto" : t }))}
          />
        </Row>
        <Row label="Context length" hint="Longer context needs more RAM. Capped by each model’s limit.">
          <Segmented
            label="Context length"
            value={String(s.context_length)}
            onChange={(v) => set({ context_length: Number(v) })}
            options={["2048", "4096", "8192", "32768"].map((v) => ({ value: v, label: Number(v).toLocaleString() }))}
          />
        </Row>
        <Row label="KV cache type" hint="q8_0 roughly halves cache memory with a small quality cost.">
          <Segmented
            label="KV cache type"
            value={s.kv_cache_type}
            onChange={(kv_cache_type) => set({ kv_cache_type })}
            options={[
              { value: "f16", label: "fp16" },
              { value: "q8_0", label: "q8_0" },
            ]}
          />
        </Row>
      </section>

      <section className="set-section">
        <h2>Agent tools</h2>
        <Row label="Approvals" hint="The tool policy blocks anything not allow-listed. This sets when you are asked.">
          <Segmented
            label="Approvals"
            value={s.approvals}
            onChange={(approvals) => set({ approvals })}
            options={[
              { value: "ask", label: "Ask every time" },
              { value: "session", label: "Once per session" },
            ]}
          />
        </Row>
      </section>

      <section className="set-section">
        <h2>Updates</h2>
        <Updates />
      </section>

      <section className="set-section">
        <h2>About</h2>
        <Row label="Provider marks" hint="Monochrome marks from lobe-icons (MIT). Trademarks belong to their owners.">
          <div className="marks">
            {MARKS.map((p) => (
              <span key={p} className="mark-cell" title={PROVIDER_NAME[p]}>
                <ProviderLogo provider={p} size={16} />
              </span>
            ))}
          </div>
        </Row>
        <Row label="Interface" hint="Components inspired by Beautiful UI (MIT). See ui/CREDITS.md.">
          <span className="muted mono">ui preview</span>
        </Row>
      </section>
    </div>
  );
}
