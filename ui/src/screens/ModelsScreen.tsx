import { useState } from "react";
import type { ModelRow } from "../api";
import { Icon } from "../components/Icon";
import { Button, Chip, Dialog, FitChip, IconButton, Progress, Segmented } from "../components/primitives";
import { ProviderLogo, PROVIDER_NAME } from "../components/ProviderLogo";
import { useStore } from "../state/store";

type Filter = "all" | "downloaded" | "code";

function ModelCard({ m, onDelete }: { m: ModelRow; onDelete: () => void }) {
  const { state, actions } = useStore();
  const pct = state.downloads[m.name];
  const downloading = pct !== undefined;
  const loading = state.engine.phase === "loading";
  return (
    <div className={`model-row ${m.is_active ? "active" : ""}`}>
      <ProviderLogo provider={m.provider} size={18} tile title={PROVIDER_NAME[m.provider]} />
      <div className="model-main">
        <div className="model-title">
          <span className="mono">{m.name}</span>
          {m.is_active && <Chip icon="check">Active</Chip>}
          {m.task === "code" && <Chip icon="terminal">Code</Chip>}
          {m.active_params_b && <Chip>MoE</Chip>}
        </div>
        <p className="model-desc">{m.description}</p>
        <div className="model-meta mono">
          <span>{m.active_params_b ? `${m.params_b}B total · ${m.active_params_b}B active` : `${m.params_b}B`}</span>
          <span>{m.quant}</span>
          <span>{m.size_gb.toFixed(m.size_gb < 1 ? 2 : 1)} GB</span>
          <span>ctx {m.context_length.toLocaleString()}</span>
          <span className="model-source" title={`Downloaded from Hugging Face: ${m.repo_id}`}>
            <ProviderLogo provider="huggingface" size={12} />
            <span className="ellipsis">{m.repo_id}</span>
          </span>
        </div>
        {downloading && (
          <div className="model-dl">
            <Progress value={pct} label={`Downloading ${m.name}`} />
            <span className="mono tnum muted">
              {Math.round(pct)}% · {((pct / 100) * m.size_gb).toFixed(1)} of {m.size_gb.toFixed(1)} GB
            </span>
          </div>
        )}
      </div>
      <div className="model-side">
        <FitChip fit={m.fit} />
        <div className="model-actions">
          {m.downloaded ? (
            <>
              {m.is_active ? (
                <Button size="sm" disabled>
                  In use
                </Button>
              ) : (
                <Button size="sm" onClick={() => actions.use(m.name)} disabled={loading}>
                  Use
                </Button>
              )}
              <IconButton
                icon="trash"
                label={m.is_active ? "Switch to another model before deleting" : `Delete ${m.name}`}
                onClick={onDelete}
                disabled={m.is_active}
              />
            </>
          ) : downloading ? (
            <Button size="sm" disabled>
              Downloading…
            </Button>
          ) : (
            <Button size="sm" icon="download" onClick={() => actions.download(m.name)}>
              Download
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

export function ModelsScreen({ confirmDelete }: { confirmDelete?: string }) {
  const { state, actions } = useStore();
  const [filter, setFilter] = useState<Filter>("all");
  const [toDelete, setToDelete] = useState<string | null>(confirmDelete ?? null);
  const rows = state.models.filter((m) => (filter === "downloaded" ? m.downloaded : filter === "code" ? m.task === "code" : true));
  const dl = state.models.filter((m) => m.downloaded);
  const used = dl.reduce((n, m) => n + m.size_gb, 0);
  const target = state.models.find((m) => m.name === toDelete);
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Models</h1>
          <p className="muted">GGUF models from Hugging Face. Fit is an estimate for this machine, shown once a file is on disk.</p>
        </div>
      </header>
      <div className="toolbar">
        <Segmented
          label="Filter models"
          value={filter}
          onChange={setFilter}
          options={[
            { value: "all", label: `All ${state.models.length}` },
            { value: "downloaded", label: `Downloaded ${dl.length}` },
            { value: "code", label: "Code" },
          ]}
        />
        <span className="grow" />
        <span className="muted storage">
          <Icon name="folder" size={13} />
          <span className="mono">{state.settings.models_dir}</span>
          <span>· {used.toFixed(1)} GB used</span>
        </span>
      </div>
      <div className="model-list">
        {rows.map((m) => (
          <ModelCard key={m.name} m={m} onDelete={() => setToDelete(m.name)} />
        ))}
      </div>
      <Dialog
        open={!!target}
        title={`Delete ${target?.name ?? ""}?`}
        onClose={() => setToDelete(null)}
        actions={
          <>
            <Button variant="ghost" onClick={() => setToDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              icon="trash"
              onClick={() => {
                if (target) actions.remove(target.name);
                setToDelete(null);
              }}
            >
              Delete {target?.size_gb.toFixed(1)} GB
            </Button>
          </>
        }
      >
        <p>
          This removes the file from <span className="mono">{state.settings.models_dir}</span>. Your chats stay. You can download it again
          later.
        </p>
      </Dialog>
    </div>
  );
}
