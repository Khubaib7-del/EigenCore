import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { FitDetail, ModelRow, ModelSearch } from "../api";
import { Icon } from "../components/Icon";
import { Button, Chip, Dialog, FitChip, IconButton, Progress, Segmented, Switch } from "../components/primitives";
import { ProviderLogo, PROVIDER_NAME } from "../components/ProviderLogo";
import { useStore } from "../state/store";
import "../styles/catalog.css";

type Filter = "all" | "downloaded" | "code";

/** 131072 -> "128K" */
function ctxLabel(n: number): string {
  return n >= 1024 && n % 1024 === 0 ? `${n / 1024}K` : n.toLocaleString();
}

/** "org/Some-Model-GGUF:file.gguf" -> "Some-Model" (search results and models added from search). */
function displayName(m: ModelRow): string {
  if (!m.name.includes(":")) return m.name;
  const repo = m.name.split(":")[0];
  return (repo.split("/").pop() ?? repo).replace(/[-_]GGUF$/i, "");
}

/** Tooltip for the fit chip: what the estimate is made of (every number is an estimate). */
function fitTitle(f: FitDetail | null | undefined): string {
  if (!f) return "Fit is estimated once the model's shape is known";
  if (f.verdict === "unknown") return "Fit from the file size only. Check fit reads the model's header (a few MB) for the full estimate.";
  const parts = [
    `Estimate at ${ctxLabel(f.context_checked)} context against ${f.budget_gb} GB: ${f.total_ram_gb} GB RAM minus ${f.system_reserve_gb} GB kept for the OS and other apps.`,
  ];
  if (f.total_gb != null) parts.push(`Weights ${f.weights_gb} GB + KV cache ${f.kv_cache_gb} GB + runtime ≈ ${f.total_gb} GB.`);
  else parts.push(`The file alone is ${f.weights_gb} GB.`);
  if (f.decode_ceiling_tok_s != null) parts.push(`Decode ceiling ≈ ${f.decode_ceiling_tok_s} tok/s (memory bandwidth ÷ weights read per token; real speed is lower).`);
  if (f.available_ram_gb != null && f.total_gb != null && f.total_gb > f.available_ram_gb && f.verdict !== "too_big")
    parts.push(`Only ${f.available_ram_gb} GB is free right now: close other apps before loading it.`);
  return parts.join(" ");
}

function LicenceChip({ m }: { m: ModelRow }) {
  if (m.noncommercial || m.licence_kind === "noncommercial")
    return (
      <span title={`Non-commercial licence: ${m.license ?? "see model card"}`}>
        <Chip tone="warn">NC</Chip>
      </span>
    );
  if (!m.license) return null;
  if (m.licence_kind === "restricted")
    return (
      <span className="lic-restricted" title={`Custom licence terms (${m.license}): read them before commercial use`}>
        <Chip icon="alert" mono>
          {m.license}
        </Chip>
      </span>
    );
  return <Chip mono>{m.license}</Chip>;
}

function ModelCard({
  m,
  onDelete,
  onCheckFit,
  checking,
}: {
  m: ModelRow;
  onDelete: () => void;
  onCheckFit?: () => void;
  checking?: boolean;
}) {
  const { state, actions } = useStore();
  const pct = state.downloads[m.name];
  const downloading = pct !== undefined;
  const loading = state.engine.phase === "loading";
  return (
    <div className={`model-row ${m.is_active ? "active" : ""}`}>
      <ProviderLogo provider={m.provider} size={18} tile title={PROVIDER_NAME[m.provider]} />
      <div className="model-main">
        <div className="model-title">
          <span className="mono" title={m.name}>
            {displayName(m)}
          </span>
          {m.is_active && <Chip icon="check">Active</Chip>}
          {m.task === "code" && <Chip icon="terminal">Code</Chip>}
          {(m.active_params_b || m.tags?.includes("moe")) && <Chip>MoE</Chip>}
          <LicenceChip m={m} />
          {m.trusted === false && (
            <span title="Community upload: not from a trusted quantiser or the model maker, and not reviewed by EigenCore">
              <Chip tone="warn">Community</Chip>
            </span>
          )}
          {m.fit_detail && !m.fit_detail.runtime_ok && (
            <span title={m.fit_detail.runtime_note ?? "Needs a newer llama-cpp-python"}>
              <Chip tone="warn" icon="alert">
                Runtime update
              </Chip>
            </span>
          )}
        </div>
        <p className="model-desc">{m.description}</p>
        <div className="model-meta mono">
          <span>{m.active_params_b ? `${m.params_b}B total · ${m.active_params_b}B active` : `${m.params_b}B`}</span>
          <span>{m.quant}</span>
          <span>{m.size_gb.toFixed(m.size_gb < 1 ? 2 : 1)} GB</span>
          <span>ctx {m.context_length.toLocaleString()}</span>
          {m.fit_detail?.max_context ? (
            <span className="fit-ctx" title="Largest context whose KV cache fits next to the weights in this machine's RAM budget (estimate)">
              ≈{ctxLabel(m.fit_detail.max_context)} ctx fits here
            </span>
          ) : null}
          {m.ram_class ? <span title="Smallest RAM size this model is listed for">{m.ram_class} GB RAM+</span> : null}
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
        <span title={fitTitle(m.fit_detail)}>
          <FitChip fit={m.fit} />
        </span>
        <div className="model-actions">
          {onCheckFit && m.fit === "unknown" && !m.downloaded && (
            <Button size="sm" variant="ghost" icon="gauge" onClick={onCheckFit} disabled={checking}>
              {checking ? "Checking…" : "Check fit"}
            </Button>
          )}
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

/** Hugging Face search: debounced, latest query wins, results keep their own fit checks. */
function useHfSearch(query: string, community: boolean) {
  const [res, setRes] = useState<ModelSearch | null>(null);
  const [busy, setBusy] = useState(false);
  const seq = useRef(0);
  useEffect(() => {
    const q = query.trim();
    const n = ++seq.current;
    if (q.length < 2) {
      setRes(null);
      setBusy(false);
      return;
    }
    setBusy(true);
    const t = window.setTimeout(() => {
      api
        .searchModels(q, community)
        .then((r) => n === seq.current && setRes(r))
        .catch((e) => n === seq.current && setRes({ query: q, results: [], error: String(e) }))
        .finally(() => n === seq.current && setBusy(false));
    }, 450);
    return () => window.clearTimeout(t);
  }, [query, community]);
  return { res, busy, setRes };
}

export function ModelsScreen({ confirmDelete }: { confirmDelete?: string }) {
  const { state, actions } = useStore();
  const [filter, setFilter] = useState<Filter>("all");
  const [toDelete, setToDelete] = useState<string | null>(confirmDelete ?? null);
  const [query, setQuery] = useState("");
  const [community, setCommunity] = useState(false);
  const { res, busy, setRes } = useHfSearch(query, community);
  const [checking, setChecking] = useState<Record<string, boolean>>({});
  const checkFit = (name: string) => {
    setChecking((c) => ({ ...c, [name]: true }));
    api
      .checkFit(name)
      .then((r) =>
        setRes((prev) =>
          prev ? { ...prev, results: prev.results.map((m) => (m.name === name ? { ...m, fit: r.fit, fit_detail: r.fit_detail ?? m.fit_detail } : m)) } : prev,
        ),
      )
      .finally(() => setChecking((c) => ({ ...c, [name]: false })));
  };
  const listed = new Set(state.models.map((m) => m.name));
  const hfRows = (res?.results ?? []).filter((m) => !listed.has(m.name));
  const rows = state.models.filter((m) => (filter === "downloaded" ? m.downloaded : filter === "code" ? m.task === "code" : true));
  const dl = state.models.filter((m) => m.downloaded);
  const used = dl.reduce((n, m) => n + m.size_gb, 0);
  const target = state.models.find((m) => m.name === toDelete);
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Models</h1>
          <p className="muted">
            GGUF models from Hugging Face: a curated list, or search for any model. Fit is an estimate for this machine's RAM (minus a share kept for the OS and other apps), checked before you
            download.
          </p>
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
        <label className="catalog-search">
          <Icon name="search" size={14} />
          <input
            type="search"
            value={query}
            placeholder="Search Hugging Face"
            aria-label="Search Hugging Face for GGUF models"
            spellCheck={false}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && setQuery("")}
          />
        </label>
        <span className="catalog-community" title="Search shows uploads from trusted quantisers and model makers. Turn on to include community uploads (not reviewed by EigenCore).">
          <Switch checked={community} onChange={setCommunity} label="Include community uploads in search" />
          <span>Community uploads</span>
        </span>
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
      {query.trim().length >= 2 && (
        <section className="catalog-results" aria-live="polite">
          <div className="catalog-head">
            <ProviderLogo provider="huggingface" size={14} />
            <span>Hugging Face</span>
            <span className="muted">
              {busy
                ? "Searching…"
                : res?.error
                  ? `Search failed: ${res.error}`
                  : `${hfRows.length} ${hfRows.length === 1 ? "result" : "results"}${(res?.results.length ?? 0) > hfRows.length ? " (curated matches are listed above)" : ""}`}
            </span>
          </div>
          {hfRows.length > 0 && (
            <div className="model-list">
              {hfRows.map((m) => (
                <ModelCard key={m.name} m={m} onDelete={() => setToDelete(m.name)} onCheckFit={() => checkFit(m.name)} checking={!!checking[m.name]} />
              ))}
            </div>
          )}
          <p className="muted catalog-note">
            {community
              ? "Community uploads are included. They are not reviewed by EigenCore: check the licence and the model card before using one."
              : "Showing trusted uploaders only (known quantisers and the model makers). Turn on Community uploads to see the rest."}
          </p>
        </section>
      )}
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
