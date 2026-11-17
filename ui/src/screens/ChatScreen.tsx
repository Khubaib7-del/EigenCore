import { useEffect, useMemo, useRef, useState } from "react";
import { ApprovalCard, RollingWord, ThinkingFold, ThinkingOrb, ToolSteps, type OrbState } from "../components/agent";
import { Icon, type IconName } from "../components/Icon";
import { Markdown, openFence, splitThinking } from "../components/Markdown";
import { Chip, IconButton, useElapsed } from "../components/primitives";
import { ContextMeter, ModelPicker, PromptBar, type SlashCommand } from "../components/PromptBar";
import { ProviderLogo } from "../components/ProviderLogo";
import { Wordmark } from "../components/RetroComputer";
import { useStore, type Message } from "../state/store";

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? "Good evening" : h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
}

const SUGGESTIONS: { icon: IconName; short: string; text: string }[] = [
  { icon: "terminal", short: "rename photos", text: "Write a Python script that renames photos by date" },
  { icon: "file", short: "summarise", text: "Summarise a long document in five bullet points" },
  { icon: "sparkles", short: "quantization", text: "Explain how quantization makes models smaller" },
  { icon: "globe", short: "translate", text: "Translate a paragraph into Urdu" },
];

/* ── Output stats as small bars ─────────────────────────────────────────
 * Each bar is measured against something real: speed against the fastest
 * reply this session, tokens against the reply's max_tokens, first token
 * against the reply's total time.
 */
function Meter({ icon, value, unit, frac, title }: { icon: IconName; value: string; unit: string; frac: number | null; title: string }) {
  return (
    <span className="meter" title={title}>
      <Icon name={icon} size={13} />
      <span className="meter-val mono tnum">
        {value}
        <small>{unit}</small>
      </span>
      <span className="meter-bar" aria-hidden>
        <span style={{ width: frac == null ? 0 : `${Math.max(4, Math.min(100, frac * 100))}%` }} />
      </span>
    </span>
  );
}

function Stats({ m }: { m: Message }) {
  const { state } = useStore();
  const [copied, setCopied] = useState(false);
  const st = m.stats;
  if (!st || m.status === "reading" || m.status === "streaming" || m.status === "awaiting_approval") return null;
  const measured = m.statsKind !== "placeholder";
  const rate = st.decode_tok_per_sec ?? st.tok_per_sec;
  const peak = Math.max(rate, ...state.turns.map((t) => t.decode_tok_per_sec ?? t.tok_per_sec));
  const kind = m.statsKind === "mock" ? "mock stream" : "measured";
  const copy = () => {
    navigator.clipboard?.writeText(splitThinking(m.content).answer).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };
  return (
    <div className="msg-stats">
      {m.statsKind === "mock" && <Chip>mock stream</Chip>}
      {m.status === "stopped" && <Chip tone="warn">Stopped</Chip>}
      {m.status === "error" && <Chip tone="err">Error</Chip>}
      <Meter
        icon="bolt"
        value={measured ? rate.toFixed(1) : "—"}
        unit="tok/s"
        frac={measured && peak > 0 ? rate / peak : null}
        title={measured ? `Output speed after the first token (${kind}). Bar: against the fastest reply this session.` : "Speed is measured once the UI is connected to the engine"}
      />
      <Meter
        icon="hash"
        value={st.tokens.toLocaleString()}
        unit="tokens"
        frac={m.budget ? st.tokens / m.budget : null}
        title={m.budget ? `${st.tokens} of the ${m.budget} max tokens for this reply` : "Tokens in this reply"}
      />
      <Meter
        icon="clock"
        value={measured && st.first_token_s != null ? st.first_token_s.toFixed(2) : "—"}
        unit="s first"
        frac={measured && st.first_token_s != null && st.seconds > 0 ? st.first_token_s / st.seconds : null}
        title={measured ? `Time to first token (${kind}); bar: share of the ${st.seconds.toFixed(1)} s total` : "Measured once the UI is connected to the engine"}
      />
      <span className="grow" />
      <IconButton icon={copied ? "check" : "copy"} label={copied ? "Copied" : "Copy reply"} size={14} onClick={copy} />
      <IconButton icon="refresh" label="Regenerate (coming soon)" size={14} disabled />
    </div>
  );
}

function orbFor(m: Message, thinkingLive: boolean): { orb: OrbState; words: string[] } {
  const { answer } = splitThinking(m.content);
  switch (m.status) {
    case "reading":
      return { orb: "thinking", words: ["Thinking…", "Analyzing…", "Reading prompt…"] };
    case "streaming":
      if (thinkingLive) return { orb: "thinking", words: ["Reasoning…"] };
      return { orb: "writing", words: [openFence(answer) ? "Writing code…" : "Writing…"] };
    case "awaiting_approval":
      return m.approval?.decision ? { orb: "thinking", words: ["Thinking…"] } : { orb: "waiting", words: ["Waiting for approval"] };
    case "stopped":
      return { orb: "stopped", words: ["Stopped"] };
    case "error":
      return { orb: "error", words: ["Error"] };
    default:
      return { orb: "done", words: ["Done"] };
  }
}

function AgentLine({ m, thinkingLive }: { m: Message; thinkingLive: boolean }) {
  const { state } = useStore();
  const model = state.models.find((x) => x.name === m.model);
  const live = m.status === "reading" || m.status === "streaming" || (m.status === "awaiting_approval" && !!m.approval?.decision);
  const elapsed = useElapsed(m.startedAt, live);
  const { orb, words } = orbFor(m, thinkingLive);
  return (
    <div className={`agent-line ${orb}`}>
      <ThinkingOrb state={orb} />
      {model && <ProviderLogo provider={model.provider} size={13} />}
      <span className="mono agent-model">{m.model ?? "model"}</span>
      <span className="agent-sep" aria-hidden />
      {live ? <RollingWord words={words} /> : <span className="agent-word">{words[0]}</span>}
      {live && m.startedAt !== undefined && <span className="mono muted tnum agent-time">{elapsed}</span>}
      {!live && m.stats && m.stats.seconds > 0 && <span className="mono muted tnum agent-time">{m.stats.seconds.toFixed(1)} s</span>}
    </div>
  );
}

function AssistantMessage({ m, onApprove }: { m: Message; onApprove: (d: "once" | "session" | "deny") => void }) {
  const { thinking, answer, thinkingOpen } = splitThinking(m.content);
  const live = m.status === "streaming" || m.status === "reading";
  const pending = m.approval && !m.approval.decision ? m.approval.tool_name : null;
  return (
    <article className="msg assistant">
      <AgentLine m={m} thinkingLive={thinkingOpen && live} />
      <div className="agent-body">
        {((m.tools && m.tools.length > 0) || pending) && <ToolSteps calls={m.tools ?? []} pending={pending} />}
        {m.approval && <ApprovalCard request={m.approval} decision={m.approval.decision} onDecide={onApprove} />}
        {thinking !== null && <ThinkingFold text={thinking} live={thinkingOpen && live} />}
        {answer && <Markdown text={answer} streaming={m.status === "streaming" && !thinkingOpen} />}
        {m.status === "error" && m.error && !answer && (
          <div className="msg-error">
            <Icon name="alert" size={14} /> {m.error}
          </div>
        )}
        <Stats m={m} />
      </div>
    </article>
  );
}

export function ChatScreen() {
  const { state, dispatch, actions } = useStore();
  const chat = state.chats.find((c) => c.id === state.activeChatId);
  const busy = state.engine.phase === "reading" || state.engine.phase === "writing";
  const scroller = useRef<HTMLDivElement>(null);
  const last = chat?.messages[chat.messages.length - 1];
  const [fill, setFill] = useState<{ text: string; n: number } | null>(null);
  const [draft, setDraft] = useState("");
  const putInComposer = (text: string) => setFill((f) => ({ text, n: (f?.n ?? 0) + 1 }));
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [last?.content, last?.status, chat?.messages.length]);

  const active = state.models.find((m) => m.name === state.activeModel);
  const commands = useMemo<SlashCommand[]>(
    () => [
      { cmd: "/new", hint: "Start a new chat", icon: "plus", run: () => dispatch({ type: "newChat" }) },
      { cmd: "/models", hint: "Download or switch models", icon: "models", run: () => dispatch({ type: "nav", screen: "models" }) },
      { cmd: "/monitor", hint: "Engine state and speed", icon: "activity", run: () => dispatch({ type: "nav", screen: "monitor" }) },
      { cmd: "/profile", hint: "This PC's hardware", icon: "cpu", run: () => dispatch({ type: "nav", screen: "profile" }) },
      { cmd: "/settings", hint: "Theme, models folder, updates", icon: "settings", run: () => dispatch({ type: "nav", screen: "settings" }) },
      {
        cmd: "/theme",
        hint: "Switch dark / light",
        icon: "sun",
        run: () => dispatch({ type: "settings", patch: { theme: state.settings.theme === "light" ? "dark" : "light" } }),
      },
    ],
    [dispatch, state.settings.theme],
  );
  const picker = (
    <ModelPicker
      models={state.models}
      active={state.activeModel}
      loading={state.engine.phase === "loading"}
      onPick={actions.use}
      onManage={() => dispatch({ type: "nav", screen: "models" })}
    />
  );
  const used = Math.round((chat?.messages.reduce((n, m) => n + m.content.length, 0) ?? 0) / 4);
  const meter = active ? <ContextMeter used={used} total={active.context_length} /> : null;
  const bar = (
    <PromptBar
      onSend={actions.send}
      onStop={actions.stop}
      busy={busy}
      disabled={!state.activeModel || state.engine.phase === "loading"}
      placeholder={state.activeModel ? "Message EigenCore" : "Download a model in Models to start"}
      picker={picker}
      meter={meter}
      options={state.genOptions}
      onOptions={(patch) => dispatch({ type: "genOptions", patch })}
      contextLength={active?.context_length}
      commands={commands}
      fill={fill}
      onTextChange={setDraft}
      autoFocus
    />
  );

  if (!chat || chat.messages.length === 0) {
    const p = state.profile;
    return (
      <div className="chat-empty">
        <div className="chat-empty-inner">
          <Wordmark
            version={state.version}
            model={active ? { name: active.name, quant: active.quant, context: active.context_length, fit: active.fit } : null}
            loading={state.engine.phase === "loading"}
            cpu={p.cpu_name.replace(/\(R\)|\(TM\)|CPU|@.*$/gi, "").replace(/\s+/g, " ").replace(/^(Intel|AMD)\s+(Core\s+)?/i, "").trim().slice(0, 13)}
            threads={`${p.recommended_threads}/${p.logical_cores} threads`}
            typed={draft}
            tps={state.tps}
            ram={`${Math.round(p.total_ram_gb)} GB`}
          />
          <h1 className="greeting">
            {greeting()}. <span>What should we work on?</span>
          </h1>
          {bar}
          <div className="suggestions">
            {SUGGESTIONS.map((s) => (
              <button key={s.text} className="suggestion" onClick={() => putInComposer(s.text)} title="Put this in the message box">
                <Icon name={s.icon} size={14} />
                <span>{s.text}</span>
              </button>
            ))}
          </div>
          <p className="local-note muted">
            <Icon name="shieldCheck" size={13} /> Runs on your CPU. Chats stay on this device.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat">
      <div className="chat-head">
        <Icon name="chat" size={15} className="muted" />
        <h1 className="ellipsis">{chat.title}</h1>
        <span className="grow" />
        {active && (
          <span className="chip">
            <ProviderLogo provider={active.provider} size={12} />
            <span className="mono">{active.name}</span>
          </span>
        )}
        <IconButton icon="dots" label="Chat options (coming soon)" disabled />
      </div>
      <div className="chat-scroll" ref={scroller}>
        <div className="thread">
          {chat.messages.map((m) =>
            m.role === "user" ? (
              <article key={m.id} className="msg user">
                <div className="bubble">{m.content}</div>
              </article>
            ) : (
              <AssistantMessage key={m.id} m={m} onApprove={(d) => m.approval && actions.approve(m.id, m.approval.id, d)} />
            ),
          )}
        </div>
      </div>
      <div className="chat-composer">{bar}</div>
    </div>
  );
}
