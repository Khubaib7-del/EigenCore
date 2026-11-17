import { useEffect, useMemo, useRef, useState } from "react";
import { ApprovalCard, RollingWord, ThinkingFold, ToolSteps } from "../components/agent";
import { GridLoader, type LoaderKind } from "../components/GridLoader";
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

const SUGGESTIONS: { icon: IconName; tint: string; short: string; text: string }[] = [
  { icon: "terminal", tint: "tint-green", short: "rename photos", text: "Write a Python script that renames photos by date" },
  { icon: "file", tint: "tint-blue", short: "summarise", text: "Summarise a long document in five bullet points" },
  { icon: "sparkles", tint: "tint-purple", short: "quantization", text: "Explain how quantization makes models smaller" },
  { icon: "globe", tint: "tint-teal", short: "translate", text: "Translate a paragraph into Urdu" },
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

/* ── Reply footer ────────────────────────────────────────────────────────
 * The time always shows. The model, the output meters, copy and regenerate
 * slide in when the pointer is over the reply (or keyboard focus is inside).
 */
function ReplyFoot({ m }: { m: Message }) {
  const { state } = useStore();
  const [copied, setCopied] = useState(false);
  if (m.status === "reading" || m.status === "streaming" || m.status === "awaiting_approval") return null;
  const st = m.stats;
  const model = state.models.find((x) => x.name === m.model);
  const measured = m.statsKind !== "placeholder";
  const rate = st ? (st.decode_tok_per_sec ?? st.tok_per_sec) : 0;
  const peak = st ? Math.max(rate, ...state.turns.map((t) => t.decode_tok_per_sec ?? t.tok_per_sec)) : 0;
  const kind = m.statsKind === "mock" ? "mock stream" : "measured";
  const copy = () => {
    navigator.clipboard?.writeText(splitThinking(m.content).answer).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };
  return (
    <div className="reply-foot">
      {m.status === "stopped" && <GridLoader kind="stopped" size={11} />}
      {m.status === "error" && <GridLoader kind="error" size={11} />}
      {m.status === "stopped" && <Chip tone="warn">Stopped</Chip>}
      {m.status === "error" && <Chip tone="err">Error</Chip>}
      {st && st.seconds > 0 && (
        <span className="mono tnum reply-time" title="Time for the whole reply">
          <Icon name="clock" size={12} />
          {st.seconds.toFixed(1)} s
        </span>
      )}
      <div className="reply-more">
        <span className="reply-model">
          {model && <ProviderLogo provider={model.provider} size={12} />}
          <span className="mono">{m.model ?? "model"}</span>
        </span>
        {st && (
          <>
            {m.statsKind === "mock" && <Chip>mock stream</Chip>}
            <Meter
              icon="bolt"
              value={measured ? rate.toFixed(1) : "—"}
              unit="tok/s"
              frac={measured && peak > 0 ? rate / peak : null}
              title={
                measured
                  ? `Output speed after the first token (${kind}). Bar: against the fastest reply this session.`
                  : "Speed is measured once the UI is connected to the engine"
              }
            />
            <Meter
              icon="hash"
              value={st.tokens.toLocaleString()}
              unit="tokens"
              frac={m.budget ? st.tokens / m.budget : null}
              title={m.budget ? `${st.tokens} of the ${m.budget} max tokens for this reply` : "Tokens in this reply"}
            />
            <Meter
              icon="timer"
              value={measured && st.first_token_s != null ? st.first_token_s.toFixed(2) : "—"}
              unit="s first"
              frac={measured && st.first_token_s != null && st.seconds > 0 ? st.first_token_s / st.seconds : null}
              title={
                measured
                  ? `Time to first token (${kind}); bar: share of the ${st.seconds.toFixed(1)} s total`
                  : "Measured once the UI is connected to the engine"
              }
            />
          </>
        )}
        <span className="grow" />
        <IconButton icon={copied ? "check" : "copy"} label={copied ? "Copied" : "Copy reply"} size={14} onClick={copy} className="tint-blue" />
        <IconButton icon="refresh" label="Regenerate (coming soon)" size={14} disabled />
      </div>
    </div>
  );
}

function liveKind(m: Message, thinkingLive: boolean): { kind: LoaderKind; words: string[] } {
  const { answer } = splitThinking(m.content);
  switch (m.status) {
    case "reading":
      return { kind: "thinking", words: ["Thinking…", "Analyzing…", "Reading prompt…"] };
    case "streaming":
      if (thinkingLive) return { kind: "reasoning", words: ["Reasoning…"] };
      return openFence(answer) ? { kind: "code", words: ["Writing code…"] } : { kind: "writing", words: ["Writing…"] };
    case "awaiting_approval":
      return m.approval?.decision ? { kind: "tool", words: ["Running tool…"] } : { kind: "approval", words: ["Waiting for approval"] };
    default:
      return { kind: "done", words: ["Done"] };
  }
}

/** Shown only while the reply is in progress: loader, rolling word, timer. */
function LiveLine({ m, thinkingLive }: { m: Message; thinkingLive: boolean }) {
  const elapsed = useElapsed(m.startedAt, true);
  const { kind, words } = liveKind(m, thinkingLive);
  return (
    <div className={`live-line k-${kind}`}>
      <GridLoader kind={kind} size={14} />
      <RollingWord words={words} />
      {m.startedAt !== undefined && <span className="mono muted tnum agent-time">{elapsed}</span>}
    </div>
  );
}

function AssistantMessage({ m, onApprove }: { m: Message; onApprove: (d: "once" | "session" | "deny") => void }) {
  const { thinking, answer, thinkingOpen } = splitThinking(m.content);
  const live = m.status === "streaming" || m.status === "reading";
  const working = live || m.status === "awaiting_approval";
  const pending = m.approval && !m.approval.decision ? m.approval.tool_name : null;
  return (
    <article className="msg assistant" tabIndex={-1}>
      {working && <LiveLine m={m} thinkingLive={thinkingOpen && live} />}
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
      </div>
      <ReplyFoot m={m} />
    </article>
  );
}

function UserMessage({ m }: { m: Message }) {
  const [copied, setCopied] = useState(false);
  return (
    <article className="msg user">
      <div className="query">{m.content}</div>
      <div className="query-foot">
        <IconButton
          icon={copied ? "check" : "copy"}
          label={copied ? "Copied" : "Copy message"}
          size={13}
          onClick={() => {
            navigator.clipboard?.writeText(m.content).catch(() => {});
            setCopied(true);
            setTimeout(() => setCopied(false), 1400);
          }}
        />
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
  /* Follow the reply only while the reader is at the bottom. Scrolling up
   * frees the view; the arrow button (or sending) sticks it again. */
  const [stuck, setStuck] = useState(true);
  const stuckRef = useRef(true);
  const lastTop = useRef(0);
  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    const wentUp = el.scrollTop < lastTop.current - 2;
    lastTop.current = el.scrollTop;
    // only the reader scrolling up unsticks; content growing never does
    if (stuckRef.current && !atBottom && !wentUp) return;
    if (atBottom !== stuckRef.current) {
      stuckRef.current = atBottom;
      setStuck(atBottom);
    }
  };
  const toBottom = (smooth = true) => {
    const el = scroller.current;
    if (!el) return;
    stuckRef.current = true;
    setStuck(true);
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "instant" });
  };
  useEffect(() => {
    if (stuckRef.current) {
      const el = scroller.current;
      if (el) el.scrollTo({ top: el.scrollHeight, behavior: "instant" });
    }
  }, [last?.content, last?.status]);
  useEffect(() => {
    toBottom(false);
  }, [chat?.messages.length, chat?.id]);

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
      onSend={(t) => {
        stuckRef.current = true;
        actions.send(t);
      }}
      onStop={actions.stop}
      busy={busy}
      disabled={!state.activeModel || state.engine.phase === "loading"}
      placeholder={
        state.activeModel
          ? "Message EigenCore"
          : state.engine.phase === "loading"
            ? "Loading the model…"
            : state.models.some((m) => m.downloaded)
              ? "Choose a model to start"
              : "Download a model in Models to start"
      }
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
            cpu={p.cpu_name
              .replace(/\(R\)|\(TM\)|CPU|@.*$/gi, "")
              .replace(/\s+/g, " ")
              .replace(/^(Intel|AMD)\s+(Core\s+)?/i, "")
              .trim()
              .slice(0, 13)}
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
                <Icon name={s.icon} size={14} className={s.tint} />
                <span>{s.text}</span>
              </button>
            ))}
          </div>
          <p className="local-note muted">
            <Icon name="shieldCheck" size={13} className="tint-green" /> Runs on your CPU. Chats stay on this device.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat">
      <div className="chat-head">
        <Icon name="chat" size={15} className="tint-blue" />
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
      <div className="chat-scroll" ref={scroller} onScroll={onScroll}>
        <div className="thread">
          {chat.messages.map((m) =>
            m.role === "user" ? (
              <UserMessage key={m.id} m={m} />
            ) : (
              <AssistantMessage key={m.id} m={m} onApprove={(d) => m.approval && actions.approve(m.id, m.approval.id, d)} />
            ),
          )}
        </div>
      </div>
      <div className="chat-composer">
        <button
          className={`to-bottom ${stuck ? "" : "show"} ${busy ? "live" : ""}`}
          onClick={() => toBottom()}
          aria-label="Scroll to the latest message"
          title="Jump to latest"
          tabIndex={stuck ? -1 : 0}
        >
          {busy && !stuck && <GridLoader kind="writing" size={10} />}
          <Icon name="arrowDown" size={16} />
        </button>
        {bar}
      </div>
    </div>
  );
}
