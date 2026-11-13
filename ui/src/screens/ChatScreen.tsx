import { useEffect, useRef } from "react";
import { ApprovalCard, ThinkingFold, ToolChips } from "../components/agent";
import { Icon } from "../components/Icon";
import { Markdown, splitThinking } from "../components/Markdown";
import { Chip, IconButton, LoadingState } from "../components/primitives";
import { ContextMeter, ModelPicker, PromptBar } from "../components/PromptBar";
import { ProviderLogo } from "../components/ProviderLogo";
import { useStore, type Message } from "../state/store";

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? "Good evening" : h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
}

const SUGGESTIONS = [
  { icon: "terminal" as const, text: "Write a Python script that renames photos by date" },
  { icon: "file" as const, text: "Summarise a long document in five bullet points" },
  { icon: "sparkle" as const, text: "Explain how quantization makes models smaller" },
  { icon: "globe" as const, text: "Translate a paragraph into Urdu" },
];

function Stats({ m }: { m: Message }) {
  const st = m.stats;
  if (!st || m.status === "reading" || m.status === "streaming" || m.status === "awaiting_approval") return null;
  const measured = m.statsKind !== "placeholder";
  const tps = measured ? `${st.tok_per_sec.toFixed(1)} tok/s` : "— tok/s";
  const ft = measured && st.first_token_s != null ? `first token ${st.first_token_s.toFixed(2)} s` : "first token —";
  return (
    <div className="msg-stats">
      {m.statsKind === "mock" && <Chip>mock stream</Chip>}
      {m.status === "stopped" && <Chip tone="warn">Stopped</Chip>}
      {m.status === "error" && <Chip tone="err">Error</Chip>}
      <span
        className="mono tnum"
        title={measured ? "Timed from the stream the UI received" : "Speed is measured once the UI is connected to the engine"}
      >
        {tps} · {st.tokens.toLocaleString()} tokens · {ft}
      </span>
      <span className="grow" />
      <IconButton icon="copy" label="Copy reply" size={14} onClick={() => navigator.clipboard?.writeText(splitThinking(m.content).answer)} />
      <IconButton icon="refresh" label="Regenerate (coming later)" size={14} disabled />
    </div>
  );
}

function AssistantMessage({ m, onApprove }: { m: Message; onApprove: (d: "once" | "session" | "deny") => void }) {
  const { state } = useStore();
  const { thinking, answer, thinkingOpen } = splitThinking(m.content);
  const model = state.models.find((x) => x.name === m.model);
  const live = m.status === "streaming" || m.status === "reading";
  return (
    <article className="msg assistant">
      <div className="msg-meta">
        {model && <ProviderLogo provider={model.provider} size={13} />}
        <span className="mono">{m.model}</span>
      </div>
      {m.tools && m.tools.length > 0 && <ToolChips calls={m.tools} />}
      {m.approval && <ApprovalCard request={m.approval} decision={m.approval.decision} onDecide={onApprove} />}
      {thinking !== null && <ThinkingFold text={thinking} live={thinkingOpen && live} />}
      {m.status === "reading" && <LoadingState label="Reading prompt" since={m.startedAt} />}
      {m.status === "awaiting_approval" && !m.approval?.decision && (
        <div className="muted waiting">
          <Icon name="clock" size={13} /> Waiting for your approval
        </div>
      )}
      {answer && <Markdown text={answer} streaming={m.status === "streaming" && !thinkingOpen} />}
      {m.status === "streaming" && !answer && !thinkingOpen && <LoadingState label="Writing" since={m.startedAt} />}
      <Stats m={m} />
    </article>
  );
}

export function ChatScreen() {
  const { state, dispatch, actions } = useStore();
  const chat = state.chats.find((c) => c.id === state.activeChatId);
  const busy = state.engine.phase === "reading" || state.engine.phase === "writing";
  const scroller = useRef<HTMLDivElement>(null);
  const last = chat?.messages[chat.messages.length - 1];
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [last?.content, last?.status, chat?.messages.length]);

  const active = state.models.find((m) => m.name === state.activeModel);
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
      autoFocus
    />
  );

  if (!chat || chat.messages.length === 0) {
    return (
      <div className="chat-empty">
        <div className="chat-empty-inner">
          <h1 className="greeting">{greeting()}</h1>
          <p className="greeting-sub">What should we work on?</p>
          {bar}
          <div className="suggestions">
            {SUGGESTIONS.map((s) => (
              <button key={s.text} className="suggestion" onClick={() => actions.send(s.text)} disabled={!state.activeModel}>
                <Icon name={s.icon} size={14} />
                <span>{s.text}</span>
              </button>
            ))}
          </div>
          <p className="local-note muted">
            <Icon name="shield" size={13} /> Runs on your CPU. Chats stay on this device.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat">
      <div className="chat-head">
        <h1 className="ellipsis">{chat.title}</h1>
        <span className="grow" />
        {active && (
          <span className="chip">
            <ProviderLogo provider={active.provider} size={12} />
            <span className="mono">{active.name}</span>
          </span>
        )}
        <IconButton icon="dots" label="Chat options" />
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
