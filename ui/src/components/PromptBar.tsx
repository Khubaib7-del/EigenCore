import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { ModelRow } from "../api";
import { Icon } from "./Icon";
import { FitChip, GlideMenu, Kbd, Ring } from "./primitives";
import { ProviderLogo } from "./ProviderLogo";

/* Composer (after Beautiful UI's Prompt Bar, MIT; layout studied from
 * affectUI's AI Chat screens, no code used). Enter sends, Shift+Enter adds a
 * newline, the send button turns into Stop while a reply streams. */

export function ModelPicker({
  models,
  active,
  onPick,
  onManage,
  loading,
}: {
  models: ModelRow[];
  active: string | null;
  onPick: (name: string) => void;
  onManage: () => void;
  loading: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const downloaded = models.filter((m) => m.downloaded);
  const cur = models.find((m) => m.name === active);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", esc);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", esc);
    };
  }, [open]);
  return (
    <div className="model-picker" ref={ref}>
      <button className="model-trigger" onClick={() => setOpen((o) => !o)} aria-haspopup="listbox" aria-expanded={open}>
        {cur ? <ProviderLogo provider={cur.provider} size={14} /> : <Icon name="models" size={14} />}
        <span className="mono">{loading ? "Loading…" : cur?.name ?? "Choose a model"}</span>
        <Icon name="chevronDown" size={13} />
      </button>
      {open && (
        <div className="popover model-menu" role="listbox" aria-label="Downloaded models">
          <div className="popover-label">Downloaded</div>
          <GlideMenu>
            {downloaded.map((m) => (
              <button
                key={m.name}
                data-glide
                role="option"
                aria-selected={m.name === active}
                className="model-option"
                onClick={() => {
                  onPick(m.name);
                  setOpen(false);
                }}
              >
                <ProviderLogo provider={m.provider} size={15} />
                <span className="model-option-text">
                  <span className="mono">{m.name}</span>
                  <span className="muted">
                    {m.params_b}B · {m.quant} · ctx {m.context_length.toLocaleString()}
                  </span>
                </span>
                <FitChip fit={m.fit} />
                <span className="model-option-check">{m.name === active && <Icon name="check" size={14} />}</span>
              </button>
            ))}
          </GlideMenu>
          <div className="popover-sep" />
          <button
            className="popover-link"
            onClick={() => {
              onManage();
              setOpen(false);
            }}
          >
            <Icon name="download" size={14} /> Get more models
          </button>
        </div>
      )}
    </div>
  );
}

export function ContextMeter({ used, total }: { used: number; total: number }) {
  const v = total ? used / total : 0;
  return (
    <span className="context-meter" title={`Context: ${used.toLocaleString()} of ${total.toLocaleString()} tokens (approximate)`}>
      <Ring value={v} size={16} stroke={2.5} />
      <span className="mono tnum">
        {used.toLocaleString()} / {total.toLocaleString()}
      </span>
    </span>
  );
}

export function PromptBar({
  onSend,
  onStop,
  busy,
  disabled,
  placeholder,
  picker,
  meter,
  autoFocus,
}: {
  onSend: (t: string) => void;
  onStop: () => void;
  busy: boolean;
  disabled?: boolean;
  placeholder?: string;
  picker: ReactNode;
  meter?: ReactNode;
  autoFocus?: boolean;
}) {
  const [text, setText] = useState("");
  const ta = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = Math.min(200, el.scrollHeight) + "px";
  }, [text]);
  const send = () => {
    if (busy || disabled || !text.trim()) return;
    onSend(text.trim());
    setText("");
  };
  return (
    <div className={`promptbar ${busy ? "busy" : ""}`}>
      <textarea
        ref={ta}
        rows={1}
        value={text}
        autoFocus={autoFocus}
        placeholder={placeholder ?? "Message EigenCore"}
        aria-label="Message"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            send();
          }
        }}
      />
      <div className="promptbar-row">
        <button className="icon-btn" aria-label="Attach a file (coming later)" title="Attach a file (coming later)" disabled>
          <Icon name="paperclip" size={16} />
        </button>
        {picker}
        <span className="grow" />
        {meter}
        <span className="promptbar-hint muted">
          <Kbd>↵</Kbd> send <Kbd>⇧↵</Kbd> new line
        </span>
        {busy ? (
          <button className="send-btn stop" onClick={onStop} aria-label="Stop generating" title="Stop">
            <Icon name="stop" size={14} />
          </button>
        ) : (
          <button className="send-btn" onClick={send} disabled={disabled || !text.trim()} aria-label="Send" title="Send">
            <Icon name="arrowUp" size={16} />
          </button>
        )}
      </div>
    </div>
  );
}
