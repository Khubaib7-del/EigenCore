import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { ModelRow } from "../api";
import { DEFAULT_GEN, type GenOptions } from "../state/store";
import { Icon, type IconName } from "./Icon";
import { FitChip, GlideMenu, Kbd, Ring, SoonPill, useDismiss } from "./primitives";
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

export interface SlashCommand {
  cmd: string;
  hint: string;
  icon: IconName;
  run: () => void;
}

/** Preview-only: ?composer=options|mode|attach|slash opens that menu (screenshots). */
function previewOpen(): "options" | "mode" | "attach" | "slash" | null {
  const v = new URLSearchParams(location.search).get("composer");
  return v === "options" || v === "mode" || v === "attach" || v === "slash" ? v : null;
}

type Menu = "options" | "mode" | "attach" | null;

function MenuButton({
  id,
  open,
  setOpen,
  icon,
  label,
  text,
  children,
  className = "",
}: {
  id: Exclude<Menu, null>;
  open: Menu;
  setOpen: (m: Menu) => void;
  icon: IconName;
  label: string;
  text?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  const isOpen = open === id;
  const close = useCallback(() => setOpen(null), [setOpen]);
  const ref = useDismiss(isOpen, close);
  return (
    <div className={`composer-menu ${className}`} ref={ref}>
      <button
        className={`composer-btn ${text ? "with-text" : ""} ${isOpen ? "open" : ""}`}
        aria-label={label}
        title={label}
        aria-haspopup="menu"
        aria-expanded={isOpen}
        onClick={() => setOpen(isOpen ? null : id)}
      >
        <Icon name={icon} size={15} />
        {text}
        {text && <Icon name="chevronDown" size={12} className="composer-chev" />}
      </button>
      {isOpen && <div className={`popover composer-pop pop-${id}`} role="menu">{children}</div>}
    </div>
  );
}

const MAX_TOKENS = [256, 512, 1024, 2048] as const;

export function PromptBar({
  onSend,
  onStop,
  busy,
  disabled,
  placeholder,
  picker,
  meter,
  autoFocus,
  options,
  onOptions,
  contextLength,
  commands = [],
  fill,
  onTextChange,
}: {
  onSend: (t: string) => void;
  onStop: () => void;
  busy: boolean;
  disabled?: boolean;
  placeholder?: string;
  picker: ReactNode;
  meter?: ReactNode;
  autoFocus?: boolean;
  options: GenOptions;
  onOptions: (patch: Partial<GenOptions>) => void;
  contextLength?: number;
  commands?: SlashCommand[];
  /** Put this text in the box (n changes on every fill, so the same text can be put twice). */
  fill?: { text: string; n: number } | null;
  /** Told about every change to the box (the empty chat's screen echoes it). */
  onTextChange?: (t: string) => void;
}) {
  const preview = useRef(previewOpen()).current;
  const [text, setText] = useState(preview === "slash" ? "/" : "");
  const [menu, setMenu] = useState<Menu>(preview && preview !== "slash" ? preview : null);
  const [sel, setSel] = useState(0);
  const ta = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    onTextChange?.(text);
  }, [text, onTextChange]);
  useLayoutEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = Math.min(200, el.scrollHeight) + "px";
  }, [text]);
  useEffect(() => {
    if (!fill) return;
    setText(fill.text);
    requestAnimationFrame(() => {
      const el = ta.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(fill.text.length, fill.text.length);
    });
  }, [fill]);
  const slash = /^\/\S*$/.test(text) ? commands.filter((c) => c.cmd.startsWith(text.toLowerCase())) : [];
  const runCmd = (c: SlashCommand) => {
    setText("");
    setSel(0);
    c.run();
  };
  const send = () => {
    if (busy || disabled || !text.trim()) return;
    onSend(text.trim());
    setText("");
  };
  const tempWord = options.temperature <= 0.3 ? "focused" : options.temperature <= 0.9 ? "balanced" : "creative";
  return (
    <div className={`promptbar ${busy ? "busy" : ""}`}>
      {slash.length > 0 && (
        <div className="popover slash-menu" role="listbox" aria-label="Commands">
          <div className="popover-label">Commands</div>
          {slash.map((c, i) => (
            <button
              key={c.cmd}
              role="option"
              aria-selected={i === sel}
              className={`slash-item ${i === sel ? "sel" : ""}`}
              onMouseEnter={() => setSel(i)}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => runCmd(c)}
            >
              <Icon name={c.icon} size={14} />
              <span className="mono">{c.cmd}</span>
              <span className="muted">{c.hint}</span>
            </button>
          ))}
        </div>
      )}
      <textarea
        ref={ta}
        rows={1}
        value={text}
        autoFocus={autoFocus}
        placeholder={placeholder ?? "Message EigenCore"}
        aria-label="Message"
        onChange={(e) => {
          setText(e.target.value);
          setSel(0);
        }}
        onKeyDown={(e) => {
          if (slash.length) {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              return setSel((n) => Math.min(slash.length - 1, n + 1));
            }
            if (e.key === "ArrowUp") {
              e.preventDefault();
              return setSel((n) => Math.max(0, n - 1));
            }
            if ((e.key === "Enter" && !e.shiftKey) || e.key === "Tab") {
              e.preventDefault();
              return runCmd(slash[Math.min(sel, slash.length - 1)]);
            }
            if (e.key === "Escape") return setText("");
          }
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            send();
          }
        }}
      />
      <div className="promptbar-row">
        <MenuButton id="attach" open={menu} setOpen={setMenu} icon="plus" label="Attach">
          <div className="popover-label">Attach</div>
          <button className="menu-item" disabled role="menuitem">
            <Icon name="paperclip" size={14} />
            <span className="menu-text">
              <span>Files</span>
              <span className="muted">Add text or code files to the message</span>
            </span>
            <SoonPill />
          </button>
          <button className="menu-item" disabled role="menuitem">
            <Icon name="folder" size={14} />
            <span className="menu-text">
              <span>A folder</span>
              <span className="muted">Let the agent work inside a workspace</span>
            </span>
            <SoonPill />
          </button>
        </MenuButton>
        <MenuButton id="mode" open={menu} setOpen={setMenu} icon="chat" label="Mode" text={<span>Chat</span>}>
          <div className="popover-label">Mode</div>
          <button className="menu-item on" role="menuitemradio" aria-checked onClick={() => setMenu(null)}>
            <Icon name="chat" size={14} />
            <span className="menu-text">
              <span>Chat</span>
              <span className="muted">Talk to the local model</span>
            </span>
            <Icon name="check" size={14} />
          </button>
          <button className="menu-item" role="menuitemradio" aria-checked={false} disabled>
            <Icon name="bot" size={14} />
            <span className="menu-text">
              <span>Agent</span>
              <span className="muted">Uses tools, asks before each one</span>
            </span>
            <SoonPill />
          </button>
          <button className="menu-item" role="menuitemradio" aria-checked={false} disabled>
            <Icon name="layers" size={14} />
            <span className="menu-text">
              <span>Plan</span>
              <span className="muted">Writes a step-by-step plan first</span>
            </span>
            <SoonPill />
          </button>
        </MenuButton>
        {picker}
        <button className="composer-btn with-text soon-btn" disabled title="Tools: coming soon (the desktop chat has no tools yet)" aria-label="Tools (coming soon)">
          <Icon name="wrench" size={14} />
          <span>Tools</span>
          <SoonPill />
        </button>
        <MenuButton id="options" open={menu} setOpen={setMenu} icon="sliders" label="Reply options" className="options-menu">
          <div className="popover-label">Reply options</div>
          <div className="opt">
            <div className="opt-head">
              <Icon name="thermometer" size={14} />
              <span>Temperature</span>
              <span className="grow" />
              <span className="mono tnum">{options.temperature.toFixed(1)}</span>
              <span className="muted">{tempWord}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1.5}
              step={0.1}
              value={options.temperature}
              aria-label="Temperature"
              onChange={(e) => onOptions({ temperature: Number(e.target.value) })}
              style={{ ["--fill" as string]: `${(options.temperature / 1.5) * 100}%` }}
            />
          </div>
          <div className="opt">
            <div className="opt-head">
              <Icon name="hash" size={14} />
              <span>Longest reply</span>
              <span className="grow" />
              <span className="muted">tokens</span>
            </div>
            <div className="opt-chips" role="radiogroup" aria-label="Longest reply">
              {MAX_TOKENS.map((n) => (
                <button key={n} role="radio" aria-checked={options.max_tokens === n} className={options.max_tokens === n ? "on" : ""} onClick={() => onOptions({ max_tokens: n })}>
                  {n.toLocaleString()}
                </button>
              ))}
            </div>
          </div>
          <div className="opt opt-ro">
            <div className="opt-head">
              <Icon name="layers" size={14} />
              <span>Context window</span>
              <span className="grow" />
              <span className="mono tnum">{contextLength ? contextLength.toLocaleString() : "—"}</span>
              <SoonPill title="Set by the model and hardware profile; changing it is coming soon" />
            </div>
          </div>
          <div className="popover-sep" />
          <button className="popover-link" onClick={() => onOptions({ ...DEFAULT_GEN })}>
            <Icon name="refresh" size={14} /> Reset to defaults
          </button>
        </MenuButton>
        <span className="grow" />
        {meter}
        <span className="promptbar-hint muted">
          <Kbd>/</Kbd> commands
        </span>
        {busy ? (
          <button className="send-btn stop" onClick={onStop} aria-label="Stop generating" title="Stop">
            <Icon name="stop" size={14} />
          </button>
        ) : (
          <button className="send-btn" onClick={send} disabled={disabled || !text.trim()} aria-label="Send" title="Send (Enter) · new line (Shift+Enter)">
            <Icon name="arrowUp" size={16} />
          </button>
        )}
      </div>
    </div>
  );
}
