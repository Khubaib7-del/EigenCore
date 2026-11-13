import type { ReactNode } from "react";
import { CodeBlock } from "./CodeBlock";

/**
 * Tiny markdown renderer for model replies (no dependency). Handles fenced
 * code (including an unclosed fence while streaming), headings, lists,
 * paragraphs, **bold**, *em*, `code` and [links](url). Never renders HTML.
 */

function inline(text: string, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*)|(\[[^\]]+\]\([^)\s]+\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    const key = `${keyBase}-${k++}`;
    if (m[1]) out.push(<code key={key} className="inline-code">{t.slice(1, -1)}</code>);
    else if (m[2]) out.push(<strong key={key}>{t.slice(2, -2)}</strong>);
    else if (m[3]) out.push(<em key={key}>{t.slice(1, -1)}</em>);
    else {
      const [, label, href] = /\[([^\]]+)\]\(([^)]+)\)/.exec(t)!;
      out.push(
        <a key={key} href={href} target="_blank" rel="noreferrer">
          {label}
        </a>,
      );
    }
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ text, streaming }: { text: string; streaming?: boolean }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let k = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = /^```(\w*)\s*$/.exec(line);
    if (fence) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) body.push(lines[i++]);
      const closed = i < lines.length;
      i++;
      blocks.push(<CodeBlock key={k++} code={body.join("\n")} lang={fence[1]} streaming={streaming && !closed} />);
      continue;
    }
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) {
      const Tag = (`h${h[1].length + 2}` as "h3" | "h4" | "h5");
      blocks.push(<Tag key={k++}>{inline(h[2], `h${k}`)}</Tag>);
      i++;
      continue;
    }
    if (/^\s*[-*]\s+/.test(line) || /^\s*\d+\.\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items: string[] = [];
      while (i < lines.length && (/^\s*[-*]\s+/.test(lines[i]) || /^\s*\d+\.\s+/.test(lines[i]))) {
        items.push(lines[i].replace(/^\s*(?:[-*]|\d+\.)\s+/, ""));
        i++;
      }
      const L = ordered ? "ol" : "ul";
      blocks.push(
        <L key={k++}>
          {items.map((it, j) => (
            <li key={j}>{inline(it, `li${k}-${j}`)}</li>
          ))}
        </L>,
      );
      continue;
    }
    if (!line.trim()) {
      i++;
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^```/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s+/.test(lines[i]) && !/^#{1,3}\s/.test(lines[i])) {
      para.push(lines[i++]);
    }
    blocks.push(<p key={k++}>{inline(para.join(" "), `p${k}`)}</p>);
  }
  return (
    <div className={`md ${streaming ? "is-streaming" : ""}`}>
      {blocks}
      {streaming && (blocks.length === 0 || lines[lines.length - 1].startsWith("```")) && <span className="caret" aria-hidden />}
    </div>
  );
}

/** Split a raw reply into its <think> part and the visible answer. */
export function splitThinking(raw: string): { thinking: string | null; answer: string; thinkingOpen: boolean } {
  const start = raw.indexOf("<think>");
  if (start === -1) return { thinking: null, answer: raw, thinkingOpen: false };
  const end = raw.indexOf("</think>");
  if (end === -1) return { thinking: raw.slice(start + 7), answer: "", thinkingOpen: true };
  return { thinking: raw.slice(start + 7, end).trim(), answer: (raw.slice(0, start) + raw.slice(end + 8)).replace(/^\s+/, ""), thinkingOpen: false };
}
