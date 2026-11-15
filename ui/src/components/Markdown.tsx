import type { ReactNode } from "react";
import { TreeList, type TreeRow } from "./agent";
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

/** A list item that is only a path (`src/app.ts`, `ui/`), for file trees. */
const PATH_ITEM = /^`?(~?[\w@.+-]+(?:\/[\w@.+-]+)*\/?)`?$/;
function isPathItem(t: string) {
  const m = PATH_ITEM.exec(t.trim());
  return !!m && (m[1].endsWith("/") || /\.[A-Za-z0-9]{1,6}$/.test(m[1]));
}
const CODE_EXT = /\.(py|ts|tsx|js|jsx|mjs|rs|go|c|h|cpp|hpp|cs|java|kt|rb|php|sh|ps1|toml|json|ya?ml|css|html|sql|lua)$/i;

function FileTree({ items }: { items: { indent: number; text: string }[] }) {
  const base = Math.min(...items.map((i) => i.indent));
  const depths = items.map((i) => Math.round((i.indent - base) / 2));
  const rows: TreeRow[] = items.map((it, i) => {
    const name = it.text.trim().replace(/^`|`$/g, "");
    const dir = name.endsWith("/") || (depths[i + 1] ?? -1) > depths[i];
    return {
      depth: depths[i],
      icon: dir ? "folderOpen" : CODE_EXT.test(name) ? "fileCode" : "file",
      label: <span className={`mono ${dir ? "tree-dir" : ""}`}>{name}</span>,
    };
  });
  return (
    <div className="md-tree">
      <TreeList rows={rows} label="Files" />
    </div>
  );
}

const FENCE_OPEN = /^\s{0,3}(`{3,}|~{3,})\s*([^\s`]*)[^`]*$/;

/** Language of the code fence still open at the end of `text`, or null. */
export function openFence(text: string): string | null {
  let open: { mark: string; lang: string } | null = null;
  for (const line of text.split("\n")) {
    if (open) {
      const c = /^\s{0,3}(`{3,}|~{3,})\s*$/.exec(line);
      if (c && c[1][0] === open.mark[0] && c[1].length >= open.mark.length) open = null;
    } else {
      const f = FENCE_OPEN.exec(line);
      if (f) open = { mark: f[1], lang: f[2] };
    }
  }
  return open ? open.lang || "text" : null;
}

export function Markdown({ text, streaming }: { text: string; streaming?: boolean }) {
  const lines = text.split("\n");
  // While streaming, a fence that is still arriving ("`", "``") is not text yet.
  if (streaming && /^\s{0,3}`{1,2}$/.test(lines[lines.length - 1])) lines.pop();
  const blocks: ReactNode[] = [];
  let i = 0;
  let k = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = FENCE_OPEN.exec(line);
    if (fence) {
      const mark = fence[1];
      const body: string[] = [];
      i++;
      let closed = false;
      while (i < lines.length) {
        const c = /^\s{0,3}(`{3,}|~{3,})\s*$/.exec(lines[i]);
        if (c && c[1][0] === mark[0] && c[1].length >= mark.length) {
          closed = true;
          break;
        }
        body.push(lines[i++]);
      }
      i++;
      // a closing fence that is half-way in ("``") is not code
      if (!closed && streaming && body.length && /^\s*[`~]{1,2}$/.test(body[body.length - 1])) body.pop();
      blocks.push(<CodeBlock key={k++} code={body.join("\n")} lang={fence[2]} streaming={streaming && !closed} />);
      continue;
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      const Tag = `h${Math.min(5, h[1].length + 2)}` as "h3" | "h4" | "h5";
      blocks.push(<Tag key={k++}>{inline(h[2], `h${k}`)}</Tag>);
      i++;
      continue;
    }
    if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) {
      blocks.push(<hr key={k++} />);
      i++;
      continue;
    }
    if (/^\s*>/.test(line)) {
      const quote: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
      blocks.push(<blockquote key={k++}>{inline(quote.join(" "), `q${k}`)}</blockquote>);
      continue;
    }
    if (LIST_ITEM.test(line)) {
      const ordered = /^\s*\d+[.)]/.test(line);
      const items: { indent: number; text: string }[] = [];
      while (i < lines.length && LIST_ITEM.test(lines[i])) {
        const m = LIST_ITEM.exec(lines[i])!;
        items.push({ indent: m[1].length, text: lines[i].slice(m[0].length) });
        i++;
      }
      if (!ordered && items.length >= 2 && items.every((it) => isPathItem(it.text))) {
        blocks.push(<FileTree key={k++} items={items} />);
        continue;
      }
      const L = ordered ? "ol" : "ul";
      blocks.push(
        <L key={k++}>
          {items.map((it, j) => (
            <li key={j} className={it.indent >= 2 ? "nested" : undefined}>
              {inline(it.text, `li${k}-${j}`)}
            </li>
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
    while (
      i < lines.length &&
      lines[i].trim() &&
      !FENCE_OPEN.test(lines[i]) &&
      !LIST_ITEM.test(lines[i]) &&
      !/^#{1,4}\s/.test(lines[i]) &&
      !/^\s*>/.test(lines[i])
    ) {
      para.push(lines[i++]);
    }
    blocks.push(<p key={k++}>{inline(para.join(" "), `p${k}`)}</p>);
  }
  return (
    <div className={`md ${streaming ? "is-streaming" : ""}`}>
      {blocks}
      {streaming && blocks.length === 0 && <span className="caret" aria-hidden />}
    </div>
  );
}

const LIST_ITEM = /^(\s*)(?:[-*+]|\d+[.)])\s+/;

/** Split a raw reply into its <think> part and the visible answer. */
export function splitThinking(raw: string): { thinking: string | null; answer: string; thinkingOpen: boolean } {
  const start = raw.indexOf("<think>");
  if (start === -1) return { thinking: null, answer: raw, thinkingOpen: false };
  const end = raw.indexOf("</think>");
  if (end === -1) return { thinking: raw.slice(start + 7), answer: "", thinkingOpen: true };
  return { thinking: raw.slice(start + 7, end).trim(), answer: (raw.slice(0, start) + raw.slice(end + 8)).replace(/^\s+/, ""), thinkingOpen: false };
}
