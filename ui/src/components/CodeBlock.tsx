import { useState, type ReactNode } from "react";
import { Icon } from "./Icon";

/*
 * Code block with line numbers and copy (layout after Beautiful UI's Code
 * Block, MIT; code written for EigenCore). Highlighting is monochrome: the
 * tokenizer only varies weight and ink level, so it works in both themes.
 */

const KW = new Set(
  "import from def return with as for in if else elif while class try except finally raise lambda yield async await const let var function export new true false null None True False and or not is pass break continue type interface".split(
    " ",
  ),
);

function highlight(line: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(#.*$|\/\/.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`[^`]*`)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|(.)/g;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(line))) {
    const [t, com, str, num, word] = m;
    if (com) out.push(<span key={k++} className="tk-com">{t}</span>);
    else if (str) out.push(<span key={k++} className="tk-str">{t}</span>);
    else if (num) out.push(<span key={k++} className="tk-num">{t}</span>);
    else if (word && KW.has(word)) out.push(<span key={k++} className="tk-kw">{t}</span>);
    else if (word && line[re.lastIndex] === "(") out.push(<span key={k++} className="tk-fn">{t}</span>);
    else out.push(t);
  }
  return out;
}

export function CodeBlock({ code, lang, streaming }: { code: string; lang?: string; streaming?: boolean }) {
  const [copied, setCopied] = useState(false);
  const lines = code.replace(/\n$/, "").split("\n");
  const copy = () => {
    navigator.clipboard?.writeText(code).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className="codeblock">
      <div className="codeblock-head">
        <span className="mono">{lang || "text"}</span>
        <button className="codeblock-copy" onClick={copy} disabled={streaming} aria-label="Copy code">
          <Icon name={copied ? "check" : "copy"} size={13} />
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        <code>
          {lines.map((l, i) => (
            <span className="code-line" key={i}>
              <span className="ln" aria-hidden>
                {i + 1}
              </span>
              <span>{highlight(l)}</span>
              {"\n"}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
