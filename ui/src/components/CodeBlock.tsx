import { useState, type ReactNode } from "react";
import { Icon } from "./Icon";

/*
 * Code block with a header (file icon, language, line count, copy), line
 * numbers and horizontal scroll (layout after Beautiful UI's Code Block, MIT;
 * code written for EigenCore). It renders the same way while the fence is
 * still open and the model is writing: the header says "writing" and a caret
 * sits at the end of the last line. Highlighting is monochrome: the tiny
 * tokenizer only varies weight and ink level, so it works in both themes.
 */

const KW = new Set(
  (
    "import from def return with as for in if else elif while class try except finally raise lambda yield async await " +
    "const let var function export default new true false null undefined None True False and or not is pass break continue " +
    "type interface enum struct impl fn pub use mut match where package func go defer chan select switch case public private " +
    "protected static void int float double bool char string auto namespace template typename echo fi then do done esac local"
  ).split(" "),
);

const LANG: Record<string, string> = {
  py: "Python",
  python: "Python",
  python3: "Python",
  ts: "TypeScript",
  tsx: "TSX",
  typescript: "TypeScript",
  js: "JavaScript",
  jsx: "JSX",
  javascript: "JavaScript",
  sh: "Shell",
  bash: "Bash",
  zsh: "Shell",
  shell: "Shell",
  console: "Shell",
  ps1: "PowerShell",
  powershell: "PowerShell",
  json: "JSON",
  yaml: "YAML",
  yml: "YAML",
  toml: "TOML",
  rs: "Rust",
  rust: "Rust",
  go: "Go",
  c: "C",
  cpp: "C++",
  "c++": "C++",
  cs: "C#",
  java: "Java",
  html: "HTML",
  css: "CSS",
  sql: "SQL",
  md: "Markdown",
  markdown: "Markdown",
  diff: "Diff",
  text: "Text",
  txt: "Text",
};

/** Hash-comment languages vs slash-comment languages (others get both). */
const HASH = new Set(["py", "python", "python3", "sh", "bash", "zsh", "shell", "console", "ps1", "powershell", "yaml", "yml", "toml", "rb", "ruby"]);
const SLASH = new Set(["ts", "tsx", "typescript", "js", "jsx", "javascript", "rs", "rust", "go", "c", "cpp", "c++", "cs", "java", "css", "kt", "swift"]);

function highlight(line: string, lang: string): ReactNode[] {
  const out: ReactNode[] = [];
  if (lang === "diff" && /^[+-]/.test(line)) return [<span key={0} className={line[0] === "+" ? "tk-add" : "tk-del"}>{line}</span>];
  const com = HASH.has(lang) ? "#.*$" : SLASH.has(lang) ? "\\/\\/.*$|\\/\\*.*?(?:\\*\\/|$)" : "#.*$|\\/\\/.*$";
  const re = new RegExp(
    `(${com})|("(?:[^"\\\\]|\\\\.)*"?|'(?:[^'\\\\]|\\\\.)*'?|\`[^\`]*\`?)|(\\b\\d+(?:\\.\\d+)?\\b)|([A-Za-z_][A-Za-z0-9_]*)|(\\s+)|(.)`,
    "g",
  );
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(line))) {
    const [t, c, str, num, word] = m;
    if (c) out.push(<span key={k++} className="tk-com">{t}</span>);
    else if (str) out.push(<span key={k++} className="tk-str">{t}</span>);
    else if (num) out.push(<span key={k++} className="tk-num">{t}</span>);
    else if (word && KW.has(word)) out.push(<span key={k++} className="tk-kw">{t}</span>);
    else if (word && line[re.lastIndex] === "(") out.push(<span key={k++} className="tk-fn">{t}</span>);
    else out.push(t);
  }
  return out;
}

export function langLabel(lang?: string) {
  const l = (lang || "").toLowerCase();
  return LANG[l] ?? (l ? l : "Text");
}

export function CodeBlock({ code, lang, streaming }: { code: string; lang?: string; streaming?: boolean }) {
  const [copied, setCopied] = useState(false);
  const l = (lang || "").toLowerCase();
  const lines = code.replace(/\n$/, "").split("\n");
  const copy = () => {
    navigator.clipboard?.writeText(code).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className={`codeblock ${streaming ? "streaming" : ""}`}>
      <div className="codeblock-head">
        <Icon name={l === "sh" || l === "bash" || l === "shell" || l === "console" || l === "powershell" ? "terminal" : "fileCode"} size={14} />
        <span className="codeblock-lang">{langLabel(lang)}</span>
        <span className="codeblock-lines mono">
          {lines.length} line{lines.length === 1 ? "" : "s"}
        </span>
        {streaming && (
          <span className="codeblock-live">
            <span className="eq" aria-hidden>
              <i />
              <i />
              <i />
            </span>
            writing
          </span>
        )}
        <button className="codeblock-copy" onClick={copy} disabled={streaming} aria-label="Copy code" title={streaming ? "Copy when the block is finished" : "Copy code"}>
          <Icon name={copied ? "check" : "copy"} size={13} />
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        <code>
          {lines.map((ln, i) => (
            <span className="code-line" key={i}>
              <span className="ln" aria-hidden>
                {i + 1}
              </span>
              <span className="lc">
                {highlight(ln, l)}
                {streaming && i === lines.length - 1 && <span className="caret code-caret" aria-hidden />}
              </span>
              {"\n"}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
