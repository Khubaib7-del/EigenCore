# EigenCore review: TUI and agent sandboxing

Scope: a read-only checkout of `main` at c92e3af. Probes ran out-of-tree (probe scripts not committed) on Linux with rich 15.0.0. Tags: **[verified]** means reproduced or read in source/docs, **[inferred]** means reasoned from evidence, **[unverified]** means not checked against a primary source.

## A. TUI (terminal chat)

### What exists
- `src/eigencore/tui/chat.py:42-196`: a line-oriented REPL that streams tokens. It uses a Rich `Console` (`:67-70`), or plain `print`/`out` when `plain=True`. It has 5 slash commands with difflib suggestions (`:128-154`), a header with the fit verdict (`:97-110`), and a measured tok/s line (`:192-196`). Its docstring admits that the full-screen layout, approval cards, tool calls and isolation are **not** implemented (`:17-18`).
- `cli/main.py:236-282`: the `chat` command. Plain mode turns on when `--plain` is passed or when stdin is not a TTY (`:277`).
- `tests/test_tui_chat.py`: 5 tests, all with `plain=True` (`:72`, `:124`). The Rich path has no test coverage.

### Bugs and overclaims
1. **Model output is parsed as Rich markup** (`chat.py:83`, `console.print(text, end="")`). If a token is `[/]`, the client raises `MarkupError: closing tag '[/]' ... has nothing to close` and the session crashes. A token like `[bold red]` gets styled instead of printed. **[verified]** by probe. Rich documents the injection risk and points to `escape()` or `markup=False` (https://rich.readthedocs.io/en/stable/markup.html). Emoji codes like `:warning:` get substituted the same way **[inferred]**.
2. **Terminal escape sequences pass through untouched, in both modes.** Plain mode writes raw `ESC]0;…BEL` (OSC title set) and `ESC[2J` straight to the terminal. Rich mode still emits the `\x1b` byte. **[verified]** This contradicts the design doc rule to "escape terminal control sequences" (`docs/design/terminal-chat.md:61`).
3. **Ctrl-C during generation is not caught.** The `try` only wraps `_input` (`chat.py:160-164`). A `KeyboardInterrupt` mid-stream escapes as a traceback, which breaks the "Ctrl-C ends cleanly" claim (`:15`). History is also left with a user turn that has no reply. **[verified]**
4. **The `AttributeError` fallback duplicates output** (`:186-191`). Any `AttributeError` raised *inside* `stream_messages` after some tokens triggers a second generation through `engine.stream(user_input)`. The probe printed `HelloHello again`, which breaks the docstring promise that output is never duplicated (`:43-44`). The fallback also drops the conversation history. **[verified]**
5. **The header label is misleading.** `policy_label="read-only (policy engine attached in P4)"` (`cli/main.py:275`), but chat exposes no tools and attaches no policy. It should read "no tools". **[verified]**
6. **The design doc status is stale.** It says "proposed… not implemented" (`terminal-chat.md:3`), yet a chat core with plain mode and slash commands exists. **[verified]**
7. **Minor issues:**
   - tok/s counts stream *chunks* rather than tokens (`:194`).
   - `/context` estimates tokens as chars/4 (`:142`).
   - History grows without bound and is never trimmed to the context window (`:173,193`).
   - `NO_COLOR` is not checked by EigenCore itself. Rich has supported it since v11.2.0 (https://no-color.org/), so the Rich path inherits it, but this is untested. **[inferred]**

### Comparable projects
- **Claude Code** has an opt-in screen-reader mode (`--ax-screen-reader`, env var, or setting). It prints flat labelled lines (`you:`, `claude:`, `tool:`, `Permission Required:`) with no box drawing or redraws, and reads diffs line-by-line as `+`/`-`. Menus become numbered lists, and there is a `prefersReducedMotion` setting. https://docs.anthropic.com/en/docs/claude-code/accessibility **[verified]**
- **Textual** runs on Linux, macOS and Windows, and recommends Windows Terminal. https://textual.textualize.io/getting_started/ **[verified]** Whether it behaves well over SSH or in a legacy conhost is **[unverified]**.
- **Not checked against primary sources [unverified]:**
  - Aider (prompt_toolkit input plus Rich output).
  - Codex CLI's TUI (Rust/ratatui).
  - opencode (Go/Bubble Tea).
  - llama.cpp's `llama-cli` (plain line REPL).

### Alternatives
| Option | Pros | Cons |
|---|---|---|
| Keep Rich, line-mode | Already a dependency; scrollback-friendly; SSH-safe | No real full-screen layout |
| Rich `Live` | Footer/status line | Repaint flicker over SSH **[inferred]** |
| prompt_toolkit input + Rich output | History, multiline, key bindings, Windows support **[inferred]** | Adds a dependency |
| Textual | Full-screen widgets; headless `Pilot` tests | Heavy; alternate screen hurts screen readers **[inferred]** |

**Recommendation [inferred]:** keep line-mode Rich and add prompt_toolkit only for input. Defer Textual until approval cards actually need full-screen layout, and keep plain mode as the accessibility path.

### Next steps and acceptance tests
1. Render model text as `Text(token)` or with `markup=False, emoji=False, highlight=False`.
   - *Test:* stream `["[/]", "[bold]x", ":warning:"]` through the Rich path into a `Console(file=StringIO, force_terminal=True)`. It must not raise, and the literal text must appear.
2. Add a `sanitize()` that strips C0/C1 controls except `\n\t`, plus CSI, OSC and DCS, before any write.
   - *Test:* a reply containing `\x1b]0;x\x07\x1b[2J\x9b` produces output with no `\x1b` or `\x9b`, in both modes.
3. Wrap the stream loop in `try/except KeyboardInterrupt`, keep the partial reply, and print `[interrupted]`.
   - *Test:* an engine that raises `KeyboardInterrupt` after one token leaves `run()` returning normally, and history ends with an assistant message.
4. Choose the streaming path with `hasattr(engine, "stream_messages")` instead of catching `AttributeError`.
   - *Test:* an engine that raises `AttributeError` mid-stream propagates or reports the error, with no second generation.
5. Fix the label (`"no tools"`) and update the doc status.
   - *Test:* the header string from the CLI wiring contains "no tools".
6. Honour `NO_COLOR` and `TERM=dumb` by switching to plain mode.
   - *Test:* with env `NO_COLOR=1`, output contains no SGR codes.

## B. Agent sandboxing

### What exists
- **`policy.py`:** a capability allow-list (`:156-170`) and path confinement for parameter names in `_PATH_PARAM_NAMES` (`:26-39`, `:172-187`). It resolves paths and rejects symlinks or `..` that leave the root (`:127-140`). It keeps a JSONL audit log with redacted arguments (`:109-121`). Its docstring is honest that it is "NOT a sandbox" (`:10-11`). `default_policy()` allows only READ (`:203-211`).
- **`tool.py:68`:** a tool's capability defaults to EXECUTE, which is good because it denies when unknown.
- **`executor.py:76-133`:** the policy check runs *before* validation, but only `if self.policy is not None` (`:102`).

### Bugs and overclaims
1. **The timeout does not bound wall time.** `with ThreadPoolExecutor(...)` returns from inside the block, and `__exit__` calls `shutdown(wait=True)` (`executor.py:138-162`). The Python docs confirm that the context manager waits as if `shutdown(wait=True)` (https://docs.python.org/3/library/concurrent.futures.html). `Future.cancel()` cannot stop a running call.
   - Probe: a 3 s handler with `timeout=0.2` returned `TIMEOUT`, reported `elapsed_seconds=0.2`, and took **3.0 s** of wall time. **[verified]**
   - `tests/test_executor.py:99` passes only because it checks the status; it silently burns 10 s (`_slow` sleeps 10 s, `:17-18`). **[verified]**
2. **Allow-all is the default on every shipped path.**
   - `ToolExecutor(policy=None)` performs no enforcement.
   - `Forge.agent()` builds `ToolExecutor(tools)` (`forge.py:177`), and `ReActAgent` defaults to the same (`react.py:122`).
   - `default_policy` is exported but never wired in anywhere.
   - Probe: an EXECUTE tool returned `SUCCESS` with no policy, and `POLICY_DENIED` with `default_policy`. This contradicts "Deny-by-default" (`tool.py:12`). **[verified]**
3. **Path confinement is keyed on parameter names.** A READ tool whose parameter is `src` was called with `/etc/passwd` under `default_policy(workspace_root=tmp)` and returned **SUCCESS** with the file contents. **[verified]**
4. **The check is separate from the use (TOCTOU).** The resolved path is discarded, and the handler gets the raw string (`policy.py:180` vs `executor.py:139`). A symlink swapped in between the check and the open escapes the workspace. **[inferred]**
5. **Other issues:**
   - Successful output is not redacted and not control-sanitised; `redact_text` covers only errors (`executor.py:168`). That output then flows into the prompt via `react.py:203`. **[verified]**
   - Truncation only happens on success. **[verified]**

### How comparable tools isolate
- **Claude Code** uses OS enforcement: Seatbelt on macOS and bubblewrap (plus an optional seccomp filter) on Linux and WSL2. Network goes through a proxy with a domain allow-list. Native Windows runs **unsandboxed**. File tools run outside the sandbox and are governed only by permission rules. https://docs.anthropic.com/en/docs/claude-code/sandboxing **[verified]**
- **Codex** uses Seatbelt on macOS and bubblewrap on Linux/WSL2. Native Windows has three modes: MXC, `elevated` (dedicated low-privilege users, ACLs and firewall rules) and `unelevated` (restricted token plus ACLs). https://developers.openai.com/codex/sandboxing.md, https://developers.openai.com/codex/windows/windows-sandbox.md **[verified]** The brief's mention of Landlock/seccomp in Codex is **not** on these pages, so it is **[unverified]**.
- **gemini-cli** offers Seatbelt profiles, Docker/Podman, gVisor `runsc`, LXC, and a Windows native sandbox based on `icacls` Low integrity level. https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/sandbox.md **[verified]**
- **Landlock** lets an unprivileged process restrict itself (filesystem from kernel 5.13; TCP from ABI v4). It inherits to children, so the intended use is "check the ABI and enforce what is available" on a best-effort basis. https://docs.kernel.org/userspace-api/landlock.html **[verified]**
- **Windows Job Objects** provide kill-on-close, memory and CPU-rate limits, and process trees that cannot break away. https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects **[verified]**
- **Not opened [unverified]:** nsjail, firejail, AppContainer, Windows Sandbox, wasmtime-py/pyodide. A WASM sandbox is the strongest option for pure-Python *compute* tools, but it cannot run arbitrary file or OS tools without WASI capability plumbing **[inferred]**.

### Recommended minimal design for a small project [inferred]
1. **Deny by default in code.** Make `policy` required, or default it to `default_policy()`, in `ToolExecutor`, `ReActAgent` and `Forge.agent`. Allow-all must be an explicit, opt-in `PolicyEngine.allow_all()`.
2. **Declare path parameters.** Add `ToolParam.kind="path"` instead of matching on names. The executor passes the **resolved `Path`** to the handler, and file tools open it with `O_NOFOLLOW` where available.
3. **Run each tool in a child process.** Use `subprocess`/`multiprocessing` spawn with a JSON-in/JSON-out worker, a scrubbed environment, `cwd=workspace`, and a hard kill on timeout.
   - On POSIX: `start_new_session=True`, `os.killpg`, and `resource.setrlimit` (CPU, address space, file size, number of files).
   - On Windows: a Job Object with `KILL_ON_JOB_CLOSE` and a memory limit (via `pywin32` or `ctypes`).
4. **Add best-effort OS confinement.** On Linux, apply Landlock in the worker when available (read-only root, read-write workspace, no TCP connect unless NETWORK is allowed), with bubblewrap as an optional backend. On macOS, `sandbox-exec`. On Windows, the Job Object only, labelled honestly as "resource-limited, not filesystem-isolated".
5. **Sanitise and redact all tool output** before it reaches the prompt, the audit log or the TUI.

### Acceptance tests
- `ToolExecutor(reg)` with no policy denies an EXECUTE tool; `Forge.agent()` denies it too.
- A handler sleeping 10 s with `timeout=0.5` returns in under 1.5 s wall time, and its process is gone afterwards (check by PID). Run on ubuntu and windows CI.
- A READ tool with `ToolParam("src", kind="path")` and `/etc/passwd` or `C:\Windows\win.ini` is denied, while an in-workspace file is allowed.
- The handler receives a `Path` equal to `resolve_in_workspace(...)`, and a symlink swapped in after the check fails to open.
- A worker allocating 2 GB with a 256 MB limit fails with an error, not a host crash.
- On Linux with Landlock, a worker write to `$HOME/x` gets `EACCES`; if Landlock is unavailable the test is skipped and the capability is reported.
- Tool output `password=hunter2\x1b[2J` comes out as `password=***REDACTED***` with no ESC, in the prompt, audit and UI.

## Gaps
- Codex's Linux Landlock/seccomp use, plus Aider, opencode, llama-cli, nsjail, firejail, AppContainer, wasmtime and pyodide, were not verified from primary pages.
- The probes ran on Linux only, not on Windows.
