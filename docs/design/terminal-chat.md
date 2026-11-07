# EigenCore terminal chat design

**Status:** proposed UX specification, not implemented. The repository currently exposes a Click CLI (`eigencore chat`) and a Python API; this document describes a future adaptive terminal interface over those capabilities.

## Design goals

1. Start quickly in PowerShell, Windows Terminal, bash, zsh, and common SSH terminals.
2. Help users choose a model that fits their actual memory and speed needs without promising a class-wide tok/s value.
3. Keep chat readable in narrow, low-color, and screen-reader-friendly terminals.
4. Make model loading, context pressure, tool access, and approvals legible; a visual confirmation is never the enforcement boundary.
5. Keep UI rendering separate from inference, policy, and tool execution so dropped/repainted frames cannot drop or duplicate events.

## Entry and first run

`eigencore` is the current executable. `eigencore chat` is the existing interactive command; typing bare `eigencore` currently invokes CLI help, not a welcome wizard. A future version may make the bare command an adaptive home screen only after a backward-compatible CLI decision.

Proposed startup:

1. Detect terminal capabilities, width, color, Unicode, and interactive input; offer `--plain`/`--no-ui` for scripts and redirected streams.
2. Show one static EigenCore mark and concise version/runtime status. Avoid mandatory animation; honor reduced-motion settings and never delay the prompt to play a splash.
3. On first interactive run, explain that model downloads are user-initiated and local inference can use substantial memory/power. Do not send prompts or diagnostics to a service by default.
4. Run a transparent hardware probe and present available RAM/VRAM, supported backend, and unknowns. Do not describe a broad “Core i7” profile as a performance measurement.
5. Offer model choices grouped by fit confidence: **fits with margin**, **tight/estimate**, and **not recommended**. Explain quant, expected memory components, context, source/size, and that speed is an estimate until measured locally. Let the user change context and GPU-offload settings.
6. Confirm download and license/source details before fetching a model. Show progress, cancel safely, and verify completion/integrity according to the model source’s available metadata.
7. Start a blank chat with a short command hint, not a forced multi-page questionnaire. Persist preferences only with clear local-file disclosure.

## Chat screen

```text
┌ EigenCore  ·  chat: default                           local · ready ┐
│ model: <name> / <quant>   ctx: 4.2k / 16k   CPU/GPU: <backend>      │
├────────────────────────────────────────────────────────────────────┤
│ You                                                                │
│   Explain this function and suggest a small refactor.               │
│                                                                    │
│ EigenCore                                                          │
│   ... response ...                                                 │
│   38 tok/s decode · 0.8s prefill · generated locally                │
├────────────────────────────────────────────────────────────────────┤
│ > Type a message · @file · /help · /model · /context · /clear      │
└────────────────────────────────────────────────────────────────────┘
```

At narrow widths, remove decoration and wrap content; at very short heights, fall back to a scrollback-friendly prompt/response layout. Keep streaming output stable, separate prefill from decode status, and allow Ctrl-C to stop generation without losing the conversation. Do not animate token-by-token status beyond the actual stream.

### Status and commands

- Header: active model + quant, local/backend indicator, context use, and active permission mode.
- Footer: concise input hint and busy/cancel state. Show throughput only when measured for the current session, label the metric, and avoid false precision.
- Proposed slash commands: `/help`, `/model`, `/context`, `/sessions`, `/clear`, `/tools`, `/permissions`, `/exit`. Unknown commands should produce suggestions, not disappear into a prompt.
- Proposed input helpers: `@path` can only read files after the user chooses/approves the path; completion must not disclose outside the workspace. All features need a plain text alternative.

## Agent and tool interaction

Tools are not currently exposed in the CLI chat flow. Before enabling them:

- Default to no tools; expose a named capability list and explain each tool’s scope.
- Separate read-only, write, network, and process execution capabilities. Restrict file paths to an explicitly trusted workspace, resolve symlinks, and defend against traversal/races.
- Present a pre-execution card with the exact operation, arguments, target paths, and expected effects. For edits, show a diff. Allow one-shot approval, session-scoped approval, or deny; no vague “approve all” default.
- Enforce policy in a service below the TUI, not in prompt text or rendering. Isolate handlers in OS-backed processes with least privilege and resource/time limits; a timeout alone does not stop side effects.
- Treat model output, retrieved files, tool output, and terminal escape sequences as untrusted. Escape terminal control sequences; never let tool output alter the UI or policy state.
- Record tool name, decision, duration, exit status, and bounded/redacted output in a local audit trail with a clear retention policy.

Example approval card:

```text
Tool request · write file
Path: ./src/example.py
Action: replace 3 lines (diff below)
Reason: requested refactor
[Approve once] [Deny] [Show full diff]
```

## Visual system and accessibility

- Use the repository logo’s graphite/navy, cool teal/blue accents, and warm amber only for caution/approval. Color is supplementary; pair every state with text/icon/shape.
- Respect terminal colors and contrast; provide monochrome mode. Keep Unicode borders optional and test PowerShell/legacy code pages with ASCII rendering.
- Keyboard-first navigation, visible focus, predictable Tab/Enter/Escape behavior, no mouse-only actions, and no rapid motion. Announce important state changes in readable text.
- Do not clear the user’s scrollback on startup or exit. Never echo secrets from environment/config in diagnostic displays.

## Implementation slices and acceptance

1. **Layout prototype:** terminal capability probe, static splash/header/footer, plain fallback, keyboard and resize handling.
2. **Chat integration:** adapt the existing stream API; test interruption, errors, empty output, Unicode, long lines, narrow terminals, and non-TTY behavior.
3. **Onboarding/model fit:** explain the profile inputs and uncertainty; test inadequate RAM/VRAM, unavailable GPU, failed downloads, cancel, and unknown model metadata.
4. **Tool UX only after policy enforcement exists:** test deny/approval, stale approvals, malicious tool output/escape sequences, path traversal, cancellation, process timeout, and audit redaction.

Release gate: works at 80×24 and 120×40, in a plain 80-column terminal, and over SSH; no network request occurs without an explicit action; no tool runs without policy authorization; model fit warnings are accurate on a versioned fixture set; interactive controls have a plain-mode equivalent. These are proposed criteria, not completed verification.
