# EigenCore design handbook

> Mock-ups use ‹angle-bracket› placeholders; no value in this file is a measurement.

**Status:** specification, v2 identity. Nothing here is implemented yet. The name stays **EigenCore**.
**Audience:** hoplite (implementer) and anyone touching the README, docs, terminal UI or slides.
**Sources:** the brand kit in `assets/brand/` (identical to the `eigencore-brand-kit.zip` the owner supplied), `docs/design/terminal-chat.md`, and the review in `docs/review/`. The kit contains logo, lockup, banner and favicon files only; it has **no terminal or UI reference screens**, so the TUI sections below are derived from `terminal-chat.md` and the review, not from kit references.

---

## 1. What the current kit gets right, and what it gets wrong

**Keep**

- The silhouette: a rounded "die" with a notch cut out of its top-right, and a bright square core sitting in that notch. It reads as "a chip with something alive in its corner", which is exactly the product: a runtime that knows its own silicon.
- Circuit traces that run towards the core.
- The outlined wordmark (paths, not live text), so lockups render without fonts.
- Graphite and slate neutrals.

**Fix**

| Problem in v1 | Evidence | Consequence |
| --- | --- | --- |
| Fake 3D built from ~30 stacked copies of the same path, plus 20 stacked core squares | `eigencore-logo.svg`: 33 `<path>` elements, 15.6 KB for one mark | Heavy file, blurry edges at small sizes, impossible to recolour by hand |
| Gradients and a radial glow carry the core | `coreL`, `glowL` gradients | Collapses to a smudge at 16 px; prints badly; disappears in one-colour use |
| Light lockup only recolours the wordmark | `eigencore-lockup-light.svg` differs from dark only in the final fill | Dark mark sits heavily on light pages |
| Saturated indigo `#3C4FB8`-`#5F77F3` as the only accent | kit README palette | Fights projector colour, clashes with any warm warning colour |
| Six traces with dark endpoint rings | logo SVG | Too busy below 64 px |
| No meaning in the mark beyond "chip" | | Misses the "eigen" idea |

## 2. The v2 mark

File: [`eigencore-mark-v2.svg`](eigencore-mark-v2.svg) (proposal, text SVG, ~2 KB). The v1 files stay in `assets/brand/` until the owner approves v2.

**Construction**

1. **Die:** the v1 outer path, unchanged geometry, one flat fill `#424B5E` with a 3 px rim `#6A7489`.
2. **Depth:** one flat offset copy of the die, translated (+14, +16), fill `#15181E`. One shadow, not thirty.
3. **Core:** two flat squares (front `#5B6CF0`, offset back `#2E3A9E`) and an inner square `#C9CFFF`. No gradient, no glow.
4. **Traces:** three, not six; stroke `#20252F`, 12 px, round caps; endpoints are solid dots `#8C98FF`.
5. **Eigen-axis (the new idea):** a dotted line in core indigo from the die's lower-left to the centre of the core. An eigenvector is the one direction a transformation leaves pointing the same way; the axis says "everything here is aligned to the core". It is the only diagonal in the mark, so it also gives the mark motion when still.

**Variants to produce (spec)**

| Variant | Use | Rule |
| --- | --- | --- |
| Full colour | README, docs, slides | As constructed |
| Small (≤ 32 px, favicon) | Browser tab, terminal title icons | Drop traces and axis; keep die, rim, core, inner square |
| One-colour ink `#1B1E22` | Print, light single-colour | Die solid, core knocked out (transparent), inner square solid |
| One-colour light `#E8E6E1` | Dark single-colour | Same, inverted |
| Terminal glyph | TUI header | `◩` or ASCII `[#]`, never an image |

**Clear space and size:** clear space = the core's width on every side. Minimum size 16 px (small variant), 48 px for the full mark.

**Do not:** add gradients or glows back; rotate the mark; change the axis angle; recolour the core warm (amber, clay, gold); place the full-colour mark on indigo or mid-grey grounds; animate the core's colour.

**Lockup:** mark + wordmark "EigenCore", wordmark set in Space Grotesk Bold (or the existing outlined paths), cap height = 0.42 x mark height, gap = 0.5 x core width. "Eigen" in ink/light, "Core" in core indigo **only** on dark grounds; on light grounds the whole word is ink.

## 3. Palette

All values hex. Contrast ratios computed against the ground named (WCAG 2.x).

### Neutrals

| Token | Hex | Use |
| --- | --- | --- |
| `ink` | `#1B1E22` | Text on light; 13.8:1 on `stone` |
| `graphite` | `#1F2226` | Dark ground (slides anchor, docs dark mode) |
| `terminal-bg` (reference only) | `#14171B` | Mock-ups; the real TUI uses the user's terminal background |
| `stone` | `#ECE9E2` | Light ground |
| `paper` | `#F6F4EF` | Cards on stone |
| `muted` | `#585E66` | Secondary text on light, 5.4:1 |
| `muted-dark` | `#9AA1AB` | Secondary text on dark, 6.1:1 |
| `light` | `#E8E6E1` | Text on dark, 12.8:1 |
| `die` / `die-rim` / `die-shadow` | `#424B5E` / `#6A7489` / `#15181E` | Mark only |

### Brand

| Token | Hex | Use |
| --- | --- | --- |
| `core` | `#5B6CF0` | Logo core, focus ring, selection on dark. Not body text on light (3.6:1). |
| `core-text` | `#3F4BC4` | Links and indigo text on light, 5.7:1 |
| `core-ice` | `#8C98FF` | Indigo text on dark, 6.1:1 |
| `core-deep` / `core-glint` | `#2E3A9E` / `#C9CFFF` | Mark only |

### Signals (state colours; never decoration)

| Token | Light ground | Dark ground | Meaning |
| --- | --- | --- | --- |
| `ok` (teal) | `#1F6B66` (5.2:1) | `#4FB3A9` (6.4:1) | SHIPPED, measured, `fits_with_margin`, allowed |
| `caution` (amber) | `#8A5D0A` large text only | `#E0A84A` (7.5:1) | `tight`, needs approval, estimate |
| `danger` (clay) | `#B04E2C` (4.4:1: 14 pt bold or larger) | `#E07A5A` (5.4:1) | `does_not_fit`, denied, error, "not an OS sandbox" |
| `planned` | `#585E66` + hatching | `#9AA1AB` + hatching | PLANNED |

**Rules:** one brand accent (`core`) per surface. Signals only on things that have that state. Amber never appears on the same screen as the full-colour mark (it muddies the core). Every colour is paired with a word or symbol, so the UI works in monochrome.

### Status label system (docs, slides, TUI)

| Label | Visual | Text form |
| --- | --- | --- |
| SHIPPED | solid `ok` chip | `SHIPPED` |
| REPORTED-ONLY | `danger` outline chip, no fill | `REPORTED-ONLY` |
| PLANNED | `planned` hatched chip | `PLANNED` |
| Estimate | `caution` text on dark / `muted` italic on light | "ceiling estimate", never a bare number |

## 4. Typography

| Role | Face | Fallback | Notes |
| --- | --- | --- | --- |
| Display (titles, big numerals) | Space Grotesk Bold | Inter SemiBold | Tight tracking (−1%), never all caps for more than 3 words |
| UI and body | Inter (Regular, SemiBold) | system-ui | 1.4 line height in docs |
| Code, numbers in tables, TUI mock-ups | JetBrains Mono | the terminal's font | Tabular figures for every number that changes |

All three are SIL OFL. The TUI never picks a font; it must look right in Cascadia Mono, Consolas, Menlo, DejaVu Sans Mono and the Linux console.

## 5. Terminal UI

### 5.1 Principles

1. The renderer never decides policy. Approvals are recorded by the policy layer; the UI only shows them.
2. Model and tool output is untrusted text: escape Rich markup, strip C0/C1 controls and OSC/CSI sequences before printing.
3. No number without its unit and its kind: "decode ‹n› tok/s (measured, this turn)", never a bare number.
4. Plain mode is a first-class product, not a fallback.
5. Do not clear scrollback on start or exit.

### 5.2 Colour mapping

| Token | Truecolor | 256-colour | 16-colour | `NO_COLOR` / `--plain` |
| --- | --- | --- | --- | --- |
| `core` | `#8C98FF` | 105 | bright blue | none |
| `ok` | `#4FB3A9` | 73 | cyan | `[ok]` |
| `caution` | `#E0A84A` | 179 | yellow | `[!]` |
| `danger` | `#E07A5A` | 173 | red | `[x]` |
| `muted` | `#9AA1AB` | 247 | default dim | none |

Detect: `NO_COLOR` set → no colour. `TERM=dumb` or non-TTY → plain. `COLORTERM=truecolor` → truecolor. Windows: use Rich's detection; test Windows Terminal, PowerShell 5.1 conhost, and legacy code pages with `--ascii`.

### 5.3 Layout

**Wide (≥ 100 columns)**

```text
◩ EigenCore 0.3 · chat                              local · ready
model qwen2.5-7b Q4_K_M   ctx ‹used› / ‹set›   fit ok +‹h› GB   tools off
────────────────────────────────────────────────────────────────────
you
  Explain this function and suggest a small refactor.

eigencore
  ...response streams here...
  decode ‹n› tok/s · prefill ‹t› s · measured this turn
────────────────────────────────────────────────────────────────────
› message   /help /model /context /clear /exit        ctrl-c stops
```

**Narrow (60-99 columns):** header collapses to two short lines, no rules, footer shows only `› message · /help`.

**Plain (`--plain`, non-TTY, `TERM=dumb`):**

```text
EigenCore 0.3 | qwen2.5-7b Q4_K_M | ctx 4096 | fit: fits_with_margin (+‹h› GB) | tools: off
you> Explain this function.
eigencore> ...
[decode ‹n› tok/s measured | prefill ‹t› s]
```

Header fields, in order: model + quant, context used / context set, fit verdict + headroom, tools state. Tools state reads exactly what is attached: `tools off`, `tools read-only (policy)`, never a promise of a future phase.

### 5.4 States

| State | Wide UI | Plain |
| --- | --- | --- |
| Loading model | `loading qwen2.5-7b · ‹read› / ‹size› GB` with a static bar; no spinner if load reports progress | `loading... ‹p›%` lines at most every 2 s |
| Will page from SSD | `caution` line: `model ‹size› GB > free RAM ‹free› GB · will read from SSD · decode will be much slower` and a confirm | same text, `[!]` prefix |
| Prefill | `reading prompt · ‹n› tokens` + elapsed | `[prefill]` |
| Decoding | text streams; footer `ctrl-c stops` | text streams |
| Cancelled | `stopped · ‹n› tokens kept` in `muted` | `[stopped]` |
| Context near full (≥ 85%) | header ctx turns `caution`, one-line hint `/context to trim` | `[!] context ‹p›%` |
| Error | `danger` one-liner + `/help errors`; never a traceback unless `--debug` | `[x] ...` |
| Fit verdicts | `fit ok`, `fit tight`, `fit no` with headroom | full verdict string |
| Estimate shown | `≈ ‹n› tok/s ceiling (estimate)` in `caution` | `(estimate)` suffix |

### 5.5 Onboarding (first interactive run)

1. Static mark line `◩ EigenCore` and version. No splash animation, no delay.
2. One paragraph: runs locally, downloads only when you say so, can use a lot of RAM and power.
3. Hardware probe with progress text, then a table: RAM free/total, measured bandwidth (multithreaded test), ISA, GPU if any, and what is unknown.
4. Model choices grouped **fits with margin / tight / not recommended**, each with weights + KV + reserve, context, licence, size, and "speed unknown until measured". MoE models show both "resident" and "read per token".
5. Confirm download (source, licence, size, SHA-256 if published). Progress with cancel.
6. Optional 30-second benchmark ("measure this machine") that writes `run.json`.
7. Blank chat with one hint line. Preferences saved only after saying where (`~/.config/eigencore/config.toml` or `%APPDATA%\eigencore`).

### 5.6 Approval flow (after process isolation exists)

```text
┌ tool request · write file ───────────────────────── needs approval ┐
│ path    ./src/example.py   (inside workspace)                       │
│ action  replace 3 lines                                             │
│ - old line                                                          │
│ + new line                                                          │
│ runs in isolated child · 10 s limit · no network                    │
├─────────────────────────────────────────────────────────────────────┤
│ [a] approve once   [s] approve for session   [d] deny   [v] diff     │
└─────────────────────────────────────────────────────────────────────┘
```

- Border and title in `caution`; denied result in `danger`; completed in `ok`.
- Default focus is **deny**. Enter on an unfocused card does nothing.
- "Approve for session" is scoped to this tool + this path prefix, and listed in `/permissions`.
- A stale card (model moved on, file changed since the diff) cannot be approved; it shows `stale · re-request`.
- Plain mode: `approve write ./src/example.py? [y/N]` with the diff printed above.
- Every decision is written to the audit log by the policy layer before the handler starts.

### 5.7 Motion

- Terminal: at most one spinner, 8-10 fps, only while nothing else updates. Honour `EIGENCORE_REDUCED_MOTION=1` and plain mode (no spinner). Never animate token counters faster than the stream.
- README banner (SVG): keep the existing animated banner's structure, retimed: traces draw towards the core over 1.2 s, the eigen-axis dots step in 6 beats, the core fades from `core-deep` to `core` once. Total ≤ 3 s, then static. Provide a static PNG and respect `prefers-reduced-motion` with a `@media` rule inside the SVG.
- Slides: fades and morphs only; builds auto-advance after the transition (owner's rule from the v3 deck).

## 6. Docs and README

- Banner, then one sentence, then the status table with SHIPPED / REPORTED-ONLY / PLANNED.
- Every speed figure is labelled "measured on <machine>" or "ceiling estimate".
- Tables over prose for status. Mermaid diagrams use `ink`, `core`, `ok`, `danger` only.

## 7. Hand-off checklist for hoplite

- [ ] Owner approves `eigencore-mark-v2.svg`; then regenerate PNG, favicon and lockups from it (binary assets are generated, not hand-drawn).
- [ ] Introduce a single `eigencore/tui/theme.py` holding the tokens in 5.2; no hex literals elsewhere.
- [ ] Sanitiser for model/tool text with tests (`[/]`, OSC title, `ESC[2J`).
- [ ] Plain mode parity test for every state in 5.4.
- [ ] Header label reflects the attached policy object, read from it, not a string literal.
