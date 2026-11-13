# Credits and third-party notices — `ui/`

Everything in `ui/` is written for EigenCore and released under the
repository's licence, except the parts listed below.

## Beautiful UI (MIT) — component patterns

- Project: Beautiful UI by Shane Levine / Turbo — https://www.beautifului.dev
- Source studied: https://github.com/slev12397/beautiful-ui (commit `44a274e`)
- Licence: MIT, Copyright (c) 2026 Shane Levine

Our components were built from scratch in plain React + CSS variables (no
Tailwind, no copied files), after studying these Beautiful UI primitives:
Loading State, Thinking, Tool Chips, Approval Card, Prompt Bar, Code Block,
Sidebar Nav (GlideMenu), Search and Recommendation Card.

Parts that follow Beautiful UI's code closely, and so carry its licence:

| Our file | What was ported |
| --- | --- |
| `src/components/primitives.tsx` — `PixelLoader` | 3×3 grid, chevron and orbit delay tables, 650 / 950 ms cycles |
| `src/styles/base.css` — `@keyframes pixel-on`, `@keyframes shimmer-text` | keyframe stops |
| `src/components/primitives.tsx` — `Shimmer`, `useElapsed`, `LoadingState` | shimmer gradient (ink-3 → ink → ink-3) and the elapsed-time format |
| `src/components/primitives.tsx` — `GlideMenu` | one highlight that follows the hovered row (220 ms, `cubic-bezier(0.23,1,0.32,1)`) |

```
MIT License

Copyright (c) 2026 Shane Levine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Provider marks — lobe-icons (MIT)

- Project: lobe-icons by LobeHub — https://github.com/lobehub/lobe-icons
- Package: `@lobehub/icons-static-svg` 1.102.0 (npm), licence MIT,
  Copyright (c) 2023 LobeHub
- Files were copied from the package's `icons/` folder; we removed the fixed
  `1em` width/height and inline style so they size from CSS. They already use
  `fill="currentColor"`, so they recolour with the theme.

| File in `src/icons/providers/` | Source file | Used for |
| --- | --- | --- |
| `meta.svg` | `icons/meta.svg` | Meta / Llama |
| `microsoft.svg` | `icons/microsoft.svg` (title changed from "Azure" to "Microsoft") | Phi-3 |
| `qwen.svg` | `icons/qwen.svg` | Qwen (Alibaba) |
| `mistral.svg` | `icons/mistral.svg` | Mistral |
| `gemma.svg` | `icons/gemma.svg` | Google Gemma |
| `deepseek.svg` | `icons/deepseek.svg` | DeepSeek |
| `huggingface.svg` | `icons/huggingface.svg` | Hugging Face (download source) |
| `ai2.svg` | `icons/ai2.svg` | Ai2 / OLMo |
| `tinyllama.svg` | drawn for EigenCore (not from lobe-icons) | TinyLlama — a neutral "TL" monogram, because TinyLlama is a community model with no official mark and is not a Meta model |

Company and model names and logos are trademarks of their owners. They are
used only to say where a model comes from, and do not mean the owners endorse
EigenCore. The MIT licence covers the SVG files, not the trademarks.

## Fonts (SIL Open Font License 1.1)

- Inter — `@fontsource-variable/inter` (OFL-1.1)
- JetBrains Mono — `@fontsource-variable/jetbrains-mono` (OFL-1.1)

Both are npm dependencies. Vite copies the `.woff2` files into `dist/` at
build time, so the app loads no font from a CDN.

## Studied for look only (no code or assets used)

- affectUI (https://affectui.com, paid): the AI Chat / Start greeting with
  composer, App Shell sidebar, monochrome dark and light themes. We looked at
  the public pages only and did not copy code, CSS or assets.
- Colibri (https://github.com/JustVugg/colibri): only as a list of the
  screens a local-LLM app needs (per-message stats, a Monitor screen). Its
  look was not copied.

## In-house

- UI icons in `src/components/Icon.tsx` were drawn for EigenCore.
