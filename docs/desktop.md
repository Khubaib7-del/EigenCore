# EigenCore desktop app (Windows-first)

The desktop app is a Flet 1.x GUI over the same engine, registry and fit
estimator as the CLI — one codebase, no WebView2 dependency. Dark theme from
the design handbook (graphite ground, single indigo accent, state colours
always paired with a word).

## Run it

```powershell
# Windows (PowerShell), with the prebuilt CPU backend wheel:
pip install "eigencore[desktop,inference]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
eigencore-gui            # or: python -m eigencore.desktop
```

## Screens

- **Chat** — model dropdown (downloaded models), streaming replies rendered as
  Markdown, stop button, status line with *measured* tok/s per turn. Enter
  sends, Shift+Enter newline.
- **Models** — every registry model with params/quant/size/context, fit chip
  (fits / tight / does-not-fit / unknown), downloaded/active state, Download
  (with live progress) and Delete (confirm dialog; deletes the files from
  `~/.eigencore/models` on this device; the active model is protected).
- **Profile** — detected CPU/RAM/GPU and the recommendation card. Speed there
  is labeled *estimate* until you run `eigencore benchmark`.

## Pack a Windows .exe

On the Windows machine:

```powershell
pip install "flet>=1.0,<2" pyinstaller
flet pack (python -c "import eigencore.desktop.app as a; print(a.__file__)") `
  --name EigenCore --icon src/eigencore/desktop/assets/eigencore.ico `
  --product-version 0.4.3 --file-version 0.4.3.0 --add-data "src/eigencore/desktop/assets;assets"
```

Output lands in `dist/` as a single-folder app; zip `dist/EigenCore` for
distribution. Inference still needs `llama-cpp-python` — bundle it by adding
the wheel from the CPU index to the build environment.

## Layout

- `src/eigencore/desktop/backend.py` — UI-agnostic bridge (registry, engine,
  worker threads, event queue). Fully unit-tested headlessly.
- `src/eigencore/desktop/views.py` — Flet views (constructible without a display).
- `src/eigencore/desktop/theme.py` — brand tokens.
- `src/eigencore/desktop/app.py` — entry point (`eigencore-gui`).
- `tests/test_desktop.py` — 12 headless tests.

## Not verified here

The Linux CI/sandbox cannot open a window, so visual/interaction behavior on
Windows is verified by the owner manually; all logic is covered by tests.
