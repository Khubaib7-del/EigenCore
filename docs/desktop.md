# EigenCore desktop app (Windows-first)

Since v0.4.6 the desktop app is the web interface in `ui/` (Vite + React +
TypeScript) shown in a native window by [pywebview](https://pywebview.flowrl.com/),
which uses Microsoft Edge **WebView2** on Windows. It runs on the same
engine, model registry and fit estimator as the CLI. The previous Flet app is
still included as a fallback.

## Install it (Windows)

Every `v*` release attaches two files:

| File | What it is |
| --- | --- |
| `EigenCore-Setup.exe` | **Recommended.** Installer built with Inno Setup from `packaging/windows/EigenCore.iss`. Installs per user into `%LOCALAPPDATA%\Programs\EigenCore` (no admin prompt) and adds a Start menu entry plus an entry in Settings > Apps > Installed apps, with an uninstaller. |
| `EigenCore.exe` | Portable single file. Nothing is installed, so it is not listed in Installed apps and cannot update itself. |

Direct links that always point to the newest release:
`https://github.com/Khubaib7-del/EigenCore/releases/latest/download/EigenCore-Setup.exe` and
`.../latest/download/EigenCore.exe`.

The window needs the **WebView2 Runtime**. It ships with Windows 11 and is
on most Windows 10 PCs (it comes with Microsoft Edge). If it is missing,
EigenCore opens the previous (Flet) design instead; installing the runtime
from <https://developer.microsoft.com/microsoft-edge/webview2/> brings the
new UI back.

### Updates

`desktop/updater.py` drives the update button at the bottom of the sidebar
(and the Updates section in Settings):

1. At startup the app checks `api.github.com/repos/Khubaib7-del/EigenCore/releases/latest` once. The button shows **Up to date** or **Update to vX**.
2. On click (installed copy only), it streams `EigenCore-Setup.exe` to a temp folder with a progress percentage, then starts it with `/SILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS`.
3. The installer closes the app, installs over the same folder (same `AppId`, so Windows treats it as an update, not a second app) and reopens EigenCore.

An install counts as "installed" when `unins000.exe` sits next to the running exe. For the portable exe and pip installs, the button opens the release page instead.

`~/.eigencore` (models, settings, saved chats) is never touched by install, update or uninstall.

### From Python instead

```powershell
pip install "eigencore[desktop,inference]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
eigencore-gui
```

The web UI files (`ui/dist`) are built by Node and are not in the PyPI
package, so a pip install opens the Flet app unless you run from a source
checkout with the UI built (see Develop below).

## What you see

- **Chat**: greeting and suggestions when empty, model picker under the
  message box, streaming Markdown with code blocks, a fold for reasoning
  (`<think>…</think>`), Stop, and per-reply stats measured by the backend:
  output speed after the first token, number of streamed pieces (one per
  token in practice) and time to first token. Chats start empty and are
  saved locally to `~/.eigencore/desktop-ui.json`, with the theme.
- **Models**: every registry model with size, quant, context, Hugging Face
  repo and fit chip (Fits / Tight / Does not fit / Unknown; Unknown until the
  file is on disk). Download with live progress, Use, Delete (confirm dialog;
  the active model is protected).
- **Monitor**: engine state (Idle, Loading, Reading prompt, Writing), the
  last reply's speed and first-token time, session tokens, a speed sparkline,
  an approximate context gauge, threads in use and recent turns. Nothing is
  shown before the first reply. Engine RAM is an estimate (model file size).
- **Profile**: CPU, instruction sets, RAM, GPU, and the profiler's memory
  bandwidth and model-size recommendation, both labelled as estimates.
  **Re-scan** runs the profiler again.
- **Settings**: theme (dark, light, system), the models folder, and the
  updater.

Shared storage with the CLI: models live in `~/.eigencore/models`, so a
model downloaded with `eigencore download` shows up in the app and the other
way round.

### Not in the desktop UI yet

- Threads, context length, KV cache type, tool approvals and changing the
  models folder are shown in Settings as **Coming soon** and can't be changed:
  the desktop engine uses the profiler's choices.
- **Run benchmark** in Profile is "coming soon"; run `eigencore benchmark` in
  a terminal.
- The desktop chat has no tool calls, so there are no approval prompts or
  tool chips. Regenerate and chat options are disabled.

## The previous app (`--flet`)

```powershell
eigencore-gui --flet        # pip / source
EigenCore.exe --flet        # installed or portable app
```

The launcher (`desktop/launcher.py`) also opens the Flet app by itself when
the web UI can't start: pywebview missing, no WebView2 runtime, `ui/dist`
missing, or the page doesn't call back within 25 seconds. The reason is
written to `~/.eigencore/logs/desktop.log`.

## How it fits together

```
ui/ (React)  --window.pywebview.api.*-->  desktop/web.py WebBridge  -->  desktop/backend.py DesktopBackend
             <--CustomEvent("eigencore")--  (event pump thread)     <--  events queue
```

- `desktop/web.py` creates the frameless window (`easy_drag=False`; the title
  bar is the `pywebview-drag-region`, its buttons excluded) and exposes
  `WebBridge` as `js_api`. Its methods call the existing backend and updater;
  blocking work runs on threads. Backend event tuples are sent to the page in
  small batches with `window.evaluate_js`, as
  `CustomEvent("eigencore", {detail: [[...], ...]})`.
- `ui/src/api/bridge.ts` implements `EigenCoreApi` over the bridge and turns
  each tuple into a typed event with `fromTuple` (`ui/src/api/types.ts`).
  `initApi()` uses it when the page runs inside the desktop window and the
  mock API anywhere else.
- `ui/dist` is found at runtime by `web.ui_dist_dir()`: `EIGENCORE_UI_DIST`,
  then `sys._MEIPASS/ui` in a packaged build, then the repo's `ui/dist`.

## Develop

Needs Node 20+ and Python 3.10+.

```bash
pip install -e ".[dev,desktop]"
cd ui && npm ci && npm run build && cd ..
eigencore-gui                    # or: python -m eigencore.desktop
eigencore-gui --debug            # with WebView dev tools
```

Hot reload against the real backend: run `npm run dev` in `ui/`, then start
the app with `EIGENCORE_UI_URL=http://localhost:5173 eigencore-gui`. In a
normal browser, `npm run dev` shows the UI on mock data (marked **Preview ·
mock data**); see `ui/README.md` for its preview switches.

On Linux, pywebview needs Qt or GTK (for example
`pip install PyQt6 PyQt6-WebEngine qtpy`); the Windows build uses WebView2.

Tests: `pytest tests/test_desktop.py tests/test_desktop_web.py` runs the
backend, the bridge (with a fake backend window and updater) and the
launcher headlessly; `npm run build` in `ui/` type-checks the UI.

## Build the Windows app yourself

The release workflow (`.github/workflows/release.yml`, job `windows-app`)
does exactly this on `windows-latest`, and the `Desktop build` workflow does
the same on pull requests and runs the packaged exe's self-test. Neither
uploads Actions artifacts.

```powershell
cd ui; npm ci; npm run build; cd ..
pip install -e ".[desktop]" "flet-cli>=1.0,<2" "flet-desktop>=1.0,<2" pyinstaller
pip install "llama-cpp-python>=0.3.0" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 1. one-folder build for the installer -> dist\EigenCore\EigenCore.exe
pyinstaller packaging/windows/EigenCore.spec --noconfirm --distpath dist

# 2. installer -> dist\EigenCore-Setup.exe (Inno Setup 6)
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" /DAppVersion=0.4.6 "/DSourceDir=$PWD\dist\EigenCore" packaging\windows\EigenCore.iss

# 3. portable single file -> dist-portable\EigenCore.exe
$env:EIGENCORE_ONEFILE = "1"; pyinstaller packaging/windows/EigenCore.spec --noconfirm --distpath dist-portable
```

`EigenCore.spec` bundles `ui/dist` (as `ui`), the icons, pywebview with its
WebView2 DLLs and pythonnet, `llama_cpp` with its native DLL, and Flet with
its desktop client (archived the way `flet pack` does it), so `--flet` works
offline. Check a build without opening a window:

```powershell
Start-Process dist\EigenCore\EigenCore.exe -ArgumentList "--self-test","report.json" -Wait; Get-Content report.json
```

Keep `AppId` in `EigenCore.iss` unchanged forever: it is how Windows recognises a new installer as an update.

## Layout

- `ui/` — the desktop UI (React). `src/api/bridge.ts` talks to Python.
- `src/eigencore/desktop/web.py` — pywebview window and `WebBridge` (`js_api`).
- `src/eigencore/desktop/launcher.py` — entry point (`eigencore-gui`, `python -m eigencore.desktop`, the exe): web UI, `--flet`, fallback, `--self-test`.
- `src/eigencore/desktop/backend.py` — UI-agnostic backend (registry, engine, worker threads, event queue).
- `src/eigencore/desktop/updater.py` — release check, installer download, silent in-place update.
- `src/eigencore/desktop/app.py`, `views.py`, `theme.py` — the Flet app (fallback).
- `packaging/windows/EigenCore.spec`, `eigencore_app.py` — PyInstaller build.
- `packaging/windows/EigenCore.iss` — Inno Setup installer script.
- `tests/test_desktop.py`, `tests/test_desktop_web.py` — headless tests.

## Not verified on Windows by the maintainers' sandbox

The development sandbox is Linux. There the web UI was run in pywebview's Qt
backend under Xvfb against the real backend, and a Linux PyInstaller build of
the spec was checked. The WebView2 window itself (frameless controls, drag,
resize grips, maximise), the Windows packaged builds and the installer update
from 0.4.5 have to be checked on Windows; the `Desktop build` workflow covers
the packaging part on each pull request.
