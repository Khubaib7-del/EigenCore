# EigenCore desktop app (Windows-first)

The desktop app is a Flet 1.x GUI over the same engine, registry and fit
estimator as the CLI — one codebase, no WebView2 dependency. Dark theme from
the design handbook (graphite ground, single indigo accent, state colours
always paired with a word).

## Install it (Windows)

Every `v*` release attaches two files:

| File | What it is |
| --- | --- |
| `EigenCore-Setup.exe` | **Recommended.** Installer built with Inno Setup from `packaging/windows/EigenCore.iss`. Installs per user into `%LOCALAPPDATA%\Programs\EigenCore` (no admin prompt) and adds a Start menu entry plus an entry in Settings > Apps > Installed apps, with an uninstaller. |
| `EigenCore.exe` | Portable single file. Nothing is installed, so it is not listed in Installed apps and cannot update itself. |

Direct links that always point to the newest release:
`https://github.com/Khubaib7-del/EigenCore/releases/latest/download/EigenCore-Setup.exe` and
`.../latest/download/EigenCore.exe`.

### Updates

`desktop/updater.py` drives the sidebar button:

1. At startup the app checks `api.github.com/repos/Khubaib7-del/EigenCore/releases/latest` once. The button shows **Up to date** or **Update to vX**.
2. On click (installed copy only), it streams `EigenCore-Setup.exe` to a temp folder with a progress percentage, then starts it with `/SILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS`.
3. The installer closes the app, installs over the same folder (same `AppId`, so Windows treats it as an update, not a second app) and reopens EigenCore.

An install counts as "installed" when `unins000.exe` sits next to the running exe. For the portable exe and pip installs, the button opens the release page instead.

`~/.eigencore` (models and settings) is never touched by install, update or uninstall.

### From Python instead

```powershell
pip install "eigencore[desktop,inference]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
eigencore-gui
```

## Screens

- **Shared storage with the CLI:** the app uses the same `~/.eigencore/`
  folder as `eigencore` CLI — models live in `~/.eigencore/models`, settings
  in `~/.eigencore/config.json`. Models downloaded via the CLI appear in the
  app (and vice versa); deleting in the app frees the same disk the CLI uses.

- **Chat** — model dropdown (downloaded models), streaming replies rendered as
  Markdown, stop button, status line with *measured* tok/s per turn. Enter
  sends, Shift+Enter newline.
- **Models** — every registry model with params/quant/size/context, fit chip
  (fits / tight / does-not-fit / unknown), downloaded/active state, Download
  (with live progress) and Delete (confirm dialog; deletes the files from
  `~/.eigencore/models` on this device; the active model is protected).
- **Profile** — detected CPU/RAM/GPU and the recommendation card. Speed there
  is labeled *estimate* until you run `eigencore benchmark`.

## Build the Windows app yourself

The release workflow (`.github/workflows/release.yml`, job `windows-app`) does exactly this on `windows-latest`. It uploads straight to the GitHub Release and uses no Actions artifacts.

```powershell
pip install -e . "flet>=1.0,<2" "flet-cli>=1.0,<2" pyinstaller
pip install "llama-cpp-python>=0.3.0" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 1. one-folder build for the installer -> dist\EigenCore\EigenCore.exe
flet pack src/eigencore/desktop/app.py --onedir -y --name EigenCore `
  --icon src/eigencore/desktop/assets/eigencore.ico `
  --product-version 0.4.5 --file-version 0.4.5.0 `
  --add-data "src/eigencore/desktop/assets;assets" `
  --pyinstaller-build-args=--collect-all=llama_cpp --distpath dist

# 2. installer -> dist\EigenCore-Setup.exe (Inno Setup 6)
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" /DAppVersion=0.4.5 "/DSourceDir=$PWD\dist\EigenCore" packaging\windows\EigenCore.iss
```

For the portable single file, run step 1 without `--onedir` and with `--distpath dist-portable`.

`--collect-all=llama_cpp` bundles the native `llama_cpp/lib` DLL. Without it, the app starts but can't load any model. The `=` form is required because argparse treats a leading `--` value as another flag.

Keep `AppId` in `EigenCore.iss` unchanged forever: it is how Windows recognises a new installer as an update.

## Layout

- `src/eigencore/desktop/backend.py` — UI-agnostic bridge (registry, engine,
  worker threads, event queue). Fully unit-tested headlessly.
- `src/eigencore/desktop/views.py` — Flet views (constructible without a display).
- `src/eigencore/desktop/theme.py` — brand tokens.
- `src/eigencore/desktop/app.py` — entry point (`eigencore-gui`), title bar, sidebar update button.
- `src/eigencore/desktop/updater.py` — release check, installer download, silent in-place update.
- `packaging/windows/EigenCore.iss` — Inno Setup installer script.
- `tests/test_desktop.py` — headless tests (backend, views, updater, close button).

## Not verified here

The Linux CI/sandbox cannot open a window, so visual/interaction behavior on
Windows is verified by the owner manually; all logic is covered by tests.
