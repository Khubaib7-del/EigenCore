# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the EigenCore Windows app.

Build the web UI first (``npm ci && npm run build`` in ui/), then, from the
repo root:

    pyinstaller packaging/windows/EigenCore.spec --noconfirm --distpath dist
        -> dist/EigenCore/EigenCore.exe + _internal/   (one folder, for the installer)
    EIGENCORE_ONEFILE=1 pyinstaller packaging/windows/EigenCore.spec --noconfirm --distpath dist-portable
        -> dist-portable/EigenCore.exe                  (single portable exe)

Bundles: ui/dist (as "ui", found at runtime via sys._MEIPASS by
eigencore.desktop.web.ui_dist_dir), the app icons, pywebview with its WebView2
DLLs and pythonnet, llama_cpp (inference), and Flet with its desktop client
so ``--flet`` and the automatic fallback work offline.
"""

import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parents[1]  # noqa: F821 (SPECPATH is set by PyInstaller)
SRC = ROOT / "src"
ONEFILE = os.environ.get("EIGENCORE_ONEFILE") == "1"
UI_DIST = ROOT / "ui" / "dist"
ASSETS = SRC / "eigencore" / "desktop" / "assets"
ICON = ASSETS / "eigencore.ico"

VERSION = os.environ.get("EIGENCORE_VERSION") or re.search(
    r'__version__\s*=\s*"([^"]+)"', (SRC / "eigencore" / "__init__.py").read_text()
).group(1)

if not (UI_DIST / "index.html").is_file():
    raise SystemExit(f"{UI_DIST} is missing: run `npm ci && npm run build` in ui/ first")

datas = [
    (str(UI_DIST), "ui"),
    (str(ASSETS), "assets"),  # Flet app (and legacy layout)
    (str(ASSETS), "eigencore/desktop/assets"),  # Path(__file__).parent / "assets"
]
binaries = []
hiddenimports = collect_submodules("eigencore") + ["clr"]

for pkg in ("webview", "llama_cpp", "pythonnet", "clr_loader", "flet", "flet_desktop"):
    try:
        d, b, h = collect_all(pkg)
    except Exception as e:  # not installed on this build machine
        print(f"[EigenCore.spec] skipping {pkg}: {e}")
        continue
    datas += d
    binaries += b
    hiddenimports += h


def bundle_flet_client() -> None:
    """Ship the Flet desktop client the way `flet pack` does.

    flet_cli's PyInstaller hook (hook-flet_desktop) bundles
    ``hook_config.temp_bin_dir`` as flet_desktop/app; at runtime
    flet_desktop.ensure_client_cached() unpacks the archive found there.
    Without it the Flet fallback would download its client on first use.
    """
    try:
        import flet_cli.__pyinstaller.config as hook_config
        from flet_cli.__pyinstaller.utils import copy_flet_bin
        from flet_cli.commands.pack import Command
        from flet_desktop import get_artifact_filename

        tmp = copy_flet_bin()
        if not tmp:
            print("[EigenCore.spec] no Flet client found; --flet will download it on first use")
            return
        for daemon in ("fletd.exe", "fletd"):
            p = os.path.join(tmp, daemon)
            if os.path.exists(p):
                os.remove(p)
        # same archive + fingerprint `flet pack` writes (methods use no instance state)
        Command.compress_flet_client_dir(Command.__new__(Command), tmp, get_artifact_filename())
        hook_config.temp_bin_dir = tmp
        print(f"[EigenCore.spec] bundling Flet client from {tmp}")
    except Exception as e:
        print(f"[EigenCore.spec] could not bundle the Flet client ({e}); --flet will download it")


bundle_flet_client()

version_info = None
if sys.platform == "win32":
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo,
        StringFileInfo,
        StringStruct,
        StringTable,
        VarFileInfo,
        VarStruct,
        VSVersionInfo,
    )

    nums = tuple(int(x) for x in re.findall(r"\d+", VERSION)[:3])
    nums = (nums + (0, 0, 0, 0))[:4]
    desc = "EigenCore (portable)" if ONEFILE else "EigenCore"
    version_info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums),
        kids=[
            StringFileInfo(
                [
                    StringTable(
                        "040904B0",
                        [
                            StringStruct("CompanyName", "EigenCore"),
                            StringStruct("FileDescription", desc),
                            StringStruct("FileVersion", f"{VERSION}.0"),
                            StringStruct("InternalName", "EigenCore"),
                            StringStruct("LegalCopyright", "MIT License"),
                            StringStruct("OriginalFilename", "EigenCore.exe"),
                            StringStruct("ProductName", "EigenCore"),
                            StringStruct("ProductVersion", VERSION),
                        ],
                    )
                ]
            ),
            VarFileInfo([VarStruct("Translation", [1033, 1200])]),
        ],
    )

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "windows" / "eigencore_app.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "gi"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

common = dict(
    name="EigenCore",
    console=False,
    icon=str(ICON),
    version=version_info,
    upx=False,
)

if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], **common)  # noqa: F821
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **common)  # noqa: F821
    coll = COLLECT(exe, a.binaries, a.datas, name="EigenCore", upx=False)  # noqa: F821
