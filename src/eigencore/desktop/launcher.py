"""Start the EigenCore desktop app.

``eigencore-gui``, ``python -m eigencore.desktop`` and the Windows EigenCore.exe
all land here. The new web UI (pywebview + WebView2) opens by default; the
previous Flet app is kept as a fallback:

- ``--flet`` opens the Flet app directly.
- If the web UI cannot start (pywebview missing, no WebView2 runtime, the
  built UI missing, or the page never loads) the Flet app opens instead and
  the reason is written to ``~/.eigencore/logs/desktop.log``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional

from eigencore import __version__

log = logging.getLogger("eigencore.desktop")


def log_path() -> Path:
    return Path.home() / ".eigencore" / "logs" / "desktop.log"


def setup_logging(path: Optional[Path] = None) -> None:
    if getattr(log, "_eigencore_configured", False):
        return
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        path = path or log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(path, maxBytes=512 * 1024, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError:
        pass
    if sys.stderr is not None and getattr(sys.stderr, "name", "") != os.devnull:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    log._eigencore_configured = True  # type: ignore[attr-defined]


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eigencore-gui", description="EigenCore desktop app")
    p.add_argument("--flet", action="store_true", help="open the previous (Flet) desktop app")
    p.add_argument("--debug", action="store_true", help="web UI: enable the WebView dev tools")
    p.add_argument("--version", action="version", version=f"EigenCore {__version__}")
    # packaging check used by CI: writes a JSON report, opens no window
    p.add_argument("--self-test", metavar="REPORT.json", help=argparse.SUPPRESS)
    # PyInstaller / Windows may pass extra arguments (e.g. -psn, multiprocessing)
    args, _unknown = p.parse_known_args(argv)
    return args


def _run_web(debug: bool) -> None:
    from eigencore.desktop import web

    web.run(debug=debug)


def _run_flet() -> None:
    from eigencore.desktop.app import cli as flet_cli

    flet_cli()


def self_test() -> dict:
    """What a packaged build can load, without opening a window.

    ``ok`` is True when the web UI could start here: the built UI is found,
    pywebview imports, and on Windows the WebView2 runtime is present.
    """
    from eigencore.desktop import web

    report: dict = {
        "version": __version__,
        "frozen": bool(getattr(sys, "frozen", False)),
        "platform": sys.platform,
    }
    dist = web.ui_dist_dir()
    report["ui_dist"] = str(dist) if dist else None
    try:
        import webview  # noqa: F401

        report["pywebview"] = True
    except Exception as e:
        report["pywebview"] = f"{type(e).__name__}: {e}"
    if sys.platform == "win32" and report["pywebview"] is True:
        try:
            web._require_webview2()
            report["webview2"] = True
        except Exception as e:
            report["webview2"] = str(e)
    try:
        import llama_cpp

        report["llama_cpp"] = getattr(llama_cpp, "__version__", True)
    except Exception as e:
        report["llama_cpp"] = f"{type(e).__name__}: {e}"
    try:
        import flet_desktop

        bundled = Path(flet_desktop.get_package_bin_dir()) / flet_desktop.get_artifact_filename()
        report["flet_client_bundled"] = bundled.exists()
    except Exception as e:
        report["flet_client_bundled"] = f"{type(e).__name__}: {e}"
    try:
        import eigencore.desktop.app  # noqa: F401  (Flet fallback imports)

        report["flet_app"] = True
    except Exception as e:
        report["flet_app"] = f"{type(e).__name__}: {e}"
    report["ok"] = bool(
        report["ui_dist"]
        and report["pywebview"] is True
        and report.get("webview2", True) is True
        and report["flet_app"] is True
    )
    return report


def main(
    argv: Optional[list] = None,
    run_web: Callable[[bool], None] = _run_web,
    run_flet: Callable[[], None] = _run_flet,
) -> str:
    """Open the desktop app; returns which UI ran ("web" or "flet")."""
    # windowed (no console) builds have no stdout/stderr; some libraries print
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        report = self_test()
        Path(args.self_test).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return "self-test-ok" if report["ok"] else "self-test-failed"
    setup_logging()
    log.info("EigenCore %s starting (%s)", __version__, "flet" if args.flet else "web")
    if not args.flet:
        try:
            run_web(args.debug)
            return "web"
        except Exception as e:  # WebUIUnavailable or anything unexpected
            log.warning("web UI unavailable, opening the Flet app instead: %s", e, exc_info=True)
    run_flet()
    return "flet"


def cli() -> None:
    if main() == "self-test-failed":
        sys.exit(1)


if __name__ == "__main__":
    cli()
