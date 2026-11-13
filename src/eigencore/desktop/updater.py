"""In-app updates for the installed Windows app.

The Windows release ships ``EigenCore-Setup.exe`` (an Inno Setup installer).
It installs per user into ``%LOCALAPPDATA%\\Programs\\EigenCore``, adds a
Start menu entry and registers EigenCore under Settings > Apps > Installed
apps, so it behaves like any other installed program.

Updating from inside the app:

1. ``check()`` asks GitHub for the latest release and finds the installer asset.
2. ``download()`` streams the installer to a temp folder, reporting progress.
3. ``install()`` starts it silently. Inno Setup closes the running app,
   installs over the same folder (same AppId) and starts the new version.

Models and settings live in ``~/.eigencore`` and are never touched by an
update or by uninstalling.

Everything here is plain Python (no Flet), so it is unit-tested headlessly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO = "Khubaib7-del/EigenCore"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_URL = f"https://github.com/{REPO}/releases/latest"
INSTALLER_NAME = "EigenCore-Setup.exe"
INSTALLER_URL = f"https://github.com/{REPO}/releases/latest/download/{INSTALLER_NAME}"

# Inno Setup switches for an in-place update: no wizard pages, close the
# running app, no reboot. The installer's [Run] entry starts the new version.
SILENT_ARGS = ["/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"]


@dataclass
class Release:
    version: str  # "0.4.5"
    page_url: str  # release page
    installer_url: Optional[str]  # direct download of EigenCore-Setup.exe
    installer_size: int = 0  # bytes, 0 when unknown


def _parse_version(text: str) -> tuple:
    parts = []
    for p in text.strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_newer(tag: str, current: str) -> bool:
    """True when release ``tag`` (v0.4.5) is newer than ``current`` (0.4.4)."""
    try:
        return _parse_version(tag) > _parse_version(current)
    except ValueError:
        return False


def _http_get(url: str, timeout: float = 8.0) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "eigencore-desktop", "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def parse_release(data: dict) -> Release:
    """Build a Release from the GitHub ``releases/latest`` JSON."""
    installer_url, size = None, 0
    for asset in data.get("assets") or []:
        if asset.get("name") == INSTALLER_NAME:
            installer_url = asset.get("browser_download_url")
            size = int(asset.get("size") or 0)
            break
    return Release(
        version=str(data.get("tag_name", "")).lstrip("vV"),
        page_url=data.get("html_url") or RELEASES_URL,
        installer_url=installer_url,
        installer_size=size,
    )


def check(fetcher: Optional[Callable[[str], str]] = None) -> Optional[Release]:
    """Latest release, or None when offline / rate-limited / malformed."""
    fetcher = fetcher or _http_get
    try:
        return parse_release(json.loads(fetcher(API_LATEST)))
    except Exception:
        return None


def is_installed(exe: Optional[str] = None) -> bool:
    """True when running from the installer's folder (not a portable .exe or pip)."""
    if exe is None:
        if not getattr(sys, "frozen", False):
            return False
        exe = sys.executable
    folder = Path(exe).resolve().parent
    return (folder / "unins000.exe").exists()


def download(
    url: str,
    dest_dir: Optional[Path] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    opener: Callable = urllib.request.urlopen,
    chunk: int = 256 * 1024,
) -> Path:
    """Stream the installer to ``dest_dir`` and return its path.

    ``on_progress`` gets 0..100. Writes to a ``.part`` file and renames at the
    end, so a cancelled or failed download never leaves a broken installer.
    """
    dest_dir = Path(dest_dir or tempfile.mkdtemp(prefix="eigencore-update-"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / INSTALLER_NAME
    part = final.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "eigencore-desktop"})
    with opener(req, timeout=30) as resp, open(part, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if on_progress and total:
                on_progress(min(100.0, done * 100.0 / total))
    if total and done != total:
        part.unlink(missing_ok=True)
        raise OSError(f"download incomplete: {done} of {total} bytes")
    os.replace(part, final)
    if on_progress:
        on_progress(100.0)
    return final


def install_command(installer: Path) -> list:
    return [str(installer), *SILENT_ARGS]


def install(installer: Path, popen: Callable = subprocess.Popen) -> None:
    """Start the installer detached; the caller then closes the app."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    popen(install_command(installer), creationflags=flags, close_fds=True)
