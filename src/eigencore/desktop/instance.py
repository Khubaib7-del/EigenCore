"""One running desktop app at a time, and a local port only it answers on.

Why this exists (v0.4.8): after updating to 0.4.7 the app showed the new
version (from Python) on the old UI. pywebview serves the built UI from a local
HTTP server on 127.0.0.1:42001 by default, a port every pywebview app without
private mode shares, and that server sets ``SO_REUSEADDR``, which on Windows
lets a second process bind a port another process is already listening on. So
whatever else answered on 42001 (another app, or an EigenCore that had not
exited) could serve the page. web.py also busts WebView2's caches.

- :class:`SingleInstance` stops a second copy from starting: a named mutex on
  Windows, an exclusive lock file elsewhere.
- :func:`pick_port` chooses the UI server port up front: EigenCore's own
  47613 only when no one else is on it (so the page origin stays the same
  from run to run), else 47614-47622, else any free port.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

log = logging.getLogger("eigencore.desktop")

MUTEX_NAME = "Local\\EigenCore-Desktop"
LOCK_NAME = "desktop.lock"
# EigenCore's own port. Not pywebview's default 42001, which every pywebview
# app without private mode shares.
PREFERRED_PORT = 47613
FALLBACK_PORTS = range(47614, 47623)
HOST = "127.0.0.1"
ERROR_ALREADY_EXISTS = 183


# ----- port ---------------------------------------------------------------------


def _someone_listening(host: str, port: int, timeout: float = 0.2) -> bool:
    """True if a server accepts connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def port_is_free(port: int, host: str = HOST) -> bool:
    """True if this process could own ``host:port`` alone.

    On Windows the test bind uses ``SO_EXCLUSIVEADDRUSE`` (and never
    ``SO_REUSEADDR``), so it fails when any other socket holds the port, even
    one bound with ``SO_REUSEADDR``. Elsewhere ``SO_REUSEADDR`` is set like the
    real server does (it only skips TIME_WAIT; a listening port still fails).
    A port where something accepts connections is never free.
    """
    if _someone_listening(host, port):
        return False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if excl is not None:
                s.setsockopt(socket.SOL_SOCKET, excl, 1)
        else:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        return True
    except OSError:
        return False
    finally:
        s.close()


def _random_free_port(host: str = HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def pick_port(
    preferred: int = PREFERRED_PORT,
    fallbacks: Iterable[int] = FALLBACK_PORTS,
    is_free: Callable[[int], bool] = port_is_free,
    random_port: Callable[[], int] = _random_free_port,
) -> int:
    """Port for the UI server: ``preferred``, else a fallback, else a random one."""
    if is_free(preferred):
        return preferred
    for p in fallbacks:
        if is_free(p):
            log.warning("port %d is in use; serving the UI on %d instead", preferred, p)
            return p
    for _ in range(20):
        p = random_port()
        if is_free(p):
            log.warning("port %d and its fallbacks are in use; serving the UI on %d", preferred, p)
            return p
    return random_port()


# ----- single instance ------------------------------------------------------------


class SingleInstance:
    """Hold this while the app runs; ``acquire()`` is False if another copy has it."""

    def __init__(self, lock_dir: Optional[Path] = None, mutex_name: str = MUTEX_NAME) -> None:
        self._lock_dir = lock_dir or (Path.home() / ".eigencore")
        self._mutex_name = mutex_name
        self._handle = None  # Windows mutex handle
        self._fd: Optional[int] = None  # lock file elsewhere
        self.acquired = False

    def acquire(self) -> bool:
        if sys.platform == "win32":
            self.acquired = self._acquire_mutex()
        else:
            self.acquired = self._acquire_lockfile()
        return self.acquired

    def _acquire_mutex(self) -> bool:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CreateMutexW.restype = wintypes.HANDLE
        handle = k32.CreateMutexW(None, False, self._mutex_name)
        err = ctypes.get_last_error()
        if not handle:
            log.warning("could not create the single-instance mutex (error %d)", err)
            return True  # never block startup on a guard failure
        if err == ERROR_ALREADY_EXISTS:
            k32.CloseHandle(wintypes.HANDLE(handle))
            return False
        self._handle = handle  # released by Windows when the process ends
        return True

    def _acquire_lockfile(self) -> bool:
        try:
            import fcntl
        except ImportError:
            return True
        try:
            self._lock_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self._lock_dir / LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as e:
            log.warning("could not open the single-instance lock: %s", e)
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            try:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception:
                pass
            os.close(self._fd)
            self._fd = None
        if self._handle is not None:
            import ctypes
            from ctypes import wintypes

            ctypes.WinDLL("kernel32").CloseHandle(wintypes.HANDLE(self._handle))
            self._handle = None
        self.acquired = False


def focus_existing_window() -> bool:
    """Windows: bring an open EigenCore window to the front. True if one was found."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        found: list = []
        proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def visit(hwnd, _lparam):
            n = user32.GetWindowTextLengthW(hwnd)
            if n and user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                # the window title is "EigenCore <version>" (web.run)
                t = buf.value
                if t.startswith("EigenCore ") and t[10:11].isdigit():
                    found.append(hwnd)
                    return False
            return True

        user32.EnumWindows(proto(visit), 0)
        if not found:
            return False
        user32.ShowWindow(found[0], 9)  # SW_RESTORE
        user32.SetForegroundWindow(found[0])
        return True
    except Exception:
        log.debug("could not focus the running window", exc_info=True)
        return False


ALREADY_RUNNING = (
    "EigenCore is already running.\n\n"
    "If you cannot see its window, end EigenCore in Task Manager and open it again."
)


def tell_already_running() -> None:
    """Explain why nothing opened (message box on Windows, log elsewhere)."""
    log.warning("another EigenCore desktop app is already running; exiting")
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.WinDLL("user32").MessageBoxW(None, ALREADY_RUNNING, "EigenCore", 0x40)
        except Exception:
            pass
    elif sys.stderr is not None:
        try:
            print(ALREADY_RUNNING.replace("\n\n", " "), file=sys.stderr)
        except Exception:
            pass
