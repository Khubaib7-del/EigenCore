"""Single-instance guard and UI server port choice (eigencore.desktop.instance)."""

from __future__ import annotations

import json
import socket
import sys

import pytest

from eigencore.desktop import instance, launcher
from eigencore.desktop.instance import SingleInstance, pick_port, port_is_free


def _listen(host="127.0.0.1", port=0, reuse=False):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if reuse:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    return s


# ----- port ---------------------------------------------------------------------------


def test_listening_port_is_not_free():
    s = _listen()
    try:
        assert port_is_free(s.getsockname()[1]) is False
    finally:
        s.close()


def test_listening_port_with_reuseaddr_is_not_free():
    # what pywebview's server (wsgiref) does: SO_REUSEADDR on the listener
    s = _listen(reuse=True)
    try:
        assert port_is_free(s.getsockname()[1]) is False
    finally:
        s.close()


def test_closed_port_is_free():
    s = _listen()
    port = s.getsockname()[1]
    s.close()
    assert port_is_free(port) is True


def test_preferred_port_is_eigencores_own():
    # never pywebview's shared default (42001)
    assert instance.PREFERRED_PORT == 47613
    assert 42001 not in instance.FALLBACK_PORTS
    assert pick_port(is_free=lambda p: True) == 47613


def test_pick_port_busy_preferred_falls_back():
    busy = {47613, 47614}
    assert pick_port(is_free=lambda p: p not in busy) == 47615


def test_pick_port_all_busy_uses_random():
    fixed = set(range(47613, 47623))
    port = pick_port(is_free=lambda p: p not in fixed, random_port=lambda: 51234)
    assert port == 51234


def test_pick_port_real_sockets():
    a = _listen(reuse=True)
    b = _listen(reuse=True)
    pa, pb = a.getsockname()[1], b.getsockname()[1]
    try:
        free = _listen()
        pf = free.getsockname()[1]
        free.close()
        assert pick_port(preferred=pa, fallbacks=[pb, pf]) == pf
    finally:
        a.close()
        b.close()


# ----- single instance ----------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="lock file guard (Windows uses a mutex)")
def test_second_instance_is_refused(tmp_path):
    first = SingleInstance(tmp_path)
    second = SingleInstance(tmp_path)
    try:
        assert first.acquire() is True
        assert second.acquire() is False
        first.release()
        assert second.acquire() is True
    finally:
        first.release()
        second.release()


@pytest.mark.skipif(sys.platform != "win32", reason="named mutex is Windows only")
def test_second_instance_is_refused_windows():
    name = "Local\\EigenCore-Desktop-test"
    first = SingleInstance(mutex_name=name)
    second = SingleInstance(mutex_name=name)
    try:
        assert first.acquire() is True
        assert second.acquire() is False
        first.release()
        assert second.acquire() is True
    finally:
        first.release()
        second.release()


def test_launcher_exits_when_already_running(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "log_path", lambda: tmp_path / "desktop.log")
    monkeypatch.setattr(launcher, "lock_dir", lambda: tmp_path)
    monkeypatch.setattr(instance.SingleInstance, "acquire", lambda self: False)
    told = []
    monkeypatch.setattr(instance, "focus_existing_window", lambda: False)
    monkeypatch.setattr(instance, "tell_already_running", lambda: told.append(1))
    ran = []
    which = launcher.main(
        [], run_web=lambda d: ran.append("web"), run_flet=lambda: ran.append("flet")
    )
    assert which == "already-running" and ran == [] and told == [1]


def test_launcher_releases_guard_after_close(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "log_path", lambda: tmp_path / "desktop.log")
    monkeypatch.setattr(launcher, "lock_dir", lambda: tmp_path)
    assert launcher.main([], run_web=lambda d: None, run_flet=lambda: None) == "web"
    assert launcher.main([], run_web=lambda d: None, run_flet=lambda: None) == "web"


def test_self_test_skips_the_guard(monkeypatch, tmp_path):
    # CI runs --self-test on the packaged exe; it must never be refused
    monkeypatch.setattr(instance.SingleInstance, "acquire", lambda self: False)
    out = tmp_path / "report.json"
    which = launcher.main(["--self-test", str(out)])
    assert which.startswith("self-test")
    assert json.loads(out.read_text())["version"]


def test_exit_soon_forces_exit():
    codes = []
    t = launcher.exit_soon(0.01, hard_exit=codes.append)
    t.join(2)
    assert codes == [0]
