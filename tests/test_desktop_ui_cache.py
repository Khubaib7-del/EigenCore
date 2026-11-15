"""v0.4.8: an update always shows the new UI (start URL, WebView2 cache, prefs)."""

from __future__ import annotations

import os
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

import pytest

from eigencore import __version__
from eigencore.desktop import web
from eigencore.desktop.web import (
    VERSION_MARKER,
    WEBVIEW_CACHE_DIRS,
    WebBridge,
    clear_stale_webview_cache,
    ui_start_url,
)

# ----- (b) cache-busting start URL ---------------------------------------------------


def _dist(tmp_path: Path) -> Path:
    d = tmp_path / "ui"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(
        '<html><script type="module" src="./assets/index-abc.js"></script></html>'
    )
    (d / "assets" / "index-abc.js").write_text("console.log('ui 0.4.8')")
    return d


def test_start_url_carries_version(tmp_path):
    d = _dist(tmp_path)
    url = ui_start_url(d, "0.4.8")
    assert url == f"{d / 'index.html'}?v=0.4.8"
    assert ui_start_url(d, "1.0 beta+x").endswith("?v=1.0%20beta%2Bx")


def test_start_url_resolves_like_pywebview(tmp_path):
    # pywebview 5/6: common path of the local URL; not a folder -> its dirname;
    # the page is that folder's relative path joined to the server address.
    d = _dist(tmp_path)
    url = ui_start_url(d, __version__)
    common = os.path.commonpath([url.split("#")[0]])
    if not os.path.isdir(common):
        common = os.path.dirname(common)
    assert os.path.normpath(common) == os.path.normpath(str(d))
    rel = os.path.relpath(url, common)
    page = urljoin("http://127.0.0.1:47613/", rel.replace(os.sep, "/"))
    assert page == f"http://127.0.0.1:47613/index.html?v={__version__}"
    # relative assets resolve next to index.html, without the query
    assert urljoin(page, "./assets/index-abc.js") == "http://127.0.0.1:47613/assets/index-abc.js"


def test_pywebview_server_serves_page_and_assets(tmp_path):
    http = pytest.importorskip("webview.http")
    from eigencore.desktop.instance import pick_port

    d = _dist(tmp_path)
    url = ui_start_url(d, "0.4.8")
    port = pick_port()
    addr, common, _server = http.BottleServer.start_server([url], port)
    page = urljoin(addr, os.path.relpath(url, common).replace(os.sep, "/"))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    html = None
    for _ in range(50):
        try:
            html = opener.open(page, timeout=2).read().decode()
            break
        except OSError:
            time.sleep(0.05)
    assert html is not None and "index-abc.js" in html
    src = re.search(r'src="\./([^"]+)"', html).group(1)
    js = opener.open(urljoin(page, "./" + src), timeout=2).read().decode()
    assert "ui 0.4.8" in js


# ----- (c) WebView2 cache cleared when the version changes -----------------------------


def _profile(store: Path) -> None:
    for parts in WEBVIEW_CACHE_DIRS:
        d = store.joinpath(*parts)
        d.mkdir(parents=True)
        (d / "data").write_text("old")
    keep = store / "EBWebView" / "Default"
    for name in ("Local Storage", "IndexedDB"):
        (keep / name).mkdir()
        (keep / name / "data").write_text("user data")


def test_cache_cleared_on_new_version_keeps_storage(tmp_path):
    store = tmp_path / "webview"
    _profile(store)
    (store / VERSION_MARKER).write_text("0.4.7")
    assert clear_stale_webview_cache(store, "0.4.8") is True
    for parts in WEBVIEW_CACHE_DIRS:
        assert not store.joinpath(*parts).exists()
    default = store / "EBWebView" / "Default"
    assert (default / "Local Storage" / "data").read_text() == "user data"
    assert (default / "IndexedDB" / "data").read_text() == "user data"
    assert (store / VERSION_MARKER).read_text() == "0.4.8"


def test_cache_cleared_when_no_marker(tmp_path):
    # first run of 0.4.8 over a profile from 0.4.6 / 0.4.7 (no marker yet)
    store = tmp_path / "webview"
    _profile(store)
    assert clear_stale_webview_cache(store, "0.4.8") is True
    assert not store.joinpath(*WEBVIEW_CACHE_DIRS[0]).exists()


def test_cache_kept_on_same_version(tmp_path):
    store = tmp_path / "webview"
    _profile(store)
    (store / VERSION_MARKER).write_text("0.4.8")
    assert clear_stale_webview_cache(store, "0.4.8") is False
    assert store.joinpath(*WEBVIEW_CACHE_DIRS[0], "data").exists()


def test_marker_not_written_if_a_folder_is_locked(tmp_path, monkeypatch):
    store = tmp_path / "webview"
    _profile(store)
    monkeypatch.setattr(web.shutil, "rmtree", lambda *a, **k: None)  # "in use"
    clear_stale_webview_cache(store, "0.4.8")
    assert not (store / VERSION_MARKER).exists()  # retried next start


def test_fresh_profile(tmp_path):
    store = tmp_path / "webview"
    assert clear_stale_webview_cache(store, "0.4.8") is True
    assert (store / VERSION_MARKER).read_text() == "0.4.8"


# ----- (d) reply options saved by the app, not per page origin --------------------------


def test_ui_prefs_round_trip(tmp_path):
    state = tmp_path / "desktop-ui.json"
    br = WebBridge(backend=object(), state_path=state)
    assert br.load_ui_prefs() == {}
    assert br.save_ui_prefs({"gen": {"temperature": 0.2, "max_tokens": 1024}}) is True
    br.save_chats([{"id": "c1", "messages": []}])
    br2 = WebBridge(backend=object(), state_path=state)
    assert br2.load_ui_prefs() == {"gen": {"temperature": 0.2, "max_tokens": 1024}}
    assert br2.load_chats() == [{"id": "c1", "messages": []}]


def test_ui_prefs_rejects_junk(tmp_path):
    br = WebBridge(backend=object(), state_path=tmp_path / "s.json")
    assert br.save_ui_prefs("nope") is False  # type: ignore[arg-type]
    assert br.save_ui_prefs({"big": "x" * 20000}) is False
    assert br.load_ui_prefs() == {}
