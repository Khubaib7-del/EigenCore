"""Tests for the v2 chat TUI pieces (docs/design/TUI.md §3, §5):

header snapshots at 80/100/120 columns + plain, streaming-safe markdown with
framed code blocks, the agent state line and measured stats, sanitiser
acceptance strings, tools label, SSD-paging warning, approval card, tree
component, scrolling, capability probe.
"""

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from rich.console import Console
from test_tui_app import FakeEngine, _drain, _make_app, _mark_downloaded, _profile, _spec

from eigencore import __version__
from eigencore.agent.policy import default_policy
from eigencore.fit import ModelGeometry, estimate_fit
from eigencore.tui import keys as K
from eigencore.tui import theme as T
from eigencore.tui.app import SsdWarningOverlay, pick_glyphs
from eigencore.tui.chat import ChatUI, ChatUiConfig
from eigencore.tui.components import (
    ApprovalCard,
    ApprovalCardModel,
    SsdWarning,
    ask_ssd_warning,
    plain_approval,
    tools_label,
)
from eigencore.tui.render import (
    TurnStats,
    open_fence,
    render_markdown,
    render_tree,
    stats_from_run,
    tree_from_paths,
)
from eigencore.tui.sanitize import sanitize

GEO = ModelGeometry(num_layers=28, hidden_size=3584, num_attention_heads=28, num_kv_heads=4)
NASTY = [
    "[/]",
    "[bold]x",
    "\x1b]0;pwn\x07",
    "\x1b[2J",
    "\x9b31m",
    "\x1bP1;2|evil\x1b\\",
    "a\rb\x07c",
]


@pytest.fixture(autouse=True)
def _no_gguf_probe(monkeypatch):
    """Test model files are fake bytes: keep the fit estimate on spec params so
    header snapshots do not depend on what other tests did to the GGUF reader."""
    import eigencore.tui.app as app_mod

    monkeypatch.setattr(app_mod, "read_gguf_model_info", lambda path: None)


def _unicode_app(tmp_path, **kw):
    app, registry, engine = _make_app(tmp_path, **kw)
    app.geometry = GEO
    app._fit_cache.clear()
    app.ascii = False
    app.glyphs = pick_glyphs(False)
    return app, registry, engine


def _text(renderable, width=100, height=None) -> str:
    con = Console(
        width=width,
        height=height,
        record=True,
        file=io.StringIO(),
        color_system=None,
        legacy_windows=False,
    )
    con.print(renderable)
    return con.export_text()


def _frame(app, width, height=30) -> str:
    con = Console(
        width=width,
        height=height,
        record=True,
        file=io.StringIO(),
        color_system=None,
        legacy_windows=False,
    )
    app.console = con
    con.print(app._frame())
    return con.export_text()


# ----- header snapshots (TUI.md §5 acceptance) ------------------------------------


def test_header_snapshot_120(tmp_path):
    app, _, _ = _unicode_app(tmp_path)
    assert _text(app._header_render(120), 120) == (
        f"◩ EigenCore {__version__} · chat".ljust(107)
        + "local · ready\n"
        + "model test-model Q4_K_M    ctx ≈0 / 4096 tok    fit ok +5.9 GB est.    tools off\n"
    )


def test_header_snapshot_100(tmp_path):
    app, _, _ = _unicode_app(tmp_path)
    assert _text(app._header_render(100), 100) == (
        f"◩ EigenCore {__version__} · chat".ljust(87)
        + "local · ready\n"
        + "model test-model Q4_K_M    ctx ≈0 / 4096 tok    fit ok +5.9 GB est.    tools off\n"
    )


def test_header_snapshot_80_narrow(tmp_path):
    app, _, _ = _unicode_app(tmp_path)
    assert _text(app._header_render(80), 80) == (
        f"◩ EigenCore {__version__} · chat".ljust(75)
        + "ready\n"
        + "test-model Q4_K_M · ctx ≈0/4096 · fit ok · tools off\n"
    )


def test_header_snapshot_plain():
    ui = ChatUI(
        FakeEngine(),
        _spec(),
        _profile(),
        geometry=GEO,
        cfg=ChatUiConfig(plain=True),
        out=io.StringIO(),
    )
    assert ui.header() == (
        f"EigenCore {__version__} | test-model Q4_K_M | ctx 4096 | "
        "fit: fits_with_margin (+5.9 GB est.) | tools: off"
    )


def test_header_ascii_fallback_has_no_unicode(tmp_path):
    app, _, _ = _make_app(tmp_path)  # StringIO console → not UTF-8 → ASCII glyphs
    assert app.ascii
    out = _frame(app, 120)
    assert "[#] EigenCore" in out and "> message" in out
    assert all(ord(ch) < 128 for ch in out), [ch for ch in out if ord(ch) >= 128][:5]


def test_frame_wide_has_rules_footer_hints_and_ctx_bar(tmp_path):
    app, _, _ = _unicode_app(tmp_path)
    out = _frame(app, 120)
    lines = out.splitlines()
    assert lines[2] == "─" * 120  # rule under header
    assert lines[-2] == "─" * 120  # rule above footer
    assert lines[-1].startswith("› message   /help /model /context /clear /exit")
    assert lines[-1].endswith("ctrl-c stops")
    assert "ctx ━━━━━━━━━━ ≈0% of 4096 tok" in lines[-3]


def test_frame_narrow_collapses(tmp_path):
    app, _, _ = _unicode_app(tmp_path)
    out = _frame(app, 80)
    assert "─" * 80 not in out  # no rules
    assert out.splitlines()[-1].rstrip() == "› message · /help"


def test_ctx_caution_at_85_percent(tmp_path):
    app, _, _ = _unicode_app(tmp_path)
    app.messages = [{"role": "user", "content": "x" * 4 * 3600}]  # ≈3600 of 4096
    out = _frame(app, 120)
    assert "/context to trim" in out and "≈88%" in out


# ----- tools label ----------------------------------------------------------------


def test_tools_label_reads_policy():
    assert tools_label(None) == "off"
    assert tools_label(default_policy(), tools=None) == "off"  # policy but no tools → off
    assert tools_label(default_policy(), tools=["read_file"]) == "read-only (policy)"


def test_cli_chat_does_not_hardcode_tools_label():
    src = (Path(__file__).parent.parent / "src/eigencore/cli/main.py").read_text()
    assert "policy engine attached in P4" not in src
    assert "render_approval_card" not in src


# ----- sanitiser ------------------------------------------------------------------


@pytest.mark.parametrize("bad", NASTY)
def test_sanitize_strips_all_controls(bad):
    out = sanitize("ok " + bad + " end\n\tz")
    assert not any(ord(c) < 32 and c not in "\n\t" for c in out)
    assert not any(0x7F <= ord(c) <= 0x9F for c in out)
    assert out.endswith(" end\n\tz")


def test_full_screen_survives_nasty_model_text(tmp_path):
    engine = FakeEngine(reply=tuple(NASTY) + ("```py\n[/] x\n",))
    app, _, _ = _unicode_app(tmp_path, engine=engine)
    app.submit_line("hi")
    _drain(app)
    out = _frame(app, 120)
    assert "\x1b" not in out and "\x9b" not in out and "\x07" not in out
    assert "[/]" in out and "[bold]x" in out


def test_plain_mode_survives_nasty_model_text():
    out = io.StringIO()
    inputs = iter(["hi", "/exit"])
    ui = ChatUI(
        FakeEngine(reply=tuple(NASTY)),
        _spec(),
        _profile(),
        geometry=GEO,
        cfg=ChatUiConfig(plain=True),
        out=out,
        inp=lambda p: next(inputs),
    )
    ui.run()
    text = out.getvalue()
    assert "\x1b" not in text and "\x9b" not in text and "\x07" not in text
    assert "[/]" in text and "[bold]x" in text


# ----- markdown / code blocks --------------------------------------------------------


def test_open_fence_detection():
    assert open_fence("hi\n```python\nx = 1")
    assert not open_fence("hi\n```python\nx = 1\n```\n")
    assert open_fence("```\na\n```\n~~~\nb")
    assert not open_fence("no code here")


def test_code_block_is_framed_with_language_label():
    out = _text(render_markdown("Use:\n\n```python\nprint('x')\n```\n\n- one\n- `two`\n"), 60)
    assert "╭─  python  ─" in out
    assert "│ print('x')" in out
    assert "```" not in out
    assert "• one" in out and "two" in out and "`two`" not in out


def test_streaming_unclosed_fence_renders_as_open_panel_not_backticks():
    partial = "Here:\n\n```python\ndef f(x):\n    return"
    out = _text(render_markdown(partial, streaming=True), 60)
    assert "```" not in out
    assert "python" in out and "def f(x):" in out and "writing…" in out
    # once closed and finished, no "writing" marker
    done = _text(render_markdown(partial + " x\n```\n", streaming=False), 60)
    assert "writing" not in done and "return x" in done


def test_markdown_ascii_box_for_legacy_consoles():
    out = _text(render_markdown("```sh\nls\n```", ascii=True), 40)
    assert all(ord(c) < 128 for c in out)
    assert "+- sh " in out or "+-- sh" in out or " sh " in out


# ----- agent state line + measured stats ------------------------------------------------


class SlowEngine:
    """Waits (prefill), then streams until released."""

    def __init__(self):
        import threading

        self.release = threading.Event()
        self.started = threading.Event()

    def count_tokens(self, text):
        return len(text.split())

    def stream_messages(self, messages, config):
        time.sleep(0.05)
        yield "Sure, "
        yield "here:\n\n```python\nx = 1\n"
        self.started.set()
        self.release.wait(5)
        yield "```\n"


def test_state_line_orb_verb_elapsed_then_stats(tmp_path):
    engine = SlowEngine()
    app, _, _ = _unicode_app(tmp_path, engine=engine)
    app.submit_line("go")
    # before the first token: prefill verb
    out = _frame(app, 120)
    assert any(v + "…" in out for v in ("Thinking", "Analyzing", "Reading prompt"))
    assert "local · thinking" in out
    assert engine.started.wait(5)
    deadline = time.time() + 5
    while "x = 1" not in app.blocks[-1]["text"] and time.time() < deadline:
        app.tick()
        time.sleep(0.01)
    out = _frame(app, 120)
    assert "Writing code…" in out and " s" in out
    assert any(o in out for o in T.UNICODE_GLYPHS["orb"])
    assert "local · writing" in out
    engine.release.set()
    _drain(app)
    out = _frame(app, 120)
    assert "Writing" not in out
    assert "measured this turn" in out and "tokens" in out and "tok/s" in out
    assert "local · ready" in out


def test_reduced_motion_static_orb(tmp_path, monkeypatch):
    monkeypatch.setenv("EIGENCORE_REDUCED_MOTION", "1")
    engine = SlowEngine()
    app, _, _ = _unicode_app(tmp_path, engine=engine)
    assert app.reduced_motion
    app.submit_line("go")
    out = _frame(app, 120)
    assert "● " in out
    engine.release.set()
    _drain(app)


def test_stats_prefer_engine_tokens_over_chunks():
    class E:
        last_stream_stats = {"completion_tokens": 50, "prefill_s": 0.5, "decode_s": 2.0}

    s = stats_from_run(E(), "abc", chunks=10, t_start=0, t_first=0.4, t_end=3.0, interrupted=False)
    assert s.unit == "tokens" and s.units == 50
    assert s.decode_rate == pytest.approx(49 / 2.0)
    assert s.wide() == "decode 24.5 tok/s · prefill 0.50 s · 50 tokens · measured this turn"
    assert s.plain() == "[decode 24.5 tok/s measured | prefill 0.50 s | 50 tokens]"


def test_stats_without_token_count_are_labelled_chunks():
    s = stats_from_run(
        object(), "abc", chunks=11, t_start=0, t_first=1.0, t_end=2.0, interrupted=False
    )
    assert s.unit == "chunks"
    assert "chunks/s" in s.wide() and "tok/s" not in s.wide()
    assert "not tokens" in s.wide()


def test_stats_interrupted():
    s = TurnStats(units=7, unit="tokens", prefill_s=0.2, decode_s=1, total_s=1.2, interrupted=True)
    assert s.wide().startswith("stopped · 7 tokens kept")


def test_app_error_mid_stream_keeps_partial_no_regen(tmp_path):
    class ErrorEngine:
        calls = 0

        def stream_messages(self, messages, config):
            ErrorEngine.calls += 1
            yield "abc"
            raise RuntimeError("boom")

    app, _, _ = _unicode_app(tmp_path, engine=ErrorEngine())
    app.submit_line("hi")
    _drain(app)
    assert ErrorEngine.calls == 1
    assert app.blocks[-1]["text"] == "abc"
    assert (
        "stream error" in app.blocks[-1]["meta"] and "partial reply kept" in app.blocks[-1]["meta"]
    )
    assert app.status[0] == "err"
    out = _frame(app, 120)
    assert out.count("abc") == 1


def test_ctrl_c_stops_and_keeps_session(tmp_path):
    app, _, _ = _unicode_app(tmp_path, engine=FakeEngine(infinite=True))
    app.submit_line("go")
    app.tick()
    app.handle_key(K.CTRL_C)
    _drain(app)
    assert app.running  # Ctrl-C never exits the app
    assert "stopped" in app.blocks[-1]["meta"]
    app.submit_line("again")  # session continues
    assert app._gen_active
    app.handle_key(K.CTRL_C)
    _drain(app)


# ----- transcript layout ----------------------------------------------------------------


def test_role_labels_and_indented_bodies(tmp_path):
    app, _, _ = _unicode_app(tmp_path)
    app.submit_line("hello world")
    _drain(app)
    lines = _frame(app, 120).splitlines()
    i = lines.index("you".ljust(120))
    assert lines[i + 1].startswith("  hello world")
    j = lines.index("eigencore".ljust(120))
    assert lines[j + 1].startswith("  Hello there!")


def test_viewport_follows_tail_and_pgup_scrolls(tmp_path):
    app, _, _ = _unicode_app(tmp_path)
    for i in range(30):
        app.blocks.append({"kind": "user", "text": f"message number {i}"})
    out = _frame(app, 120, 30)
    assert "message number 29" in out and "message number 0\n" not in out
    app.handle_key(K.PGUP)
    out = _frame(app, 120, 30)
    assert app.scroll > 0 and "PgDn returns" in out
    assert "message number 29" not in out
    app.handle_key(K.PGDN)
    app.handle_key(K.PGDN)
    assert app.scroll == 0


def test_slash_suggestions_in_footer(tmp_path):
    app, _, _ = _unicode_app(tmp_path)
    for ch in "/c":
        app.handle_key(ch)
    last = _frame(app, 120).splitlines()[-1]
    assert "/context" in last and "/clear" in last and "tab completes" in last
    app.handle_key(K.TAB)
    assert app.input.buf in ("/context", "/clear")


# ----- SSD-paging warning (§3.4) ---------------------------------------------------------


def test_ssd_warning_text_and_keys():
    w = SsdWarning("big-model", 21.0, 12.0)
    out = _text(w.render(), 120)
    assert "[!]" in out and "model 21.0 GB > free RAM 12.0 GB" in out
    assert "read the rest from the SSD on every token" in out
    assert "[c] continue" in out and "[q] pick another model" in out and "[x] cancel" in out
    assert SsdWarning.decide("c") == "continue"
    assert SsdWarning.decide(K.ENTER) is None  # no default into paging
    assert SsdWarning.decide(K.ESC) == "cancel"
    lines = []
    assert ask_ssd_warning(w, ask=lambda p: "", out=lines.append) == "cancel"  # default
    assert ask_ssd_warning(w, ask=lambda p: "c", out=lines.append) == "continue"
    assert lines[0].startswith("[!] big-model: model 21.0 GB > free RAM 12.0 GB")


def test_ssd_warning_needed_from_fit():
    est = estimate_fit(30.0, "Q4_K_M", 4096, 4.0, GEO)
    assert SsdWarning.needed(est)
    assert not SsdWarning.needed(estimate_fit(1.0, "Q4_K_M", 4096, 16.0, GEO))


def test_switch_to_model_bigger_than_ram_asks_first(tmp_path):
    app, registry, _ = _unicode_app(tmp_path)
    big = next(m for m in registry.list_models() if m.size_gb > 12)
    _mark_downloaded(registry, big)
    switched = []
    app.on_model_change = lambda s: switched.append(s.name)
    app.switch_model(big)
    assert isinstance(app.overlay, SsdWarningOverlay) and switched == []
    out = _frame(app, 120)
    assert "> free RAM" in out
    app.handle_key(K.ENTER)  # ignored
    assert isinstance(app.overlay, SsdWarningOverlay)
    app.handle_key("x")
    assert app.overlay is None and switched == []
    app.switch_model(big)
    app.handle_key("c")
    assert switched == [big.name] and app.spec.name == big.name


# ----- approval card (§3.5, component only) -------------------------------------------------


def _card_model(**kw):
    return ApprovalCardModel(
        tool="write file",
        path="./src/example.py",
        action="replace 3 lines",
        diff=["- total = sum([x for x in xs if x > 0])", "+ total = sum(x for x in xs if x > 0)"],
        limits="runs in isolated child · 10 s limit · no network",
        **kw,
    )


def test_approval_card_render_matches_design():
    out = _text(ApprovalCard(_card_model()).render(width=80), 90)
    assert "tool request · write file" in out and "needs approval" in out
    assert "./src/example.py" in out and "(inside workspace)" in out
    assert "- total = sum([x" in out and "+ total = sum(x" in out
    assert "runs in isolated child · 10 s limit · no network" in out
    for label in ("[a] approve once", "[s] approve for session", "[d] deny", "[v] diff"):
        assert label in out
    assert "default focus: deny" in out


def test_approval_card_never_invents_isolation():
    model = ApprovalCardModel(tool="write file", path="./a.py", action="create")
    assert "isolated" not in _text(ApprovalCard(model).render(width=80), 90)


def test_approval_card_default_deny_and_enter_does_nothing():
    card = ApprovalCard(_card_model())
    assert card.focus == "d"
    assert card.handle_key(K.ENTER) is None and card.result is None  # unfocused
    card.handle_key(K.RIGHT)  # move focus: d → v
    card.handle_key(K.LEFT)  # back to deny
    assert card.handle_key(K.ENTER) == "deny"
    assert ApprovalCard(_card_model()).handle_key("a") == "once"
    assert ApprovalCard(_card_model()).handle_key("s") == "session"
    assert ApprovalCard(_card_model()).handle_key(K.ESC) == "deny"


def test_approval_card_stale_cannot_be_approved():
    card = ApprovalCard(_card_model(stale=True))
    assert card.handle_key("a") is None and card.handle_key("s") is None
    assert "stale · re-request" in _text(card.render(width=80), 90)
    assert card.handle_key("d") == "deny"


def test_approval_card_sanitises_and_ascii():
    model = _card_model()
    model.path = "./x\x1b]0;pwn\x07.py"
    out = _text(ApprovalCard(model).render(ascii=True, width=70), 80)
    assert "\x1b" not in out and "./x.py" in out
    assert all(ord(c) < 128 for c in out)


def test_plain_approval_y_n_default_deny():
    lines = []
    assert (
        plain_approval(_card_model(), ask=lambda p: lines.append(p) or "", out=lines.append)
        is False
    )
    assert "approve write ./src/example.py? [y/N] " in lines
    assert "- total = sum([x for x in xs if x > 0])" in lines
    assert plain_approval(_card_model(), ask=lambda p: "y", out=lambda s: None) is True
    assert plain_approval(_card_model(stale=True), ask=lambda p: "y", out=lambda s: None) is False


def test_chat_never_shows_approval_card(tmp_path):
    app, _, _ = _unicode_app(tmp_path)
    app.submit_line("please write a file")
    _drain(app)
    out = _frame(app, 120)
    assert "tool request" not in out and "needs approval" not in out
    assert "tools off" in out


# ----- tree component --------------------------------------------------------------


def test_tree_renderer_unicode_and_ascii():
    nodes = tree_from_paths(["src/app.py", "src/tui/keys.py", "README.md"])
    out = render_tree(nodes).plain
    assert out == "├─ src/\n│  ├─ tui/\n│  │  └─ keys.py\n│  └─ app.py\n└─ README.md"
    asc = render_tree(nodes, ascii=True).plain
    assert asc.startswith("|-- src/") and "`-- README.md" in asc
    assert all(ord(c) < 128 for c in asc)


def test_tree_steps():
    out = render_tree([("read ./a.py", ["12 lines"]), "done"]).plain
    assert out == "├─ read ./a.py\n│  └─ 12 lines\n└─ done"


# ----- capability probe ---------------------------------------------------------------


def test_no_color_and_dumb_term_force_plain(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert T.probe_terminal(width=120).plain
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert T.probe_terminal(width=120).plain


def test_ascii_glyphs_on_legacy_windows_console(monkeypatch):
    monkeypatch.setattr(T.os, "name", "nt")
    monkeypatch.delenv("WT_SESSION", raising=False)
    assert T.legacy_windows_console()

    class S:
        encoding = "utf-8"

    assert T.use_ascii(S())
    monkeypatch.setenv("WT_SESSION", "x")
    assert not T.use_ascii(S())


# ----- key reader ------------------------------------------------------------------


def test_incomplete_escape_detection():
    from eigencore.tui.keys import incomplete_escape

    assert incomplete_escape("\x1b")
    assert incomplete_escape("ab\x1b[")
    assert incomplete_escape("\x1b[1")
    assert not incomplete_escape("\x1b[B")
    assert not incomplete_escape("abc")
    assert not incomplete_escape("\x1bx")


def test_key_reader_joins_split_arrow_key(monkeypatch):
    from eigencore.tui.keys import KeyReader

    reader = KeyReader.__new__(KeyReader)
    from collections import deque

    reader._buf = deque()
    chunks = iter(["\x1b", "[", "B", ""])
    reader._read_raw = lambda timeout: next(chunks, "")
    assert reader.poll(0) == K.DOWN
    assert reader.poll(0) is None
