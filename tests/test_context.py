"""Tests for the attention-aware context manager."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eigencore.context.manager import ContextManager, Message


def test_no_compression_when_under_budget():
    mgr = ContextManager(max_context_tokens=10000)
    mgr.add("system", "You are a helpful assistant.")
    mgr.add("user", "Hello!")
    mgr.add("assistant", "Hi there!")

    ctx = mgr.get_context()
    assert len(ctx.messages) == 3
    assert ctx.compression_ratio == 1.0
    assert ctx.messages_compressed == 0


def test_sink_tokens_preserved():
    """First messages (attention sinks) must survive compression."""
    mgr = ContextManager(max_context_tokens=200)

    mgr.add("system", "You are a coding assistant. " * 10)
    mgr.add("user", "First question about Python. " * 5)

    # add many middle messages to force compression
    for i in range(20):
        role = "user" if i % 2 == 0 else "assistant"
        mgr.add(role, f"Middle message {i} with some filler content here. " * 3)

    mgr.add("user", "Latest question that should be preserved.")

    ctx = mgr.get_context()

    # first two messages (sink) should be the originals
    assert ctx.messages[0].role == "system"
    assert "coding assistant" in ctx.messages[0].content

    # recent messages should be preserved
    assert ctx.messages[-1].content == "Latest question that should be preserved."

    assert ctx.total_tokens <= ctx.max_tokens


def test_compression_maintains_order():
    mgr = ContextManager(max_context_tokens=300)

    mgr.add("system", "System prompt here.")
    mgr.add("user", "First user message.")
    for i in range(15):
        mgr.add("user" if i % 2 == 0 else "assistant", f"Message {i}. " * 10)
    mgr.add("user", "Final message.")

    ctx = mgr.get_context()

    # check ordering: system first, then chronological
    assert ctx.messages[0].role == "system"
    assert "Final message" in ctx.messages[-1].content


def test_smart_truncation():
    text = "First sentence here. Second sentence about code. Third one about math. Fourth about physics. Fifth conclusion."
    result = ContextManager._smart_truncate(text, 60)
    assert len(result) <= 80  # some slack for [...] marker
    assert "..." in result or "[...]" in result


def test_empty_context():
    mgr = ContextManager(max_context_tokens=2048)
    ctx = mgr.get_context()
    assert len(ctx.messages) == 0
    assert ctx.total_tokens == 0


def test_token_estimation():
    msg = Message(role="user", content="Hello world, this is a test message.")
    assert msg.token_estimate > 0
    assert msg.token_estimate == len(msg.content) // 4


def test_session_persistence(tmp_path):
    db_path = tmp_path / "test.db"

    mgr = ContextManager(max_context_tokens=2048, db_path=db_path)
    mgr.add("system", "Test system prompt.")
    mgr.add("user", "Test user message.")
    mgr.add("assistant", "Test response.")
    mgr.save_session("test-session-1")
    mgr._db.close()

    mgr2 = ContextManager(max_context_tokens=2048, db_path=db_path)
    count = mgr2.load_session("test-session-1")
    assert count == 3
    assert mgr2.messages[0].role == "system"
    assert mgr2.messages[1].content == "Test user message."
    mgr2._db.close()


if __name__ == "__main__":
    import tempfile

    test_no_compression_when_under_budget()
    print("test_no_compression_when_under_budget PASSED")

    test_sink_tokens_preserved()
    print("test_sink_tokens_preserved PASSED")

    test_compression_maintains_order()
    print("test_compression_maintains_order PASSED")

    test_smart_truncation()
    print("test_smart_truncation PASSED")

    test_empty_context()
    print("test_empty_context PASSED")

    test_token_estimation()
    print("test_token_estimation PASSED")

    with tempfile.TemporaryDirectory() as tmp:
        test_session_persistence(Path(tmp))
    print("test_session_persistence PASSED")

    print("\nAll context manager tests passed.")
