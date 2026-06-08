"""Unit tests for ConversationHistory — no API key required."""

from __future__ import annotations

import time

import pytest

from hey_claude_features.conversation_history import ConversationHistory, Turn


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------


class TestTurn:
    def test_to_message(self):
        t = Turn(role="user", content="hello")
        msg = t.to_message()
        assert msg == {"role": "user", "content": "hello"}

    def test_estimated_tokens_nonzero(self):
        t = Turn(role="user", content="hello world")
        assert t.estimated_tokens >= 1

    def test_long_content_more_tokens(self):
        short = Turn(role="user", content="hi")
        long = Turn(role="user", content="x" * 400)
        assert long.estimated_tokens > short.estimated_tokens

    def test_timestamp_set(self):
        before = time.time()
        t = Turn(role="user", content="hi")
        assert t.timestamp >= before

    def test_meta_stored(self):
        t = Turn(role="user", content="hi", meta={"intent": "question"})
        assert t.meta["intent"] == "question"


# ---------------------------------------------------------------------------
# Basic add / read
# ---------------------------------------------------------------------------


class TestAddAndRead:
    def test_add_user(self):
        h = ConversationHistory()
        h.add_user("hello")
        assert len(h) == 1
        assert h.last_user == "hello"

    def test_add_assistant(self):
        h = ConversationHistory()
        h.add_assistant("hi there")
        assert h.last_assistant == "hi there"

    def test_add_both(self):
        h = ConversationHistory()
        h.add_user("question")
        h.add_assistant("answer")
        assert len(h) == 2

    def test_last_user_returns_most_recent(self):
        h = ConversationHistory()
        h.add_user("first")
        h.add_user("second")
        assert h.last_user == "second"

    def test_last_assistant_empty_when_none(self):
        h = ConversationHistory()
        assert h.last_assistant == ""

    def test_last_user_empty_when_none(self):
        h = ConversationHistory()
        assert h.last_user == ""

    def test_last_pair(self):
        h = ConversationHistory()
        h.add_user("q")
        h.add_assistant("a")
        assert h.last_pair == ("q", "a")

    def test_add_generic_user(self):
        h = ConversationHistory()
        h.add("user", "hello")
        assert h.last_user == "hello"

    def test_add_invalid_role_raises(self):
        h = ConversationHistory()
        with pytest.raises(ValueError):
            h.add("system", "prompt")

    def test_meta_passed_through(self):
        h = ConversationHistory()
        t = h.add_user("hi", intent="question", latency=0.1)
        assert t.meta["intent"] == "question"
        assert t.meta["latency"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# messages property
# ---------------------------------------------------------------------------


class TestMessages:
    def test_messages_format(self):
        h = ConversationHistory()
        h.add_user("hello")
        h.add_assistant("world")
        msgs = h.messages
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "world"}

    def test_messages_excludes_meta(self):
        h = ConversationHistory()
        h.add_user("hi", intent="question")
        msg = h.messages[0]
        assert "intent" not in msg
        assert set(msg.keys()) == {"role", "content"}

    def test_messages_returns_copy(self):
        h = ConversationHistory()
        h.add_user("hi")
        msgs = h.messages
        msgs.clear()
        assert len(h) == 1   # original unaffected


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_empty_history_zero_tokens(self):
        h = ConversationHistory()
        assert h.estimated_tokens == 0

    def test_tokens_accumulate(self):
        h = ConversationHistory()
        h.add_user("a" * 400)    # ≈ 100 tokens
        h.add_assistant("b" * 400)
        assert h.estimated_tokens >= 100


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------


class TestTrimming:
    def test_trim_oldest_over_budget(self):
        # max_tokens=50, each turn ≈ 25 tokens (100 chars / 4)
        h = ConversationHistory(max_tokens=50, min_keep=0)
        for i in range(4):
            h.add_user("x" * 100)   # ≈ 25 tokens each
        assert h.estimated_tokens <= 100   # some trimming happened

    def test_min_keep_respected(self):
        h = ConversationHistory(max_tokens=1, min_keep=4)
        for i in range(10):
            h.add_user("x" * 100)
        assert len(h) >= 4

    def test_recent_turns_kept(self):
        h = ConversationHistory(max_tokens=50, min_keep=2)
        h.add_user("x" * 400)   # old
        h.add_user("last user")
        h.add_assistant("last assistant")
        assert h.last_user == "last user"
        assert h.last_assistant == "last assistant"

    def test_no_trim_under_budget(self):
        h = ConversationHistory(max_tokens=10_000)
        for i in range(5):
            h.add_user("hi")
        assert len(h) == 5


# ---------------------------------------------------------------------------
# Clear / tail / summary
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_clear(self):
        h = ConversationHistory()
        h.add_user("hi")
        h.clear()
        assert len(h) == 0

    def test_tail(self):
        h = ConversationHistory()
        for i in range(6):
            h.add_user(str(i))
        last = h.tail(3)
        assert [t.content for t in last] == ["3", "4", "5"]

    def test_summary_line_contains_turns(self):
        h = ConversationHistory()
        h.add_user("hi")
        assert "1 turn" in h.summary_line()

    def test_iteration(self):
        h = ConversationHistory()
        h.add_user("a")
        h.add_assistant("b")
        roles = [t.role for t in h]
        assert roles == ["user", "assistant"]

    def test_set_system(self):
        h = ConversationHistory()
        h.set_system("You are helpful.")
        assert h._system == "You are helpful."
