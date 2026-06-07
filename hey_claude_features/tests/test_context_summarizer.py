"""Unit tests for SummarizingHistory — no API key required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hey_claude_features.context_summarizer import SummarizingHistory


def _make_client(summary_text: str = "A summary.") -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.content = [MagicMock(text=summary_text)]
    client.messages.create.return_value = resp
    return client


def _turn(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def _fill(history: SummarizingHistory, n: int) -> None:
    for i in range(n):
        history.append(_turn("user", f"msg {i}"))
        history.append(_turn("assistant", f"reply {i}"))


class TestAppend:
    def test_appends_messages(self):
        h = SummarizingHistory(client=_make_client(), max_before_summary=100)
        h.append(_turn("user", "hello"))
        assert len(h) == 1

    def test_no_compression_below_threshold(self):
        client = _make_client()
        h = SummarizingHistory(client=client, keep_recent=6, max_before_summary=20)
        _fill(h, 9)  # 18 turns — under threshold
        client.messages.create.assert_not_called()
        assert len(h) == 18

    def test_compression_triggered_at_threshold(self):
        client = _make_client("Summary text.")
        h = SummarizingHistory(client=client, keep_recent=4, max_before_summary=10)
        _fill(h, 5)  # 10 turns — hits threshold
        client.messages.create.assert_called_once()
        # After compression, only keep_recent turns remain in buffer
        assert len(h) == h.keep_recent


class TestMessages:
    def test_messages_without_summary(self):
        h = SummarizingHistory(client=_make_client(), max_before_summary=100)
        h.append(_turn("user", "hi"))
        msgs = h.messages
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hi"

    def test_messages_with_summary_prepends_summary_block(self):
        client = _make_client("The user asked about the weather.")
        h = SummarizingHistory(client=client, keep_recent=2, max_before_summary=4)
        _fill(h, 2)  # triggers compression
        msgs = h.messages
        # First two entries are the synthetic summary exchange
        assert msgs[0]["role"] == "user"
        assert "[Conversation summary]" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"
        assert "weather" in msgs[1]["content"]

    def test_returns_copy_not_internal_list(self):
        h = SummarizingHistory(client=_make_client(), max_before_summary=100)
        h.append(_turn("user", "hello"))
        msgs = h.messages
        msgs.append(_turn("user", "extra"))
        assert len(h) == 1  # internal state untouched


class TestClear:
    def test_clear_resets_state(self):
        client = _make_client("summary")
        h = SummarizingHistory(client=client, keep_recent=2, max_before_summary=4)
        _fill(h, 2)
        h.clear()
        assert len(h) == 0
        assert h._summary is None
        assert h.messages == []


class TestSummarizationFailure:
    def test_graceful_on_api_error(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        h = SummarizingHistory(client=client, keep_recent=2, max_before_summary=4)
        _fill(h, 2)  # should not raise
        # History is kept raw when summarization fails
        assert len(h) == 4


class TestBuildTranscript:
    def test_plain_content(self):
        turns = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        transcript = SummarizingHistory._build_transcript(turns)
        assert "USER: hello" in transcript
        assert "ASSISTANT: hi there" in transcript

    def test_list_content_blocks(self):
        turns = [{"role": "user", "content": [{"text": "block text"}]}]
        transcript = SummarizingHistory._build_transcript(turns)
        assert "block text" in transcript
