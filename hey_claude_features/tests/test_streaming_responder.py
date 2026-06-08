"""Unit tests for StreamingResponder — no API key required."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from hey_claude_features.streaming_responder import (
    StreamChunk,
    StreamingResponder,
    _split_on_boundary,
)


# ---------------------------------------------------------------------------
# _split_on_boundary
# ---------------------------------------------------------------------------


class TestSplitOnBoundary:
    def test_no_boundary_short_text(self):
        chunk, remainder = _split_on_boundary("Hello world", max_chunk=200)
        assert chunk == ""
        assert remainder == "Hello world"

    def test_splits_at_sentence_end(self):
        chunk, remainder = _split_on_boundary("Hello world. How are you?", max_chunk=200)
        assert "Hello world." in chunk
        assert "How are you?" in remainder

    def test_hard_split_at_max_chunk(self):
        long = "a" * 300
        chunk, remainder = _split_on_boundary(long, max_chunk=200)
        assert len(chunk) <= 200
        assert len(remainder) > 0

    def test_exclamation_boundary(self):
        chunk, _ = _split_on_boundary("Amazing! Next sentence.", max_chunk=200)
        assert "Amazing!" in chunk

    def test_question_boundary(self):
        chunk, _ = _split_on_boundary("Is it raining? I think so.", max_chunk=200)
        assert "Is it raining?" in chunk

    def test_empty_string(self):
        chunk, remainder = _split_on_boundary("", max_chunk=200)
        assert chunk == ""
        assert remainder == ""

    def test_multiple_sentences_yields_last_before_max(self):
        text = "First sentence. Second sentence. Third sentence."
        chunk, remainder = _split_on_boundary(text, max_chunk=200)
        # Should flush up to the last boundary
        assert "sentence." in chunk


# ---------------------------------------------------------------------------
# StreamChunk
# ---------------------------------------------------------------------------


class TestStreamChunk:
    def test_str(self):
        c = StreamChunk("hello")
        assert str(c) == "hello"

    def test_repr(self):
        c = StreamChunk("hello", is_final=True)
        assert "final=True" in repr(c)

    def test_is_final_default_false(self):
        assert not StreamChunk("hi").is_final


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_stream_context(tokens: list[str]):
    """Build a mock context manager that yields text deltas."""
    ctx = MagicMock()
    ctx.text_stream = iter(tokens)

    final_msg = MagicMock()
    final_msg.usage.input_tokens = 10
    final_msg.usage.output_tokens = len(tokens)
    ctx.get_final_message.return_value = final_msg

    # Make it work as a context manager
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _make_client(tokens: list[str]) -> MagicMock:
    client = MagicMock()
    client.messages.stream.return_value = _make_stream_context(tokens)
    return client


# ---------------------------------------------------------------------------
# StreamingResponder.stream()
# ---------------------------------------------------------------------------


class TestStream:
    def test_yields_chunks(self):
        tokens = ["Hello", " world", ". ", "How are you", "?"]
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        chunks = list(r.stream([{"role": "user", "content": "hi"}]))
        texts = [c.text for c in chunks if c.text]
        assert len(texts) >= 1
        full = "".join(texts)
        assert "Hello world" in full
        assert "How are you" in full

    def test_last_chunk_is_final(self):
        tokens = ["Hello world."]
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        chunks = list(r.stream([{"role": "user", "content": "hi"}]))
        assert chunks[-1].is_final

    def test_full_text_reconstructable(self):
        tokens = ["The sky ", "is blue. ", "Clouds are white."]
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        chunks = list(r.stream([{"role": "user", "content": "hi"}]))
        full = "".join(c.text for c in chunks)
        assert "sky is blue" in full
        assert "Clouds are white" in full

    def test_empty_stream_yields_final(self):
        tokens = []
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        chunks = list(r.stream([{"role": "user", "content": "hi"}]))
        # No crash; may yield nothing or a final empty chunk
        for c in chunks:
            assert isinstance(c, StreamChunk)

    def test_chunk_size_respected(self):
        tokens = ["a" * 250]  # exceeds default max_chunk_chars=200
        client = _make_client(tokens)
        r = StreamingResponder(client=client, max_chunk_chars=200)
        chunks = list(r.stream([{"role": "user", "content": "hi"}]))
        non_final = [c for c in chunks if not c.is_final]
        for c in non_final:
            assert len(c.text) <= 200

    def test_custom_system_prompt(self):
        tokens = ["ok"]
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        list(r.stream([{"role": "user", "content": "hi"}], system="Be brief."))
        call_kwargs = client.messages.stream.call_args
        assert call_kwargs.kwargs.get("system") == "Be brief."

    def test_stream_raises_on_api_error(self):
        client = MagicMock()
        client.messages.stream.side_effect = RuntimeError("API down")
        r = StreamingResponder(client=client)
        with pytest.raises(RuntimeError):
            list(r.stream([{"role": "user", "content": "hi"}]))


# ---------------------------------------------------------------------------
# stream_with_callback()
# ---------------------------------------------------------------------------


class TestStreamWithCallback:
    def test_callback_called_for_each_chunk(self):
        tokens = ["Hello world. ", "How are you?"]
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        received: list[str] = []
        r.stream_with_callback(
            [{"role": "user", "content": "hi"}],
            on_chunk=received.append,
        )
        assert len(received) >= 1
        assert "Hello world" in "".join(received)

    def test_returns_full_text(self):
        tokens = ["The answer is 42. ", "Always."]
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        full = r.stream_with_callback(
            [{"role": "user", "content": "what"}],
            on_chunk=lambda _: None,
        )
        assert "42" in full
        assert "Always" in full


# ---------------------------------------------------------------------------
# respond() — non-streaming convenience wrapper
# ---------------------------------------------------------------------------


class TestRespond:
    def test_returns_string(self):
        tokens = ["The sky is blue. "]
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        result = r.respond([{"role": "user", "content": "hi"}])
        assert isinstance(result, str)
        assert "sky is blue" in result

    def test_concatenates_all_chunks(self):
        tokens = ["Part one. ", "Part two. ", "Part three."]
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        result = r.respond([{"role": "user", "content": "hi"}])
        assert "Part one" in result
        assert "Part three" in result


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


class TestTokenTracking:
    def test_input_tokens_captured(self):
        tokens = ["reply"]
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        list(r.stream([{"role": "user", "content": "hi"}]))
        assert r.last_input_tokens == 10

    def test_output_tokens_captured(self):
        tokens = ["a", "b", "c"]
        client = _make_client(tokens)
        r = StreamingResponder(client=client)
        list(r.stream([{"role": "user", "content": "hi"}]))
        assert r.last_output_tokens == 3
