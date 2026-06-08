"""
Streaming responder for Hey-Claude.

Instead of waiting for the full Opus reply before starting TTS, this module
streams the response token-by-token and yields text chunks as soon as they
arrive.  A sentence-boundary detector decides when enough text has
accumulated to hand off to TTS — minimising perceived latency.

How it works
------------
1. Open a streaming Anthropic messages request
2. Buffer incoming text_delta events
3. Yield a chunk whenever a sentence boundary is detected OR the buffer
   exceeds ``max_chunk_chars``
4. Flush the remaining buffer when the stream ends
5. Optionally call an ``on_chunk`` callback so callers can pipe each chunk
   to TTS without a generator

Sentence boundaries
-------------------
A chunk is flushed on:  ``.``, ``!``, ``?``  followed by whitespace or end-of-stream
and when the buffer exceeds ``max_chunk_chars`` (default 200).

Usage in app.py
---------------
    from hey_claude_features.streaming_responder import StreamingResponder

    responder = StreamingResponder(client=anthropic_client)

    # Generator style — yields chunks as they arrive:
    for chunk in responder.stream(messages, model="claude-opus-4-8"):
        tts.speak_async(chunk)

    # Callback style — same but via on_chunk:
    responder.stream_with_callback(
        messages,
        model="claude-opus-4-8",
        on_chunk=lambda text: tts.speak_async(text),
    )

    # Full response (no streaming) — convenience wrapper:
    full_text = responder.respond(messages, model="claude-opus-4-8")
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Sentence-ending punctuation followed by whitespace or end-of-string
_SENTENCE_END_RE = re.compile(r"[.!?][)\"]?\s")


def _split_on_boundary(text: str, max_chunk: int) -> tuple[str, str]:
    """
    Split *text* at the last sentence boundary before *max_chunk* chars.

    Returns (chunk_to_yield, remainder).
    """
    if len(text) < max_chunk:
        # Look for a sentence boundary anywhere in the text
        m = None
        for m in _SENTENCE_END_RE.finditer(text):
            pass  # keep last match
        if m:
            end = m.end()
            return text[:end].strip(), text[end:]
        return "", text  # no boundary yet

    # Buffer is full — find the last boundary before max_chunk
    search_window = text[:max_chunk]
    m = None
    for m in _SENTENCE_END_RE.finditer(search_window):
        pass
    if m:
        end = m.end()
        return text[:end].strip(), text[end:]

    # No boundary in window — hard split at max_chunk
    return text[:max_chunk].strip(), text[max_chunk:]


# ---------------------------------------------------------------------------
# StreamChunk
# ---------------------------------------------------------------------------


class StreamChunk:
    """A text chunk yielded during streaming."""

    def __init__(self, text: str, is_final: bool = False) -> None:
        self.text = text
        self.is_final = is_final

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"StreamChunk({self.text!r}, final={self.is_final})"


# ---------------------------------------------------------------------------
# StreamingResponder
# ---------------------------------------------------------------------------


class StreamingResponder:
    """
    Wraps the Anthropic streaming API to yield sentence-sized text chunks.

    Parameters
    ----------
    client:
        Anthropic client instance.
    max_chunk_chars:
        Hard upper limit on chunk size (chars).  A chunk is also flushed
        at sentence boundaries before this limit.
    system_prompt:
        Default system prompt prepended to every request.
    """

    def __init__(
        self,
        client: Any,
        max_chunk_chars: int = 200,
        system_prompt: str = "You are Hey-Claude, a helpful voice assistant. Keep replies concise.",
    ) -> None:
        self._client = client
        self._max_chunk = max_chunk_chars
        self._system = system_prompt

        # Populated after each stream completes
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_full_text: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[dict[str, Any]],
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Yield StreamChunk objects as the model generates text.

        Chunks are sentence-boundary-aligned where possible.
        The final chunk has ``is_final=True``.
        """
        system_prompt = system if system is not None else self._system
        buffer = ""
        full_text_parts: list[str] = []

        try:
            with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            ) as stream_ctx:
                for text_delta in stream_ctx.text_stream:
                    buffer += text_delta
                    full_text_parts.append(text_delta)

                    chunk_text, buffer = _split_on_boundary(buffer, self._max_chunk)
                    if chunk_text:
                        yield StreamChunk(chunk_text, is_final=False)

                # Flush remaining buffer
                if buffer.strip():
                    yield StreamChunk(buffer.strip(), is_final=True)
                elif full_text_parts:
                    # Emit empty final sentinel so callers know the stream ended
                    yield StreamChunk("", is_final=True)

                # Capture usage from the final message
                try:
                    final_msg = stream_ctx.get_final_message()
                    self.last_input_tokens = final_msg.usage.input_tokens
                    self.last_output_tokens = final_msg.usage.output_tokens
                except Exception:
                    pass  # usage not critical

        except Exception:
            logger.exception("Streaming request failed")
            raise

        self.last_full_text = "".join(full_text_parts)

    def stream_with_callback(
        self,
        messages: list[dict[str, Any]],
        on_chunk: Callable[[str], None],
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> str:
        """
        Stream and call *on_chunk* for each non-empty chunk.

        Returns the full concatenated text when done.
        """
        parts: list[str] = []
        for chunk in self.stream(messages, model=model, max_tokens=max_tokens, system=system):
            if chunk.text:
                on_chunk(chunk.text)
                parts.append(chunk.text)
        return "".join(parts)

    def respond(
        self,
        messages: list[dict[str, Any]],
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> str:
        """
        Non-streaming convenience wrapper — returns the full reply as a string.

        Uses the streaming API internally so token counts are still captured.
        """
        parts: list[str] = []
        for chunk in self.stream(messages, model=model, max_tokens=max_tokens, system=system):
            if chunk.text:
                parts.append(chunk.text)
        return "".join(parts)
