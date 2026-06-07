"""
Auto-summarizes old conversation turns to keep token costs low while preserving context.

Drop-in integration with responder.py: replace the raw history list with
SummarizingHistory before passing it to the Anthropic client.

Usage in responder.py:
    from hey_claude_features.context_summarizer import SummarizingHistory

    history = SummarizingHistory(
        client=anthropic_client,
        model="claude-haiku-4-5",
        keep_recent=6,       # keep last 6 turns verbatim
        max_before_summary=20,  # summarize once history exceeds 20 turns
    )
    history.append({"role": "user", "content": "..."})
    history.append({"role": "assistant", "content": "..."})

    # pass history.messages to the API call
    client.messages.create(..., messages=history.messages)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

MessageDict = dict[str, Any]

_SUMMARIZE_PROMPT = (
    "You are summarizing a voice assistant conversation to save context space. "
    "Write a concise factual summary (3-6 sentences) of the key information, "
    "decisions, and facts established in the conversation below. "
    "Focus on details the assistant might need later. Be terse."
)


@dataclass
class SummarizingHistory:
    """
    Conversation history that automatically compresses old turns.

    Keeps `keep_recent` turns verbatim and, once the buffer exceeds
    `max_before_summary` turns, uses Haiku to summarize the older portion
    into a single synthetic user/assistant exchange injected at the front.

    Thread-safety: not thread-safe — Hey-Claude processes queries serially.
    """

    client: anthropic.Anthropic
    model: str = "claude-haiku-4-5"
    keep_recent: int = 6
    max_before_summary: int = 20

    _turns: list[MessageDict] = field(default_factory=list, init=False, repr=False)
    _summary: str | None = field(default=None, init=False, repr=False)

    def append(self, message: MessageDict) -> None:
        self._turns.append(message)
        if len(self._turns) >= self.max_before_summary:
            self._compress()

    def extend(self, messages: list[MessageDict]) -> None:
        for m in messages:
            self.append(m)

    @property
    def messages(self) -> list[MessageDict]:
        """Returns the message list to pass directly to the Anthropic API."""
        if self._summary is None:
            return list(self._turns)

        summary_block: list[MessageDict] = [
            {"role": "user", "content": "[Conversation summary]"},
            {"role": "assistant", "content": self._summary},
        ]
        recent = self._turns[-self.keep_recent :]
        return summary_block + recent

    def __len__(self) -> int:
        return len(self._turns)

    def clear(self) -> None:
        self._turns.clear()
        self._summary = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compress(self) -> None:
        """Summarize all but the most recent turns using Haiku."""
        to_summarize = self._turns[: -self.keep_recent]
        if not to_summarize:
            return

        # If there's already a summary, prepend it so context accumulates.
        prior = f"Prior summary:\n{self._summary}\n\n" if self._summary else ""
        transcript = self._build_transcript(to_summarize)
        prompt = f"{prior}Conversation to summarize:\n{transcript}"

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_SUMMARIZE_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            new_summary = resp.content[0].text.strip()
        except Exception:
            logger.exception("Context summarization failed; keeping raw history")
            return

        self._summary = new_summary
        # Drop old turns, keep only recent ones in the live buffer.
        self._turns = self._turns[-self.keep_recent :]
        logger.debug("History compressed. Summary: %s", self._summary[:120])

    @staticmethod
    def _build_transcript(turns: list[MessageDict]) -> str:
        lines = []
        for t in turns:
            role = t.get("role", "?")
            content = t.get("content", "")
            if isinstance(content, list):
                # Handle content blocks (tool results etc.)
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)
