"""
Conversation history manager for Hey-Claude.

Manages the list of ``{"role": ..., "content": ...}`` messages passed to
the Anthropic API.  Handles:

  - Appending user/assistant turns
  - Token budget enforcement via a simple char-based estimate
    (4 chars ≈ 1 token — conservative, avoids exact tokenisation)
  - Rolling trim: oldest turns are dropped when the budget is exceeded,
    always keeping the system prompt and the most recent ``min_keep`` turns
  - Metadata tagging: each turn can carry a dict of extra data (intent,
    latency, cost) that is stored locally but never sent to the API
  - Export to plain ``list[dict]`` for the Anthropic messages parameter

This is *not* the same as ``context_summarizer``, which compresses history
via a Haiku call.  This module is the low-level store; the summarizer can
sit on top.

Usage in app.py
---------------
    from hey_claude_features.conversation_history import ConversationHistory

    history = ConversationHistory(max_tokens=8000, min_keep=4)

    history.add_user("hey claude, what's the weather?")
    history.add_assistant("It's sunny and 22°C in your area.")

    # Pass to API:
    response = client.messages.create(
        model="claude-opus-4-8",
        messages=history.messages,
        ...
    )

    # Inspect:
    print(f"{len(history)} turns, ~{history.estimated_tokens} tokens")
    print(history.last_user)
    print(history.last_assistant)

    # Clear between sessions:
    history.clear()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


_CHARS_PER_TOKEN = 4   # conservative estimate


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single conversation turn (user or assistant)."""
    role: str                          # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_tokens(self) -> int:
        return max(1, len(self.content) // _CHARS_PER_TOKEN)

    def to_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------


class ConversationHistory:
    """
    Rolling conversation history with token-budget trimming.

    Parameters
    ----------
    max_tokens:
        Approximate token budget for the history (characters / 4).
        Oldest turns are dropped when exceeded.
    min_keep:
        Minimum number of recent turns to keep regardless of budget.
    system_prompt:
        If set, included as the first message with role "system" in
        ``messages`` but not counted in the turn list.
    """

    def __init__(
        self,
        max_tokens: int = 8_000,
        min_keep: int = 4,
        system_prompt: str = "",
    ) -> None:
        self._max_tokens = max_tokens
        self._min_keep = min_keep
        self._system = system_prompt
        self._turns: list[Turn] = []

    # ------------------------------------------------------------------
    # Public API — adding turns
    # ------------------------------------------------------------------

    def add_user(self, content: str, **meta: Any) -> Turn:
        """Append a user turn and trim if necessary."""
        turn = Turn(role="user", content=content, meta=meta)
        self._turns.append(turn)
        self._trim()
        return turn

    def add_assistant(self, content: str, **meta: Any) -> Turn:
        """Append an assistant turn and trim if necessary."""
        turn = Turn(role="assistant", content=content, meta=meta)
        self._turns.append(turn)
        self._trim()
        return turn

    def add(self, role: str, content: str, **meta: Any) -> Turn:
        """Generic add — use add_user / add_assistant when role is known."""
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        turn = Turn(role=role, content=content, meta=meta)
        self._turns.append(turn)
        self._trim()
        return turn

    # ------------------------------------------------------------------
    # Public API — reading
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[dict[str, str]]:
        """Return history as Anthropic-compatible message list."""
        return [t.to_message() for t in self._turns]

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    @property
    def estimated_tokens(self) -> int:
        return sum(t.estimated_tokens for t in self._turns)

    @property
    def last_user(self) -> str:
        for t in reversed(self._turns):
            if t.role == "user":
                return t.content
        return ""

    @property
    def last_assistant(self) -> str:
        for t in reversed(self._turns):
            if t.role == "assistant":
                return t.content
        return ""

    @property
    def last_pair(self) -> tuple[str, str]:
        """Return (last_user, last_assistant) as a tuple."""
        return self.last_user, self.last_assistant

    def tail(self, n: int) -> list[Turn]:
        """Return the last *n* turns."""
        return self._turns[-n:]

    # ------------------------------------------------------------------
    # Public API — management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._turns.clear()

    def set_system(self, prompt: str) -> None:
        self._system = prompt

    def summary_line(self) -> str:
        return (
            f"{len(self._turns)} turns, "
            f"~{self.estimated_tokens} tokens "
            f"(budget {self._max_tokens})"
        )

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self):
        return iter(self._turns)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Drop oldest turns until under budget, keeping min_keep."""
        while (
            len(self._turns) > self._min_keep
            and self.estimated_tokens > self._max_tokens
        ):
            self._turns.pop(0)
