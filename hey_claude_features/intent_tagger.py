"""
Intent tagger for Hey-Claude queries.

Tags each incoming query with a single intent category before it reaches the
Haiku gate or Opus responder.  The tag enables smarter downstream routing
(e.g. skip Haiku for obvious commands) and per-intent session analytics.

Intent categories
-----------------
    question   — the user wants information ("what is…", "how do I…")
    command    — the user wants Hey-Claude to do something ("set a timer")
    reminder   — explicitly a reminder request ("remind me to…")
    search     — searching notes or the web ("search for…", "find…")
    chit_chat  — casual conversation ("hey", "thanks", "how are you")
    unknown    — couldn't classify with sufficient confidence

Two-tier classification
-----------------------
1. Fast regex rules cover the most common patterns with no API cost.
2. Anything ambiguous is sent to Haiku for structured JSON classification.

Usage in app.py
---------------
    from hey_claude_features.intent_tagger import IntentTagger, Intent

    tagger = IntentTagger(client=anthropic_client)
    tag = tagger.tag("remind me to call Sarah at 3pm")
    # → IntentTag(intent=Intent.REMINDER, confidence=1.0, method="regex")

    if tag.intent == Intent.REMINDER:
        reminder_system.set_reminder(query)
    elif tag.intent == Intent.SEARCH:
        results = conversation_search.search(query)
    else:
        # normal Haiku → Opus pipeline
        ...

Analytics
---------
    tagger.session_counts()   # {"question": 5, "command": 3, …}
    tagger.session_report()   # formatted string
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    QUESTION = "question"
    COMMAND = "command"
    REMINDER = "reminder"
    SEARCH = "search"
    CHIT_CHAT = "chit_chat"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Regex patterns — ordered most-specific first
# ---------------------------------------------------------------------------

_REMINDER_RE = re.compile(
    r"\b(remind me|set a reminder|reminder for|don't let me forget|alert me)\b",
    re.IGNORECASE,
)

_SEARCH_RE = re.compile(
    r"\b(search (for|my|the)|find (me|my|the|a|an)|look (up|for)|where (did i|do i)|"
    r"have i (talked|mentioned|said|asked))\b",
    re.IGNORECASE,
)

_COMMAND_RE = re.compile(
    r"\b(set|start|stop|cancel|play|pause|resume|open|close|turn (on|off)|"
    r"mute|unmute|send|call|schedule|add|create|delete|remove|show|hide|"
    r"increase|decrease|volume|timer|alarm)\b",
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(
    r"^(what|who|where|when|why|how|which|whose|is|are|was|were|do|does|did|"
    r"can|could|will|would|should|has|have|had)\b",
    re.IGNORECASE,
)

_CHIT_CHAT_RE = re.compile(
    r"^(hi|hey|hello|good morning|good afternoon|good evening|good night|"
    r"thanks|thank you|ok|okay|cool|great|awesome|bye|goodbye|see you|"
    r"how are you|you there|you awake)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class IntentTag:
    """Classification result for a single query."""
    intent: Intent
    confidence: float = 1.0
    method: str = "regex"   # "regex" | "haiku" | "fallback"
    raw_query: str = ""

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.75

    def __str__(self) -> str:
        return f"[{self.intent.value}|{self.confidence:.2f}|{self.method}]"


# ---------------------------------------------------------------------------
# IntentTagger
# ---------------------------------------------------------------------------


class IntentTagger:
    """
    Classifies voice queries into intent categories.

    Parameters
    ----------
    client:
        Anthropic client (used for Haiku fallback classification).
    model:
        Haiku model ID for ambiguous queries.
    confidence_threshold:
        Minimum confidence to trust Haiku's classification; below this
        the result is Intent.UNKNOWN.
    track_history:
        If True, all tagged queries are stored for session analytics.
    """

    _SYSTEM = (
        "Classify the user's voice query into exactly one intent category. "
        "Respond with JSON only: {\"intent\": \"<category>\", \"confidence\": <0.0-1.0>}. "
        "Categories: question, command, reminder, search, chit_chat, unknown."
    )

    def __init__(
        self,
        client: Any,
        model: str = "claude-haiku-4-5-20251001",
        confidence_threshold: float = 0.75,
        track_history: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._threshold = confidence_threshold
        self._track = track_history
        self._history: list[IntentTag] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag(self, query: str) -> IntentTag:
        """
        Classify *query* and return an IntentTag.

        Tries fast regex rules first; falls back to Haiku only when the
        query doesn't match any pattern.
        """
        result = self._regex_classify(query)
        if result is None:
            result = self._haiku_classify(query)

        result.raw_query = query
        if self._track:
            self._history.append(result)
        return result

    def session_counts(self) -> dict[str, int]:
        """Return counts per intent for the current session."""
        c: Counter[str] = Counter(t.intent.value for t in self._history)
        return dict(c)

    def session_report(self) -> str:
        """Formatted session breakdown."""
        if not self._history:
            return "No queries tagged yet."
        counts = self.session_counts()
        total = len(self._history)
        lines = [f"Intent breakdown — {total} queries"]
        for intent in Intent:
            n = counts.get(intent.value, 0)
            if n:
                pct = n / total * 100
                lines.append(f"  {intent.value:<10s}  {n:3d}  ({pct:.0f}%)")
        return "\n".join(lines)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------
    # Internal — regex tier
    # ------------------------------------------------------------------

    def _regex_classify(self, query: str) -> IntentTag | None:
        """Return IntentTag if a regex matches, else None."""
        if _REMINDER_RE.search(query):
            return IntentTag(intent=Intent.REMINDER, confidence=1.0, method="regex")
        if _SEARCH_RE.search(query):
            return IntentTag(intent=Intent.SEARCH, confidence=1.0, method="regex")
        if _CHIT_CHAT_RE.match(query):
            return IntentTag(intent=Intent.CHIT_CHAT, confidence=1.0, method="regex")
        if _COMMAND_RE.search(query):
            return IntentTag(intent=Intent.COMMAND, confidence=0.9, method="regex")
        if _QUESTION_RE.match(query):
            return IntentTag(intent=Intent.QUESTION, confidence=0.9, method="regex")
        return None

    # ------------------------------------------------------------------
    # Internal — Haiku tier
    # ------------------------------------------------------------------

    def _haiku_classify(self, query: str) -> IntentTag:
        """Send *query* to Haiku and parse the JSON intent response."""
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=64,
                system=self._SYSTEM,
                messages=[{"role": "user", "content": query}],
            )
            text = resp.content[0].text.strip()
            data = json.loads(text)
            intent_str = str(data.get("intent", "unknown")).lower()
            confidence = float(data.get("confidence", 0.0))

            try:
                intent = Intent(intent_str)
            except ValueError:
                intent = Intent.UNKNOWN

            if confidence < self._threshold:
                intent = Intent.UNKNOWN

            return IntentTag(intent=intent, confidence=confidence, method="haiku")
        except Exception:
            logger.exception("Haiku intent classification failed for %r", query)
            return IntentTag(intent=Intent.UNKNOWN, confidence=0.0, method="fallback")
