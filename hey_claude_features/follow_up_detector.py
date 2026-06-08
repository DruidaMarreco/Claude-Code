"""
Follow-up query detector for Hey-Claude.

Detects when a user's query is a continuation of the previous exchange
(e.g. "and in Paris?", "what about tomorrow?", "how about for kids?")
and automatically enriches it with the prior context so the responder
can answer correctly without re-reading the full history.

Why this matters
----------------
Hey-Claude processes each utterance somewhat independently. Without context
injection, "what about tomorrow?" arrives at Opus with no idea what "it" is.
This module bridges that gap cheaply.

Pipeline
--------
1. Free heuristic check  — regex patterns catch obvious follow-ups instantly
2. Haiku classification  — for ambiguous queries, ask Haiku (cheap)
3. Context injection     — enrich the query with a one-line context summary

Usage in app.py / responder.py
-------------------------------
    from hey_claude_features.follow_up_detector import FollowUpDetector

    detector = FollowUpDetector(client=anthropic_client)

    # After wake detection, before sending to Opus:
    enriched = detector.enrich(query, history=self._history.messages)
    reply = responder.respond(enriched)

The enriched query is only modified when a follow-up is detected; otherwise
the original string is returned unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristics — free, instant
# ---------------------------------------------------------------------------

# Queries that almost certainly reference something said previously
_FOLLOW_UP_RE = re.compile(
    r"^("
    r"and\b|but\b|also\b|what about\b|how about\b|"
    r"what if\b|why\b(?! not)|"
    r"(what|how|when|where|who|which)\s+(about|else|instead|too)\b|"
    r"(and\s+)?(in|for|at|on|with|without|from|to)\s+\w|"  # "in Paris?", "for kids?"
    r"(is\s+)?(it|that|this|he|she|they|there)\b|"          # pronoun references
    r"(tell me more|more details|explain|elaborate|why so|really\?)"
    r")",
    re.I,
)

# Queries that are clearly standalone — skip Haiku
_STANDALONE_RE = re.compile(
    r"^(hey claude|ok claude|what time|what date|calculate|remind me|set a reminder)",
    re.I,
)

_CLASSIFY_SYSTEM = """\
You are a follow-up detector for a voice assistant conversation.
Given the LAST EXCHANGE and a NEW QUERY, decide if the new query references
the previous exchange (i.e. it can only be understood with that context).

Respond with JSON only:
{"is_followup": true/false, "confidence": 0.0-1.0}
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    is_followup: bool
    confidence: float = 1.0
    enriched_query: str = ""   # set only when is_followup=True


# ---------------------------------------------------------------------------
# FollowUpDetector
# ---------------------------------------------------------------------------


class FollowUpDetector:
    """
    Detects follow-up queries and injects prior context into them.

    Parameters
    ----------
    client:
        Anthropic client (used only when heuristics are inconclusive).
    model:
        Haiku model for classification.
    confidence_threshold:
        Minimum Haiku confidence to trust a follow-up classification.
    context_turns:
        How many of the most recent history turns to use as context window.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-haiku-4-5",
        confidence_threshold: float = 0.80,
        context_turns: int = 4,
    ) -> None:
        self._client = client
        self._model = model
        self._threshold = confidence_threshold
        self._context_turns = context_turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, query: str, history: list[dict]) -> str:
        """
        Return the query, potentially enriched with prior context.

        If the query is detected as a follow-up, returns a version of the
        query that includes a brief recap of the last exchange:
            "[Context: <prior topic>] <original query>"

        If not a follow-up, returns *query* unchanged.
        """
        result = self.detect(query, history)
        if result.is_followup and result.enriched_query:
            return result.enriched_query
        return query

    def detect(self, query: str, history: list[dict]) -> DetectionResult:
        """
        Classify a query and return a DetectionResult.

        Standalone detection skips Haiku even if history is available.
        """
        if not history:
            return DetectionResult(is_followup=False)

        # Fast-path: obviously standalone
        if _STANDALONE_RE.match(query.strip()):
            return DetectionResult(is_followup=False)

        # Heuristic follow-up match
        if _FOLLOW_UP_RE.match(query.strip()):
            enriched = self._build_enriched(query, history)
            return DetectionResult(is_followup=True, confidence=0.95, enriched_query=enriched)

        # Haiku classification for ambiguous cases
        return self._haiku_classify(query, history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _haiku_classify(self, query: str, history: list[dict]) -> DetectionResult:
        last_exchange = self._last_exchange_text(history)
        if not last_exchange:
            return DetectionResult(is_followup=False)

        prompt = f"Last exchange:\n{last_exchange}\n\nNew query: \"{query}\""
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=64,
                system=_CLASSIFY_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(resp.content[0].text.strip())
            is_followup = bool(data.get("is_followup", False))
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            logger.exception("Follow-up Haiku classification failed; assuming standalone")
            return DetectionResult(is_followup=False)

        if is_followup and confidence >= self._threshold:
            enriched = self._build_enriched(query, history)
            return DetectionResult(is_followup=True, confidence=confidence, enriched_query=enriched)

        return DetectionResult(is_followup=False, confidence=confidence)

    def _build_enriched(self, query: str, history: list[dict]) -> str:
        """Build a context-enriched version of the query."""
        last_exchange = self._last_exchange_text(history)
        if not last_exchange:
            return query
        # Prepend a compact context marker that Opus will pick up
        return f"[Continuing from: {last_exchange[:200].strip()}] {query}"

    def _last_exchange_text(self, history: list[dict]) -> str:
        """Extract the most recent user+assistant turn as a compact string."""
        relevant = [
            m for m in history
            if m.get("role") in ("user", "assistant")
        ][-self._context_turns * 2:]

        lines = []
        for m in relevant:
            role = m.get("role", "").upper()
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
