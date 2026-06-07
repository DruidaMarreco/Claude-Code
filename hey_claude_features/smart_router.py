"""
Smart query router that skips Opus for queries answerable by a single cheap tool.

The two-tier architecture (Haiku gate → Opus responder) is great for complex
queries, but asking Opus "what time is it?" wastes ~10-50x the tokens needed.

This router sits between wake detection and the responder:

    wake result → QueryRouter.route() → RouteDecision
        FAST_TOOL   → execute tool directly, skip Opus entirely
        NEEDS_OPUS  → send to full Opus responder as usual

How it works:
    1. Pattern match against known single-tool queries (free, instant).
    2. If ambiguous, ask Haiku to classify (cheap, ~1-2 cents per 1k queries).
    3. Opus is only called when real language understanding is needed.

Usage in app.py:
    from hey_claude_features.smart_router import QueryRouter, RouteDecision

    router = QueryRouter(client=anthropic_client, tool_registry=registry)

    decision = router.route(query)
    if decision.kind == RouteDecision.FAST_TOOL:
        result = registry.run(decision.tool_name, decision.tool_args)
        speak(result)
    else:
        responder.respond(query)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RouteKind(Enum):
    FAST_TOOL = auto()
    NEEDS_OPUS = auto()


@dataclass
class RouteDecision:
    FAST_TOOL = RouteKind.FAST_TOOL
    NEEDS_OPUS = RouteKind.NEEDS_OPUS

    kind: RouteKind
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    @property
    def is_fast(self) -> bool:
        return self.kind == RouteKind.FAST_TOOL


# ---------------------------------------------------------------------------
# Static patterns (zero cost)
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(
    r"\b(what'?s?\s+(the\s+)?time|current\s+time|time\s+is\s+it|what\s+time)\b", re.I
)
_DATE_RE = re.compile(
    r"\b(what'?s?\s+(the\s+)?date|today'?s?\s+date|current\s+date|what\s+day)\b", re.I
)
_CALC_RE = re.compile(
    r"\b(calculate|compute|what\s+is|what'?s)\s+[\d\s\.\+\-\*\/\(\)]+", re.I
)

_STATIC_PATTERNS: list[tuple[re.Pattern[str], str, dict[str, Any]]] = [
    (_TIME_RE, "get_current_time", {}),
    (_DATE_RE, "get_current_date", {}),
]


def _extract_calc_expr(query: str) -> str | None:
    """Extract a safe arithmetic expression from a query string."""
    m = re.search(r"[\d\s\.\+\-\*\/\(\)]{3,}", query)
    if m:
        return m.group().strip()
    return None


# ---------------------------------------------------------------------------
# Haiku classification prompt
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """\
You are a query classifier for a voice assistant. Given a user query, decide
whether it can be answered by exactly ONE of the available tools without any
additional reasoning, or whether it needs the full language model responder.

Available tools: get_current_time, get_current_date, calculate

Respond with JSON only (no extra text):
{"route": "fast_tool", "tool": "<tool_name>", "args": {}, "confidence": 0.0-1.0}
  or
{"route": "needs_opus", "confidence": 0.0-1.0}

Rules:
- Use "fast_tool" only when the query maps unambiguously to a single tool.
- For calculate, extract the expression into args: {"expression": "2+2"}.
- Use "needs_opus" for anything requiring context, opinions, or multiple steps.
- confidence must reflect how certain you are.
"""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class QueryRouter:
    """
    Routes a query to either a fast tool path or the full Opus responder.

    Parameters
    ----------
    client:
        An Anthropic client instance (used only when static patterns don't match).
    haiku_model:
        Model used for classification. Defaults to the cheapest Haiku.
    confidence_threshold:
        Minimum Haiku confidence to trust a fast-tool classification.
        Below this, falls back to Opus.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        haiku_model: str = "claude-haiku-4-5",
        confidence_threshold: float = 0.85,
    ) -> None:
        self._client = client
        self._haiku_model = haiku_model
        self._confidence_threshold = confidence_threshold

    def route(self, query: str) -> RouteDecision:
        """Classify a query and return a RouteDecision."""
        # 1. Free static check
        static = self._static_route(query)
        if static is not None:
            logger.debug("Static route matched: %s", static.tool_name)
            return static

        # 2. Ask Haiku
        return self._haiku_classify(query)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _static_route(self, query: str) -> RouteDecision | None:
        for pattern, tool_name, args in _STATIC_PATTERNS:
            if pattern.search(query):
                return RouteDecision(kind=RouteKind.FAST_TOOL, tool_name=tool_name, tool_args=args)

        # Calculator: detect expression in query
        if _CALC_RE.search(query):
            expr = _extract_calc_expr(query)
            if expr:
                return RouteDecision(
                    kind=RouteKind.FAST_TOOL,
                    tool_name="calculate",
                    tool_args={"expression": expr},
                )

        return None

    def _haiku_classify(self, query: str) -> RouteDecision:
        try:
            resp = self._client.messages.create(
                model=self._haiku_model,
                max_tokens=128,
                system=_CLASSIFY_SYSTEM,
                messages=[{"role": "user", "content": query}],
            )
            raw = resp.content[0].text.strip()
            data = json.loads(raw)
        except Exception:
            logger.exception("Haiku classification failed; defaulting to Opus")
            return RouteDecision(kind=RouteKind.NEEDS_OPUS, confidence=0.0)

        confidence = float(data.get("confidence", 0.0))
        route = data.get("route", "needs_opus")

        if route == "fast_tool" and confidence >= self._confidence_threshold:
            return RouteDecision(
                kind=RouteKind.FAST_TOOL,
                tool_name=data.get("tool"),
                tool_args=data.get("args", {}),
                confidence=confidence,
            )

        return RouteDecision(kind=RouteKind.NEEDS_OPUS, confidence=confidence)
