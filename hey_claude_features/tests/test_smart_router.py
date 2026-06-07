"""Unit tests for QueryRouter — no API key required."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from hey_claude_features.smart_router import QueryRouter, RouteDecision, RouteKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(route: str = "needs_opus", tool: str = "", args: dict | None = None, confidence: float = 0.95) -> MagicMock:
    client = MagicMock()
    payload: dict = {"route": route, "confidence": confidence}
    if tool:
        payload["tool"] = tool
    if args is not None:
        payload["args"] = args
    resp = MagicMock()
    resp.content = [MagicMock(text=json.dumps(payload))]
    client.messages.create.return_value = resp
    return client


def _router(client=None, threshold=0.85) -> QueryRouter:
    return QueryRouter(client=client or MagicMock(), confidence_threshold=threshold)


# ---------------------------------------------------------------------------
# Static routing (no API call)
# ---------------------------------------------------------------------------


class TestStaticRouting:
    @pytest.mark.parametrize("query", [
        "What time is it?",
        "what's the time",
        "current time please",
        "What time is it right now?",
    ])
    def test_time_queries(self, query):
        router = _router()
        decision = router.route(query)
        assert decision.is_fast
        assert decision.tool_name == "get_current_time"
        router._client.messages.create.assert_not_called()

    @pytest.mark.parametrize("query", [
        "What's the date today?",
        "what is today's date",
        "current date",
        "what day is it",
    ])
    def test_date_queries(self, query):
        router = _router()
        decision = router.route(query)
        assert decision.is_fast
        assert decision.tool_name == "get_current_date"
        router._client.messages.create.assert_not_called()

    @pytest.mark.parametrize("query,expected_expr", [
        ("calculate 2 + 2", "2 + 2"),
        ("what is 10 * 5", "10 * 5"),
        ("compute 100 / 4", "100 / 4"),
    ])
    def test_calc_queries(self, query, expected_expr):
        router = _router()
        decision = router.route(query)
        assert decision.is_fast
        assert decision.tool_name == "calculate"
        assert decision.tool_args.get("expression") is not None
        router._client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Haiku classification fallback
# ---------------------------------------------------------------------------


class TestHaikuRouting:
    def test_complex_query_falls_to_haiku_then_opus(self):
        client = _make_client(route="needs_opus", confidence=0.95)
        router = _router(client=client)
        decision = router.route("Tell me a joke about programming")
        assert decision.kind == RouteKind.NEEDS_OPUS
        client.messages.create.assert_called_once()

    def test_haiku_fast_tool_above_threshold(self):
        client = _make_client(route="fast_tool", tool="get_current_time", args={}, confidence=0.95)
        router = _router(client=client, threshold=0.85)
        decision = router.route("Hey, any idea about the time?")
        assert decision.is_fast
        assert decision.tool_name == "get_current_time"

    def test_haiku_fast_tool_below_threshold_falls_back_to_opus(self):
        client = _make_client(route="fast_tool", tool="get_current_time", args={}, confidence=0.60)
        router = _router(client=client, threshold=0.85)
        decision = router.route("Hey, any idea about the time?")
        assert decision.kind == RouteKind.NEEDS_OPUS

    def test_haiku_api_failure_falls_back_to_opus(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("network error")
        router = _router(client=client)
        decision = router.route("some query")
        assert decision.kind == RouteKind.NEEDS_OPUS
        assert decision.confidence == 0.0

    def test_haiku_invalid_json_falls_back_to_opus(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content = [MagicMock(text="not json at all")]
        client.messages.create.return_value = resp
        router = _router(client=client)
        decision = router.route("something complex")
        assert decision.kind == RouteKind.NEEDS_OPUS


# ---------------------------------------------------------------------------
# RouteDecision helpers
# ---------------------------------------------------------------------------


class TestRouteDecision:
    def test_is_fast_true_for_fast_tool(self):
        d = RouteDecision(kind=RouteKind.FAST_TOOL, tool_name="get_current_time")
        assert d.is_fast is True

    def test_is_fast_false_for_needs_opus(self):
        d = RouteDecision(kind=RouteKind.NEEDS_OPUS)
        assert d.is_fast is False

    def test_class_level_constants_match_enum(self):
        assert RouteDecision.FAST_TOOL == RouteKind.FAST_TOOL
        assert RouteDecision.NEEDS_OPUS == RouteKind.NEEDS_OPUS
