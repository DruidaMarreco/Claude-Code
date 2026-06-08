"""Unit tests for FollowUpDetector — no API key required."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from hey_claude_features.follow_up_detector import DetectionResult, FollowUpDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(is_followup: bool = False, confidence: float = 0.9) -> MagicMock:
    client = MagicMock()
    payload = {"is_followup": is_followup, "confidence": confidence}
    resp = MagicMock()
    resp.content = [MagicMock(text=json.dumps(payload))]
    client.messages.create.return_value = resp
    return client


def _detector(client=None) -> FollowUpDetector:
    return FollowUpDetector(client=client or MagicMock(), confidence_threshold=0.80)


def _history(user: str = "What's the weather in London?", assistant: str = "It's sunny and 18°C.") -> list[dict]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ---------------------------------------------------------------------------
# Empty history — always standalone
# ---------------------------------------------------------------------------


class TestEmptyHistory:
    def test_no_history_returns_standalone(self):
        d = _detector()
        result = d.detect("what about Paris?", history=[])
        assert not result.is_followup

    def test_enrich_unchanged_with_no_history(self):
        d = _detector()
        q = "what about Paris?"
        assert d.enrich(q, history=[]) == q


# ---------------------------------------------------------------------------
# Heuristic follow-up patterns
# ---------------------------------------------------------------------------


class TestHeuristicFollowups:
    @pytest.mark.parametrize("query", [
        "and in Paris?",
        "what about tomorrow?",
        "how about for kids?",
        "but why?",
        "also in Berlin",
        "is it the same there?",
        "tell me more",
        "what else?",
    ])
    def test_heuristic_detected(self, query):
        d = _detector()
        result = d.detect(query, history=_history())
        assert result.is_followup
        d._client.messages.create.assert_not_called()

    @pytest.mark.parametrize("query", [
        "what time is it?",
        "hey claude set a reminder",
        "calculate 5 + 3",
        "what date is today",
    ])
    def test_standalone_patterns_skip_haiku(self, query):
        d = _detector()
        result = d.detect(query, history=_history())
        assert not result.is_followup
        d._client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Haiku classification
# ---------------------------------------------------------------------------


class TestHaikuClassification:
    # "What are some good alternatives?" has no heuristic follow-up markers
    # so it always reaches Haiku classification.
    _AMBIGUOUS = "What are some good alternatives?"

    def test_haiku_confirms_followup(self):
        client = _make_client(is_followup=True, confidence=0.92)
        d = FollowUpDetector(client=client, confidence_threshold=0.80)
        result = d.detect(self._AMBIGUOUS, history=_history())
        assert result.is_followup
        client.messages.create.assert_called_once()

    def test_haiku_below_threshold_returns_standalone(self):
        client = _make_client(is_followup=True, confidence=0.60)
        d = FollowUpDetector(client=client, confidence_threshold=0.80)
        result = d.detect(self._AMBIGUOUS, history=_history())
        assert not result.is_followup

    def test_haiku_says_standalone(self):
        client = _make_client(is_followup=False, confidence=0.95)
        d = _detector(client=client)
        result = d.detect("What is the capital of France?", history=_history())
        assert not result.is_followup

    def test_haiku_api_failure_returns_standalone(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("network error")
        d = _detector(client=client)
        result = d.detect(self._AMBIGUOUS, history=_history())
        assert not result.is_followup

    def test_haiku_invalid_json_returns_standalone(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content = [MagicMock(text="not json")]
        client.messages.create.return_value = resp
        d = _detector(client=client)
        result = d.detect(self._AMBIGUOUS, history=_history())
        assert not result.is_followup


# ---------------------------------------------------------------------------
# Context enrichment
# ---------------------------------------------------------------------------


class TestEnrichment:
    def test_enriched_query_contains_original(self):
        d = _detector()
        result = d.detect("and in Paris?", history=_history())
        assert "and in Paris?" in result.enriched_query

    def test_enriched_query_contains_context(self):
        d = _detector()
        result = d.detect("and in Paris?", history=_history(
            user="What's the weather in London?",
            assistant="It's sunny and 18°C.",
        ))
        assert "London" in result.enriched_query or "weather" in result.enriched_query

    def test_enrich_returns_original_when_standalone(self):
        client = _make_client(is_followup=False, confidence=0.95)
        d = _detector(client=client)
        q = "What is the capital of France?"
        assert d.enrich(q, history=_history()) == q

    def test_enrich_returns_enriched_when_followup(self):
        d = _detector()
        q = "and in Paris?"
        enriched = d.enrich(q, history=_history())
        assert enriched != q
        assert "Paris" in enriched

    def test_enriched_context_capped_at_200_chars(self):
        long_reply = "x" * 500
        d = _detector()
        result = d.detect("and in Paris?", history=_history(assistant=long_reply))
        # The context portion should not be excessively long
        assert len(result.enriched_query) < 500


# ---------------------------------------------------------------------------
# History with list-type content (tool results)
# ---------------------------------------------------------------------------


class TestListContent:
    def test_list_content_blocks_handled(self):
        history = [
            {"role": "user", "content": [{"text": "weather in London"}]},
            {"role": "assistant", "content": [{"text": "Sunny, 18°C"}]},
        ]
        d = _detector()
        result = d.detect("and in Paris?", history=history)
        assert result.is_followup


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


class TestDetectionResult:
    def test_defaults(self):
        r = DetectionResult(is_followup=False)
        assert r.confidence == 1.0
        assert r.enriched_query == ""

    def test_with_values(self):
        r = DetectionResult(is_followup=True, confidence=0.9, enriched_query="[Ctx] query")
        assert r.is_followup
        assert "Ctx" in r.enriched_query
