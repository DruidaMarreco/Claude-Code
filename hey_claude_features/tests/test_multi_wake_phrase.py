"""Unit tests for multi-wake-phrase — no API key required."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from hey_claude_features.multi_wake_phrase import (
    MultiPhrasePrefilter,
    MultiPhraseWakeDetector,
    WakeResult,
    _looks_like_phrase,
    _simple_ratio,
    phrases_from_env,
)


# ---------------------------------------------------------------------------
# _simple_ratio
# ---------------------------------------------------------------------------


class TestSimpleRatio:
    def test_identical_strings(self):
        assert _simple_ratio("hey claude", "hey claude") == 1.0

    def test_completely_different(self):
        assert _simple_ratio("abc", "xyz") < 0.3

    def test_empty_strings(self):
        assert _simple_ratio("", "") == 1.0

    def test_one_empty(self):
        assert _simple_ratio("hello", "") == 0.0


# ---------------------------------------------------------------------------
# _looks_like_phrase
# ---------------------------------------------------------------------------


class TestLooksLikePhrase:
    def test_exact_substring(self):
        assert _looks_like_phrase("hey claude what time is it", "hey claude")

    def test_case_insensitive(self):
        assert _looks_like_phrase("HEY CLAUDE", "hey claude")

    def test_homophone_fuzzy(self):
        # "hay cloud" is close enough to "hey claude"
        assert _looks_like_phrase("hay cloud what's the weather", "hey claude")

    def test_unrelated_text_fails(self):
        assert not _looks_like_phrase("the quick brown fox", "hey claude")

    def test_ok_claude(self):
        assert _looks_like_phrase("ok claude set a timer", "ok claude")


# ---------------------------------------------------------------------------
# phrases_from_env
# ---------------------------------------------------------------------------


class TestPhrasesFromEnv:
    def test_parses_comma_separated(self, monkeypatch):
        monkeypatch.setenv("HEY_CLAUDE_WAKE_PHRASES", "hey claude, ok claude, yo claude")
        result = phrases_from_env()
        assert result == ["hey claude", "ok claude", "yo claude"]

    def test_empty_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("HEY_CLAUDE_WAKE_PHRASES", raising=False)
        assert phrases_from_env() is None

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("HEY_CLAUDE_WAKE_PHRASES", "  hey claude  ,  ok claude  ")
        result = phrases_from_env()
        assert result == ["hey claude", "ok claude"]


# ---------------------------------------------------------------------------
# MultiPhrasePrefilter
# ---------------------------------------------------------------------------


class TestMultiPhrasePrefilter:
    def test_default_phrases(self):
        pf = MultiPhrasePrefilter()
        assert "hey claude" in pf.phrases
        assert "ok claude" in pf.phrases

    def test_custom_phrases(self):
        pf = MultiPhrasePrefilter(phrases=["yo bot", "wake up"])
        assert pf.phrases == ["yo bot", "wake up"]

    def test_env_phrases_used_when_no_arg(self, monkeypatch):
        monkeypatch.setenv("HEY_CLAUDE_WAKE_PHRASES", "custom phrase")
        pf = MultiPhrasePrefilter()
        assert pf.phrases == ["custom phrase"]

    def test_looks_like_wake_first_phrase(self):
        pf = MultiPhrasePrefilter(phrases=["hey claude", "ok claude"])
        assert pf.looks_like_wake("hey claude what's the weather")

    def test_looks_like_wake_second_phrase(self):
        pf = MultiPhrasePrefilter(phrases=["hey claude", "ok claude"])
        assert pf.looks_like_wake("ok claude set a timer")

    def test_no_match_returns_false(self):
        pf = MultiPhrasePrefilter(phrases=["hey claude", "ok claude"])
        assert not pf.looks_like_wake("I was just talking to myself")

    def test_matching_phrase_returns_first_match(self):
        pf = MultiPhrasePrefilter(phrases=["hey claude", "ok claude"])
        assert pf.matching_phrase("hey claude do a thing") == "hey claude"

    def test_matching_phrase_returns_none_on_miss(self):
        pf = MultiPhrasePrefilter(phrases=["hey claude"])
        assert pf.matching_phrase("nothing relevant") is None

    def test_multiple_phrases_any_match(self):
        pf = MultiPhrasePrefilter(phrases=["alpha phrase", "beta phrase", "gamma phrase"])
        assert pf.looks_like_wake("gamma phrase do this")
        assert not pf.looks_like_wake("the quick brown fox jumps")


# ---------------------------------------------------------------------------
# MultiPhraseWakeDetector
# ---------------------------------------------------------------------------


def _make_client(detected=True, phrase="hey claude", query="what time is it"):
    client = MagicMock()
    payload = {"detected": detected, "phrase": phrase, "query": query}
    resp = MagicMock()
    resp.content = [MagicMock(text=json.dumps(payload))]
    client.messages.create.return_value = resp
    return client


class TestMultiPhraseWakeDetector:
    def test_no_match_skips_haiku(self):
        client = _make_client()
        detector = MultiPhraseWakeDetector(client=client, phrases=["hey claude"])
        result = detector.detect("just some background noise")
        assert not result.detected
        client.messages.create.assert_not_called()

    def test_match_calls_haiku(self):
        client = _make_client(detected=True, phrase="hey claude", query="what time is it")
        detector = MultiPhraseWakeDetector(client=client, phrases=["hey claude"])
        result = detector.detect("hey claude what time is it")
        assert result.detected
        assert result.query == "what time is it"
        assert result.matched_phrase == "hey claude"
        client.messages.create.assert_called_once()

    def test_haiku_says_not_detected(self):
        client = _make_client(detected=False, phrase="", query="")
        detector = MultiPhraseWakeDetector(client=client, phrases=["hey claude"])
        result = detector.detect("hey cloud I'm talking about clouds")
        assert not result.detected

    def test_second_phrase_detected(self):
        client = _make_client(detected=True, phrase="ok claude", query="set a timer")
        detector = MultiPhraseWakeDetector(
            client=client, phrases=["hey claude", "ok claude"]
        )
        result = detector.detect("ok claude set a timer for 5 minutes")
        assert result.detected
        assert result.matched_phrase == "ok claude"
        assert result.query == "set a timer"

    def test_haiku_failure_falls_back_gracefully(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("network error")
        detector = MultiPhraseWakeDetector(client=client, phrases=["hey claude"])
        result = detector.detect("hey claude help me")
        # Falls back to detected=True rather than crashing
        assert result.detected

    def test_haiku_invalid_json_falls_back(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content = [MagicMock(text="not json")]
        client.messages.create.return_value = resp
        detector = MultiPhraseWakeDetector(client=client, phrases=["hey claude"])
        result = detector.detect("hey claude test")
        assert result.detected


# ---------------------------------------------------------------------------
# WakeResult
# ---------------------------------------------------------------------------


class TestWakeResult:
    def test_defaults(self):
        r = WakeResult(detected=False)
        assert r.query == ""
        assert r.matched_phrase == ""

    def test_with_values(self):
        r = WakeResult(detected=True, query="what time", matched_phrase="ok claude")
        assert r.detected
        assert r.query == "what time"
        assert r.matched_phrase == "ok claude"
