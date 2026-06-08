"""Unit tests for IntentTagger — no API key required."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from hey_claude_features.intent_tagger import Intent, IntentTag, IntentTagger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(intent: str = "question", confidence: float = 0.9) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.content = [MagicMock(text=json.dumps({"intent": intent, "confidence": confidence}))]
    client.messages.create.return_value = resp
    return client


def _tagger(client=None) -> IntentTagger:
    return IntentTagger(client=client or MagicMock(), confidence_threshold=0.75)


# ---------------------------------------------------------------------------
# IntentTag
# ---------------------------------------------------------------------------


class TestIntentTag:
    def test_str_format(self):
        t = IntentTag(intent=Intent.QUESTION, confidence=0.9, method="regex")
        assert "question" in str(t)
        assert "0.90" in str(t)

    def test_is_confident_above_threshold(self):
        assert IntentTag(intent=Intent.QUESTION, confidence=0.8).is_confident

    def test_is_confident_below_threshold(self):
        assert not IntentTag(intent=Intent.UNKNOWN, confidence=0.5).is_confident


# ---------------------------------------------------------------------------
# Regex — reminder
# ---------------------------------------------------------------------------


class TestReminderRegex:
    @pytest.mark.parametrize("query", [
        "remind me to call Sarah",
        "set a reminder for 3pm",
        "reminder for my dentist appointment",
        "don't let me forget the meeting",
        "alert me when it's 5 o'clock",
    ])
    def test_reminder_detected(self, query):
        t = _tagger()
        tag = t.tag(query)
        assert tag.intent == Intent.REMINDER
        assert tag.method == "regex"
        t._client.messages.create.assert_not_called()

    def test_reminder_confidence_is_1(self):
        tag = _tagger().tag("remind me to buy milk")
        assert tag.confidence == 1.0


# ---------------------------------------------------------------------------
# Regex — search
# ---------------------------------------------------------------------------


class TestSearchRegex:
    @pytest.mark.parametrize("query", [
        "search for my notes on Python",
        "find me a good recipe",
        "look up the weather",
        "where did I leave off",
        "have I mentioned this before",
    ])
    def test_search_detected(self, query):
        tag = _tagger().tag(query)
        assert tag.intent == Intent.SEARCH
        assert tag.method == "regex"

    def test_search_confidence_is_1(self):
        tag = _tagger().tag("search for my old notes")
        assert tag.confidence == 1.0


# ---------------------------------------------------------------------------
# Regex — chit-chat
# ---------------------------------------------------------------------------


class TestChitChatRegex:
    @pytest.mark.parametrize("query", [
        "hey",
        "hi there",
        "hello",
        "good morning",
        "thanks",
        "thank you",
        "okay",
        "bye",
    ])
    def test_chit_chat_detected(self, query):
        tag = _tagger().tag(query)
        assert tag.intent == Intent.CHIT_CHAT
        assert tag.method == "regex"


# ---------------------------------------------------------------------------
# Regex — command
# ---------------------------------------------------------------------------


class TestCommandRegex:
    @pytest.mark.parametrize("query", [
        "set a timer for 5 minutes",
        "play some music",
        "turn off the lights",
        "create a new note",
        "open my calendar",
    ])
    def test_command_detected(self, query):
        tag = _tagger().tag(query)
        assert tag.intent == Intent.COMMAND
        assert tag.method == "regex"


# ---------------------------------------------------------------------------
# Regex — question
# ---------------------------------------------------------------------------


class TestQuestionRegex:
    @pytest.mark.parametrize("query", [
        "what time is it",
        "how do I make pasta",
        "where is the Eiffel Tower",
        "who invented the telephone",
        "why is the sky blue",
        "is it going to rain today",
        "can you help me",
    ])
    def test_question_detected(self, query):
        tag = _tagger().tag(query)
        assert tag.intent == Intent.QUESTION
        assert tag.method == "regex"


# ---------------------------------------------------------------------------
# Haiku tier
# ---------------------------------------------------------------------------


class TestHaikuClassification:
    # Phrase that doesn't match any regex pattern
    _AMBIGUOUS = "Tell me something interesting about penguins"

    def test_haiku_called_for_ambiguous(self):
        client = _make_client(intent="question", confidence=0.88)
        t = IntentTagger(client=client)
        tag = t.tag(self._AMBIGUOUS)
        assert tag.intent == Intent.QUESTION
        assert tag.method == "haiku"
        client.messages.create.assert_called_once()

    def test_haiku_below_threshold_returns_unknown(self):
        client = _make_client(intent="question", confidence=0.50)
        t = IntentTagger(client=client, confidence_threshold=0.75)
        tag = t.tag(self._AMBIGUOUS)
        assert tag.intent == Intent.UNKNOWN

    def test_haiku_invalid_intent_returns_unknown(self):
        client = _make_client(intent="gossip", confidence=0.95)
        t = IntentTagger(client=client)
        tag = t.tag(self._AMBIGUOUS)
        assert tag.intent == Intent.UNKNOWN

    def test_haiku_api_failure_returns_unknown(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("network error")
        t = IntentTagger(client=client)
        tag = t.tag(self._AMBIGUOUS)
        assert tag.intent == Intent.UNKNOWN
        assert tag.method == "fallback"

    def test_haiku_invalid_json_returns_unknown(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content = [MagicMock(text="not json")]
        client.messages.create.return_value = resp
        t = IntentTagger(client=client)
        tag = t.tag(self._AMBIGUOUS)
        assert tag.intent == Intent.UNKNOWN

    def test_regex_priority_over_haiku(self):
        client = _make_client(intent="command", confidence=0.99)
        t = IntentTagger(client=client)
        t.tag("remind me to call mom")   # regex catches this
        client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# History and analytics
# ---------------------------------------------------------------------------


class TestSessionAnalytics:
    def test_history_grows(self):
        t = _tagger()
        t.tag("what time is it")
        t.tag("remind me to buy milk")
        assert len(t) == 2

    def test_session_counts(self):
        t = _tagger()
        t.tag("what time is it")
        t.tag("what day is today")
        t.tag("remind me to call")
        counts = t.session_counts()
        assert counts["question"] == 2
        assert counts["reminder"] == 1

    def test_session_report_no_data(self):
        t = _tagger()
        assert "No queries" in t.session_report()

    def test_session_report_contains_intents(self):
        t = _tagger()
        t.tag("what time is it")
        t.tag("remind me to buy milk")
        report = t.session_report()
        assert "question" in report
        assert "reminder" in report

    def test_clear_resets_history(self):
        t = _tagger()
        t.tag("hi")
        t.clear()
        assert len(t) == 0

    def test_track_history_false(self):
        t = IntentTagger(client=MagicMock(), track_history=False)
        t.tag("hi")
        assert len(t) == 0

    def test_raw_query_stored_in_tag(self):
        t = _tagger()
        tag = t.tag("what is the weather")
        assert tag.raw_query == "what is the weather"


# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------


class TestIntentEnum:
    def test_all_values_accessible(self):
        for intent in Intent:
            assert intent.value in ("question", "command", "reminder", "search", "chit_chat", "unknown")

    def test_string_comparison(self):
        assert Intent.QUESTION == "question"
