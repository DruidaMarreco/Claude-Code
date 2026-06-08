"""Unit tests for TranscriptCleaner — no API key or audio required."""

from __future__ import annotations

import pytest

from hey_claude_features.transcript_cleaner import (
    CleanResult,
    TranscriptCleaner,
    _replace_number_words,
)


# ---------------------------------------------------------------------------
# _replace_number_words (standalone helper)
# ---------------------------------------------------------------------------


class TestReplaceNumberWords:
    def test_single_digit(self):
        assert _replace_number_words("three") == "3"

    def test_teen(self):
        assert _replace_number_words("thirteen") == "13"

    def test_tens(self):
        assert _replace_number_words("twenty") == "20"

    def test_in_context(self):
        assert _replace_number_words("remind me at three PM") == "remind me at 3 PM"

    def test_case_insensitive(self):
        assert _replace_number_words("FIVE") == "5"

    def test_no_match_unchanged(self):
        assert _replace_number_words("remind me later") == "remind me later"

    def test_hundred(self):
        assert _replace_number_words("one hundred") == "1 100"


# ---------------------------------------------------------------------------
# CleanResult
# ---------------------------------------------------------------------------


class TestCleanResult:
    def test_changed_true(self):
        r = CleanResult(original="um hello", cleaned="hello", changes=["fillers"])
        assert r.changed

    def test_changed_false(self):
        r = CleanResult(original="hello", cleaned="hello")
        assert not r.changed

    def test_str(self):
        r = CleanResult(original="um hello", cleaned="hello")
        assert str(r) == "hello"


# ---------------------------------------------------------------------------
# Filler removal
# ---------------------------------------------------------------------------


class TestFillerRemoval:
    @pytest.mark.parametrize("filler,query", [
        ("um", "um what time is it"),
        ("uh", "uh set a timer"),
        ("er", "er remind me"),
        ("ah", "ah hello"),
        ("hmm", "hmm okay"),
        ("like", "I like want to know"),
        ("you know", "you know what I mean"),
        ("i mean", "i mean remind me"),
        ("basically", "basically set a timer"),
    ])
    def test_filler_removed(self, filler, query):
        c = TranscriptCleaner()
        result = c.clean_str(query)
        assert filler.lower() not in result.lower()

    def test_filler_disabled(self):
        c = TranscriptCleaner(remove_fillers=False)
        assert c.clean_str("um hello") == "um hello"

    def test_filler_mid_sentence(self):
        c = TranscriptCleaner()
        result = c.clean_str("what um is the weather")
        assert "um" not in result
        assert "weather" in result

    def test_extra_fillers(self):
        c = TranscriptCleaner(extra_fillers=("blimey",))
        assert "blimey" not in c.clean_str("blimey that was fast")

    def test_filler_does_not_remove_substrings(self):
        # "um" should not strip inside "umbrella"
        c = TranscriptCleaner()
        result = c.clean_str("open umbrella")
        assert "umbrella" in result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_repeated_word(self):
        c = TranscriptCleaner()
        assert c.clean_str("I I want to go") == "I want to go"

    def test_repeated_word_case_insensitive(self):
        c = TranscriptCleaner()
        result = c.clean_str("the The weather")
        words = result.lower().split()
        assert words.count("the") == 1

    def test_triple_repeat_collapses(self):
        c = TranscriptCleaner()
        result = c.clean_str("go go go")
        assert result == "go"

    def test_no_false_dedup(self):
        c = TranscriptCleaner()
        assert c.clean_str("the cat sat") == "the cat sat"

    def test_dedup_disabled(self):
        c = TranscriptCleaner(deduplicate_words=False)
        assert c.clean_str("I I want") == "I I want"


# ---------------------------------------------------------------------------
# Number normalisation
# ---------------------------------------------------------------------------


class TestNumberNormalisation:
    def test_basic(self):
        c = TranscriptCleaner()
        assert "3" in c.clean_str("set a timer for three minutes")

    def test_disabled(self):
        c = TranscriptCleaner(normalize_numbers=False)
        result = c.clean_str("three minutes")
        assert "three" in result

    @pytest.mark.parametrize("word,digit", [
        ("one", "1"), ("five", "5"), ("ten", "10"),
        ("fifteen", "15"), ("twenty", "20"), ("fifty", "50"),
    ])
    def test_various_numbers(self, word, digit):
        c = TranscriptCleaner()
        assert digit in c.clean_str(f"I need {word} items")


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------


class TestWhitespaceNormalisation:
    def test_multiple_spaces_collapsed(self):
        c = TranscriptCleaner()
        assert c.clean_str("hello   world") == "hello world"

    def test_leading_trailing_stripped(self):
        c = TranscriptCleaner()
        assert c.clean_str("  hello  ") == "hello"

    def test_disabled(self):
        c = TranscriptCleaner(normalize_whitespace=False)
        assert c.clean_str("  hello  ") == "  hello  "


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_full_clean(self):
        c = TranscriptCleaner()
        raw = "uh I I want to set a timer for three minutes"
        result = c.clean_str(raw)
        assert "uh" not in result
        assert "I I" not in result
        assert "3" in result

    def test_clean_result_tracks_changes(self):
        c = TranscriptCleaner()
        r = c.clean("um um remind me at three PM")
        assert r.changed
        assert len(r.changes) > 0

    def test_already_clean_no_changes(self):
        c = TranscriptCleaner()
        r = c.clean("set a timer for 5 minutes")
        assert not r.changed

    def test_empty_string(self):
        c = TranscriptCleaner()
        assert c.clean_str("") == ""


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------


class TestCustomRules:
    def test_add_rule_applied(self):
        c = TranscriptCleaner()
        c.add_rule(r"\bhey claude\b", "", description="wake_word")
        result = c.clean_str("hey claude set a timer")
        assert "hey claude" not in result
        assert "set a timer" in result

    def test_custom_rule_runs_before_fillers(self):
        c = TranscriptCleaner()
        c.add_rule(r"\bsorry about that\b", "", description="apology")
        result = c.clean_str("sorry about that um remind me")
        assert "sorry about that" not in result
        assert "remind me" in result

    def test_remove_rule(self):
        c = TranscriptCleaner()
        c.add_rule(r"\btest\b", "", description="test_rule")
        c.remove_rule("test_rule")
        assert c.clean_str("test the app") == "test the app"

    def test_remove_rule_not_found_returns_false(self):
        c = TranscriptCleaner()
        assert not c.remove_rule("nonexistent")

    def test_rule_without_description_uses_pattern(self):
        c = TranscriptCleaner()
        c.add_rule(r"\bwhoops\b", "")
        r = c.clean("whoops set a timer")
        assert "whoops" not in r.cleaned
