"""Unit tests for CommandShortcuts — no API key required."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hey_claude_features.command_shortcuts import (
    CommandShortcuts,
    Shortcut,
    ShortcutResult,
    _fuzzy_match,
    _normalise,
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_lowercases(self):
        assert _normalise("STOP") == "stop"

    def test_strips_punctuation(self):
        assert _normalise("stop!") == "stop"

    def test_strips_whitespace(self):
        assert _normalise("  volume up  ") == "volume up"

    def test_contraction_stripped(self):
        assert _normalise("never mind.") == "never mind"


class TestFuzzyMatch:
    def test_identical(self):
        assert _fuzzy_match("stop", "stop") == 1.0

    def test_partial_overlap(self):
        score = _fuzzy_match("volume up please", "volume up")
        assert score > 0.5

    def test_no_overlap(self):
        assert _fuzzy_match("stop", "play music") == 0.0

    def test_empty_strings(self):
        assert _fuzzy_match("", "stop") == 0.0


# ---------------------------------------------------------------------------
# Shortcut matching
# ---------------------------------------------------------------------------


class TestShortcutMatches:
    def test_exact_match(self):
        sc = Shortcut(name="stop", triggers=["stop"], voice_reply="Stopping.")
        assert sc.matches("stop")

    def test_no_match(self):
        sc = Shortcut(name="stop", triggers=["stop"], voice_reply="Stopping.")
        assert not sc.matches("play music")

    def test_fuzzy_match_close(self):
        sc = Shortcut(name="stop", triggers=["stop now"], voice_reply=".", fuzzy_threshold=0.5)
        assert sc.matches("stop")

    def test_fuzzy_below_threshold(self):
        sc = Shortcut(name="stop", triggers=["stop"], voice_reply=".", fuzzy_threshold=0.99)
        assert not sc.matches("stopping")


# ---------------------------------------------------------------------------
# ShortcutResult
# ---------------------------------------------------------------------------


class TestShortcutResult:
    def test_miss(self):
        r = ShortcutResult.miss()
        assert not r.matched

    def test_execute_calls_callback(self):
        cb = MagicMock()
        sc = Shortcut(name="x", triggers=["x"], voice_reply=".", callback=cb)
        r = ShortcutResult(matched=True, shortcut=sc, voice_reply=".")
        r.execute()
        cb.assert_called_once()

    def test_execute_no_callback_no_error(self):
        sc = Shortcut(name="x", triggers=["x"], voice_reply=".")
        r = ShortcutResult(matched=True, shortcut=sc, voice_reply=".")
        r.execute()  # should not raise

    def test_execute_callback_exception_swallowed(self):
        sc = Shortcut(name="x", triggers=["x"], voice_reply=".", callback=MagicMock(side_effect=RuntimeError))
        r = ShortcutResult(matched=True, shortcut=sc, voice_reply=".")
        r.execute()  # should not raise


# ---------------------------------------------------------------------------
# CommandShortcuts built-ins
# ---------------------------------------------------------------------------


class TestBuiltins:
    def test_five_builtins_loaded(self):
        sc = CommandShortcuts()
        assert len(sc) == 5

    @pytest.mark.parametrize("phrase,name", [
        ("stop", "stop"),
        ("cancel", "stop"),
        ("never mind", "stop"),
        ("mute", "mute"),
        ("be quiet", "mute"),
        ("repeat", "repeat"),
        ("say that again", "repeat"),
        ("volume up", "volume_up"),
        ("louder", "volume_up"),
        ("volume down", "volume_down"),
        ("quieter", "volume_down"),
    ])
    def test_builtin_exact(self, phrase, name):
        sc = CommandShortcuts()
        r = sc.match(phrase)
        assert r.matched
        assert r.shortcut.name == name

    def test_punctuation_stripped_before_match(self):
        sc = CommandShortcuts()
        r = sc.match("stop!")
        assert r.matched

    def test_case_insensitive(self):
        sc = CommandShortcuts()
        r = sc.match("STOP")
        assert r.matched

    def test_no_match_returns_miss(self):
        sc = CommandShortcuts()
        r = sc.match("what is the weather today")
        assert not r.matched

    def test_empty_transcript_misses(self):
        sc = CommandShortcuts()
        assert not sc.match("").matched


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_on_stop(self):
        cb = MagicMock()
        sc = CommandShortcuts()
        sc.on_stop(cb)
        r = sc.match("stop")
        r.execute()
        cb.assert_called_once()

    def test_on_mute(self):
        cb = MagicMock()
        sc = CommandShortcuts()
        sc.on_mute(cb)
        sc.match("mute").execute()
        cb.assert_called_once()

    def test_on_repeat(self):
        cb = MagicMock()
        sc = CommandShortcuts()
        sc.on_repeat(cb)
        sc.match("repeat").execute()
        cb.assert_called_once()

    def test_on_volume_up(self):
        cb = MagicMock()
        sc = CommandShortcuts()
        sc.on_volume_up(cb)
        sc.match("louder").execute()
        cb.assert_called_once()

    def test_on_volume_down(self):
        cb = MagicMock()
        sc = CommandShortcuts()
        sc.on_volume_down(cb)
        sc.match("volume down").execute()
        cb.assert_called_once()

    def test_voice_reply_correct(self):
        sc = CommandShortcuts()
        r = sc.match("stop")
        assert "Stopping" in r.voice_reply


# ---------------------------------------------------------------------------
# Custom shortcuts
# ---------------------------------------------------------------------------


class TestCustomShortcuts:
    def test_add_and_match(self):
        sc = CommandShortcuts()
        cb = MagicMock()
        sc.add("lights_on", triggers=["lights on", "turn on the lights"], voice_reply="Lights on.", callback=cb, persist=False)
        r = sc.match("lights on")
        assert r.matched
        r.execute()
        cb.assert_called_once()

    def test_add_without_triggers_uses_name(self):
        sc = CommandShortcuts()
        sc.add("pause", voice_reply="Paused.", persist=False)
        assert sc.match("pause").matched

    def test_remove_existing(self):
        sc = CommandShortcuts()
        sc.add("custom", persist=False)
        assert sc.remove("custom")
        assert not sc.match("custom").matched

    def test_remove_nonexistent_returns_false(self):
        sc = CommandShortcuts()
        assert not sc.remove("nonexistent")

    def test_list_names_includes_custom(self):
        sc = CommandShortcuts()
        sc.add("foo", persist=False)
        assert "foo" in sc.list_names()

    def test_get_returns_shortcut(self):
        sc = CommandShortcuts()
        assert sc.get("stop") is not None

    def test_get_missing_returns_none(self):
        sc = CommandShortcuts()
        assert sc.get("ghost") is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_custom_shortcut_persisted(self, tmp_path):
        path = tmp_path / "shortcuts.json"
        sc1 = CommandShortcuts(persist_path=path)
        sc1.add("disco", triggers=["disco mode"], voice_reply="Disco!", persist=True)

        sc2 = CommandShortcuts(persist_path=path)
        assert sc2.match("disco mode").matched

    def test_builtin_not_duplicated_after_load(self, tmp_path):
        path = tmp_path / "shortcuts.json"
        sc1 = CommandShortcuts(persist_path=path)
        sc1.add("custom1", persist=True)
        sc2 = CommandShortcuts(persist_path=path)
        # Builtins + 1 custom = 6 total
        assert len(sc2) == 6

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "shortcuts.json"
        path.write_text("{bad json")
        sc = CommandShortcuts(persist_path=path)
        assert len(sc) == 5  # only builtins

    def test_remove_persists(self, tmp_path):
        path = tmp_path / "shortcuts.json"
        sc1 = CommandShortcuts(persist_path=path)
        sc1.add("temp", persist=True)
        sc1.remove("temp")

        sc2 = CommandShortcuts(persist_path=path)
        assert not sc2.match("temp").matched


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


class TestIteration:
    def test_iter_yields_shortcuts(self):
        sc = CommandShortcuts()
        names = [s.name for s in sc]
        assert "stop" in names
        assert "mute" in names
