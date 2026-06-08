"""Unit tests for PromptBuilder — no API key required."""

from __future__ import annotations

import datetime

import pytest

from hey_claude_features.prompt_builder import PromptBuilder, ToolSpec, _VOICE_HINTS


# ---------------------------------------------------------------------------
# Fixed datetime for deterministic tests
# ---------------------------------------------------------------------------

_FIXED_DT = datetime.datetime(2025, 6, 9, 14, 32, 0)


def _builder(**kw) -> PromptBuilder:
    defaults = dict(datetime_fn=lambda: _FIXED_DT)
    defaults.update(kw)
    return PromptBuilder(**defaults)


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_str_format(self):
        t = ToolSpec(name="set_reminder", description="Sets a reminder")
        assert "set_reminder" in str(t)
        assert "Sets a reminder" in str(t)


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------


class TestPersona:
    def test_default_persona_in_output(self):
        b = _builder()
        assert "Hey-Claude" in b.build()

    def test_custom_persona(self):
        b = _builder(persona="You are a pirate assistant.")
        assert "pirate" in b.build()

    def test_set_persona_updates(self):
        b = _builder()
        b.set_persona("You are a chef assistant.")
        assert "chef" in b.build()

    def test_empty_persona_uses_default(self):
        b = _builder(persona="")
        assert "Hey-Claude" in b.build()


# ---------------------------------------------------------------------------
# Date/time
# ---------------------------------------------------------------------------


class TestDatetime:
    def test_datetime_included_by_default(self):
        b = _builder()
        assert "2025-06-09" in b.build()
        assert "14:32" in b.build()

    def test_datetime_disabled(self):
        b = _builder(include_datetime=False)
        assert "2025" not in b.build()

    def test_day_of_week_included(self):
        b = _builder()
        assert "Monday" in b.build()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class TestTools:
    def test_no_tools_no_capabilities_section(self):
        b = _builder()
        assert "capabilities" not in b.build()

    def test_added_tool_in_output(self):
        b = _builder()
        b.add_tool("set_reminder", "Set a timed reminder")
        result = b.build()
        assert "set_reminder" in result
        assert "Set a timed reminder" in result

    def test_multiple_tools(self):
        b = _builder()
        b.add_tool("search_notes", "Search conversation notes")
        b.add_tool("set_reminder", "Set a reminder")
        result = b.build()
        assert "search_notes" in result
        assert "set_reminder" in result

    def test_remove_tool(self):
        b = _builder()
        b.add_tool("search_notes", "Search")
        b.remove_tool("search_notes")
        assert "search_notes" not in b.build()

    def test_remove_nonexistent_returns_false(self):
        b = _builder()
        assert not b.remove_tool("ghost")

    def test_len_counts_tools(self):
        b = _builder()
        b.add_tool("a", "A")
        b.add_tool("b", "B")
        assert len(b) == 2


# ---------------------------------------------------------------------------
# One-shot instructions
# ---------------------------------------------------------------------------


class TestInstructions:
    def test_instruction_in_output(self):
        b = _builder()
        b.add_instruction("The user mentioned a dentist appointment today.")
        assert "dentist" in b.build()

    def test_multiple_instructions(self):
        b = _builder()
        b.add_instruction("Reminder: meeting at 3pm.")
        b.add_instruction("User prefers metric units.")
        result = b.build()
        assert "meeting" in result
        assert "metric" in result

    def test_clear_instructions(self):
        b = _builder()
        b.add_instruction("temporary note")
        b.clear_instructions()
        assert "temporary note" not in b.build()

    def test_empty_instruction_ignored(self):
        b = _builder()
        b.add_instruction("")
        b.add_instruction("   ")
        b.add_instruction("real instruction")
        assert b._instructions == ["real instruction"]


# ---------------------------------------------------------------------------
# Custom sections
# ---------------------------------------------------------------------------


class TestCustomSections:
    def test_custom_section_in_output(self):
        b = _builder()
        b.add_section("User preferences", "Prefers short answers.")
        assert "User preferences" in b.build()
        assert "Prefers short answers" in b.build()

    def test_remove_section(self):
        b = _builder()
        b.add_section("debug", "some debug info")
        b.remove_section("debug")
        assert "debug" not in b.build()

    def test_remove_nonexistent_returns_false(self):
        b = _builder()
        assert not b.remove_section("ghost")

    def test_empty_section_omitted(self):
        b = _builder()
        b.add_section("empty", "")
        assert "empty" not in b.build()


# ---------------------------------------------------------------------------
# Voice hints
# ---------------------------------------------------------------------------


class TestVoiceHints:
    def test_voice_hints_enabled_by_default(self):
        b = _builder()
        assert "markdown" in b.build().lower()

    def test_voice_hints_disabled(self):
        b = _builder(voice_hints=False)
        assert "markdown" not in b.build().lower()

    def test_voice_hints_last_section(self):
        b = _builder()
        result = b.build()
        # Voice hints should appear after persona content
        persona_pos = result.find("Hey-Claude")
        hints_pos = result.find("markdown")
        assert hints_pos > persona_pos


# ---------------------------------------------------------------------------
# build() output structure
# ---------------------------------------------------------------------------


class TestBuild:
    def test_sections_separated_by_blank_lines(self):
        b = _builder()
        b.add_tool("t", "desc")
        result = b.build()
        assert "\n\n" in result

    def test_no_empty_sections(self):
        b = _builder(include_datetime=False, voice_hints=False)
        result = b.build()
        # No trailing blank lines or doubled blanks
        assert "\n\n\n" not in result

    def test_returns_string(self):
        assert isinstance(_builder().build(), str)

    def test_non_empty_output(self):
        assert len(_builder().build()) > 0


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_keys(self):
        b = _builder()
        s = b.snapshot()
        assert "persona" in s
        assert "tools" in s
        assert "instructions" in s

    def test_snapshot_tools_list(self):
        b = _builder()
        b.add_tool("t1", "desc1")
        s = b.snapshot()
        assert s["tools"][0]["name"] == "t1"

    def test_snapshot_datetime_string(self):
        b = _builder()
        s = b.snapshot()
        assert "2025-06-09" in s["datetime"]

    def test_snapshot_datetime_none_when_disabled(self):
        b = _builder(include_datetime=False)
        assert b.snapshot()["datetime"] is None
