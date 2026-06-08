"""Unit tests for DebugConsole — no API key or terminal required."""

from __future__ import annotations

import pytest

from hey_claude_features.debug_console import DebugConsole, PipelineState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _console(**kw) -> DebugConsole:
    defaults = dict(enabled=True, color=False)
    defaults.update(kw)
    return DebugConsole(**defaults)


# ---------------------------------------------------------------------------
# PipelineState defaults
# ---------------------------------------------------------------------------


class TestPipelineState:
    def test_default_stage(self):
        assert PipelineState().stage == "IDLE"

    def test_default_transcript_empty(self):
        assert PipelineState().transcript == ""

    def test_default_latency_empty(self):
        assert PipelineState().latency == {}

    def test_default_circuits_empty(self):
        assert PipelineState().circuits == {}


# ---------------------------------------------------------------------------
# DebugConsole.update()
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_stage_updated(self):
        c = _console()
        c.update(stage="SPEAKING")
        assert c.state.stage == "SPEAKING"

    def test_transcript_updated(self):
        c = _console()
        c.update(transcript="hello world")
        assert c.state.transcript == "hello world"

    def test_latency_updated(self):
        c = _console()
        c.update(latency={"stt": 45, "opus": 800})
        assert c.state.latency["stt"] == 45

    def test_circuits_updated(self):
        c = _console()
        c.update(circuits={"haiku": "CLOSED"})
        assert c.state.circuits["haiku"] == "CLOSED"

    def test_unknown_key_stored_in_extra(self):
        c = _console()
        c.update(my_custom_field="hello")
        assert c.state.extra["my_custom_field"] == "hello"

    def test_disabled_update_is_noop(self):
        c = _console(enabled=False)
        c.update(stage="SPEAKING")
        assert c.state.stage == "IDLE"   # unchanged

    def test_multiple_fields_at_once(self):
        c = _console()
        c.update(stage="SPEAKING", transcript="hi", query_cost=0.001)
        assert c.state.stage == "SPEAKING"
        assert c.state.transcript == "hi"
        assert c.state.query_cost == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# DebugConsole.render()
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_returns_string(self):
        c = _console()
        result = c.render()
        assert isinstance(result, str)

    def test_render_contains_box_chars(self):
        c = _console()
        result = c.render()
        assert "┌" in result
        assert "┘" in result

    def test_render_contains_title(self):
        c = _console()
        result = c.render()
        assert "Hey-Claude Debug" in result

    def test_render_contains_stage(self):
        c = _console()
        c.update(stage="SPEAKING")
        assert "SPEAKING" in c.render()

    def test_render_contains_transcript(self):
        c = _console()
        c.update(transcript="set a timer")
        assert "set a timer" in c.render()

    def test_render_contains_intent(self):
        c = _console()
        c.update(intent="command", intent_method="regex")
        result = c.render()
        assert "command" in result
        assert "regex" in result

    def test_render_contains_latency(self):
        c = _console()
        c.update(latency={"stt": 45, "opus": 812})
        result = c.render()
        assert "45ms" in result
        assert "812ms" in result

    def test_render_contains_cost(self):
        c = _console()
        c.update(query_cost=0.0012, session_cost=0.0087)
        result = c.render()
        assert "0.0012" in result
        assert "0.0087" in result

    def test_render_contains_circuits(self):
        c = _console()
        c.update(circuits={"haiku": "CLOSED", "opus": "OPEN"})
        result = c.render()
        assert "haiku" in result
        assert "OPEN" in result

    def test_render_contains_vad(self):
        c = _console()
        c.update(vad_rms=0.0231, vad_threshold=0.015)
        result = c.render()
        assert "0.0231" in result

    def test_render_extra_fields(self):
        c = _console()
        c.update(my_debug="test_value")
        assert "test_value" in c.render()

    def test_disabled_render_empty(self):
        c = _console(enabled=False)
        assert c.render() == ""

    def test_long_transcript_truncated(self):
        c = _console()
        c.update(transcript="x" * 200)
        result = c.render()
        assert "…" in result

    def test_render_no_crash_all_defaults(self):
        c = _console()
        c.render()   # should not raise


# ---------------------------------------------------------------------------
# DebugConsole.reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self):
        c = _console()
        c.update(stage="SPEAKING", transcript="hello")
        c.reset()
        assert c.state.stage == "IDLE"
        assert c.state.transcript == ""

    def test_reset_clears_extra(self):
        c = _console()
        c.update(my_field="value")
        c.reset()
        assert c.state.extra == {}


# ---------------------------------------------------------------------------
# _strip_ansi
# ---------------------------------------------------------------------------


class TestStripAnsi:
    def test_strips_color_codes(self):
        text = "\033[32mhello\033[0m"
        assert DebugConsole._strip_ansi(text) == "hello"

    def test_plain_text_unchanged(self):
        assert DebugConsole._strip_ansi("hello world") == "hello world"


# ---------------------------------------------------------------------------
# Color disabled (default in tests)
# ---------------------------------------------------------------------------


class TestColorDisabled:
    def test_no_escape_codes_in_output(self):
        c = _console(color=False)
        c.update(circuits={"haiku": "OPEN"})
        result = c.render()
        assert "\033[" not in result
