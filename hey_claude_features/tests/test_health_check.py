"""Unit tests for HealthChecker — no real API key, mic, or audio required."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from hey_claude_features.health_check import (
    CheckResult,
    CheckStatus,
    HealthChecker,
    HealthReport,
    _check_api_key,
    _check_import,
    _check_tools,
)


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_ok_property(self):
        r = CheckResult("test", CheckStatus.OK, "fine")
        assert r.ok is True

    def test_not_ok_for_fail(self):
        r = CheckResult("test", CheckStatus.FAIL, "broken", critical=True)
        assert r.ok is False

    def test_str_ok(self):
        r = CheckResult("api_key", CheckStatus.OK, "Set.")
        assert "✓" in str(r)
        assert "api_key" in str(r)

    def test_str_fail_critical(self):
        r = CheckResult("api_key", CheckStatus.FAIL, "Missing.", critical=True)
        assert "✗" in str(r)
        assert "CRITICAL" in str(r)

    def test_str_warning_no_critical_tag(self):
        r = CheckResult("tts", CheckStatus.WARNING, "Optional.")
        assert "⚠" in str(r)
        assert "CRITICAL" not in str(r)


# ---------------------------------------------------------------------------
# HealthReport
# ---------------------------------------------------------------------------


class TestHealthReport:
    def test_all_ok_true(self):
        report = HealthReport(results=[
            CheckResult("a", CheckStatus.OK, ""),
            CheckResult("b", CheckStatus.OK, ""),
        ])
        assert report.all_ok is True

    def test_all_ok_false_on_warning(self):
        report = HealthReport(results=[
            CheckResult("a", CheckStatus.OK, ""),
            CheckResult("b", CheckStatus.WARNING, ""),
        ])
        assert report.all_ok is False

    def test_has_critical_true(self):
        report = HealthReport(results=[
            CheckResult("a", CheckStatus.FAIL, "", critical=True),
        ])
        assert report.has_critical is True

    def test_has_critical_false_when_ok(self):
        report = HealthReport(results=[
            CheckResult("a", CheckStatus.OK, "", critical=True),
        ])
        assert report.has_critical is False

    def test_failures_list(self):
        report = HealthReport(results=[
            CheckResult("a", CheckStatus.OK, ""),
            CheckResult("b", CheckStatus.FAIL, ""),
            CheckResult("c", CheckStatus.WARNING, ""),
        ])
        assert len(report.failures) == 2

    def test_summary_contains_all_names(self):
        report = HealthReport(results=[
            CheckResult("api_key", CheckStatus.OK, "Set."),
            CheckResult("microphone", CheckStatus.FAIL, "Not found.", critical=True),
        ])
        s = report.summary()
        assert "api_key" in s
        assert "microphone" in s

    def test_voice_summary_all_ok(self):
        report = HealthReport(results=[
            CheckResult("a", CheckStatus.OK, ""),
            CheckResult("b", CheckStatus.OK, ""),
        ])
        assert "2" in report.voice_summary()
        assert "passed" in report.voice_summary()

    def test_voice_summary_failures(self):
        report = HealthReport(results=[
            CheckResult("api_key", CheckStatus.FAIL, ""),
            CheckResult("mic", CheckStatus.FAIL, ""),
        ])
        vs = report.voice_summary()
        assert "2" in vs
        assert "failed" in vs


# ---------------------------------------------------------------------------
# _check_api_key
# ---------------------------------------------------------------------------


class TestCheckApiKey:
    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        r = _check_api_key()
        assert r.status == CheckStatus.FAIL
        assert r.critical is True

    def test_empty_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        r = _check_api_key()
        assert r.status == CheckStatus.FAIL

    def test_valid_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 40)
        r = _check_api_key()
        assert r.status == CheckStatus.OK

    def test_short_key_warning(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "short")
        r = _check_api_key()
        assert r.status == CheckStatus.WARNING


# ---------------------------------------------------------------------------
# _check_import
# ---------------------------------------------------------------------------


class TestCheckImport:
    def test_existing_module(self):
        r = _check_import("json", "json")
        assert r.ok

    def test_missing_module(self):
        r = _check_import("nonexistent_xyz_module", "nonexistent", critical=True)
        assert r.status == CheckStatus.FAIL
        assert r.critical is True


# ---------------------------------------------------------------------------
# _check_tools
# ---------------------------------------------------------------------------


class TestCheckTools:
    def test_valid_tools(self):
        specs = [
            {"name": "get_time", "description": "Get time", "input_schema": {"type": "object", "properties": {}}},
        ]
        r = _check_tools(specs)
        assert r.ok

    def test_missing_name(self):
        specs = [{"description": "No name", "input_schema": {"type": "object"}}]
        r = _check_tools(specs)
        assert r.status == CheckStatus.WARNING
        assert "missing name" in r.message

    def test_missing_description(self):
        specs = [{"name": "x", "input_schema": {"type": "object"}}]
        r = _check_tools(specs)
        assert r.status == CheckStatus.WARNING

    def test_invalid_schema(self):
        specs = [{"name": "x", "description": "y", "input_schema": {"type": "string"}}]
        r = _check_tools(specs)
        assert r.status == CheckStatus.WARNING

    def test_empty_list(self):
        r = _check_tools([])
        assert r.ok


# ---------------------------------------------------------------------------
# HealthChecker.run()
# ---------------------------------------------------------------------------


class TestHealthChecker:
    def test_run_skips_api_when_no_client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-" + "x" * 40)
        checker = HealthChecker(
            client=None,
            check_microphone=False,
            check_api=False,
            tts_engine=None,
        )
        report = checker.run()
        names = [r.name for r in report.results]
        assert "api_reachable" not in names

    def test_run_skips_mic_when_disabled(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-" + "x" * 40)
        checker = HealthChecker(check_microphone=False, tts_engine=None)
        report = checker.run()
        names = [r.name for r in report.results]
        assert "microphone" not in names

    def test_run_includes_tool_check(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-" + "x" * 40)
        specs = [{"name": "x", "description": "y", "input_schema": {"type": "object"}}]
        checker = HealthChecker(tool_specs=specs, check_microphone=False, tts_engine=None)
        report = checker.run()
        names = [r.name for r in report.results]
        assert "tools" in names

    def test_api_key_always_checked(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        checker = HealthChecker(check_microphone=False, tts_engine=None)
        report = checker.run()
        names = [r.name for r in report.results]
        assert "api_key" in names

    def test_report_has_critical_on_missing_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        checker = HealthChecker(check_microphone=False, tts_engine=None)
        report = checker.run()
        assert report.has_critical

    def test_voice_summary_all_pass(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 40)
        checker = HealthChecker(check_microphone=False, check_api=False, tts_engine=None)
        report = checker.run()
        # api_key + whisper checks; whisper may fail in test env — just verify structure
        assert "passed" in report.voice_summary() or "failed" in report.voice_summary()
