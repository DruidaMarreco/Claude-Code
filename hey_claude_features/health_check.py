"""
Startup health checker for Hey-Claude.

Runs a set of lightweight diagnostic checks before the main loop starts
and reports any problems clearly — so users know immediately if something
is misconfigured rather than getting cryptic errors mid-conversation.

Checks performed
----------------
- api_key       : ANTHROPIC_API_KEY env var is set and non-empty
- api_reachable : A minimal Haiku API call succeeds (verifies network + key)
- microphone    : sounddevice can list input devices
- whisper       : whisper module is importable
- tts           : pyttsx3 or piper-tts is importable (based on config)
- tools         : All registered tools have valid name/description/schema

Usage in app.py
---------------
    from hey_claude_features.health_check import HealthChecker, CheckStatus

    checker = HealthChecker(client=anthropic_client, config=config)
    report = checker.run()

    if not report.all_ok:
        print(report.summary())
        if report.has_critical:
            sys.exit(1)
    else:
        print("All systems go.")

Or for a voice-ready string:
    speak(report.voice_summary())
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class CheckStatus(Enum):
    OK = auto()
    WARNING = auto()
    FAIL = auto()


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    critical: bool = False

    @property
    def ok(self) -> bool:
        return self.status == CheckStatus.OK

    def __str__(self) -> str:
        icon = {"OK": "✓", "WARNING": "⚠", "FAIL": "✗"}[self.status.name]
        crit = " [CRITICAL]" if self.critical and not self.ok else ""
        return f"{icon} {self.name}: {self.message}{crit}"


@dataclass
class HealthReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def has_critical(self) -> bool:
        return any(r.critical and not r.ok for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.ok]

    def summary(self) -> str:
        lines = ["Hey-Claude Health Check", "-" * 30]
        for r in self.results:
            lines.append(str(r))
        passed = sum(1 for r in self.results if r.ok)
        lines.append(f"\n{passed}/{len(self.results)} checks passed")
        return "\n".join(lines)

    def voice_summary(self) -> str:
        """Compact voice-friendly summary."""
        if self.all_ok:
            return f"All {len(self.results)} startup checks passed."
        failures = self.failures
        names = ", ".join(r.name for r in failures)
        return f"{len(failures)} startup check{'s' if len(failures) != 1 else ''} failed: {names}."


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def _check_api_key() -> CheckResult:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return CheckResult(
            name="api_key",
            status=CheckStatus.FAIL,
            message="ANTHROPIC_API_KEY is not set.",
            critical=True,
        )
    if len(key) < 20:
        return CheckResult(
            name="api_key",
            status=CheckStatus.WARNING,
            message="ANTHROPIC_API_KEY looks unusually short.",
        )
    return CheckResult(name="api_key", status=CheckStatus.OK, message="Set.")


def _check_api_reachable(client: Any) -> CheckResult:
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return CheckResult(
            name="api_reachable",
            status=CheckStatus.OK,
            message=f"Haiku responded (stop_reason={resp.stop_reason}).",
        )
    except Exception as exc:
        return CheckResult(
            name="api_reachable",
            status=CheckStatus.FAIL,
            message=f"API call failed: {exc}",
            critical=True,
        )


def _check_import(module: str, display_name: str, critical: bool = False) -> CheckResult:
    try:
        importlib.import_module(module)
        return CheckResult(
            name=display_name,
            status=CheckStatus.OK,
            message=f"{module} importable.",
        )
    except ImportError:
        return CheckResult(
            name=display_name,
            status=CheckStatus.FAIL,
            message=f"{module} not installed. Run: pip install {module}",
            critical=critical,
        )


def _check_microphone() -> CheckResult:
    try:
        import sounddevice as sd  # type: ignore[import]
        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        if not inputs:
            return CheckResult(
                name="microphone",
                status=CheckStatus.FAIL,
                message="No input devices found.",
                critical=True,
            )
        names = [d["name"] for d in inputs[:3]]
        return CheckResult(
            name="microphone",
            status=CheckStatus.OK,
            message=f"{len(inputs)} input device(s): {', '.join(names)}",
        )
    except ImportError:
        return CheckResult(
            name="microphone",
            status=CheckStatus.WARNING,
            message="sounddevice not installed; microphone check skipped.",
        )
    except Exception as exc:
        return CheckResult(
            name="microphone",
            status=CheckStatus.FAIL,
            message=f"sounddevice error: {exc}",
            critical=True,
        )


def _check_tools(tool_specs: list[dict[str, Any]]) -> CheckResult:
    issues = []
    for i, spec in enumerate(tool_specs):
        if not spec.get("name"):
            issues.append(f"tool[{i}] missing name")
        if not spec.get("description"):
            issues.append(f"tool[{i}] missing description")
        schema = spec.get("input_schema", {})
        if not isinstance(schema, dict) or schema.get("type") != "object":
            issues.append(f"tool '{spec.get('name', i)}' has invalid input_schema")
    if issues:
        return CheckResult(
            name="tools",
            status=CheckStatus.WARNING,
            message="; ".join(issues),
        )
    return CheckResult(
        name="tools",
        status=CheckStatus.OK,
        message=f"{len(tool_specs)} tool(s) validated.",
    )


# ---------------------------------------------------------------------------
# HealthChecker
# ---------------------------------------------------------------------------


class HealthChecker:
    """
    Runs startup diagnostics and returns a HealthReport.

    Parameters
    ----------
    client:
        Anthropic client.  If None, the api_reachable check is skipped.
    tool_specs:
        List of tool descriptor dicts to validate.
    tts_engine:
        "pyttsx3", "piper", or None to skip TTS check.
    check_microphone:
        Set False to skip mic check (e.g. in --text mode).
    check_api:
        Set False to skip the live API ping (saves a Haiku call).
    """

    def __init__(
        self,
        client: Any | None = None,
        tool_specs: list[dict[str, Any]] | None = None,
        tts_engine: str | None = "pyttsx3",
        check_microphone: bool = True,
        check_api: bool = True,
    ) -> None:
        self._client = client
        self._tool_specs = tool_specs or []
        self._tts_engine = tts_engine
        self._check_microphone = check_microphone
        self._check_api = check_api

    def run(self) -> HealthReport:
        """Execute all checks and return a HealthReport."""
        report = HealthReport()

        # Always run these
        report.results.append(_check_api_key())

        if self._check_api and self._client is not None:
            report.results.append(_check_api_reachable(self._client))

        if self._check_microphone:
            report.results.append(_check_microphone())

        report.results.append(_check_import("whisper", "whisper"))

        if self._tts_engine == "pyttsx3":
            report.results.append(_check_import("pyttsx3", "tts(pyttsx3)"))
        elif self._tts_engine == "piper":
            report.results.append(_check_import("piper", "tts(piper)"))

        if self._tool_specs:
            report.results.append(_check_tools(self._tool_specs))

        for r in report.results:
            level = logging.INFO if r.ok else (logging.ERROR if r.critical else logging.WARNING)
            logger.log(level, "Health check [%s]: %s", r.name, r.message)

        return report
