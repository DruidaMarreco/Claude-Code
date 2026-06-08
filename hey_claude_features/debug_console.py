"""
Debug console for Hey-Claude's pipeline.

Displays a live snapshot of pipeline state in the terminal during
development: current stage, last transcript, intent tag, latency,
cost, circuit breaker states, and VAD energy.

No external dependencies — renders to plain text using ANSI codes so it
works in any terminal.  Can be completely disabled in production with a
single flag.

Output example
--------------
    ┌─ Hey-Claude Debug ──────────────────────────────────────────────────┐
    │ Stage : SPEAKING            VAD RMS : 0.0231 / threshold 0.0150     │
    │ Intent: command (regex)     Cost    : $0.0012  session $0.0087      │
    │ Latency stt=45ms gate=23ms opus=812ms tts=0ms  total=880ms          │
    │ Circuit: haiku=CLOSED  opus=CLOSED                                   │
    │ Transcript: "set a timer for five minutes"                           │
    └──────────────────────────────────────────────────────────────────────┘

Usage in app.py
---------------
    from hey_claude_features.debug_console import DebugConsole

    console = DebugConsole(enabled=True)

    # Update fields as the pipeline runs:
    console.update(stage="STT", vad_rms=0.023)
    console.update(transcript="set a timer for five minutes")
    console.update(intent="command", intent_method="regex")
    console.update(latency={"stt": 45, "gate": 23, "opus": 812})
    console.update(query_cost=0.0012, session_cost=0.0087)
    console.update(circuits={"haiku": "CLOSED", "opus": "CLOSED"})

    # Print the current snapshot:
    console.render()

    # Or clear terminal and re-render (live update):
    console.refresh()
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + _RESET


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ---------------------------------------------------------------------------
# PipelineState
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """Snapshot of pipeline state for one render cycle."""
    stage: str = "IDLE"
    transcript: str = ""
    intent: str = ""
    intent_method: str = ""
    intent_confidence: float = 0.0
    latency: dict[str, float] = field(default_factory=dict)   # stage → ms
    query_cost: float = 0.0
    session_cost: float = 0.0
    vad_rms: float = 0.0
    vad_threshold: float = 0.0
    circuits: dict[str, str] = field(default_factory=dict)    # name → state
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DebugConsole
# ---------------------------------------------------------------------------

_WIDTH = 72


class DebugConsole:
    """
    Renders a pipeline state snapshot to stdout.

    Parameters
    ----------
    enabled:
        If False, all methods are no-ops.
    color:
        Use ANSI color codes.  Auto-detected from tty by default.
    width:
        Box width in characters.
    """

    def __init__(
        self,
        enabled: bool = True,
        color: bool | None = None,
        width: int = _WIDTH,
    ) -> None:
        self._enabled = enabled
        self._color = color if color is not None else _supports_color()
        self._width = width
        self._state = PipelineState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, **kwargs: Any) -> None:
        """Update one or more fields in the pipeline state."""
        if not self._enabled:
            return
        for k, v in kwargs.items():
            if hasattr(self._state, k):
                setattr(self._state, k, v)
            else:
                self._state.extra[k] = v

    def render(self) -> str:
        """Build and return the rendered box as a string. Also prints it."""
        if not self._enabled:
            return ""
        text = self._build()
        print(text)
        return text

    def refresh(self) -> str:
        """Clear the terminal and re-render."""
        if not self._enabled:
            return ""
        if os.name == "nt":
            os.system("cls")
        else:
            print("\033[H\033[J", end="")
        return self.render()

    def reset(self) -> None:
        """Reset all state to defaults."""
        self._state = PipelineState()

    @property
    def state(self) -> PipelineState:
        return self._state

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _build(self) -> str:
        s = self._state
        w = self._width
        inner = w - 2   # space inside box borders

        lines: list[str] = []

        # Row 1: stage + VAD
        stage_color = _GREEN if s.stage == "SPEAKING" else _DIM
        stage_str = f"Stage : {s.stage:<12s}"
        if s.vad_rms or s.vad_threshold:
            vad_str = f"VAD : {s.vad_rms:.4f} / thr {s.vad_threshold:.4f}"
        else:
            vad_str = ""
        lines.append(self._row(stage_str, vad_str, inner, stage_color))

        # Row 2: intent + cost
        if s.intent:
            conf = f" ({s.intent_confidence:.0%})" if s.intent_confidence else ""
            intent_str = f"Intent: {s.intent}{conf} [{s.intent_method}]"
        else:
            intent_str = "Intent: —"
        cost_str = f"Cost: ${s.query_cost:.4f}  sess ${s.session_cost:.4f}" if s.query_cost else ""
        lines.append(self._row(intent_str, cost_str, inner))

        # Row 3: latency
        if s.latency:
            lat_parts = [f"{k}={v:.0f}ms" for k, v in s.latency.items()]
            lat_str = "Latency: " + "  ".join(lat_parts)
        else:
            lat_str = "Latency: —"
        lines.append(self._row(lat_str, "", inner))

        # Row 4: circuits
        if s.circuits:
            def _circuit_color(state: str) -> str:
                if not self._color:
                    return state
                if state == "CLOSED":
                    return _c(state, _GREEN)
                if state == "HALF_OPEN":
                    return _c(state, _YELLOW)
                return _c(state, _RED)

            circ_parts = [f"{n}={_circuit_color(st)}" for n, st in s.circuits.items()]
            circ_str = "Circuits: " + "  ".join(circ_parts)
        else:
            circ_str = "Circuits: —"
        lines.append(self._row(circ_str, "", inner))

        # Row 5: transcript
        if s.transcript:
            trunc = s.transcript[:inner - 14]
            if len(s.transcript) > inner - 14:
                trunc += "…"
            lines.append(self._row(f'Transcript: "{trunc}"', "", inner))

        # Extra fields
        for k, v in s.extra.items():
            lines.append(self._row(f"{k}: {v}", "", inner))

        # Box
        title = " Hey-Claude Debug "
        top = "┌─" + title + "─" * max(0, w - len(title) - 3) + "┐"
        bot = "└" + "─" * (w - 2) + "┘"

        box_lines = [top]
        for line in lines:
            box_lines.append("│" + line + "│")
        box_lines.append(bot)

        return "\n".join(box_lines)

    def _row(self, left: str, right: str, width: int, left_color: str = "") -> str:
        """Render a single row inside the box."""
        visible_left = self._strip_ansi(left)
        visible_right = self._strip_ansi(right)
        gap = width - len(visible_left) - len(visible_right)
        if gap < 1:
            # Truncate right side
            right = ""
            gap = width - len(visible_left)
        if self._color and left_color:
            left = left_color + left + _RESET

        return left + " " * max(1, gap) + right

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Return *text* with ANSI escape codes removed (for length calc)."""
        import re
        return re.sub(r"\033\[[0-9;]*m", "", text)
