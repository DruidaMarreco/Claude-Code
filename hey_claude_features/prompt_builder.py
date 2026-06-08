"""
Dynamic system prompt builder for Hey-Claude.

Assembles the Anthropic ``system`` parameter from multiple sources so
app.py never has to do manual string concatenation:

    - Persona base prompt  (from PersonaSwitcher or a plain string)
    - Date/time context    ("Today is Monday 2025-06-09, 14:32 local time")
    - Tool catalogue       (list of available tools / capabilities)
    - Custom instructions  (injected at runtime, e.g. from reminders)
    - Format hints         (voice-friendly: short sentences, no markdown)

Sections are separated by blank lines and rendered in a consistent order.
Empty sections are omitted automatically.

Usage in app.py
---------------
    from hey_claude_features.prompt_builder import PromptBuilder

    builder = PromptBuilder(
        persona="You are Hey-Claude, a helpful voice assistant.",
        voice_hints=True,
    )

    # Add tool capabilities:
    builder.add_tool("set_reminder", "Set a reminder for a specific time")
    builder.add_tool("search_notes", "Search past conversation notes")

    # Add a one-shot instruction (e.g. from reminders):
    builder.add_instruction("The user asked to be reminded about the dentist today.")

    # Build and pass to API:
    system_prompt = builder.build()
    response = client.messages.create(
        model="claude-opus-4-8",
        system=system_prompt,
        messages=history.messages,
        ...
    )

    # Clear one-shot instructions after each query:
    builder.clear_instructions()

Snapshot
--------
    builder.snapshot()  → dict with each section's content (useful for debug)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    name: str
    description: str

    def __str__(self) -> str:
        return f"- {self.name}: {self.description}"


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

_VOICE_HINTS = (
    "Speak naturally in short sentences. "
    "Do not use markdown, bullet points, or headers — your reply will be read aloud. "
    "Be concise: one or two sentences unless more depth is clearly needed."
)


class PromptBuilder:
    """
    Builds the system prompt by composing named sections.

    Parameters
    ----------
    persona:
        Base persona string.  Defaults to a minimal Hey-Claude identity.
    voice_hints:
        If True, appends a voice-formatting reminder.
    include_datetime:
        If True, injects the current date/time into the prompt.
    datetime_fn:
        Callable returning a ``datetime.datetime``; defaults to
        ``datetime.datetime.now``.  Inject a fixed value in tests.
    """

    _DEFAULT_PERSONA = (
        "You are Hey-Claude, a helpful, friendly voice assistant. "
        "You answer questions, set reminders, search notes, and handle everyday tasks."
    )

    def __init__(
        self,
        persona: str = "",
        voice_hints: bool = True,
        include_datetime: bool = True,
        datetime_fn: Any = None,
    ) -> None:
        self._persona = persona or self._DEFAULT_PERSONA
        self._voice_hints = voice_hints
        self._include_datetime = include_datetime
        self._datetime_fn = datetime_fn or datetime.datetime.now
        self._tools: list[ToolSpec] = []
        self._instructions: list[str] = []
        self._custom_sections: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API — configuration
    # ------------------------------------------------------------------

    def set_persona(self, persona: str) -> None:
        self._persona = persona

    def add_tool(self, name: str, description: str) -> None:
        """Register a tool/capability the assistant can use."""
        self._tools.append(ToolSpec(name=name, description=description))

    def remove_tool(self, name: str) -> bool:
        before = len(self._tools)
        self._tools = [t for t in self._tools if t.name != name]
        return len(self._tools) < before

    def add_instruction(self, instruction: str) -> None:
        """Add a one-shot instruction for the next query."""
        if instruction.strip():
            self._instructions.append(instruction.strip())

    def clear_instructions(self) -> None:
        """Remove all one-shot instructions (call after each query)."""
        self._instructions.clear()

    def add_section(self, title: str, content: str) -> None:
        """Add a custom named section."""
        self._custom_sections[title] = content

    def remove_section(self, title: str) -> bool:
        existed = title in self._custom_sections
        self._custom_sections.pop(title, None)
        return existed

    # ------------------------------------------------------------------
    # Public API — building
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Assemble and return the full system prompt string."""
        sections: list[str] = []

        # 1. Persona
        sections.append(self._persona.strip())

        # 2. Date/time
        if self._include_datetime:
            dt = self._datetime_fn()
            dt_str = dt.strftime("%A %Y-%m-%d, %H:%M")
            sections.append(f"Current date and time: {dt_str}.")

        # 3. Available tools
        if self._tools:
            tool_lines = ["You have access to the following capabilities:"]
            tool_lines.extend(str(t) for t in self._tools)
            sections.append("\n".join(tool_lines))

        # 4. One-shot instructions
        if self._instructions:
            instr_block = "Additional context for this query:\n" + "\n".join(
                f"- {i}" for i in self._instructions
            )
            sections.append(instr_block)

        # 5. Custom sections
        for title, content in self._custom_sections.items():
            if content.strip():
                sections.append(f"{title}:\n{content.strip()}")

        # 6. Voice hints (always last)
        if self._voice_hints:
            sections.append(_VOICE_HINTS)

        return "\n\n".join(s for s in sections if s.strip())

    def snapshot(self) -> dict[str, Any]:
        """Return a dict representation of all sections (for debugging)."""
        dt = self._datetime_fn() if self._include_datetime else None
        return {
            "persona": self._persona,
            "datetime": dt.strftime("%A %Y-%m-%d, %H:%M") if dt else None,
            "tools": [{"name": t.name, "description": t.description} for t in self._tools],
            "instructions": list(self._instructions),
            "custom_sections": dict(self._custom_sections),
            "voice_hints": _VOICE_HINTS if self._voice_hints else None,
        }

    def __len__(self) -> int:
        """Number of tools registered."""
        return len(self._tools)
