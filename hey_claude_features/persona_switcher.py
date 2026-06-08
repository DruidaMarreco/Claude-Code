"""
Named assistant persona switcher for Hey-Claude.

Personas are named system-prompt presets that change how the assistant
responds.  Switching is instant — no restart needed.

Built-in presets
----------------
- default    : The standard Hey-Claude voice assistant personality
- concise    : Answers in one sentence max; no filler words
- professional: Formal tone, structured responses, no jokes
- creative   : Playful, expansive, explores tangents and ideas
- tutor      : Patient, explains reasoning step by step

Custom personas can be added in two ways:
1. Drop a .md file into ~/.hey-claude/personas/<name>.md (the file content
   becomes the system prompt)
2. Call PersonaSwitcher.add(name, prompt) at runtime

Usage in responder.py
---------------------
    from hey_claude_features.persona_switcher import PersonaSwitcher

    personas = PersonaSwitcher()
    # Pass current system prompt to API:
    client.messages.create(system=personas.current_prompt, ...)

    # Switch via a tool:
    registry.register(Tool(
        name="switch_persona",
        description="Change assistant personality. Options: default, concise, professional, creative, tutor.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        function=lambda name: personas.switch(name),
        requires_confirmation=False,
    ))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

DEFAULT_PERSONAS_DIR = Path("~/.hey-claude/personas").expanduser()

# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

_BUILTIN: dict[str, str] = {
    "default": (
        "You are Hey Claude, a friendly and capable voice assistant. "
        "Keep responses concise and natural for speech — avoid markdown, "
        "bullet points, and long lists unless specifically requested. "
        "Be warm but efficient."
    ),
    "concise": (
        "You are a terse voice assistant. Answer every question in one "
        "sentence. Never use filler words. If unsure, say so in five words."
    ),
    "professional": (
        "You are a professional executive assistant. Use formal language, "
        "structured responses, and precise terminology. Avoid jokes or "
        "casual phrasing. Always acknowledge the question before answering."
    ),
    "creative": (
        "You are an imaginative, enthusiastic assistant who loves exploring "
        "ideas. Feel free to offer unexpected angles, make interesting "
        "connections, and be playful with language. Answers can be expansive."
    ),
    "tutor": (
        "You are a patient tutor. Explain your reasoning step by step. "
        "Check understanding by offering to clarify. Use simple analogies "
        "for complex topics. Encourage questions."
    ),
}


# ---------------------------------------------------------------------------
# PersonaSwitcher
# ---------------------------------------------------------------------------


class PersonaSwitcher:
    """
    Manages named assistant personas (system prompt presets).

    Parameters
    ----------
    personas_dir:
        Directory scanned for custom .md persona files on init.
    initial:
        Name of the persona to activate on startup.
    """

    def __init__(
        self,
        personas_dir: Path = DEFAULT_PERSONAS_DIR,
        initial: str = "default",
    ) -> None:
        self._personas_dir = personas_dir
        self._custom: dict[str, str] = {}
        self._active: str = "default"

        self._load_custom()

        if initial != "default":
            self.switch(initial)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_prompt(self) -> str:
        """The system prompt for the active persona."""
        return self._resolve(self._active)

    @property
    def active_name(self) -> str:
        return self._active

    def switch(self, name: str) -> str:
        """
        Activate a persona by name.

        Returns a voice-friendly confirmation string.
        Raises ValueError if the name is not recognised.
        """
        name = name.strip().lower()
        if not self._exists(name):
            available = ", ".join(sorted(self.list_names()))
            raise ValueError(
                f"Unknown persona '{name}'. Available: {available}"
            )
        self._active = name
        logger.info("Persona switched to '%s'", name)
        return f"Switched to {name} mode."

    def add(self, name: str, prompt: str, persist: bool = False) -> None:
        """
        Add or replace a custom persona at runtime.

        Parameters
        ----------
        name:
            Identifier (lowercase, no spaces recommended).
        prompt:
            The system prompt text.
        persist:
            If True, save to ~/.hey-claude/personas/<name>.md so it
            survives restarts.
        """
        name = name.strip().lower()
        self._custom[name] = prompt
        if persist:
            self._save_custom(name, prompt)
        logger.debug("Persona '%s' added", name)

    def remove(self, name: str) -> bool:
        """Remove a custom persona. Returns True if it existed."""
        name = name.strip().lower()
        if name in _BUILTIN:
            raise ValueError(f"Cannot remove built-in persona '{name}'")
        existed = name in self._custom
        self._custom.pop(name, None)
        path = self._personas_dir / f"{name}.md"
        if path.exists():
            path.unlink()
        return existed

    def list_names(self) -> list[str]:
        """Return all available persona names (built-in + custom)."""
        return sorted({*_BUILTIN.keys(), *self._custom.keys()})

    def describe(self, name: str) -> str:
        """Return the system prompt for a named persona."""
        name = name.strip().lower()
        if not self._exists(name):
            raise ValueError(f"Unknown persona '{name}'")
        return self._resolve(name)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """Iterate over (name, prompt) pairs for all personas."""
        for name in self.list_names():
            yield name, self._resolve(name)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _exists(self, name: str) -> bool:
        return name in _BUILTIN or name in self._custom

    def _resolve(self, name: str) -> str:
        return self._custom.get(name) or _BUILTIN.get(name, _BUILTIN["default"])

    def _load_custom(self) -> None:
        if not self._personas_dir.exists():
            return
        for path in self._personas_dir.glob("*.md"):
            name = path.stem.lower()
            try:
                self._custom[name] = path.read_text(encoding="utf-8").strip()
                logger.debug("Loaded custom persona '%s' from %s", name, path)
            except Exception:
                logger.exception("Failed to load persona from %s", path)

    def _save_custom(self, name: str, prompt: str) -> None:
        try:
            self._personas_dir.mkdir(parents=True, exist_ok=True)
            (self._personas_dir / f"{name}.md").write_text(prompt, encoding="utf-8")
        except Exception:
            logger.exception("Failed to persist persona '%s'", name)
