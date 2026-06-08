"""
Lightning-fast voice command shortcuts for Hey-Claude.

Certain spoken commands ('stop', 'mute', 'volume up', …) should be handled
immediately — zero API calls, zero latency.  This module intercepts those
utterances before they reach the Haiku gate or Opus responder and executes
a registered Python callback instead.

How it works
------------
1. Incoming transcript is normalised (lowercase, stripped punctuation)
2. Each shortcut's trigger phrases are compared via exact match or a
   configurable fuzzy threshold
3. On match, the shortcut's callback is called and a ``ShortcutResult``
   is returned so the caller knows to skip the pipeline

Built-in shortcuts
------------------
stop / cancel / never mind  → on_stop(callback)
mute / be quiet / silence   → on_mute(callback)
repeat / say that again      → on_repeat(callback)
volume up / louder           → on_volume_up(callback)
volume down / quieter        → on_volume_down(callback)

Custom shortcuts
----------------
    sc = CommandShortcuts()
    sc.add("lights on", callback=lambda: turn_lights(True), aliases=["turn the lights on"])

Persistence
-----------
Custom shortcuts (without callbacks) can be persisted to / loaded from JSON
so they survive restarts.  Built-in callbacks are always re-registered at init.

Usage in app.py
---------------
    from hey_claude_features.command_shortcuts import CommandShortcuts, ShortcutResult

    shortcuts = CommandShortcuts()
    shortcuts.on_stop(lambda: pipeline.stop())
    shortcuts.on_mute(lambda: speaker.mute())
    shortcuts.on_repeat(lambda: speaker.repeat_last())

    result = shortcuts.match(transcript)
    if result.matched:
        result.execute()
        print(result.voice_reply)
        return   # skip Haiku + Opus
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalise(text: str) -> str:
    """Lowercase and strip punctuation/extra whitespace."""
    return _PUNCT_RE.sub("", text.lower()).strip()


def _token_set(text: str) -> set[str]:
    return set(text.split())


def _fuzzy_match(a: str, b: str) -> float:
    """Token-set overlap ratio — fast, no deps."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Shortcut:
    """A single voice shortcut definition."""
    name: str
    triggers: list[str]          # normalised canonical triggers
    voice_reply: str             # what Hey-Claude says after executing
    callback: Callable[[], None] | None = field(default=None, repr=False)
    fuzzy_threshold: float = 0.85

    def matches(self, normalised_transcript: str) -> bool:
        """Return True if *normalised_transcript* matches any trigger."""
        for trigger in self.triggers:
            if normalised_transcript == trigger:
                return True
            if _fuzzy_match(normalised_transcript, trigger) >= self.fuzzy_threshold:
                return True
        return False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "triggers": self.triggers,
            "voice_reply": self.voice_reply,
            "fuzzy_threshold": self.fuzzy_threshold,
        }


@dataclass
class ShortcutResult:
    """Outcome of a ``CommandShortcuts.match()`` call."""
    matched: bool
    shortcut: Shortcut | None = None
    voice_reply: str = ""
    matched_trigger: str = ""

    def execute(self) -> None:
        """Run the shortcut's callback (if any)."""
        if self.shortcut and self.shortcut.callback:
            try:
                self.shortcut.callback()
            except Exception:
                logger.exception("Shortcut %r callback raised", self.shortcut.name)

    @classmethod
    def miss(cls) -> "ShortcutResult":
        return cls(matched=False)


# ---------------------------------------------------------------------------
# CommandShortcuts
# ---------------------------------------------------------------------------


_BUILTIN_DEFS: list[dict] = [
    {
        "name": "stop",
        "triggers": ["stop", "cancel", "never mind", "nevermind", "abort"],
        "voice_reply": "Stopping.",
    },
    {
        "name": "mute",
        "triggers": ["mute", "be quiet", "silence", "shush", "shut up"],
        "voice_reply": "Muted.",
    },
    {
        "name": "repeat",
        "triggers": ["repeat", "say that again", "repeat that", "what did you say"],
        "voice_reply": "Repeating.",
    },
    {
        "name": "volume_up",
        "triggers": ["volume up", "louder", "speak up", "turn it up"],
        "voice_reply": "Volume increased.",
    },
    {
        "name": "volume_down",
        "triggers": ["volume down", "quieter", "softer", "turn it down"],
        "voice_reply": "Volume decreased.",
    },
]


class CommandShortcuts:
    """
    Maintains a registry of voice shortcuts and matches transcripts against them.

    Parameters
    ----------
    persist_path:
        Optional JSON file for saving/loading custom shortcuts.  Built-in
        shortcuts are always reconstructed at init.
    fuzzy_threshold:
        Default similarity threshold for new shortcuts (0.0–1.0).
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        fuzzy_threshold: float = 0.85,
    ) -> None:
        self._threshold = fuzzy_threshold
        self._persist_path = persist_path
        self._shortcuts: dict[str, Shortcut] = {}

        # Register built-ins (no callbacks until caller registers them)
        for defn in _BUILTIN_DEFS:
            self._shortcuts[defn["name"]] = Shortcut(
                name=defn["name"],
                triggers=[_normalise(t) for t in defn["triggers"]],
                voice_reply=defn["voice_reply"],
                fuzzy_threshold=fuzzy_threshold,
            )

        if persist_path and persist_path.exists():
            self._load(persist_path)

    # ------------------------------------------------------------------
    # Public API — callback registration
    # ------------------------------------------------------------------

    def on_stop(self, callback: Callable[[], None]) -> None:
        self._set_callback("stop", callback)

    def on_mute(self, callback: Callable[[], None]) -> None:
        self._set_callback("mute", callback)

    def on_repeat(self, callback: Callable[[], None]) -> None:
        self._set_callback("repeat", callback)

    def on_volume_up(self, callback: Callable[[], None]) -> None:
        self._set_callback("volume_up", callback)

    def on_volume_down(self, callback: Callable[[], None]) -> None:
        self._set_callback("volume_down", callback)

    # ------------------------------------------------------------------
    # Public API — shortcut management
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        *,
        triggers: list[str] | None = None,
        voice_reply: str = "Done.",
        callback: Callable[[], None] | None = None,
        fuzzy_threshold: float | None = None,
        persist: bool = True,
    ) -> None:
        """Register a new shortcut or replace an existing one."""
        if triggers is None:
            triggers = [name]
        sc = Shortcut(
            name=name,
            triggers=[_normalise(t) for t in triggers],
            voice_reply=voice_reply,
            callback=callback,
            fuzzy_threshold=fuzzy_threshold if fuzzy_threshold is not None else self._threshold,
        )
        self._shortcuts[name] = sc
        if persist and self._persist_path:
            self._save()

    def remove(self, name: str) -> bool:
        """Remove a shortcut by name.  Returns True if it existed."""
        existed = name in self._shortcuts
        self._shortcuts.pop(name, None)
        if existed and self._persist_path:
            self._save()
        return existed

    def list_names(self) -> list[str]:
        return list(self._shortcuts.keys())

    def get(self, name: str) -> Shortcut | None:
        return self._shortcuts.get(name)

    # ------------------------------------------------------------------
    # Public API — matching
    # ------------------------------------------------------------------

    def match(self, transcript: str) -> ShortcutResult:
        """
        Check *transcript* against all registered shortcuts.

        Returns the first match found (shortcuts checked in insertion order).
        Returns ``ShortcutResult.miss()`` if nothing matched.
        """
        norm = _normalise(transcript)
        if not norm:
            return ShortcutResult.miss()

        for sc in self._shortcuts.values():
            for trigger in sc.triggers:
                if norm == trigger:
                    return ShortcutResult(
                        matched=True,
                        shortcut=sc,
                        voice_reply=sc.voice_reply,
                        matched_trigger=trigger,
                    )
                if _fuzzy_match(norm, trigger) >= sc.fuzzy_threshold:
                    return ShortcutResult(
                        matched=True,
                        shortcut=sc,
                        voice_reply=sc.voice_reply,
                        matched_trigger=trigger,
                    )
        return ShortcutResult.miss()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Shortcut]:
        return iter(self._shortcuts.values())

    def __len__(self) -> int:
        return len(self._shortcuts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_callback(self, name: str, callback: Callable[[], None]) -> None:
        if name in self._shortcuts:
            self._shortcuts[name].callback = callback

    def _save(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            data = [sc.as_dict() for sc in self._shortcuts.values() if sc.name not in {d["name"] for d in _BUILTIN_DEFS}]
            self._persist_path.write_text(json.dumps(data, indent=2))  # type: ignore[union-attr]
        except Exception:
            logger.exception("Failed to persist shortcuts")

    def _load(self, path: Path) -> None:
        try:
            records: list[dict] = json.loads(path.read_text())
            for r in records:
                self._shortcuts[r["name"]] = Shortcut(
                    name=r["name"],
                    triggers=r["triggers"],
                    voice_reply=r.get("voice_reply", "Done."),
                    fuzzy_threshold=r.get("fuzzy_threshold", self._threshold),
                )
            logger.debug("Loaded %d custom shortcuts from %s", len(records), path)
        except Exception:
            logger.exception("Failed to load shortcuts from %s", path)
