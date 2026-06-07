"""
Persistent reminder system for Hey-Claude.

Provides two tools that integrate directly into Hey-Claude's ToolRegistry:

  set_reminder  — schedule a reminder for a future time
  list_reminders — show all pending reminders

Reminders are stored in ~/.hey-claude/reminders.json and survive restarts.
A background thread checks every 10 seconds and calls a user-supplied
`on_fire` callback when a reminder is due.

Usage in app.py
---------------
    from hey_claude_features.reminder_system import ReminderSystem

    def _speak_reminder(message: str) -> None:
        print(f"[REMINDER] {message}")
        speaker.speak(message)

    reminders = ReminderSystem(on_fire=_speak_reminder)
    reminders.start()                    # starts background checker

    # Register the tools
    for tool in reminders.tools():
        registry.register(tool)

    # On shutdown
    reminders.stop()

Natural-language time parsing
------------------------------
set_reminder accepts ISO-8601 strings ("2026-06-08T09:00") *or* delta
expressions the assistant should resolve before calling (e.g. Claude
converts "in 30 minutes" → ISO string).  A lightweight fallback parser
handles common absolute formats so the tool is usable even if the LLM
sends a string directly.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)

DEFAULT_STORE = Path("~/.hey-claude/reminders.json").expanduser()
_CHECK_INTERVAL = 10  # seconds between due-date checks


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Reminder:
    id: str
    message: str
    fire_at: float          # unix timestamp
    created_at: float = field(default_factory=time.time)
    fired: bool = False

    @property
    def fire_dt(self) -> datetime:
        return datetime.fromtimestamp(self.fire_at)

    def is_due(self, now: float | None = None) -> bool:
        return not self.fired and (now or time.time()) >= self.fire_at


# ---------------------------------------------------------------------------
# Time parsing helpers
# ---------------------------------------------------------------------------

_DELTA_PATTERNS: list[tuple[str, timedelta]] = [
    ("1 minute", timedelta(minutes=1)),
    ("5 minutes", timedelta(minutes=5)),
    ("10 minutes", timedelta(minutes=10)),
    ("15 minutes", timedelta(minutes=15)),
    ("30 minutes", timedelta(minutes=30)),
    ("1 hour", timedelta(hours=1)),
    ("2 hours", timedelta(hours=2)),
]

_ISO_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
]


def parse_time(value: str) -> datetime | None:
    """
    Parse a time string into a datetime.

    Accepts:
    - ISO-8601 strings: "2026-06-08T09:00", "2026-06-08 09:30:00"
    - Delta strings: "in 30 minutes", "in 1 hour"
    - Date only: "2026-06-08" (assumes midnight)

    Returns None if the string cannot be parsed.
    """
    v = value.strip().lower()

    # Delta expressions
    if v.startswith("in "):
        remainder = v[3:].strip()
        for label, delta in _DELTA_PATTERNS:
            if remainder.startswith(label):
                return datetime.now() + delta
        # Generic "in N minutes/hours" parsing
        parts = remainder.split()
        if len(parts) >= 2:
            try:
                n = float(parts[0])
                unit = parts[1].rstrip("s")
                if unit in ("minute", "min"):
                    return datetime.now() + timedelta(minutes=n)
                if unit in ("hour", "hr"):
                    return datetime.now() + timedelta(hours=n)
                if unit in ("second", "sec"):
                    return datetime.now() + timedelta(seconds=n)
                if unit == "day":
                    return datetime.now() + timedelta(days=n)
            except (ValueError, IndexError):
                pass
        return None

    # Absolute ISO strings
    for fmt in _ISO_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue

    return None


# ---------------------------------------------------------------------------
# ReminderSystem
# ---------------------------------------------------------------------------


class ReminderSystem:
    """
    Manages persistent reminders with a background firing thread.

    Parameters
    ----------
    on_fire:
        Called with the reminder message string when a reminder fires.
    store_path:
        JSON file for persistence.  Defaults to ~/.hey-claude/reminders.json.
    check_interval:
        How often (seconds) the background thread checks for due reminders.
    """

    def __init__(
        self,
        on_fire: Callable[[str], None],
        store_path: Path = DEFAULT_STORE,
        check_interval: float = _CHECK_INTERVAL,
    ) -> None:
        self._on_fire = on_fire
        self._store_path = store_path
        self._check_interval = check_interval
        self._lock = threading.Lock()
        self._reminders: dict[str, Reminder] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._load()

    # ------------------------------------------------------------------
    # Tool-compatible API (called by Hey-Claude's tool executor)
    # ------------------------------------------------------------------

    def set_reminder(self, message: str, when: str) -> str:
        """
        Schedule a reminder.

        Parameters
        ----------
        message:
            What to remind the user about.
        when:
            When to fire.  Accepts ISO-8601 strings or delta expressions
            like "in 30 minutes" or "in 2 hours".

        Returns a confirmation string suitable for speaking aloud.
        """
        dt = parse_time(when)
        if dt is None:
            return f"Sorry, I couldn't understand the time '{when}'. Try 'in 30 minutes' or '2026-06-08T09:00'."

        if dt <= datetime.now():
            return "That time is already in the past. Please give me a future time."

        reminder = Reminder(
            id=str(uuid4())[:8],
            message=message,
            fire_at=dt.timestamp(),
        )
        with self._lock:
            self._reminders[reminder.id] = reminder
        self._save()

        delta = dt - datetime.now()
        minutes = int(delta.total_seconds() / 60)
        time_desc = f"in {minutes} minute{'s' if minutes != 1 else ''}" if minutes < 60 else dt.strftime("at %H:%M")
        return f"Got it. I'll remind you {time_desc}: {message}"

    def list_reminders(self) -> str:
        """Return a human-readable list of all pending (unfired) reminders."""
        with self._lock:
            pending = [r for r in self._reminders.values() if not r.fired]

        if not pending:
            return "You have no pending reminders."

        pending.sort(key=lambda r: r.fire_at)
        lines = [f"You have {len(pending)} reminder{'s' if len(pending) != 1 else ''}:"]
        for r in pending:
            lines.append(f"  • {r.fire_dt.strftime('%H:%M')} — {r.message}")
        return "\n".join(lines)

    def cancel_reminder(self, reminder_id: str) -> str:
        """Cancel a reminder by ID. Returns confirmation."""
        with self._lock:
            r = self._reminders.get(reminder_id)
            if r is None or r.fired:
                return f"No active reminder with ID {reminder_id}."
            r.fired = True
        self._save()
        return f"Reminder cancelled: {r.message}"

    # ------------------------------------------------------------------
    # Tool descriptors for ToolRegistry
    # ------------------------------------------------------------------

    def tools(self) -> list[dict[str, Any]]:
        """
        Return tool descriptor dicts compatible with Hey-Claude's Tool dataclass.

        Usage:
            from hey_claude.tools import Tool
            for spec in reminders.tools():
                registry.register(Tool(**spec, function=...))

        Or with the plugin system, define TOOLS in a plugin file that imports
        this ReminderSystem instance.
        """
        return [
            {
                "name": "set_reminder",
                "description": (
                    "Schedule a reminder that will fire at a future time. "
                    "Use when the user says 'remind me to...', 'set a reminder for...', etc."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "What to remind the user about.",
                        },
                        "when": {
                            "type": "string",
                            "description": (
                                "When to fire the reminder. "
                                "ISO-8601 ('2026-06-08T09:00') or delta ('in 30 minutes')."
                            ),
                        },
                    },
                    "required": ["message", "when"],
                },
                "requires_confirmation": False,
            },
            {
                "name": "list_reminders",
                "description": "List all pending reminders.",
                "input_schema": {"type": "object", "properties": {}},
                "requires_confirmation": False,
            },
        ]

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background reminder checker thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="reminder-checker")
        self._thread.start()
        logger.debug("Reminder checker started (interval=%ss)", self._check_interval)

    def stop(self) -> None:
        """Stop the background thread gracefully."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._check_interval + 1)

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self._check_interval):
            self._check_due()

    def _check_due(self) -> None:
        now = time.time()
        fired_ids = []
        with self._lock:
            for r in self._reminders.values():
                if r.is_due(now):
                    r.fired = True
                    fired_ids.append(r.id)

        for rid in fired_ids:
            with self._lock:
                r = self._reminders[rid]
            try:
                self._on_fire(r.message)
            except Exception:
                logger.exception("on_fire callback raised for reminder %s", rid)

        if fired_ids:
            self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {rid: asdict(r) for rid, r in self._reminders.items()}
            self._store_path.write_text(json.dumps(data, indent=2))
        except Exception:
            logger.exception("Failed to save reminders to %s", self._store_path)

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        try:
            raw: dict[str, Any] = json.loads(self._store_path.read_text())
            for rid, d in raw.items():
                self._reminders[rid] = Reminder(**d)
            logger.debug("Loaded %d reminders from %s", len(self._reminders), self._store_path)
        except Exception:
            logger.exception("Failed to load reminders; starting fresh")
