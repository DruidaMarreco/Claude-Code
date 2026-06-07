"""
End-of-session note generator for Hey-Claude.

When the assistant shuts down (or on demand), uses Haiku to summarize what
was discussed and extract any action items.  Notes are saved as dated
markdown files in ~/.hey-claude/notes/.

Features
--------
- Haiku-powered summary (cheap, fast)
- Extracts action items as a bulleted checklist
- Saves to ~/.hey-claude/notes/YYYY-MM-DD_HH-MM.md
- Index file (notes/index.md) kept up-to-date for quick browsing
- `generate()` is safe to call on shutdown — errors are caught and logged

Usage in app.py
---------------
    from hey_claude_features.session_notes import SessionNotes

    notes = SessionNotes(client=anthropic_client)

    # On shutdown (e.g. KeyboardInterrupt handler):
    path = notes.generate(history=self._history.messages)
    if path:
        print(f"Session notes saved to {path}")

Or on demand via a tool:
    registry.register(Tool(
        name="save_session_notes",
        description="Summarize and save notes from this conversation.",
        input_schema={"type": "object", "properties": {}},
        function=lambda: notes.generate(history=self._history.messages) or "Notes saved.",
        requires_confirmation=False,
    ))
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_NOTES_DIR = Path("~/.hey-claude/notes").expanduser()
DEFAULT_MODEL = "claude-haiku-4-5"

_SYSTEM_PROMPT = """\
You are summarizing a voice assistant session for the user's personal notes.
Write in second person ("You asked...", "You mentioned...").

Structure your response EXACTLY as follows (use these exact headers):

## Summary
2-4 sentences covering the main topics and outcomes.

## Action Items
A markdown checklist of concrete things the user said they'd do or asked to be reminded about.
If none, write: - None

## Key Facts
Bullet points of any important information, dates, or facts that came up.
If none, write: - None

Be concise. This is a personal note, not a report.
"""


class SessionNotes:
    """
    Generates and persists end-of-session markdown notes.

    Parameters
    ----------
    client:
        Anthropic client for Haiku calls.
    model:
        Model to use for summarization.  Defaults to Haiku (cheap).
    notes_dir:
        Directory where note files are saved.
    min_turns:
        Minimum number of conversation turns before generating notes.
        Avoids creating empty/trivial note files for very short sessions.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = DEFAULT_MODEL,
        notes_dir: Path = DEFAULT_NOTES_DIR,
        min_turns: int = 2,
    ) -> None:
        self._client = client
        self._model = model
        self._notes_dir = notes_dir
        self._min_turns = min_turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        history: list[dict[str, Any]],
        title: str | None = None,
    ) -> Path | None:
        """
        Generate session notes from conversation history and save to disk.

        Parameters
        ----------
        history:
            The conversation message list (same format as Anthropic API messages).
        title:
            Optional custom title.  Defaults to the session timestamp.

        Returns the path to the saved note file, or None if generation failed
        or the session was too short.
        """
        turns = [m for m in history if m.get("role") in ("user", "assistant")]
        if len(turns) < self._min_turns:
            logger.debug("Session too short (%d turns); skipping notes", len(turns))
            return None

        transcript = self._build_transcript(turns)
        content = self._summarize(transcript)
        if content is None:
            return None

        now = datetime.now()
        slug = now.strftime("%Y-%m-%d_%H-%M")
        note_title = title or f"Session {now.strftime('%Y-%m-%d %H:%M')}"
        markdown = self._format_note(note_title, content, now)

        path = self._save(slug, markdown)
        if path:
            self._update_index(path, note_title, now)
        return path

    def list_notes(self) -> list[Path]:
        """Return all saved note paths, newest first."""
        if not self._notes_dir.exists():
            return []
        return sorted(self._notes_dir.glob("*.md"), reverse=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _summarize(self, transcript: str) -> str | None:
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Conversation transcript:\n\n{transcript}"}
                ],
            )
            return resp.content[0].text.strip()
        except Exception:
            logger.exception("Session notes summarization failed")
            return None

    def _format_note(self, title: str, content: str, dt: datetime) -> str:
        header = (
            f"# {title}\n\n"
            f"*Generated {dt.strftime('%Y-%m-%d at %H:%M')}*\n\n"
        )
        return header + content + "\n"

    def _save(self, slug: str, markdown: str) -> Path | None:
        try:
            self._notes_dir.mkdir(parents=True, exist_ok=True)
            path = self._notes_dir / f"{slug}.md"
            # Avoid collisions within the same minute
            counter = 1
            while path.exists():
                path = self._notes_dir / f"{slug}_{counter}.md"
                counter += 1
            path.write_text(markdown, encoding="utf-8")
            logger.info("Session notes saved to %s", path)
            return path
        except Exception:
            logger.exception("Failed to save session notes")
            return None

    def _update_index(self, note_path: Path, title: str, dt: datetime) -> None:
        index_path = self._notes_dir / "index.md"
        entry = f"- [{title}]({note_path.name}) — {dt.strftime('%Y-%m-%d %H:%M')}\n"
        try:
            if index_path.exists():
                existing = index_path.read_text(encoding="utf-8")
                # Prepend new entry after the header
                lines = existing.splitlines(keepends=True)
                if lines and lines[0].startswith("#"):
                    lines.insert(1, "\n" + entry)
                    index_path.write_text("".join(lines), encoding="utf-8")
                else:
                    index_path.write_text(entry + existing, encoding="utf-8")
            else:
                index_path.write_text(
                    "# Session Notes Index\n\n" + entry, encoding="utf-8"
                )
        except Exception:
            logger.exception("Failed to update notes index")

    @staticmethod
    def _build_transcript(turns: list[dict[str, Any]]) -> str:
        lines = []
        for t in turns:
            role = t.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = t.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)
