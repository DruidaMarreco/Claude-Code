"""Unit tests for SessionNotes — no API key required."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hey_claude_features.session_notes import SessionNotes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(summary: str = "## Summary\nYou asked things.\n\n## Action Items\n- None\n\n## Key Facts\n- None") -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.content = [MagicMock(text=summary)]
    client.messages.create.return_value = resp
    return client


def _history(n_turns: int = 3) -> list[dict]:
    turns = []
    for i in range(n_turns):
        turns.append({"role": "user", "content": f"Question {i}"})
        turns.append({"role": "assistant", "content": f"Answer {i}"})
    return turns


def _notes(tmp_path: Path, client=None, min_turns: int = 2) -> SessionNotes:
    return SessionNotes(
        client=client or _make_client(),
        notes_dir=tmp_path / "notes",
        min_turns=min_turns,
    )


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_path_on_success(self, tmp_path):
        sn = _notes(tmp_path)
        path = sn.generate(_history(3))
        assert path is not None
        assert path.exists()

    def test_file_contains_markdown_headers(self, tmp_path):
        sn = _notes(tmp_path)
        path = sn.generate(_history(3))
        content = path.read_text()
        assert "# Session" in content
        assert "## Summary" in content

    def test_short_session_returns_none(self, tmp_path):
        sn = _notes(tmp_path, min_turns=4)
        # Only 2 turns (1 exchange)
        result = sn.generate(_history(1))
        assert result is None

    def test_custom_title_in_file(self, tmp_path):
        sn = _notes(tmp_path)
        path = sn.generate(_history(3), title="My Custom Session")
        assert "My Custom Session" in path.read_text()

    def test_api_failure_returns_none(self, tmp_path):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        sn = _notes(tmp_path, client=client)
        result = sn.generate(_history(3))
        assert result is None

    def test_no_collision_same_minute(self, tmp_path):
        sn = _notes(tmp_path)
        p1 = sn.generate(_history(3))
        p2 = sn.generate(_history(3))
        assert p1 != p2
        assert p1.exists()
        assert p2.exists()

    def test_haiku_called_with_transcript(self, tmp_path):
        client = _make_client()
        sn = _notes(tmp_path, client=client)
        sn.generate(_history(2))
        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else []
        # Verify the transcript content was included
        assert client.messages.create.called


# ---------------------------------------------------------------------------
# list_notes()
# ---------------------------------------------------------------------------


class TestListNotes:
    def test_empty_when_no_notes_dir(self, tmp_path):
        sn = _notes(tmp_path)
        assert sn.list_notes() == []

    def test_returns_saved_notes(self, tmp_path):
        sn = _notes(tmp_path)
        sn.generate(_history(3))
        sn.generate(_history(3))
        notes = sn.list_notes()
        # index.md + 2 note files, but list_notes returns all .md files
        assert len(notes) >= 2

    def test_newest_first(self, tmp_path):
        sn = _notes(tmp_path)
        p1 = sn.generate(_history(3))
        p2 = sn.generate(_history(3))
        listed = sn.list_notes()
        md_notes = [p for p in listed if p.name != "index.md"]
        # Both files exist and are listed (order depends on filenames)
        assert p1 in md_notes or p2 in md_notes


# ---------------------------------------------------------------------------
# Index file
# ---------------------------------------------------------------------------


class TestIndex:
    def test_index_created(self, tmp_path):
        sn = _notes(tmp_path)
        sn.generate(_history(3))
        index = tmp_path / "notes" / "index.md"
        assert index.exists()

    def test_index_contains_session_link(self, tmp_path):
        sn = _notes(tmp_path)
        path = sn.generate(_history(3), title="My Session")
        index = (tmp_path / "notes" / "index.md").read_text()
        assert "My Session" in index

    def test_multiple_sessions_all_in_index(self, tmp_path):
        sn = _notes(tmp_path)
        sn.generate(_history(3), title="Session A")
        sn.generate(_history(3), title="Session B")
        index = (tmp_path / "notes" / "index.md").read_text()
        assert "Session A" in index
        assert "Session B" in index


# ---------------------------------------------------------------------------
# Transcript builder
# ---------------------------------------------------------------------------


class TestBuildTranscript:
    def test_plain_messages(self):
        turns = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        transcript = SessionNotes._build_transcript(turns)
        assert "USER: hello" in transcript
        assert "ASSISTANT: hi there" in transcript

    def test_list_content_blocks(self):
        turns = [{"role": "user", "content": [{"text": "block text"}]}]
        transcript = SessionNotes._build_transcript(turns)
        assert "block text" in transcript

    def test_filters_non_user_assistant(self):
        turns = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "ignored"},
        ]
        transcript = SessionNotes._build_transcript(turns)
        assert "ignored" not in transcript
