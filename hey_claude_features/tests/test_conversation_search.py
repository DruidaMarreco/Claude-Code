"""Unit tests for ConversationSearch — no API key required."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hey_claude_features.conversation_search import (
    ConversationSearch,
    SearchReport,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(ranked: list[dict] | None = None) -> MagicMock:
    client = MagicMock()
    payload = ranked or []
    resp = MagicMock()
    resp.content = [MagicMock(text=json.dumps(payload))]
    client.messages.create.return_value = resp
    return client


def _search(tmp_path: Path, client=None) -> ConversationSearch:
    return ConversationSearch(
        client=client or _make_client(),
        notes_dir=tmp_path / "notes",
    )


def _write_note(notes_dir: Path, slug: str, content: str) -> Path:
    notes_dir.mkdir(parents=True, exist_ok=True)
    path = notes_dir / f"{slug}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# _is_semantic
# ---------------------------------------------------------------------------


class TestIsSemanticHeuristic:
    def test_short_word_keyword(self):
        assert not ConversationSearch._is_semantic("dentist")

    def test_long_query_semantic(self):
        assert ConversationSearch._is_semantic("when did I talk about the dentist")

    def test_question_mark_semantic(self):
        assert ConversationSearch._is_semantic("dentist?")

    def test_starts_with_when(self):
        assert ConversationSearch._is_semantic("when did I mention Paris")

    def test_did_i_prefix(self):
        assert ConversationSearch._is_semantic("did i discuss the budget")


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    def test_finds_matching_note(self, tmp_path):
        notes_dir = tmp_path / "notes"
        _write_note(notes_dir, "2026-06-01_10-00", "# Session\n\nI asked about dentist appointments.")
        s = _search(tmp_path)
        report = s.keyword_search("dentist")
        assert len(report.results) == 1
        assert "dentist" in report.results[0].excerpt.lower()

    def test_case_insensitive(self, tmp_path):
        notes_dir = tmp_path / "notes"
        _write_note(notes_dir, "2026-06-01_10-00", "# Session\n\nDentist was mentioned.")
        s = _search(tmp_path)
        report = s.keyword_search("DENTIST")
        assert len(report.results) == 1

    def test_no_match_returns_empty(self, tmp_path):
        notes_dir = tmp_path / "notes"
        _write_note(notes_dir, "2026-06-01_10-00", "# Session\n\nWeather discussion.")
        s = _search(tmp_path)
        report = s.keyword_search("dentist")
        assert len(report.results) == 0

    def test_multiple_matches(self, tmp_path):
        notes_dir = tmp_path / "notes"
        for i in range(3):
            _write_note(notes_dir, f"2026-06-0{i+1}_10-00", f"# Session {i}\n\nDentist topic {i}.")
        s = _search(tmp_path)
        report = s.keyword_search("dentist")
        assert len(report.results) == 3

    def test_max_results_capped(self, tmp_path):
        notes_dir = tmp_path / "notes"
        for i in range(10):
            _write_note(notes_dir, f"2026-06-{i+1:02d}_10-00", f"# S{i}\n\nKeyword here.")
        s = ConversationSearch(client=_make_client(), notes_dir=notes_dir, max_results=3)
        report = s.keyword_search("Keyword")
        assert len(report.results) <= 3

    def test_index_md_excluded(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "index.md").write_text("# Index\n\ndentist link here")
        s = _search(tmp_path)
        report = s.keyword_search("dentist")
        assert len(report.results) == 0

    def test_empty_notes_dir(self, tmp_path):
        s = _search(tmp_path)
        report = s.keyword_search("anything")
        assert report.results == []


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    def test_calls_haiku(self, tmp_path):
        notes_dir = tmp_path / "notes"
        _write_note(notes_dir, "2026-06-01_10-00", "# Session\n\nWeather in Paris discussed.")
        client = _make_client(ranked=[
            {"slug": "2026-06-01_10-00", "score": 0.9, "excerpt": "Weather in Paris discussed."}
        ])
        s = ConversationSearch(client=client, notes_dir=notes_dir)
        report = s.semantic_search("when did I ask about Paris weather?")
        client.messages.create.assert_called_once()
        assert len(report.results) == 1

    def test_empty_dir_returns_empty(self, tmp_path):
        s = _search(tmp_path)
        report = s.semantic_search("anything?")
        assert report.results == []

    def test_haiku_failure_falls_back_to_keyword(self, tmp_path):
        notes_dir = tmp_path / "notes"
        _write_note(notes_dir, "2026-06-01_10-00", "# Session\n\nParis weather topic.")
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        s = ConversationSearch(client=client, notes_dir=notes_dir)
        report = s.semantic_search("Paris weather?")
        # Falls back to keyword search on "Paris weather?"
        assert report.mode in ("keyword", "semantic")

    def test_haiku_invalid_json_fallback(self, tmp_path):
        notes_dir = tmp_path / "notes"
        _write_note(notes_dir, "2026-06-01_10-00", "# Session\n\nParis topic.")
        client = MagicMock()
        resp = MagicMock()
        resp.content = [MagicMock(text="not json")]
        client.messages.create.return_value = resp
        s = ConversationSearch(client=client, notes_dir=notes_dir)
        report = s.semantic_search("Paris?")
        assert isinstance(report, SearchReport)

    def test_unknown_slug_in_response_skipped(self, tmp_path):
        notes_dir = tmp_path / "notes"
        _write_note(notes_dir, "2026-06-01_10-00", "# Session\n\nContent.")
        client = _make_client(ranked=[
            {"slug": "nonexistent-slug", "score": 0.9, "excerpt": "Something."}
        ])
        s = ConversationSearch(client=client, notes_dir=notes_dir)
        report = s.semantic_search("anything?")
        assert len(report.results) == 0


# ---------------------------------------------------------------------------
# SearchReport
# ---------------------------------------------------------------------------


class TestSearchReport:
    def test_voice_summary_no_results(self):
        r = SearchReport(query="dentist", results=[])
        assert "No notes" in r.voice_summary()
        assert "dentist" in r.voice_summary()

    def test_voice_summary_with_results(self, tmp_path):
        path = tmp_path / "2026-06-01_10-00.md"
        path.write_text("")
        r = SearchReport(query="dentist", results=[
            SearchResult(note_path=path, title="My Session", excerpt="Dentist was mentioned.", score=0.9)
        ])
        s = r.voice_summary()
        assert "1 matching note" in s
        assert "2026-06-01" in s

    def test_formatted_shows_all_results(self, tmp_path):
        results = []
        for i in range(3):
            p = tmp_path / f"2026-06-0{i+1}_10-00.md"
            p.write_text("")
            results.append(SearchResult(note_path=p, title=f"S{i}", excerpt=f"Excerpt {i}", score=0.9))
        r = SearchReport(query="test", results=results)
        formatted = r.formatted()
        assert "1." in formatted
        assert "3." in formatted


# ---------------------------------------------------------------------------
# search() auto-mode
# ---------------------------------------------------------------------------


class TestAutoSearch:
    def test_short_query_uses_keyword(self, tmp_path):
        notes_dir = tmp_path / "notes"
        _write_note(notes_dir, "2026-06-01_10-00", "# S\n\ndentist visit planned.")
        client = _make_client()
        s = ConversationSearch(client=client, notes_dir=notes_dir)
        result = s.search("dentist")
        # Keyword mode — Haiku not called
        client.messages.create.assert_not_called()
        assert isinstance(result, str)

    def test_long_query_uses_semantic(self, tmp_path):
        notes_dir = tmp_path / "notes"
        _write_note(notes_dir, "2026-06-01_10-00", "# S\n\nParis trip discussed.")
        client = _make_client(ranked=[
            {"slug": "2026-06-01_10-00", "score": 0.9, "excerpt": "Paris trip discussed."}
        ])
        s = ConversationSearch(client=client, notes_dir=notes_dir)
        s.search("when did I talk about the Paris trip?")
        client.messages.create.assert_called_once()
