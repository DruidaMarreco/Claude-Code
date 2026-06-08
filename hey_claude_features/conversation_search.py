"""
Search through Hey-Claude's saved session notes.

Provides two search modes:

1. Keyword search (free, instant)
   Scans note files for literal text matches.  Fast and offline.

2. Semantic search (Haiku-powered)
   Asks Haiku to identify which notes are relevant to a natural-language
   query like "when did I discuss the dentist appointment?".  Returns
   ranked excerpts.

Usage as a tool in Hey-Claude
------------------------------
    from hey_claude_features.conversation_search import ConversationSearch

    search = ConversationSearch(
        client=anthropic_client,
        notes_dir=Path("~/.hey-claude/notes").expanduser(),
    )

    # Register as a tool:
    Tool(
        name="search_notes",
        description="Search past conversation notes. Use for 'what did I ask about X?' queries.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        function=lambda query: search.search(query),
        requires_confirmation=False,
    )

Standalone usage
----------------
    results = search.keyword_search("dentist")
    results = search.semantic_search("when did I plan the trip to Lisbon?")
    print(search.search("dentist"))   # auto-chooses mode
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_NOTES_DIR = Path("~/.hey-claude/notes").expanduser()
_MAX_NOTE_CHARS = 2000   # truncate long notes before sending to Haiku
_MAX_RESULTS = 5


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    note_path: Path
    title: str
    excerpt: str
    score: float = 1.0          # 1.0 for keyword hits; Haiku relevance 0-1

    def __str__(self) -> str:
        date = self.note_path.stem[:10]  # YYYY-MM-DD from filename
        return f"[{date}] {self.title}: {self.excerpt}"


@dataclass
class SearchReport:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    mode: str = "keyword"

    def voice_summary(self) -> str:
        if not self.results:
            return f"No notes found matching '{self.query}'."
        count = len(self.results)
        top = self.results[0]
        date = top.note_path.stem[:10]
        return (
            f"Found {count} matching note{'s' if count != 1 else ''}. "
            f"Most relevant: {date} — {top.excerpt[:120]}"
        )

    def formatted(self) -> str:
        if not self.results:
            return f"No results for: {self.query}"
        lines = [f"Search results for '{self.query}' ({self.mode} mode):"]
        for i, r in enumerate(self.results, 1):
            lines.append(f"\n{i}. {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ConversationSearch
# ---------------------------------------------------------------------------


class ConversationSearch:
    """
    Searches saved Hey-Claude session notes.

    Parameters
    ----------
    client:
        Anthropic client for semantic search mode.
    notes_dir:
        Directory containing session note .md files.
    model:
        Haiku model for semantic ranking.
    max_results:
        Maximum number of results to return.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        notes_dir: Path = DEFAULT_NOTES_DIR,
        model: str = "claude-haiku-4-5",
        max_results: int = _MAX_RESULTS,
    ) -> None:
        self._client = client
        self._notes_dir = notes_dir
        self._model = model
        self._max_results = max_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        """
        Auto-select search mode and return a formatted string.

        Uses keyword search for short/simple queries and semantic search
        for natural-language questions.
        """
        if self._is_semantic(query):
            report = self.semantic_search(query)
        else:
            report = self.keyword_search(query)
        return report.voice_summary()

    def keyword_search(self, query: str) -> SearchReport:
        """Find notes containing the query as a literal substring (case-insensitive)."""
        notes = self._load_notes()
        pattern = re.compile(re.escape(query.strip()), re.I)
        results: list[SearchResult] = []

        for path, content in notes:
            if pattern.search(content):
                excerpt = self._extract_excerpt(content, query)
                title = self._extract_title(content, path)
                results.append(SearchResult(
                    note_path=path, title=title, excerpt=excerpt, score=1.0
                ))

        results = results[: self._max_results]
        return SearchReport(query=query, results=results, mode="keyword")

    def semantic_search(self, query: str) -> SearchReport:
        """
        Use Haiku to rank notes by relevance to a natural-language query.

        Falls back to keyword search if the API call fails.
        """
        notes = self._load_notes()
        if not notes:
            return SearchReport(query=query, results=[], mode="semantic")

        # Build a compact digest of all notes for Haiku
        digest = self._build_digest(notes)
        ranked = self._haiku_rank(query, digest, notes)
        return SearchReport(query=query, results=ranked[: self._max_results], mode="semantic")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_notes(self) -> list[tuple[Path, str]]:
        """Load all .md note files (excluding index.md), newest first."""
        if not self._notes_dir.exists():
            return []
        paths = sorted(
            (p for p in self._notes_dir.glob("*.md") if p.name != "index.md"),
            reverse=True,
        )
        results = []
        for path in paths:
            try:
                results.append((path, path.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("Could not read note %s", path)
        return results

    def _build_digest(self, notes: list[tuple[Path, str]]) -> str:
        """Build a compact multi-note digest for Haiku to rank."""
        parts = []
        for path, content in notes[:20]:   # cap at 20 notes to keep prompt small
            truncated = content[:_MAX_NOTE_CHARS]
            parts.append(f"=== {path.stem} ===\n{truncated}")
        return "\n\n".join(parts)

    def _haiku_rank(
        self,
        query: str,
        digest: str,
        notes: list[tuple[Path, str]],
    ) -> list[SearchResult]:
        system = (
            "You are a search assistant. Given a collection of conversation notes "
            "and a user query, identify the most relevant notes. "
            "Respond with JSON: a list of objects with keys 'slug' (the === slug ===), "
            "'score' (0.0-1.0), and 'excerpt' (1-2 sentences from the note explaining why it's relevant). "
            "Return only notes with score > 0.3, sorted by score descending. Max 5 items."
        )
        prompt = f"Query: {query}\n\nNotes:\n{digest}"
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            import json
            raw = resp.content[0].text.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()
            ranked_data: list[dict[str, Any]] = json.loads(raw)
        except Exception:
            logger.exception("Semantic search Haiku call failed; falling back to keyword")
            return self.keyword_search(query).results

        # Map slugs back to paths
        slug_map = {path.stem: (path, content) for path, content in notes}
        results: list[SearchResult] = []
        for item in ranked_data:
            slug = str(item.get("slug", "")).strip()
            if slug not in slug_map:
                continue
            path, content = slug_map[slug]
            results.append(SearchResult(
                note_path=path,
                title=self._extract_title(content, path),
                excerpt=str(item.get("excerpt", "")).strip(),
                score=float(item.get("score", 0.5)),
            ))
        return results

    @staticmethod
    def _extract_title(content: str, path: Path) -> str:
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return path.stem

    @staticmethod
    def _extract_excerpt(content: str, query: str) -> str:
        """Return the sentence/line containing the first match."""
        pattern = re.compile(re.escape(query.strip()), re.I)
        for line in content.splitlines():
            if pattern.search(line):
                return line.strip()[:200]
        return content[:200]

    @staticmethod
    def _is_semantic(query: str) -> bool:
        """Heuristic: treat questions/natural language as semantic queries."""
        q = query.strip().lower()
        return (
            len(q.split()) >= 4
            or q.endswith("?")
            or q.startswith(("when", "what", "where", "who", "how", "did i", "have i"))
        )
