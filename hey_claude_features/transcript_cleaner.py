"""
Whisper transcript post-processor for Hey-Claude.

Whisper produces raw transcripts that often contain:
  - Filler words ("um", "uh", "like", "you know")
  - Repeated words from false starts ("I I want to…")
  - Inconsistent capitalisation mid-sentence
  - Spoken numbers that should be digits ("three PM" → "3 PM")
  - Trailing ellipses and stutter artefacts

This module cleans those artefacts so the Haiku gate and Opus responder
receive a cleaner, more parseable string — improving classification accuracy
and reducing token count.

Usage in app.py
---------------
    from hey_claude_features.transcript_cleaner import TranscriptCleaner

    cleaner = TranscriptCleaner()
    clean = cleaner.clean(raw_whisper_transcript)

All transforms are applied in a fixed pipeline and are individually
toggleable via the constructor.

Custom rules
------------
    cleaner = TranscriptCleaner()
    cleaner.add_rule(r"\\bblimey\\b", "")        # regex replace
    cleaner.add_rule(r"\\bsort of\\b", "")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Number-word → digit mapping
# ---------------------------------------------------------------------------

_ONES = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}

# Build a regex that matches any single number word (word-boundary anchored)
_NUMBER_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ONES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _replace_number_words(text: str) -> str:
    return _NUMBER_WORD_RE.sub(lambda m: _ONES[m.group(0).lower()], text)


# ---------------------------------------------------------------------------
# Built-in filler words
# ---------------------------------------------------------------------------

_DEFAULT_FILLERS = (
    "um", "uh", "er", "ah", "hmm", "hm",
    "like", "you know", "i mean", "sort of", "kind of",
    "basically", "literally", "actually", "honestly",
    "right", "okay so", "so like",
)


def _filler_pattern(fillers: tuple[str, ...]) -> re.Pattern[str]:
    escaped = sorted((re.escape(f) for f in fillers), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b[,]?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CleanResult:
    """Result of a single clean() call."""
    original: str
    cleaned: str
    changes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.original != self.cleaned

    def __str__(self) -> str:
        return self.cleaned


# ---------------------------------------------------------------------------
# Custom rule
# ---------------------------------------------------------------------------


@dataclass
class CleanRule:
    pattern: re.Pattern[str]
    replacement: str
    description: str = ""


# ---------------------------------------------------------------------------
# TranscriptCleaner
# ---------------------------------------------------------------------------


class TranscriptCleaner:
    """
    Post-processes Whisper transcripts to remove noise before LLM classification.

    Parameters
    ----------
    remove_fillers:
        Strip um/uh/like/you know style filler words.
    deduplicate_words:
        Collapse immediate word repetitions ("I I want" → "I want").
    normalize_numbers:
        Replace isolated number words with digits ("three" → "3").
    normalize_whitespace:
        Collapse multiple spaces and strip leading/trailing whitespace.
    extra_fillers:
        Additional filler words to remove on top of the defaults.
    """

    def __init__(
        self,
        remove_fillers: bool = True,
        deduplicate_words: bool = True,
        normalize_numbers: bool = True,
        normalize_whitespace: bool = True,
        extra_fillers: tuple[str, ...] = (),
    ) -> None:
        self._remove_fillers = remove_fillers
        self._deduplicate = deduplicate_words
        self._normalize_numbers = normalize_numbers
        self._normalize_ws = normalize_whitespace

        fillers = _DEFAULT_FILLERS + extra_fillers
        self._filler_re = _filler_pattern(fillers)

        # Repeated-word pattern: "word word" → "word" (case-insensitive)
        self._dedup_re = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

        # Custom rules added by .add_rule()
        self._custom_rules: list[CleanRule] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, transcript: str) -> CleanResult:
        """Apply all enabled transforms and return a CleanResult."""
        text = transcript
        changes: list[str] = []

        # 1. Custom rules first (caller may want to override before defaults)
        for rule in self._custom_rules:
            new = rule.pattern.sub(rule.replacement, text)
            if new != text:
                changes.append(rule.description or f"custom:{rule.pattern.pattern}")
                text = new

        # 2. Filler removal
        if self._remove_fillers:
            new = self._filler_re.sub("", text)
            if new != text:
                changes.append("fillers")
                text = new

        # 3. Duplicate-word collapse
        if self._deduplicate:
            prev = None
            while prev != text:
                prev = text
                new = self._dedup_re.sub(r"\1", text)
                if new != text:
                    changes.append("dedup")
                text = new

        # 4. Number-word normalisation
        if self._normalize_numbers:
            new = _replace_number_words(text)
            if new != text:
                changes.append("numbers")
                text = new

        # 5. Whitespace normalisation (always last)
        if self._normalize_ws:
            new = re.sub(r" {2,}", " ", text).strip()
            if new != text:
                changes.append("whitespace")
                text = new

        return CleanResult(original=transcript, cleaned=text, changes=list(set(changes)))

    def clean_str(self, transcript: str) -> str:
        """Convenience wrapper — returns the cleaned string directly."""
        return self.clean(transcript).cleaned

    def add_rule(
        self,
        pattern: str,
        replacement: str,
        flags: int = re.IGNORECASE,
        description: str = "",
    ) -> None:
        """Register a custom regex substitution rule."""
        self._custom_rules.append(CleanRule(
            pattern=re.compile(pattern, flags),
            replacement=replacement,
            description=description or pattern,
        ))

    def remove_rule(self, description: str) -> bool:
        """Remove a custom rule by its description. Returns True if found."""
        before = len(self._custom_rules)
        self._custom_rules = [r for r in self._custom_rules if r.description != description]
        return len(self._custom_rules) < before
