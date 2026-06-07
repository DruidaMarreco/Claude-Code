"""
Multi-wake-phrase support for Hey-Claude.

Extends the existing single-phrase prefilter to support a list of phrases,
each with independent fuzzy matching.  A transcript passes the pre-filter
if it looks like ANY configured wake phrase — making it a drop-in upgrade
for prefilter.py.

Also provides MultiPhraseWakeDetector, a WakeDetector-compatible class
that wraps the model-based detector with multi-phrase awareness:
  1. Free fuzzy pre-filter against all configured phrases
  2. If any phrase matches, ask Haiku to confirm AND identify which phrase
     was used (so the extracted query is clean regardless of phrasing)

Default phrases: ["hey claude", "ok claude"]
Users can add custom phrases via config or environment variable.

Usage — replacing prefilter.py
-------------------------------
    from hey_claude_features.multi_wake_phrase import MultiPhrasePrefilter

    prefilter = MultiPhrasePrefilter(["hey claude", "ok claude", "yo claude"])
    if prefilter.looks_like_wake(transcript):
        ...  # proceed to Haiku gate

Usage — replacing WakeDetector entirely
-----------------------------------------
    from hey_claude_features.multi_wake_phrase import MultiPhraseWakeDetector

    detector = MultiPhraseWakeDetector(
        client=anthropic_client,
        phrases=["hey claude", "ok claude"],
        model="claude-haiku-4-5",
    )
    result = detector.detect(transcript)
    if result.detected:
        handle(result.query)

Environment variable
--------------------
    HEY_CLAUDE_WAKE_PHRASES=hey claude,ok claude,yo claude
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)

_DEFAULT_PHRASES = ["hey claude", "ok claude"]
_FUZZY_THRESHOLD = 0.66   # matching prefilter.py's existing threshold


# ---------------------------------------------------------------------------
# Fuzzy helpers (mirrors prefilter.py logic, extended for lists)
# ---------------------------------------------------------------------------


def _simple_ratio(a: str, b: str) -> float:
    """Dice-coefficient-like similarity: 2 * common_chars / total_chars."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    matches = sum(c in longer for c in shorter)
    return 2 * matches / (len(a) + len(b))


def _looks_like_phrase(transcript: str, phrase: str, threshold: float = _FUZZY_THRESHOLD) -> bool:
    """Return True if *transcript* contains or resembles *phrase*."""
    t = transcript.lower().strip()
    p = phrase.lower().strip()
    if p in t:
        return True
    # Check if any window of similar length fuzzy-matches
    words = t.split()
    phrase_words = p.split()
    window = len(phrase_words)
    for i in range(max(1, len(words) - window + 1)):
        chunk = " ".join(words[i : i + window])
        if _simple_ratio(chunk, p) >= threshold:
            return True
    return False


def phrases_from_env(env_var: str = "HEY_CLAUDE_WAKE_PHRASES") -> list[str] | None:
    """Parse a comma-separated list of phrases from an env var."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


# ---------------------------------------------------------------------------
# MultiPhrasePrefilter
# ---------------------------------------------------------------------------


class MultiPhrasePrefilter:
    """
    Drop-in replacement for prefilter.py's `looks_like_wake()`.

    Checks a transcript against all configured wake phrases.  Returns True
    if the transcript resembles any of them (substring OR fuzzy match).

    Parameters
    ----------
    phrases:
        List of wake phrases to check against.  If None, falls back to
        HEY_CLAUDE_WAKE_PHRASES env var, then ["hey claude", "ok claude"].
    threshold:
        Fuzzy similarity threshold (0–1).  Mirrors prefilter.py default.
    """

    def __init__(
        self,
        phrases: list[str] | None = None,
        threshold: float = _FUZZY_THRESHOLD,
    ) -> None:
        self.phrases = (
            phrases
            or phrases_from_env()
            or list(_DEFAULT_PHRASES)
        )
        self.threshold = threshold
        logger.debug("MultiPhrasePrefilter phrases: %s", self.phrases)

    def looks_like_wake(self, transcript: str) -> bool:
        """Return True if the transcript resembles any wake phrase."""
        return any(
            _looks_like_phrase(transcript, phrase, self.threshold)
            for phrase in self.phrases
        )

    def matching_phrase(self, transcript: str) -> str | None:
        """Return the first phrase that matches, or None."""
        for phrase in self.phrases:
            if _looks_like_phrase(transcript, phrase, self.threshold):
                return phrase
        return None


# ---------------------------------------------------------------------------
# WakeResult (mirrors hey_claude/wake.py)
# ---------------------------------------------------------------------------


@dataclass
class WakeResult:
    detected: bool
    query: str = ""
    matched_phrase: str = ""


# ---------------------------------------------------------------------------
# MultiPhraseWakeDetector
# ---------------------------------------------------------------------------

_DETECT_SYSTEM = """\
You are a wake-word detector for a voice assistant. The assistant responds to
multiple wake phrases. Given a transcript, determine:

1. Was a wake phrase present? (yes/no)
2. Which phrase was used?
3. What was the user's actual question or command (everything after the wake phrase)?

Respond with JSON only:
{"detected": true/false, "phrase": "<matched phrase or empty>", "query": "<extracted query or empty>"}
"""


class MultiPhraseWakeDetector:
    """
    WakeDetector-compatible detector supporting multiple wake phrases.

    Three-stage pipeline:
      1. Free fuzzy pre-filter — bail early if no phrase is close
      2. Haiku gate — confirm detection and extract the clean query
      3. Return WakeResult

    Parameters
    ----------
    client:
        Anthropic client.
    phrases:
        Wake phrases to listen for.
    model:
        Haiku model for gate confirmation.
    threshold:
        Fuzzy pre-filter threshold.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        phrases: list[str] | None = None,
        model: str = "claude-haiku-4-5",
        threshold: float = _FUZZY_THRESHOLD,
    ) -> None:
        self._client = client
        self._prefilter = MultiPhrasePrefilter(phrases=phrases, threshold=threshold)
        self._model = model

    @property
    def phrases(self) -> list[str]:
        return self._prefilter.phrases

    def detect(self, transcript: str) -> WakeResult:
        """Detect wake phrase in *transcript* and return WakeResult."""
        if not self._prefilter.looks_like_wake(transcript):
            return WakeResult(detected=False)
        return self._haiku_confirm(transcript)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _haiku_confirm(self, transcript: str) -> WakeResult:
        phrases_list = ", ".join(f'"{p}"' for p in self.phrases)
        prompt = (
            f"Wake phrases to listen for: {phrases_list}\n"
            f"Transcript: \"{transcript}\""
        )
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=128,
                system=_DETECT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            import json
            data = json.loads(resp.content[0].text.strip())
            return WakeResult(
                detected=bool(data.get("detected", False)),
                query=data.get("query", "").strip(),
                matched_phrase=data.get("phrase", "").strip(),
            )
        except Exception:
            logger.exception("Haiku multi-phrase detection failed; falling back to detected=True")
            matched = self._prefilter.matching_phrase(transcript) or ""
            return WakeResult(detected=True, query=transcript, matched_phrase=matched)
