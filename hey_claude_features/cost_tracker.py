"""
Per-query and per-session cost tracker for Hey-Claude.

Reads the `usage` object already present on every Anthropic API response and
accumulates token counts and estimated USD cost.  No extra API calls needed.

Pricing defaults match the June 2026 public rates; override via constructor
if Anthropic updates them.

Usage in responder.py
---------------------
    from hey_claude_features.cost_tracker import CostTracker

    tracker = CostTracker()

    # After every API call:
    response = client.messages.create(...)
    tracker.record(response.usage, model=config.responder_model)

    # Log a per-query summary:
    print(tracker.last_query_summary())

    # Log session totals:
    print(tracker.session_summary())

    # Persist across restarts:
    tracker = CostTracker(persist_path=Path("~/.hey-claude/costs.json").expanduser())
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# USD per million tokens (input / output) by model prefix.
# Haiku is used for gate + summarizer; Opus for responder.
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":    (0.80,   4.00),   # $0.80 / $4.00 per M tokens
    "claude-sonnet-4":     (3.00,  15.00),
    "claude-opus-4":      (15.00,  75.00),
    "claude-opus-4-8":    (15.00,  75.00),
}
_FALLBACK_PRICING = (3.00, 15.00)   # used when model not found


def _price_for(model: str) -> tuple[float, float]:
    for prefix, prices in _DEFAULT_PRICING.items():
        if model.startswith(prefix):
            return prices
    return _FALLBACK_PRICING


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class QueryCost:
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def usd(self) -> float:
        input_price, output_price = _price_for(self.model)
        return (
            self.input_tokens * input_price / 1_000_000
            + self.output_tokens * output_price / 1_000_000
        )


@dataclass
class SessionTotals:
    queries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usd: float = 0.0
    started_at: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, qc: QueryCost) -> None:
        self.queries += 1
        self.input_tokens += qc.input_tokens
        self.output_tokens += qc.output_tokens
        self.cache_read_tokens += qc.cache_read_tokens
        self.cache_write_tokens += qc.cache_write_tokens
        self.usd += qc.usd


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """
    Accumulates token usage and USD cost across API calls.

    Parameters
    ----------
    pricing:
        Override default per-model pricing.  Dict of model-prefix →
        (input_usd_per_million, output_usd_per_million).
    persist_path:
        If set, session totals are saved to this JSON file after every
        recorded query and loaded on init.
    """

    def __init__(
        self,
        pricing: dict[str, tuple[float, float]] | None = None,
        persist_path: Path | None = None,
    ) -> None:
        if pricing:
            _DEFAULT_PRICING.update(pricing)
        self._persist_path = persist_path
        self._session = SessionTotals()
        self._last: QueryCost | None = None
        if persist_path:
            self._load(persist_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, usage: Any, model: str) -> QueryCost:
        """
        Record token usage from an Anthropic API response.

        *usage* is the `response.usage` object (or any object / dict with
        `input_tokens` and `output_tokens` attributes/keys).
        """
        input_tokens = _get(usage, "input_tokens", 0)
        output_tokens = _get(usage, "output_tokens", 0)
        cache_read = _get(usage, "cache_read_input_tokens", 0)
        cache_write = _get(usage, "cache_creation_input_tokens", 0)

        qc = QueryCost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        self._last = qc
        self._session.add(qc)
        logger.debug(
            "Cost recorded: %s  in=%d out=%d  $%.5f",
            model, input_tokens, output_tokens, qc.usd,
        )
        if self._persist_path:
            self._save(self._persist_path)
        return qc

    def last_query_summary(self) -> str:
        """Human-readable one-liner for the most recent query."""
        if self._last is None:
            return "No queries recorded yet."
        q = self._last
        return (
            f"Last query — {q.model}: "
            f"{q.input_tokens} in + {q.output_tokens} out = "
            f"{q.total_tokens} tokens  (${q.usd:.4f})"
        )

    def session_summary(self) -> str:
        """Human-readable session totals."""
        s = self._session
        elapsed = time.time() - s.started_at
        mins = int(elapsed / 60)
        return (
            f"Session ({mins}m) — {s.queries} queries  "
            f"{s.input_tokens} in + {s.output_tokens} out = "
            f"{s.total_tokens} tokens  total ${s.usd:.4f}"
        )

    @property
    def session(self) -> SessionTotals:
        return self._session

    @property
    def last(self) -> QueryCost | None:
        return self._last

    def reset_session(self) -> None:
        """Start a fresh session (does not clear the persist file history)."""
        self._session = SessionTotals()
        self._last = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(self._session)
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            logger.exception("Failed to save cost tracker to %s", path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data: dict[str, Any] = json.loads(path.read_text())
            self._session = SessionTotals(**data)
            logger.debug("Loaded cost session from %s ($%.4f so far)", path, self._session.usd)
        except Exception:
            logger.exception("Failed to load cost tracker from %s; starting fresh", path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str, default: int) -> int:
    """Get an attribute or dict key, returning default if missing."""
    if isinstance(obj, dict):
        return int(obj.get(key, default))
    return int(getattr(obj, key, default))
