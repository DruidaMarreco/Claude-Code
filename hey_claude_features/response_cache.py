"""
LRU response cache for Hey-Claude's Opus responder.

Repeated or near-identical queries (e.g. "what's the weather" asked twice in a
row) skip the Anthropic API entirely and return the cached reply instantly.

Features
--------
- Exact-match cache keyed on normalized query text (lowercased, stripped)
- Configurable TTL: entries expire after `ttl_seconds` (default 5 min)
- Configurable capacity: LRU eviction once `max_size` entries are reached
- Optional disk persistence: survives assistant restarts
- Thread-safe: uses a simple lock (Hey-Claude is serial, but --serve mode is not)

Usage in responder.py
---------------------
    from hey_claude_features.response_cache import ResponseCache

    cache = ResponseCache(max_size=256, ttl_seconds=300)

    def respond(query: str) -> str:
        cached = cache.get(query)
        if cached is not None:
            return cached
        reply = _call_opus(query)
        cache.set(query, reply)
        return reply

Disk persistence
----------------
    cache = ResponseCache(persist_path=Path("~/.hey-claude/response_cache.json").expanduser())

Cache is saved on every `set()` call and loaded on init.  JSON format keeps it
human-readable so you can inspect/clear it manually.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 256
_DEFAULT_TTL = 300  # 5 minutes


@dataclass
class _Entry:
    reply: str
    expires_at: float  # unix timestamp


class ResponseCache:
    """
    Thread-safe LRU cache with TTL for voice assistant responses.

    Parameters
    ----------
    max_size:
        Maximum number of entries before LRU eviction kicks in.
    ttl_seconds:
        How long an entry stays valid.  0 = never expires.
    persist_path:
        Optional path to a JSON file for cross-restart persistence.
    """

    def __init__(
        self,
        max_size: int = _DEFAULT_MAX_SIZE,
        ttl_seconds: float = _DEFAULT_TTL,
        persist_path: Path | None = None,
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.persist_path = persist_path
        self._lock = threading.Lock()
        self._store: OrderedDict[str, _Entry] = OrderedDict()

        if persist_path:
            self._load(persist_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> str | None:
        """Return the cached reply for *query*, or None if missing/expired."""
        key = _normalize(query)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._store[key]
                return None
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            return entry.reply

    def set(self, query: str, reply: str) -> None:
        """Cache *reply* for *query*, evicting LRU entries if at capacity."""
        key = _normalize(query)
        expires_at = time.time() + self.ttl_seconds if self.ttl_seconds > 0 else float("inf")
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = _Entry(reply=reply, expires_at=expires_at)
            while len(self._store) > self.max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("LRU evicted: %s", evicted_key[:40])
        if self.persist_path:
            self._save(self.persist_path)

    def invalidate(self, query: str) -> bool:
        """Remove a single entry.  Returns True if it existed."""
        key = _normalize(query)
        with self._lock:
            existed = key in self._store
            self._store.pop(key, None)
        return existed

    def clear(self) -> None:
        """Remove all entries and wipe the persist file if configured."""
        with self._lock:
            self._store.clear()
        if self.persist_path and self.persist_path.exists():
            self.persist_path.write_text("{}")

    def stats(self) -> dict[str, Any]:
        """Return cache statistics for monitoring/debugging."""
        with self._lock:
            total = len(self._store)
            now = time.time()
            expired = sum(1 for e in self._store.values() if self._is_expired(e, now=now))
        return {"total": total, "live": total - expired, "expired": expired}

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_expired(self, entry: _Entry, now: float | None = None) -> bool:
        if entry.expires_at == float("inf"):
            return False
        return (now or time.time()) >= entry.expires_at

    def _save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {k: asdict(v) for k, v in self._store.items()}
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            logger.exception("Failed to persist response cache to %s", path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data: dict[str, Any] = json.loads(path.read_text())
            now = time.time()
            for key, raw in data.items():
                entry = _Entry(**raw)
                if not self._is_expired(entry, now=now):
                    self._store[key] = entry
            logger.debug("Loaded %d live cache entries from %s", len(self._store), path)
        except Exception:
            logger.exception("Failed to load response cache from %s; starting fresh", path)


def _normalize(query: str) -> str:
    """Normalize a query to a cache key: lowercase, collapse whitespace."""
    return " ".join(query.lower().split())
