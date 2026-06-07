"""Unit tests for ResponseCache — no API key or I/O required."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hey_claude_features.response_cache import ResponseCache, _normalize


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize("  what   time  is  it  ") == "what time is it"

    def test_empty_string(self):
        assert _normalize("") == ""


# ---------------------------------------------------------------------------
# Basic get/set
# ---------------------------------------------------------------------------


class TestGetSet:
    def test_miss_returns_none(self):
        cache = ResponseCache()
        assert cache.get("anything") is None

    def test_hit_returns_reply(self):
        cache = ResponseCache()
        cache.set("hello", "world")
        assert cache.get("hello") == "world"

    def test_case_insensitive_key(self):
        cache = ResponseCache()
        cache.set("What Time Is It", "3pm")
        assert cache.get("what time is it") == "3pm"
        assert cache.get("WHAT TIME IS IT") == "3pm"

    def test_whitespace_normalized(self):
        cache = ResponseCache()
        cache.set("  hey  claude  ", "hi")
        assert cache.get("hey claude") == "hi"

    def test_overwrite_existing(self):
        cache = ResponseCache()
        cache.set("q", "first")
        cache.set("q", "second")
        assert cache.get("q") == "second"


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


class TestTTL:
    def test_expired_entry_returns_none(self):
        cache = ResponseCache(ttl_seconds=0.01)
        cache.set("q", "answer")
        time.sleep(0.05)
        assert cache.get("q") is None

    def test_non_expired_entry_is_returned(self):
        cache = ResponseCache(ttl_seconds=60)
        cache.set("q", "answer")
        assert cache.get("q") == "answer"

    def test_zero_ttl_never_expires(self):
        cache = ResponseCache(ttl_seconds=0)
        cache.set("q", "answer")
        assert cache.get("q") == "answer"

    def test_expired_entry_removed_from_store(self):
        cache = ResponseCache(ttl_seconds=0.01)
        cache.set("q", "answer")
        time.sleep(0.05)
        cache.get("q")  # triggers removal
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_oldest_evicted_at_capacity(self):
        cache = ResponseCache(max_size=3, ttl_seconds=0)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.set("d", "4")  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("d") == "4"

    def test_recently_used_not_evicted(self):
        cache = ResponseCache(max_size=3, ttl_seconds=0)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.get("a")      # promote "a" to MRU
        cache.set("d", "4") # evicts "b" (now LRU)
        assert cache.get("a") == "1"
        assert cache.get("b") is None

    def test_len_respects_max_size(self):
        cache = ResponseCache(max_size=5, ttl_seconds=0)
        for i in range(10):
            cache.set(str(i), str(i))
        assert len(cache) == 5


# ---------------------------------------------------------------------------
# Invalidate / clear
# ---------------------------------------------------------------------------


class TestInvalidateClear:
    def test_invalidate_existing_returns_true(self):
        cache = ResponseCache()
        cache.set("q", "a")
        assert cache.invalidate("q") is True
        assert cache.get("q") is None

    def test_invalidate_missing_returns_false(self):
        cache = ResponseCache()
        assert cache.invalidate("not there") is False

    def test_clear_empties_cache(self):
        cache = ResponseCache()
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_counts(self):
        cache = ResponseCache(ttl_seconds=60)
        cache.set("a", "1")
        cache.set("b", "2")
        s = cache.stats()
        assert s["total"] == 2
        assert s["live"] == 2
        assert s["expired"] == 0

    def test_stats_expired_counted(self):
        cache = ResponseCache(ttl_seconds=0.01)
        cache.set("a", "1")
        time.sleep(0.05)
        s = cache.stats()
        assert s["expired"] == 1
        assert s["live"] == 0


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "cache.json"
        c1 = ResponseCache(ttl_seconds=60, persist_path=path)
        c1.set("hello", "world")

        c2 = ResponseCache(ttl_seconds=60, persist_path=path)
        assert c2.get("hello") == "world"

    def test_expired_entries_not_loaded(self, tmp_path):
        path = tmp_path / "cache.json"
        # Write an already-expired entry directly
        data = {"hello": {"reply": "world", "expires_at": time.time() - 1}}
        path.write_text(json.dumps(data))

        c = ResponseCache(ttl_seconds=60, persist_path=path)
        assert c.get("hello") is None

    def test_clear_wipes_file(self, tmp_path):
        path = tmp_path / "cache.json"
        c = ResponseCache(ttl_seconds=60, persist_path=path)
        c.set("q", "a")
        c.clear()
        assert json.loads(path.read_text()) == {}

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text("not valid json }{")
        c = ResponseCache(ttl_seconds=60, persist_path=path)
        assert len(c) == 0
