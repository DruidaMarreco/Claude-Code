"""Unit tests for CostTracker — no API key required."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hey_claude_features.cost_tracker import CostTracker, QueryCost, SessionTotals, _price_for


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage(input_tokens=100, output_tokens=50, cache_read=0, cache_write=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def _usage_dict(input_tokens=100, output_tokens=50):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestPricing:
    def test_haiku_price(self):
        inp, out = _price_for("claude-haiku-4-5")
        assert inp == 0.80
        assert out == 4.00

    def test_opus_price(self):
        inp, out = _price_for("claude-opus-4-8")
        assert inp == 15.00
        assert out == 75.00

    def test_unknown_model_fallback(self):
        inp, out = _price_for("claude-unknown-9-9")
        assert inp == 3.00
        assert out == 15.00


# ---------------------------------------------------------------------------
# QueryCost
# ---------------------------------------------------------------------------


class TestQueryCost:
    def test_total_tokens(self):
        qc = QueryCost(
            model="claude-haiku-4-5",
            input_tokens=100, output_tokens=50,
            cache_read_tokens=0, cache_write_tokens=0,
        )
        assert qc.total_tokens == 150

    def test_usd_calculation_haiku(self):
        qc = QueryCost(
            model="claude-haiku-4-5",
            input_tokens=1_000_000, output_tokens=0,
            cache_read_tokens=0, cache_write_tokens=0,
        )
        assert abs(qc.usd - 0.80) < 0.001

    def test_usd_calculation_opus(self):
        qc = QueryCost(
            model="claude-opus-4-8",
            input_tokens=0, output_tokens=1_000_000,
            cache_read_tokens=0, cache_write_tokens=0,
        )
        assert abs(qc.usd - 75.00) < 0.001


# ---------------------------------------------------------------------------
# CostTracker.record
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_returns_query_cost(self):
        tracker = CostTracker()
        qc = tracker.record(_usage(100, 50), model="claude-haiku-4-5")
        assert isinstance(qc, QueryCost)
        assert qc.input_tokens == 100
        assert qc.output_tokens == 50

    def test_record_dict_usage(self):
        tracker = CostTracker()
        qc = tracker.record(_usage_dict(200, 80), model="claude-haiku-4-5")
        assert qc.input_tokens == 200

    def test_session_accumulates(self):
        tracker = CostTracker()
        tracker.record(_usage(100, 50), model="claude-haiku-4-5")
        tracker.record(_usage(200, 100), model="claude-haiku-4-5")
        assert tracker.session.queries == 2
        assert tracker.session.input_tokens == 300
        assert tracker.session.output_tokens == 150

    def test_last_reflects_most_recent(self):
        tracker = CostTracker()
        tracker.record(_usage(100, 50), model="claude-haiku-4-5")
        tracker.record(_usage(999, 1), model="claude-haiku-4-5")
        assert tracker.last.input_tokens == 999

    def test_cache_tokens_recorded(self):
        tracker = CostTracker()
        tracker.record(_usage(100, 50, cache_read=20, cache_write=5), model="claude-haiku-4-5")
        assert tracker.session.cache_read_tokens == 20
        assert tracker.session.cache_write_tokens == 5


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


class TestSummaries:
    def test_last_query_summary_no_queries(self):
        tracker = CostTracker()
        assert "No queries" in tracker.last_query_summary()

    def test_last_query_summary_after_record(self):
        tracker = CostTracker()
        tracker.record(_usage(100, 50), model="claude-haiku-4-5")
        s = tracker.last_query_summary()
        assert "100" in s
        assert "50" in s
        assert "$" in s

    def test_session_summary_shows_totals(self):
        tracker = CostTracker()
        tracker.record(_usage(500, 200), model="claude-opus-4-8")
        s = tracker.session_summary()
        assert "1 queries" in s
        assert "$" in s


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_session(self):
        tracker = CostTracker()
        tracker.record(_usage(100, 50), model="claude-haiku-4-5")
        tracker.reset_session()
        assert tracker.session.queries == 0
        assert tracker.session.usd == 0.0
        assert tracker.last is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "costs.json"
        t1 = CostTracker(persist_path=path)
        t1.record(_usage(1000, 500), model="claude-opus-4-8")

        t2 = CostTracker(persist_path=path)
        assert t2.session.queries == 1
        assert t2.session.input_tokens == 1000
        assert t2.session.usd > 0

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "costs.json"
        path.write_text("{bad json")
        t = CostTracker(persist_path=path)
        assert t.session.queries == 0

    def test_missing_file_starts_fresh(self, tmp_path):
        t = CostTracker(persist_path=tmp_path / "nope.json")
        assert t.session.queries == 0

    def test_custom_pricing_override(self):
        tracker = CostTracker(pricing={"claude-haiku-4-5": (1.00, 5.00)})
        qc = tracker.record(_usage(1_000_000, 0), model="claude-haiku-4-5")
        assert abs(qc.usd - 1.00) < 0.001
