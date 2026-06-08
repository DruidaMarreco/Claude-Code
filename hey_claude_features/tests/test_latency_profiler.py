"""Unit tests for LatencyProfiler — no API key or audio required."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hey_claude_features.latency_profiler import LatencyProfiler, QueryTiming, StageStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profiler(tmp_path: Path | None = None) -> LatencyProfiler:
    if tmp_path:
        return LatencyProfiler(persist_path=tmp_path / "latency.json")
    return LatencyProfiler()


def _record(profiler: LatencyProfiler, stt=0.1, gate=0.05, opus=0.5, tts=0.3) -> QueryTiming:
    profiler._pending = {"stt": stt, "gate": gate, "opus": opus, "tts": tts}
    return profiler.record_query()


# ---------------------------------------------------------------------------
# measure() context manager
# ---------------------------------------------------------------------------


class TestMeasure:
    def test_records_elapsed_time(self):
        p = _profiler()
        with p.measure("stt"):
            time.sleep(0.01)
        assert p._pending.get("stt", 0) >= 0.005

    def test_accumulates_multiple_calls(self):
        p = _profiler()
        with p.measure("stt"):
            time.sleep(0.005)
        with p.measure("stt"):
            time.sleep(0.005)
        assert p._pending["stt"] >= 0.008

    def test_exception_still_records(self):
        p = _profiler()
        try:
            with p.measure("opus"):
                raise ValueError("oops")
        except ValueError:
            pass
        assert "opus" in p._pending


# ---------------------------------------------------------------------------
# record_query()
# ---------------------------------------------------------------------------


class TestRecordQuery:
    def test_returns_query_timing(self):
        p = _profiler()
        qt = _record(p)
        assert isinstance(qt, QueryTiming)
        assert qt.stt == pytest.approx(0.1)
        assert qt.opus == pytest.approx(0.5)

    def test_total_auto_computed(self):
        p = _profiler()
        qt = _record(p, stt=0.1, gate=0.05, opus=0.5, tts=0.3)
        assert qt.total == pytest.approx(0.95)

    def test_explicit_total_not_overwritten(self):
        p = _profiler()
        p._pending = {"stt": 0.1, "total": 1.0}
        qt = p.record_query()
        assert qt.total == pytest.approx(1.0)

    def test_clears_pending_after_record(self):
        p = _profiler()
        _record(p)
        assert p._pending == {}

    def test_history_grows(self):
        p = _profiler()
        _record(p)
        _record(p)
        assert len(p) == 2

    def test_max_history_enforced(self):
        p = LatencyProfiler(max_history=3)
        for _ in range(5):
            _record(p)
        assert len(p) == 3


# ---------------------------------------------------------------------------
# query_summary()
# ---------------------------------------------------------------------------


class TestQuerySummary:
    def test_no_queries(self):
        p = _profiler()
        assert "No queries" in p.query_summary()

    def test_shows_stage_times(self):
        p = _profiler()
        _record(p, stt=0.1, opus=0.5)
        s = p.query_summary()
        assert "stt=" in s
        assert "opus=" in s

    def test_zero_stages_omitted(self):
        p = _profiler()
        p._pending = {"stt": 0.0, "opus": 0.5}
        p.record_query()
        s = p.query_summary()
        assert "stt=" not in s


# ---------------------------------------------------------------------------
# session_report() and compute_stats()
# ---------------------------------------------------------------------------


class TestSessionReport:
    def test_no_data(self):
        p = _profiler()
        assert "No query data" in p.session_report()

    def test_report_contains_stages(self):
        p = _profiler()
        _record(p, stt=0.1, gate=0.05, opus=0.5, tts=0.3)
        _record(p, stt=0.12, gate=0.06, opus=0.45, tts=0.28)
        report = p.session_report()
        assert "stt" in report
        assert "opus" in report

    def test_stats_count_correct(self):
        p = _profiler()
        for _ in range(5):
            _record(p)
        stats = {s.stage: s for s in p.compute_stats()}
        assert stats["stt"].count == 5
        assert stats["opus"].count == 5

    def test_mean_reasonable(self):
        p = _profiler()
        _record(p, opus=1.0)
        _record(p, opus=2.0)
        stats = {s.stage: s for s in p.compute_stats()}
        assert abs(stats["opus"].mean_ms - 1500) < 1

    def test_p95_gte_median(self):
        p = _profiler()
        for i in range(10):
            _record(p, opus=0.1 * (i + 1))
        stats = {s.stage: s for s in p.compute_stats()}
        assert stats["opus"].p95_ms >= stats["opus"].median_ms


# ---------------------------------------------------------------------------
# slowest_stage()
# ---------------------------------------------------------------------------


class TestSlowestStage:
    def test_identifies_opus_as_slowest(self):
        p = _profiler()
        _record(p, stt=0.05, gate=0.03, opus=0.8, tts=0.2)
        assert p.slowest_stage() == "opus"

    def test_no_data_returns_none(self):
        p = _profiler()
        assert p.slowest_stage() is None

    def test_excludes_total_from_comparison(self):
        p = _profiler()
        _record(p, stt=0.05, gate=0.03, opus=0.8, tts=0.2)
        assert p.slowest_stage() != "total"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_saves_and_reloads(self, tmp_path):
        p1 = _profiler(tmp_path)
        _record(p1, opus=0.5)

        p2 = _profiler(tmp_path)
        assert len(p2) == 1
        assert p2._history[0].opus == pytest.approx(0.5)

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "latency.json"
        path.write_text("{bad json")
        p = LatencyProfiler(persist_path=path)
        assert len(p) == 0

    def test_clear_empties_history(self):
        p = _profiler()
        _record(p)
        p.clear()
        assert len(p) == 0
