"""
Per-stage latency profiler for Hey-Claude.

Records timing for each pipeline stage (STT, gate, Opus, TTS) across all
queries in a session and across sessions.  Helps identify where time is
being spent and whether optimisations are working.

Pipeline stages
---------------
    STT      — Whisper transcription
    GATE     — Haiku wake-word gate
    OPUS     — Opus responder (time-to-first-token + full generation)
    TTS      — Text-to-speech playback
    TOTAL    — Wall time from utterance end to first audio byte

Usage in app.py
---------------
    from hey_claude_features.latency_profiler import LatencyProfiler

    profiler = LatencyProfiler(persist_path=Path("~/.hey-claude/latency.json").expanduser())

    # Wrap each stage with the profiler's context manager:
    with profiler.measure("stt"):
        transcript = transcriber.transcribe(audio)

    with profiler.measure("gate"):
        result = wake_detector.detect(transcript)

    with profiler.measure("opus"):
        reply = responder.respond(query)

    with profiler.measure("tts"):
        speaker.speak(reply)

    profiler.record_query()   # saves the current query's timings

    # Print a summary after each query:
    print(profiler.query_summary())

    # Print session stats:
    print(profiler.session_report())
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator, Iterator

import numpy as np

logger = logging.getLogger(__name__)

STAGES = ("stt", "gate", "opus", "tts", "total")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class QueryTiming:
    """Latency breakdown for a single query (all values in seconds)."""
    stt: float = 0.0
    gate: float = 0.0
    opus: float = 0.0
    tts: float = 0.0
    total: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, float]:
        return {s: getattr(self, s) for s in STAGES}


@dataclass
class StageStats:
    stage: str
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def __str__(self) -> str:
        return (
            f"{self.stage:8s}  mean={self.mean_ms:6.0f}ms  "
            f"p50={self.median_ms:6.0f}ms  p95={self.p95_ms:6.0f}ms  "
            f"min={self.min_ms:5.0f}ms  max={self.max_ms:6.0f}ms  (n={self.count})"
        )


# ---------------------------------------------------------------------------
# LatencyProfiler
# ---------------------------------------------------------------------------


class LatencyProfiler:
    """
    Records and analyses per-stage latency for Hey-Claude queries.

    Parameters
    ----------
    persist_path:
        If set, query timings are appended to a JSON file on each
        `record_query()` call so data survives restarts.
    max_history:
        Maximum number of QueryTiming records kept in memory.
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        max_history: int = 1000,
    ) -> None:
        self._persist_path = persist_path
        self._max_history = max_history
        self._history: list[QueryTiming] = []
        self._pending: dict[str, float] = {}   # stage → elapsed seconds
        self._active_start: dict[str, float] = {}

        if persist_path:
            self._load(persist_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def measure(self, stage: str) -> Generator[None, None, None]:
        """Context manager that measures wall time for a pipeline stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._pending[stage] = self._pending.get(stage, 0.0) + elapsed

    def record_query(self) -> QueryTiming:
        """
        Commit the current pending timings as a completed QueryTiming.

        Call this once per query after all stages have run.
        Automatically computes `total` as the sum of STT + GATE + OPUS + TTS
        unless explicitly set.
        """
        if "total" not in self._pending:
            self._pending["total"] = sum(
                self._pending.get(s, 0.0) for s in ("stt", "gate", "opus", "tts")
            )

        qt = QueryTiming(**{s: self._pending.get(s, 0.0) for s in STAGES})
        self._history.append(qt)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        self._pending.clear()

        if self._persist_path:
            self._append(qt)

        return qt

    def query_summary(self) -> str:
        """One-liner for the most recently recorded query."""
        if not self._history:
            return "No queries recorded yet."
        qt = self._history[-1]
        parts = [f"{s}={getattr(qt, s)*1000:.0f}ms" for s in STAGES if getattr(qt, s) > 0]
        return "Latency: " + "  ".join(parts)

    def session_report(self) -> str:
        """Multi-line session statistics table."""
        if not self._history:
            return "No query data yet."

        lines = [f"Latency Report — {len(self._history)} queries", "-" * 70]
        for stats in self.compute_stats():
            if stats.count > 0:
                lines.append(str(stats))
        return "\n".join(lines)

    def compute_stats(self) -> list[StageStats]:
        """Return StageStats for each pipeline stage."""
        results = []
        for stage in STAGES:
            values = [getattr(qt, stage) for qt in self._history if getattr(qt, stage) > 0]
            if not values:
                results.append(StageStats(stage=stage, count=0, mean_ms=0, median_ms=0, p95_ms=0, min_ms=0, max_ms=0))
                continue
            arr = np.array(values) * 1000  # → ms
            results.append(StageStats(
                stage=stage,
                count=len(arr),
                mean_ms=float(np.mean(arr)),
                median_ms=float(np.median(arr)),
                p95_ms=float(np.percentile(arr, 95)),
                min_ms=float(np.min(arr)),
                max_ms=float(np.max(arr)),
            ))
        return results

    def slowest_stage(self) -> str | None:
        """Return the name of the slowest stage by mean latency."""
        stats = [s for s in self.compute_stats() if s.count > 0 and s.stage != "total"]
        if not stats:
            return None
        return max(stats, key=lambda s: s.mean_ms).stage

    def clear(self) -> None:
        self._history.clear()
        self._pending.clear()

    def __len__(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _append(self, qt: QueryTiming) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            existing: list[dict] = []
            if self._persist_path.exists():
                existing = json.loads(self._persist_path.read_text())
            existing.append(asdict(qt))
            self._persist_path.write_text(json.dumps(existing, indent=2))
        except Exception:
            logger.exception("Failed to persist latency data")

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            records: list[dict] = json.loads(path.read_text())
            for r in records[-self._max_history:]:
                self._history.append(QueryTiming(**r))
            logger.debug("Loaded %d latency records from %s", len(self._history), path)
        except Exception:
            logger.exception("Failed to load latency data from %s", path)
