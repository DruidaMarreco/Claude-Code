"""Unit tests for TTSQueue — no audio or API key required."""

from __future__ import annotations

import threading
import time

import pytest

from hey_claude_features.tts_queue import QueueStats, TTSQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _queue(speaker=None, **kw) -> TTSQueue:
    return TTSQueue(speaker=speaker or (lambda _: None), **kw)


def _running(speaker=None, **kw) -> TTSQueue:
    q = _queue(speaker=speaker, **kw)
    q.start()
    return q


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_and_stop(self):
        q = _running()
        q.stop()
        assert not q._running

    def test_double_start_idempotent(self):
        q = _running()
        q.start()   # second call should be safe
        q.stop()

    def test_stop_without_start_safe(self):
        q = _queue()
        q.stop()    # should not raise

    def test_worker_thread_is_daemon(self):
        q = _running()
        assert q._worker.daemon
        q.stop()


# ---------------------------------------------------------------------------
# Enqueue and speak
# ---------------------------------------------------------------------------


class TestEnqueueAndSpeak:
    def test_spoken_chunks_collected(self):
        spoken: list[str] = []
        q = _running(speaker=spoken.append)
        q.enqueue("hello")
        q.enqueue("world")
        q.join()
        q.stop()
        assert spoken == ["hello", "world"]

    def test_empty_string_ignored(self):
        spoken: list[str] = []
        q = _running(speaker=spoken.append)
        q.enqueue("")
        q.enqueue("   ")
        q.enqueue("hi")
        q.join()
        q.stop()
        assert spoken == ["hi"]

    def test_total_enqueued_counts_non_empty(self):
        q = _running()
        q.enqueue("a")
        q.enqueue("b")
        q.join()
        q.stop()
        assert q.stats().total_enqueued == 2

    def test_total_spoken_increments(self):
        spoken_count = [0]
        def sp(text):
            spoken_count[0] += 1
        q = _running(speaker=sp)
        q.enqueue("one")
        q.enqueue("two")
        q.join()
        q.stop()
        assert q.stats().total_spoken == 2

    def test_order_preserved(self):
        spoken: list[str] = []
        q = _running(speaker=spoken.append)
        for i in range(10):
            q.enqueue(str(i))
        q.join()
        q.stop()
        assert spoken == [str(i) for i in range(10)]


# ---------------------------------------------------------------------------
# Pause / resume
# ---------------------------------------------------------------------------


class TestPauseResume:
    def test_pause_blocks_pending(self):
        spoken: list[str] = []
        barrier = threading.Event()

        def slow_speaker(text):
            spoken.append(text)
            if text == "first":
                barrier.wait(timeout=2)

        q = _running(speaker=slow_speaker)
        q.pause()
        q.enqueue("first")
        q.enqueue("second")

        # Give worker time to pick up "first" then block
        time.sleep(0.05)
        q.resume()
        barrier.set()
        q.join()
        q.stop()
        assert "second" in spoken

    def test_is_paused_property(self):
        q = _running()
        assert not q.is_paused
        q.pause()
        assert q.is_paused
        q.resume()
        assert not q.is_paused
        q.stop()


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_drops_pending(self):
        barrier = threading.Event()
        spoken: list[str] = []

        def slow_speaker(text):
            spoken.append(text)
            barrier.wait(timeout=2)

        q = _running(speaker=slow_speaker)
        q.enqueue("first")
        # Wait until first is being spoken
        time.sleep(0.05)
        q.enqueue("second")
        q.enqueue("third")
        q.clear()
        barrier.set()
        q.join()
        q.stop()
        assert "second" not in spoken
        assert "third" not in spoken

    def test_clear_on_empty_queue_safe(self):
        q = _running()
        q.clear()  # should not raise
        q.stop()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_on_chunk_start_called(self):
        started: list[str] = []
        q = _running(on_chunk_start=started.append)
        q.enqueue("hello")
        q.join()
        q.stop()
        assert "hello" in started

    def test_on_chunk_finish_called(self):
        finished: list[str] = []
        q = _running(on_chunk_finish=finished.append)
        q.enqueue("world")
        q.join()
        q.stop()
        assert "world" in finished

    def test_callback_exception_does_not_crash_worker(self):
        def bad_cb(text):
            raise RuntimeError("boom")

        spoken: list[str] = []
        q = _running(speaker=spoken.append, on_chunk_start=bad_cb)
        q.enqueue("still works")
        q.join()
        q.stop()
        assert "still works" in spoken


# ---------------------------------------------------------------------------
# Speaker exception
# ---------------------------------------------------------------------------


class TestSpeakerException:
    def test_speaker_exception_does_not_stop_worker(self):
        calls: list[str] = []

        def flaky_speaker(text):
            if text == "bad":
                raise RuntimeError("TTS crashed")
            calls.append(text)

        q = _running(speaker=flaky_speaker)
        q.enqueue("bad")
        q.enqueue("good")
        q.join()
        q.stop()
        assert "good" in calls


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_type(self):
        q = _running()
        s = q.stats()
        assert isinstance(s, QueueStats)
        q.stop()

    def test_pending_count(self):
        barrier = threading.Event()

        def slow(text):
            barrier.wait(timeout=2)

        q = _running(speaker=slow)
        q.enqueue("a")
        q.enqueue("b")
        time.sleep(0.05)   # let worker pick up "a"
        assert q.pending <= 1   # "b" still pending
        barrier.set()
        q.join()
        q.stop()
