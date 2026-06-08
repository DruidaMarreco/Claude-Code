"""
TTS playback queue for Hey-Claude.

When using StreamingResponder, text chunks arrive faster than TTS can
speak them.  This module sequences them: each chunk is enqueued and a
background worker thread speaks them one at a time in arrival order.

Features
--------
- Thread-safe FIFO queue fed by the streaming pipeline
- pause() / resume() / skip() / clear() controls
- on_start / on_finish callbacks per chunk (for UI feedback)
- Pluggable speaker: any callable(text) works — built-in no-op for tests

Usage in app.py
---------------
    from hey_claude_features.tts_queue import TTSQueue

    def speak(text: str) -> None:
        tts_engine.speak_blocking(text)

    queue = TTSQueue(speaker=speak)
    queue.start()

    # Feed chunks from StreamingResponder:
    for chunk in responder.stream(messages):
        if chunk.text:
            queue.enqueue(chunk.text)

    # Wait for all speech to finish:
    queue.join()

    # Or interrupt immediately:
    queue.skip()     # skip the currently-speaking chunk
    queue.clear()    # drop all pending chunks
    queue.pause()    # pause after current chunk
    queue.resume()   # resume

    # Shutdown at app exit:
    queue.stop()
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

_SENTINEL = object()   # signals the worker to stop


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class QueueStats:
    """Snapshot of queue state."""
    pending: int
    total_enqueued: int
    total_spoken: int
    is_paused: bool
    is_speaking: bool


# ---------------------------------------------------------------------------
# TTSQueue
# ---------------------------------------------------------------------------


class TTSQueue:
    """
    Thread-safe FIFO queue that speaks text chunks in order.

    Parameters
    ----------
    speaker:
        Callable ``(text: str) -> None`` that blocks until the text has
        been spoken.  Pass ``None`` or a no-op for testing.
    maxsize:
        Maximum number of queued chunks (0 = unlimited).
    on_chunk_start:
        Called just before each chunk is spoken.
    on_chunk_finish:
        Called just after each chunk finishes speaking.
    """

    def __init__(
        self,
        speaker: Callable[[str], None] | None = None,
        maxsize: int = 0,
        on_chunk_start: Callable[[str], None] | None = None,
        on_chunk_finish: Callable[[str], None] | None = None,
    ) -> None:
        self._speaker = speaker or (lambda _: None)
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._on_start = on_chunk_start
        self._on_finish = on_chunk_finish

        self._paused = threading.Event()
        self._paused.set()          # not paused initially (set = "go")
        self._speaking = threading.Event()  # set while a chunk is being spoken

        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._running = False

        self._total_enqueued = 0
        self._total_spoken = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._run, daemon=True, name="tts-queue")
        self._worker.start()

    def stop(self) -> None:
        """Gracefully stop the worker (drains remaining queue first)."""
        if not self._running:
            return
        self._running = False
        self._paused.set()          # unblock if paused
        self._q.put(_SENTINEL)      # wake the worker
        if self._worker:
            self._worker.join(timeout=5)

    def join(self) -> None:
        """Block until all enqueued chunks have been spoken."""
        self._q.join()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, text: str) -> None:
        """Add *text* to the speak queue."""
        if not text.strip():
            return
        with self._lock:
            self._total_enqueued += 1
        self._q.put(text)

    def pause(self) -> None:
        """Pause speaking after the current chunk finishes."""
        self._paused.clear()

    def resume(self) -> None:
        """Resume speaking."""
        self._paused.set()

    def skip(self) -> None:
        """
        Interrupt the currently-speaking chunk.

        Because the speaker callable blocks, a true interrupt requires the
        speaker itself to support interruption.  Here we signal the intent;
        the worker will move to the next chunk as soon as the current one
        returns.  For immediate interruption, the caller should use a
        speaker that supports cancellation.
        """
        # Drain any pending items immediately so the next chunk is the
        # item currently being spoken (which will finish naturally).
        self.clear()

    def clear(self) -> None:
        """Drop all pending (not-yet-spoken) chunks."""
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def pending(self) -> int:
        return self._q.qsize()

    def stats(self) -> QueueStats:
        with self._lock:
            return QueueStats(
                pending=self._q.qsize(),
                total_enqueued=self._total_enqueued,
                total_spoken=self._total_spoken,
                is_paused=self.is_paused,
                is_speaking=self.is_speaking,
            )

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                self._q.task_done()
                break

            # Wait if paused
            self._paused.wait()

            text: str = item
            self._speaking.set()
            try:
                if self._on_start:
                    try:
                        self._on_start(text)
                    except Exception:
                        logger.exception("on_chunk_start callback raised")

                self._speaker(text)

                if self._on_finish:
                    try:
                        self._on_finish(text)
                    except Exception:
                        logger.exception("on_chunk_finish callback raised")

                with self._lock:
                    self._total_spoken += 1
            except Exception:
                logger.exception("TTS speaker raised for chunk %r", text)
            finally:
                self._speaking.clear()
                self._q.task_done()
