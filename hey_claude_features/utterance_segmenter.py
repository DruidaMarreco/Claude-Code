"""
Energy-based utterance segmenter for Hey-Claude.

Splits a continuous stream of audio frames into discrete utterances by
detecting when speech starts (energy rises above threshold) and ends
(energy stays below threshold for a configurable silence window).

This is the core recording loop — it decides *when* to send audio to Whisper.

State machine
-------------
    IDLE ──(energy > threshold)──► SPEAKING
    SPEAKING ──(silence < timeout)──► SPEAKING
    SPEAKING ──(silence >= timeout)──► IDLE  (emit utterance)

Usage with a live microphone
-----------------------------
    from hey_claude_features.utterance_segmenter import UtteranceSegmenter

    seg = UtteranceSegmenter(sample_rate=16000, energy_threshold=0.015)

    for frame in mic_stream:            # numpy float32 frames
        utterance = seg.push(frame)
        if utterance is not None:
            transcript = whisper.transcribe(utterance)

Usage with pre-recorded audio (testing)
----------------------------------------
    frames = [audio[i:i+512] for i in range(0, len(audio), 512)]
    utterances = seg.process_frames(frames)

Configuration tips
------------------
- Lower energy_threshold → more sensitive (picks up quiet speech)
- Higher silence_timeout → waits longer before cutting the utterance
- Use AmbientCalibrator to set energy_threshold automatically
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rms(frame: np.ndarray) -> float:
    if len(frame) == 0:
        return 0.0
    return float(math.sqrt(np.mean(frame.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class SegmenterState(str, Enum):
    IDLE = "idle"
    SPEAKING = "speaking"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Utterance:
    """A complete spoken utterance."""
    audio: np.ndarray           # concatenated frames, float32
    sample_rate: int
    frame_count: int
    peak_rms: float
    mean_rms: float

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate


@dataclass
class SegmenterStats:
    utterances_emitted: int = 0
    frames_processed: int = 0
    frames_speaking: int = 0


# ---------------------------------------------------------------------------
# UtteranceSegmenter
# ---------------------------------------------------------------------------


class UtteranceSegmenter:
    """
    Detects speech start/end in a stream of audio frames.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz.
    energy_threshold:
        RMS energy level that triggers speech-start detection.
    silence_timeout:
        Seconds of continuous silence before the utterance is emitted.
    frame_size:
        Samples per frame.  Should match the frame size of the audio source.
    pre_roll_frames:
        Number of idle frames to prepend to each utterance (captures the
        very beginning of speech that fell just below the threshold).
    min_utterance_frames:
        Minimum frames required for an utterance to be emitted (filters
        very short noise bursts).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        energy_threshold: float = 0.015,
        silence_timeout: float = 0.8,
        frame_size: int = 512,
        pre_roll_frames: int = 3,
        min_utterance_frames: int = 8,
    ) -> None:
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.frame_size = frame_size
        self.pre_roll_frames = pre_roll_frames
        self.min_utterance_frames = min_utterance_frames

        self._silence_frames = max(1, int(silence_timeout * sample_rate / frame_size))
        self._state = SegmenterState.IDLE
        self._buffer: list[np.ndarray] = []       # current utterance frames
        self._pre_roll: list[np.ndarray] = []     # recent idle frames
        self._silent_streak = 0
        self._stats = SegmenterStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> SegmenterState:
        return self._state

    @property
    def is_speaking(self) -> bool:
        return self._state == SegmenterState.SPEAKING

    def push(self, frame: np.ndarray) -> Utterance | None:
        """
        Feed one audio frame.

        Returns an ``Utterance`` when speech ends, otherwise None.
        """
        self._stats.frames_processed += 1
        rms = _rms(frame)

        if self._state == SegmenterState.IDLE:
            self._pre_roll.append(frame)
            if len(self._pre_roll) > self.pre_roll_frames:
                self._pre_roll.pop(0)

            if rms >= self.energy_threshold:
                self._state = SegmenterState.SPEAKING
                self._buffer = list(self._pre_roll)
                self._silent_streak = 0
            return None

        # SPEAKING
        self._stats.frames_speaking += 1
        self._buffer.append(frame)

        if rms < self.energy_threshold:
            self._silent_streak += 1
        else:
            self._silent_streak = 0

        if self._silent_streak >= self._silence_frames:
            return self._emit()

        return None

    def flush(self) -> Utterance | None:
        """
        Force-emit any buffered audio (call at end of recording session).
        Returns None if there is nothing worth emitting.
        """
        if self._state == SegmenterState.SPEAKING and self._buffer:
            return self._emit()
        return None

    def reset(self) -> None:
        """Reset to IDLE state, discarding any buffered audio."""
        self._state = SegmenterState.IDLE
        self._buffer = []
        self._pre_roll = []
        self._silent_streak = 0

    def process_frames(self, frames: Sequence[np.ndarray]) -> list[Utterance]:
        """
        Convenience method: feed all *frames* and return all emitted utterances.
        """
        utterances: list[Utterance] = []
        for f in frames:
            u = self.push(f)
            if u is not None:
                utterances.append(u)
        u = self.flush()
        if u is not None:
            utterances.append(u)
        return utterances

    @property
    def stats(self) -> SegmenterStats:
        return self._stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self) -> Utterance | None:
        frames = self._buffer
        self._state = SegmenterState.IDLE
        self._buffer = []
        self._pre_roll = []
        self._silent_streak = 0

        if len(frames) < self.min_utterance_frames:
            return None  # too short — noise burst

        audio = np.concatenate(frames).astype(np.float32)
        rms_values = [_rms(f) for f in frames]
        utterance = Utterance(
            audio=audio,
            sample_rate=self.sample_rate,
            frame_count=len(frames),
            peak_rms=float(max(rms_values)),
            mean_rms=float(sum(rms_values) / len(rms_values)),
        )
        self._stats.utterances_emitted += 1
        return utterance
