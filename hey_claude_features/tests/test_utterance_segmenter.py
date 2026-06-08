"""Unit tests for UtteranceSegmenter — no microphone required."""

from __future__ import annotations

import numpy as np
import pytest

from hey_claude_features.utterance_segmenter import (
    SegmenterState,
    Utterance,
    UtteranceSegmenter,
    _rms,
)

_SR = 16_000
_FRAME = 512
_THRESH = 0.05


# ---------------------------------------------------------------------------
# _rms helper
# ---------------------------------------------------------------------------


class TestRms:
    def test_sine_rms(self):
        t = np.linspace(0, 1, _SR, dtype=np.float32)
        sig = np.sin(2 * np.pi * 440 * t)
        r = _rms(sig)
        assert 0.6 < r < 0.8   # sine RMS ≈ 1/√2

    def test_silence_rms_zero(self):
        assert _rms(np.zeros(512, dtype=np.float32)) == pytest.approx(0.0)

    def test_empty_frame_zero(self):
        assert _rms(np.array([], dtype=np.float32)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _loud(n_frames: int = 10) -> list[np.ndarray]:
    """Loud frames whose RMS exceeds _THRESH."""
    return [np.full(_FRAME, 0.5, dtype=np.float32) for _ in range(n_frames)]


def _quiet(n_frames: int = 10) -> list[np.ndarray]:
    """Silent frames."""
    return [np.zeros(_FRAME, dtype=np.float32) for _ in range(n_frames)]


def _seg(**kw) -> UtteranceSegmenter:
    defaults = dict(
        sample_rate=_SR,
        energy_threshold=_THRESH,
        silence_timeout=0.1,   # short for tests
        frame_size=_FRAME,
        pre_roll_frames=2,
        min_utterance_frames=3,
    )
    defaults.update(kw)
    return UtteranceSegmenter(**defaults)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_starts_idle(self):
        assert _seg().state == SegmenterState.IDLE

    def test_loud_frame_triggers_speaking(self):
        seg = _seg()
        seg.push(_loud(1)[0])
        assert seg.state == SegmenterState.SPEAKING

    def test_quiet_frames_stay_idle(self):
        seg = _seg()
        for f in _quiet(5):
            seg.push(f)
        assert seg.state == SegmenterState.IDLE

    def test_returns_to_idle_after_silence(self):
        seg = _seg()
        for f in _loud(5):
            seg.push(f)
        assert seg.state == SegmenterState.SPEAKING
        silence_frames = int(0.1 * _SR / _FRAME) + 2
        for f in _quiet(silence_frames):
            seg.push(f)
        assert seg.state == SegmenterState.IDLE

    def test_is_speaking_property(self):
        seg = _seg()
        assert not seg.is_speaking
        seg.push(_loud(1)[0])
        assert seg.is_speaking


# ---------------------------------------------------------------------------
# Utterance emission
# ---------------------------------------------------------------------------


class TestUtteranceEmission:
    def test_emits_after_silence(self):
        seg = _seg()
        utterance = None
        for f in _loud(5):
            seg.push(f)
        silence_count = int(0.1 * _SR / _FRAME) + 2
        for f in _quiet(silence_count):
            result = seg.push(f)
            if result is not None:
                utterance = result
        assert utterance is not None
        assert isinstance(utterance, Utterance)

    def test_utterance_audio_nonempty(self):
        seg = _seg()
        for f in _loud(5):
            seg.push(f)
        silence_count = int(0.1 * _SR / _FRAME) + 2
        u = None
        for f in _quiet(silence_count):
            r = seg.push(f)
            if r:
                u = r
        assert u is not None
        assert len(u.audio) > 0

    def test_utterance_duration_positive(self):
        seg = _seg()
        for f in _loud(10):
            seg.push(f)
        silence_count = int(0.1 * _SR / _FRAME) + 2
        u = None
        for f in _quiet(silence_count):
            r = seg.push(f)
            if r:
                u = r
        assert u is not None
        assert u.duration > 0

    def test_short_burst_filtered(self):
        seg = _seg(min_utterance_frames=10)
        for f in _loud(2):   # fewer than min_utterance_frames
            seg.push(f)
        silence_count = int(0.1 * _SR / _FRAME) + 2
        results = [seg.push(f) for f in _quiet(silence_count)]
        assert all(r is None for r in results)

    def test_peak_rms_set(self):
        seg = _seg()
        for f in _loud(5):
            seg.push(f)
        silence_count = int(0.1 * _SR / _FRAME) + 2
        u = None
        for f in _quiet(silence_count):
            r = seg.push(f)
            if r:
                u = r
        assert u is not None
        assert u.peak_rms > 0

    def test_mean_rms_set(self):
        seg = _seg()
        for f in _loud(5):
            seg.push(f)
        silence_count = int(0.1 * _SR / _FRAME) + 2
        u = None
        for f in _quiet(silence_count):
            r = seg.push(f)
            if r:
                u = r
        assert u is not None
        assert u.mean_rms > 0


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_emits_buffered(self):
        seg = _seg()
        for f in _loud(5):
            seg.push(f)
        u = seg.flush()
        assert u is not None

    def test_flush_idle_returns_none(self):
        seg = _seg()
        assert seg.flush() is None

    def test_flush_resets_to_idle(self):
        seg = _seg()
        for f in _loud(5):
            seg.push(f)
        seg.flush()
        assert seg.state == SegmenterState.IDLE


# ---------------------------------------------------------------------------
# process_frames()
# ---------------------------------------------------------------------------


class TestProcessFrames:
    def _speech_silence(self, n_speech=15, n_silence=10) -> list[np.ndarray]:
        return _loud(n_speech) + _quiet(n_silence)

    def test_single_utterance(self):
        seg = _seg()
        silence_count = int(0.1 * _SR / _FRAME) + 2
        frames = _loud(10) + _quiet(silence_count)
        utterances = seg.process_frames(frames)
        assert len(utterances) == 1

    def test_two_utterances(self):
        seg = _seg()
        silence_count = int(0.1 * _SR / _FRAME) + 2
        frames = (
            _loud(10) + _quiet(silence_count) +
            _loud(10) + _quiet(silence_count)
        )
        utterances = seg.process_frames(frames)
        assert len(utterances) == 2

    def test_no_speech_no_utterances(self):
        seg = _seg()
        utterances = seg.process_frames(_quiet(50))
        assert utterances == []

    def test_speech_without_trailing_silence_flushed(self):
        seg = _seg()
        utterances = seg.process_frames(_loud(10))
        assert len(utterances) == 1  # flushed at end


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_buffer(self):
        seg = _seg()
        for f in _loud(5):
            seg.push(f)
        seg.reset()
        assert seg.state == SegmenterState.IDLE
        assert seg.flush() is None

    def test_reset_then_normal_operation(self):
        seg = _seg()
        for f in _loud(5):
            seg.push(f)
        seg.reset()
        silence_count = int(0.1 * _SR / _FRAME) + 2
        frames = _loud(10) + _quiet(silence_count)
        utterances = seg.process_frames(frames)
        assert len(utterances) == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_frames_processed_counted(self):
        seg = _seg()
        seg.process_frames(_quiet(10))
        assert seg.stats.frames_processed == 10

    def test_utterances_emitted_counted(self):
        seg = _seg()
        silence_count = int(0.1 * _SR / _FRAME) + 2
        seg.process_frames(_loud(10) + _quiet(silence_count))
        assert seg.stats.utterances_emitted == 1
