"""Unit tests for AmbientCalibrator — no microphone required."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hey_claude_features.ambient_calibrator import (
    AmbientCalibrator,
    CalibrationResult,
    _rms,
    _MIN_THRESHOLD,
    _MAX_THRESHOLD,
)


# ---------------------------------------------------------------------------
# _rms helper
# ---------------------------------------------------------------------------


class TestRms:
    def test_silence(self):
        frame = np.zeros(512, dtype=np.float32)
        assert _rms(frame) == 0.0

    def test_constant_signal(self):
        frame = np.full(512, 0.5, dtype=np.float32)
        assert abs(_rms(frame) - 0.5) < 1e-5

    def test_empty_frame(self):
        assert _rms(np.array([], dtype=np.float32)) == 0.0

    def test_sine_wave(self):
        t = np.linspace(0, 2 * math.pi, 512, dtype=np.float32)
        frame = np.sin(t)
        # RMS of sin is 1/sqrt(2) ≈ 0.707
        assert abs(_rms(frame) - 1 / math.sqrt(2)) < 0.01


# ---------------------------------------------------------------------------
# from_frames() — no I/O needed
# ---------------------------------------------------------------------------


def _make_frames(rms_level: float, n: int = 20, frame_size: int = 512) -> list[np.ndarray]:
    """Generate frames with approximately the given RMS level."""
    frames = []
    for _ in range(n):
        # White noise scaled to target RMS
        raw = np.random.randn(frame_size).astype(np.float32)
        current_rms = _rms(raw)
        if current_rms > 0:
            raw = raw * (rms_level / current_rms)
        frames.append(raw)
    return frames


class TestFromFrames:
    def test_successful_calibration(self):
        frames = _make_frames(rms_level=0.01)
        result = AmbientCalibrator.from_frames(frames)
        assert result.success

    def test_noise_floor_close_to_target(self):
        np.random.seed(42)
        frames = _make_frames(rms_level=0.01, n=50)
        result = AmbientCalibrator.from_frames(frames)
        assert abs(result.noise_floor - 0.01) < 0.005

    def test_threshold_is_margin_times_floor(self):
        np.random.seed(42)
        frames = _make_frames(rms_level=0.01, n=50)
        result = AmbientCalibrator.from_frames(frames, margin_factor=3.0)
        assert abs(result.threshold - result.noise_floor * 3.0) < 0.005

    def test_threshold_clamped_to_min(self):
        # Very quiet room — threshold would be below min
        frames = _make_frames(rms_level=0.0005, n=20)
        result = AmbientCalibrator.from_frames(frames, margin_factor=2.0)
        assert result.success
        assert result.threshold >= _MIN_THRESHOLD

    def test_threshold_clamped_to_max(self):
        # Very loud environment
        frames = _make_frames(rms_level=0.20, n=20)
        result = AmbientCalibrator.from_frames(frames, margin_factor=5.0)
        assert result.success
        assert result.threshold <= _MAX_THRESHOLD

    def test_empty_frames_fails(self):
        result = AmbientCalibrator.from_frames([])
        assert not result.success
        assert "No audio frames" in result.error

    def test_silent_frames_fails(self):
        frames = [np.zeros(512, dtype=np.float32)] * 10
        result = AmbientCalibrator.from_frames(frames)
        assert not result.success
        assert "silence" in result.error.lower()

    def test_frame_count_in_result(self):
        frames = _make_frames(rms_level=0.01, n=15)
        result = AmbientCalibrator.from_frames(frames)
        assert result.frame_count == 15

    def test_peak_rms_gte_mean(self):
        frames = _make_frames(rms_level=0.01)
        result = AmbientCalibrator.from_frames(frames)
        assert result.peak_rms >= result.noise_floor


# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------


class TestCalibrationResult:
    def test_voice_summary_success(self):
        r = CalibrationResult(success=True, noise_floor=0.008, threshold=0.024)
        s = r.voice_summary()
        assert "0.008" in s or "0.00" in s
        assert "threshold" in s.lower()

    def test_voice_summary_failure(self):
        r = CalibrationResult(success=False, error="no mic")
        s = r.voice_summary()
        assert "failed" in s.lower()
        assert "no mic" in s

    def test_default_threshold_is_reasonable(self):
        r = CalibrationResult(success=False)
        assert 0.005 <= r.threshold <= 0.10


# ---------------------------------------------------------------------------
# AmbientCalibrator constructor defaults
# ---------------------------------------------------------------------------


class TestCalibratorDefaults:
    def test_split_frames_even_division(self):
        cal = AmbientCalibrator(frame_size=512)
        audio = np.zeros(1024, dtype=np.float32)
        frames = cal._split_frames(audio)
        assert len(frames) == 2

    def test_split_frames_ignores_remainder(self):
        cal = AmbientCalibrator(frame_size=512)
        audio = np.zeros(1023, dtype=np.float32)
        frames = cal._split_frames(audio)
        assert len(frames) == 1

    def test_calibrate_fails_gracefully_without_sounddevice(self, monkeypatch):
        # Simulate sounddevice not installed
        import sys
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        cal = AmbientCalibrator()
        result = cal.calibrate()
        assert not result.success
        assert "sounddevice" in result.error.lower()
