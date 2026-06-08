"""
Ambient noise calibrator for Hey-Claude's VAD energy threshold.

Hey-Claude's EnergyVAD uses a fixed RMS threshold to decide whether audio
contains speech.  In quiet rooms this is fine, but in noisy environments
(open offices, kitchens, outdoors) it causes constant false triggers.

This module measures the ambient background noise at startup and computes
a threshold that is higher than the noise floor by a configurable margin.
The calibrated value can then be passed to EnergyVAD's threshold parameter.

How it works
------------
1. Record N seconds of silence (no one should be talking)
2. Compute the RMS energy of each audio frame
3. Set threshold = mean_rms * margin_factor  (default 3.0×)
4. Clamp to [min_threshold, max_threshold] safety bounds

Usage in app.py
---------------
    from hey_claude_features.ambient_calibrator import AmbientCalibrator

    # At startup, before starting the mic:
    calibrator = AmbientCalibrator(sample_rate=16000, duration=2.0)
    result = calibrator.calibrate()

    if result.success:
        print(f"Noise floor: {result.noise_floor:.4f}  Threshold: {result.threshold:.4f}")
        config.vad_energy_threshold = result.threshold   # override config
    else:
        print(f"Calibration failed: {result.error}; using default threshold")

No-audio fallback
-----------------
If sounddevice is not installed or no microphone is available, calibrate()
returns a CalibrationResult with success=False and a meaningful error string.
The caller should fall back to the default threshold gracefully.

Pure-numpy simulation for testing
----------------------------------
AmbientCalibrator.from_frames(frames, ...) accepts pre-recorded audio frames
so tests can run without a real microphone.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Safety bounds for the computed threshold
_MIN_THRESHOLD = 0.005   # very quiet room
_MAX_THRESHOLD = 0.30    # very loud environment
_DEFAULT_MARGIN = 3.0    # threshold = noise_floor * margin
_DEFAULT_DURATION = 2.0  # seconds of silence to record
_DEFAULT_FRAME_SIZE = 512


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    success: bool
    noise_floor: float = 0.0   # mean RMS across recorded frames
    threshold: float = 0.015   # recommended VAD threshold
    peak_rms: float = 0.0      # max RMS frame (useful for debugging)
    frame_count: int = 0
    error: str = ""

    def voice_summary(self) -> str:
        if not self.success:
            return f"Calibration failed: {self.error}"
        return (
            f"Noise floor {self.noise_floor:.4f}, "
            f"VAD threshold set to {self.threshold:.4f}."
        )


# ---------------------------------------------------------------------------
# AmbientCalibrator
# ---------------------------------------------------------------------------


class AmbientCalibrator:
    """
    Records ambient noise and computes a recommended VAD energy threshold.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz.  Should match Hey-Claude's config (16 000).
    duration:
        How many seconds to record for calibration.
    frame_size:
        Audio frames per chunk (should match UtteranceSegmenter's frame_size).
    margin_factor:
        threshold = noise_floor_rms * margin_factor.  Higher = less sensitive.
    min_threshold / max_threshold:
        Safety clamps applied after margin scaling.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        duration: float = _DEFAULT_DURATION,
        frame_size: int = _DEFAULT_FRAME_SIZE,
        margin_factor: float = _DEFAULT_MARGIN,
        min_threshold: float = _MIN_THRESHOLD,
        max_threshold: float = _MAX_THRESHOLD,
    ) -> None:
        self.sample_rate = sample_rate
        self.duration = duration
        self.frame_size = frame_size
        self.margin_factor = margin_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self) -> CalibrationResult:
        """
        Record ambient audio from the default microphone and compute threshold.

        Requires `sounddevice` to be installed.  Returns a failed result if
        the microphone is unavailable rather than raising.
        """
        try:
            import sounddevice as sd  # type: ignore[import]
        except ImportError:
            return CalibrationResult(
                success=False,
                error="sounddevice not installed; install with: pip install sounddevice",
            )

        total_samples = int(self.sample_rate * self.duration)
        try:
            logger.info("Calibrating ambient noise for %.1fs...", self.duration)
            audio = sd.rec(
                total_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocking=True,
            )
            frames = self._split_frames(audio.flatten())
            return self.from_frames(frames)
        except Exception as exc:
            logger.exception("Ambient calibration recording failed")
            return CalibrationResult(success=False, error=str(exc))

    @classmethod
    def from_frames(
        cls,
        frames: Sequence[np.ndarray],
        margin_factor: float = _DEFAULT_MARGIN,
        min_threshold: float = _MIN_THRESHOLD,
        max_threshold: float = _MAX_THRESHOLD,
    ) -> CalibrationResult:
        """
        Compute calibration from pre-recorded audio frames.

        This is the pure-computation path used in tests and when audio
        is captured externally.
        """
        if not frames:
            return CalibrationResult(success=False, error="No audio frames provided")

        rms_values = np.array([_rms(f) for f in frames])
        noise_floor = float(np.mean(rms_values))
        peak_rms = float(np.max(rms_values))

        if noise_floor == 0.0:
            return CalibrationResult(
                success=False,
                error="Recorded silence (all zeros); check microphone connection",
            )

        raw_threshold = noise_floor * margin_factor
        threshold = float(np.clip(raw_threshold, min_threshold, max_threshold))

        logger.info(
            "Calibration: noise_floor=%.4f  peak=%.4f  threshold=%.4f",
            noise_floor, peak_rms, threshold,
        )
        return CalibrationResult(
            success=True,
            noise_floor=noise_floor,
            threshold=threshold,
            peak_rms=peak_rms,
            frame_count=len(frames),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_frames(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split a flat audio array into fixed-size frames."""
        n = len(audio)
        return [
            audio[i : i + self.frame_size]
            for i in range(0, n - self.frame_size + 1, self.frame_size)
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rms(frame: np.ndarray) -> float:
    """Root mean square energy of a float32 audio frame."""
    if len(frame) == 0:
        return 0.0
    return float(math.sqrt(np.mean(frame.astype(np.float64) ** 2)))
