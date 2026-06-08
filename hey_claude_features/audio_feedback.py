"""
Non-verbal audio feedback tones for Hey-Claude.

Plays short synthesised tones at key pipeline events so the user has
immediate audio confirmation without waiting for TTS:

    WAKE       — rising two-tone "ding" when the wake phrase is detected
    THINKING   — soft low pulse while Opus is generating
    DONE       — falling two-tone "dong" when the reply is ready
    ERROR      — dissonant buzz for errors / API failures
    MUTED      — single flat blip when mute is toggled

Tones are synthesised on-the-fly with NumPy (no audio files needed).
Playback uses sounddevice if available; the module degrades gracefully
to a no-op when sounddevice or numpy is missing so the pipeline never
breaks on headless servers.

Usage in app.py
---------------
    from hey_claude_features.audio_feedback import AudioFeedback, Tone

    fb = AudioFeedback(volume=0.4)

    # On wake detection:
    fb.play(Tone.WAKE)

    # While waiting for Opus:
    fb.play(Tone.THINKING)

    # When reply ready:
    fb.play(Tone.DONE)

    # On error:
    fb.play(Tone.ERROR)

    # Mute toggled:
    fb.play(Tone.MUTED)

Pure-numpy synthesis for testing
---------------------------------
Use AudioFeedback.synthesize(tone) to get a raw numpy array without
any playback — useful in tests and for pre-recording tones to files.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 44_100


# ---------------------------------------------------------------------------
# Tone enum
# ---------------------------------------------------------------------------


class Tone(str, Enum):
    WAKE = "wake"
    THINKING = "thinking"
    DONE = "done"
    ERROR = "error"
    MUTED = "muted"


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------


def _sine(freq: float, duration: float, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    """Pure sine wave, float32, range [-1, 1]."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float64)
    return np.sin(2 * math.pi * freq * t).astype(np.float32)


def _envelope(signal: np.ndarray, attack: float = 0.01, release: float = 0.05) -> np.ndarray:
    """Apply a simple linear attack/release envelope."""
    n = len(signal)
    env = np.ones(n, dtype=np.float32)
    atk = int(attack * _SAMPLE_RATE)
    rel = int(release * _SAMPLE_RATE)
    if atk > 0:
        env[:atk] = np.linspace(0, 1, atk, dtype=np.float32)
    if rel > 0 and rel <= n:
        env[-rel:] = np.linspace(1, 0, rel, dtype=np.float32)
    return signal * env


def _concat(parts: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)


def _silence(duration: float, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tone definitions
# ---------------------------------------------------------------------------

# Each tone is defined as (freq_hz, duration_s) pairs with optional gaps
_TONE_SPECS: dict[Tone, list[tuple[float, float]]] = {
    Tone.WAKE:      [(880, 0.08), (0, 0.03), (1320, 0.12)],   # rising ding
    Tone.THINKING:  [(440, 0.06), (0, 0.08), (440, 0.06)],    # soft double pulse
    Tone.DONE:      [(1320, 0.08), (0, 0.03), (880, 0.12)],   # falling dong
    Tone.ERROR:     [(220, 0.10), (0, 0.02), (196, 0.10)],    # dissonant buzz
    Tone.MUTED:     [(660, 0.07)],                             # single blip
}


def _build_tone(tone: Tone) -> np.ndarray:
    parts: list[np.ndarray] = []
    for freq, dur in _TONE_SPECS[tone]:
        if freq == 0:
            parts.append(_silence(dur))
        else:
            parts.append(_envelope(_sine(freq, dur), attack=0.01, release=0.03))
    return _concat(parts)


# ---------------------------------------------------------------------------
# AudioFeedback
# ---------------------------------------------------------------------------


class AudioFeedback:
    """
    Plays synthesised feedback tones at pipeline events.

    Parameters
    ----------
    volume:
        Output amplitude scale (0.0–1.0).
    sample_rate:
        Audio sample rate in Hz.
    enabled:
        Master on/off switch.  Set to False to silence all tones.
    """

    def __init__(
        self,
        volume: float = 0.5,
        sample_rate: int = _SAMPLE_RATE,
        enabled: bool = True,
    ) -> None:
        self._volume = max(0.0, min(1.0, volume))
        self._sample_rate = sample_rate
        self._enabled = enabled
        # Pre-build all tones once so play() has no synthesis latency
        self._cache: dict[Tone, np.ndarray] = {t: _build_tone(t) for t in Tone}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = max(0.0, min(1.0, value))

    def play(self, tone: Tone) -> bool:
        """
        Play *tone* asynchronously.

        Returns True if playback was initiated, False if disabled or
        sounddevice unavailable.
        """
        if not self._enabled:
            return False
        try:
            import sounddevice as sd  # type: ignore[import]
        except ImportError:
            logger.debug("sounddevice not available; audio feedback skipped")
            return False

        signal = self._cache[tone] * self._volume
        try:
            sd.play(signal, samplerate=self._sample_rate)
            return True
        except Exception:
            logger.exception("Audio feedback playback failed for tone %s", tone)
            return False

    def play_blocking(self, tone: Tone) -> bool:
        """Play *tone* and block until it finishes. Useful for tests/CLI."""
        if not self._enabled:
            return False
        try:
            import sounddevice as sd  # type: ignore[import]
        except ImportError:
            logger.debug("sounddevice not available; audio feedback skipped")
            return False

        signal = self._cache[tone] * self._volume
        try:
            sd.play(signal, samplerate=self._sample_rate)
            sd.wait()
            return True
        except Exception:
            logger.exception("Audio feedback blocking playback failed for tone %s", tone)
            return False

    def synthesize(self, tone: Tone) -> np.ndarray:
        """
        Return the pre-built numpy array for *tone*, scaled by volume.

        This is the pure-computation path used in tests — no sounddevice needed.
        """
        return self._cache[tone] * self._volume

    def duration(self, tone: Tone) -> float:
        """Return the duration of *tone* in seconds."""
        return len(self._cache[tone]) / self._sample_rate
