"""Unit tests for AudioFeedback — no sounddevice or microphone required."""

from __future__ import annotations

import numpy as np
import pytest

from hey_claude_features.audio_feedback import (
    AudioFeedback,
    Tone,
    _build_tone,
    _envelope,
    _silence,
    _sine,
)

_SR = 44_100


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------


class TestSine:
    def test_length(self):
        s = _sine(440, 0.1)
        assert len(s) == pytest.approx(_SR * 0.1, abs=1)

    def test_dtype(self):
        assert _sine(440, 0.1).dtype == np.float32

    def test_amplitude_bounded(self):
        s = _sine(440, 0.5)
        assert float(np.max(np.abs(s))) <= 1.0 + 1e-6

    def test_zero_frequency_is_silence(self):
        s = _sine(0, 0.1)
        assert float(np.max(np.abs(s))) < 1e-6


class TestSilence:
    def test_all_zeros(self):
        s = _silence(0.05)
        assert np.all(s == 0)

    def test_correct_length(self):
        s = _silence(0.1)
        assert len(s) == pytest.approx(_SR * 0.1, abs=1)


class TestEnvelope:
    def test_starts_near_zero(self):
        sig = np.ones(int(_SR * 0.1), dtype=np.float32)
        env = _envelope(sig, attack=0.01)
        assert env[0] < 0.1

    def test_ends_near_zero(self):
        sig = np.ones(int(_SR * 0.1), dtype=np.float32)
        env = _envelope(sig, release=0.05)
        assert env[-1] < 0.1

    def test_peak_unchanged(self):
        sig = np.ones(int(_SR * 0.1), dtype=np.float32)
        env = _envelope(sig, attack=0.01, release=0.01)
        mid = len(env) // 2
        assert abs(env[mid] - 1.0) < 0.05


# ---------------------------------------------------------------------------
# _build_tone
# ---------------------------------------------------------------------------


class TestBuildTone:
    @pytest.mark.parametrize("tone", list(Tone))
    def test_returns_float32(self, tone):
        arr = _build_tone(tone)
        assert arr.dtype == np.float32

    @pytest.mark.parametrize("tone", list(Tone))
    def test_non_empty(self, tone):
        arr = _build_tone(tone)
        assert len(arr) > 0

    def test_wake_shorter_than_half_second(self):
        arr = _build_tone(Tone.WAKE)
        assert len(arr) / _SR < 0.5

    def test_done_has_sound(self):
        arr = _build_tone(Tone.DONE)
        assert float(np.max(np.abs(arr))) > 0.01


# ---------------------------------------------------------------------------
# AudioFeedback construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_volume(self):
        fb = AudioFeedback()
        assert fb.volume == pytest.approx(0.5)

    def test_volume_clamped_high(self):
        fb = AudioFeedback(volume=5.0)
        assert fb.volume == pytest.approx(1.0)

    def test_volume_clamped_low(self):
        fb = AudioFeedback(volume=-1.0)
        assert fb.volume == pytest.approx(0.0)

    def test_enabled_default(self):
        assert AudioFeedback().enabled

    def test_disabled(self):
        fb = AudioFeedback(enabled=False)
        assert not fb.enabled

    def test_all_tones_pre_cached(self):
        fb = AudioFeedback()
        for tone in Tone:
            assert tone in fb._cache


# ---------------------------------------------------------------------------
# synthesize() — pure numpy, no sounddevice
# ---------------------------------------------------------------------------


class TestSynthesize:
    @pytest.mark.parametrize("tone", list(Tone))
    def test_returns_array(self, tone):
        fb = AudioFeedback()
        arr = fb.synthesize(tone)
        assert isinstance(arr, np.ndarray)

    def test_volume_scaling(self):
        fb_half = AudioFeedback(volume=0.5)
        fb_full = AudioFeedback(volume=1.0)
        half = fb_half.synthesize(Tone.WAKE)
        full = fb_full.synthesize(Tone.WAKE)
        ratio = float(np.max(np.abs(full))) / float(np.max(np.abs(half)))
        assert ratio == pytest.approx(2.0, rel=0.1)

    def test_volume_zero_silence(self):
        fb = AudioFeedback(volume=0.0)
        arr = fb.synthesize(Tone.WAKE)
        assert float(np.max(np.abs(arr))) == pytest.approx(0.0)

    def test_all_tones_have_sound_at_full_volume(self):
        fb = AudioFeedback(volume=1.0)
        for tone in Tone:
            arr = fb.synthesize(tone)
            assert float(np.max(np.abs(arr))) > 0.01, f"{tone} was silent"


# ---------------------------------------------------------------------------
# duration()
# ---------------------------------------------------------------------------


class TestDuration:
    @pytest.mark.parametrize("tone", list(Tone))
    def test_duration_positive(self, tone):
        fb = AudioFeedback()
        assert fb.duration(tone) > 0

    def test_wake_duration_reasonable(self):
        fb = AudioFeedback()
        # WAKE = 80ms + 30ms gap + 120ms = 230ms
        assert 0.2 < fb.duration(Tone.WAKE) < 0.4


# ---------------------------------------------------------------------------
# play() — disabled / no sounddevice
# ---------------------------------------------------------------------------


class TestPlay:
    def test_disabled_returns_false(self):
        fb = AudioFeedback(enabled=False)
        assert fb.play(Tone.WAKE) is False

    def test_enabled_setter(self):
        fb = AudioFeedback()
        fb.enabled = False
        assert not fb.enabled
        fb.enabled = True
        assert fb.enabled

    def test_volume_setter(self):
        fb = AudioFeedback()
        fb.volume = 0.3
        assert fb.volume == pytest.approx(0.3)

    def test_play_without_sounddevice_returns_false(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        fb = AudioFeedback()
        assert fb.play(Tone.WAKE) is False

    def test_play_blocking_disabled_returns_false(self):
        fb = AudioFeedback(enabled=False)
        assert fb.play_blocking(Tone.DONE) is False
