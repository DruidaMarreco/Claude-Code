"""Unit tests for ConfigManager — no API key or audio required."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hey_claude_features.config_manager import Config, ConfigManager, _coerce


# ---------------------------------------------------------------------------
# _coerce helper
# ---------------------------------------------------------------------------


class TestCoerce:
    def test_int(self):
        assert _coerce("42", int) == 42

    def test_float(self):
        assert _coerce("3.14", float) == pytest.approx(3.14)

    def test_bool_true_variants(self):
        for v in ("1", "true", "True", "yes", "on"):
            assert _coerce(v, bool) is True

    def test_bool_false_variants(self):
        for v in ("0", "false", "no", "off"):
            assert _coerce(v, bool) is False

    def test_list_comma_separated(self):
        assert _coerce("hey claude,ok computer", list) == ["hey claude", "ok computer"]

    def test_str_passthrough(self):
        assert _coerce("hello", str) == "hello"


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_sample_rate_default(self):
        assert Config().sample_rate == 16_000

    def test_haiku_model_default(self):
        assert "haiku" in Config().haiku_model.lower()

    def test_wake_phrases_default(self):
        assert Config().wake_phrases == ["hey claude"]

    def test_budget_unlimited_by_default(self):
        c = Config()
        assert c.hourly_usd_limit == 0.0
        assert c.daily_usd_limit == 0.0

    def test_as_dict_returns_all_keys(self):
        d = Config().as_dict()
        assert "sample_rate" in d
        assert "haiku_model" in d
        assert "wake_phrases" in d


# ---------------------------------------------------------------------------
# Load from file
# ---------------------------------------------------------------------------


class TestLoadFromFile:
    def test_loads_overrides(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"sample_rate": 22050}))
        mgr = ConfigManager(path=path)
        cfg = mgr.load()
        assert cfg.sample_rate == 22050

    def test_unknown_keys_ignored(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"nonexistent_key": "value"}))
        mgr = ConfigManager(path=path)
        cfg = mgr.load()
        assert cfg.sample_rate == 16_000  # default unchanged

    def test_missing_file_uses_defaults(self, tmp_path):
        mgr = ConfigManager(path=tmp_path / "absent.json")
        cfg = mgr.load()
        assert cfg.sample_rate == 16_000

    def test_corrupt_file_uses_defaults(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{bad json")
        mgr = ConfigManager(path=path)
        cfg = mgr.load()
        assert cfg.sample_rate == 16_000

    def test_no_path_uses_defaults(self):
        mgr = ConfigManager()
        cfg = mgr.load()
        assert cfg.sample_rate == 16_000


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    def test_int_env_var(self, monkeypatch):
        monkeypatch.setenv("HEY_CLAUDE_SAMPLE_RATE", "22050")
        mgr = ConfigManager()
        cfg = mgr.load()
        assert cfg.sample_rate == 22050

    def test_float_env_var(self, monkeypatch):
        monkeypatch.setenv("HEY_CLAUDE_VAD_ENERGY_THRESHOLD", "0.025")
        mgr = ConfigManager()
        cfg = mgr.load()
        assert cfg.vad_energy_threshold == pytest.approx(0.025)

    def test_bool_env_var(self, monkeypatch):
        monkeypatch.setenv("HEY_CLAUDE_FEEDBACK_ENABLED", "false")
        mgr = ConfigManager()
        cfg = mgr.load()
        assert cfg.feedback_enabled is False

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"sample_rate": 22050}))
        monkeypatch.setenv("HEY_CLAUDE_SAMPLE_RATE", "44100")
        mgr = ConfigManager(path=path)
        cfg = mgr.load()
        assert cfg.sample_rate == 44100

    def test_invalid_env_var_ignored(self, monkeypatch):
        monkeypatch.setenv("HEY_CLAUDE_SAMPLE_RATE", "not_a_number")
        mgr = ConfigManager()
        cfg = mgr.load()
        assert cfg.sample_rate == 16_000  # falls back to default


# ---------------------------------------------------------------------------
# set() / reset() / get()
# ---------------------------------------------------------------------------


class TestSetReset:
    def test_set_changes_value(self):
        mgr = ConfigManager()
        mgr.load()
        mgr.set("sample_rate", 22050)
        assert mgr.config.sample_rate == 22050

    def test_set_unknown_key_raises(self):
        mgr = ConfigManager()
        mgr.load()
        with pytest.raises(KeyError):
            mgr.set("nonexistent", 42)

    def test_reset_reverts_to_default(self):
        mgr = ConfigManager()
        mgr.load()
        mgr.set("sample_rate", 22050)
        mgr.reset("sample_rate")
        assert mgr.config.sample_rate == 16_000

    def test_get_returns_value(self):
        mgr = ConfigManager()
        mgr.load()
        assert mgr.get("sample_rate") == 16_000

    def test_get_unknown_returns_default(self):
        mgr = ConfigManager()
        mgr.load()
        assert mgr.get("unknown_key", "fallback") == "fallback"

    def test_reset_all_clears_overrides(self):
        mgr = ConfigManager()
        mgr.load()
        mgr.set("sample_rate", 22050)
        mgr.set("feedback_volume", 0.1)
        mgr.reset_all()
        assert mgr.config.sample_rate == 16_000
        assert mgr.config.feedback_volume == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# save() / persist round-trip
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "config.json"
        mgr = ConfigManager(path=path)
        mgr.load()
        mgr.set("sample_rate", 22050)
        mgr.save()
        assert path.exists()

    def test_saved_value_reloaded(self, tmp_path):
        path = tmp_path / "config.json"
        mgr1 = ConfigManager(path=path)
        mgr1.load()
        mgr1.set("sample_rate", 22050)
        mgr1.save()

        mgr2 = ConfigManager(path=path)
        cfg = mgr2.load()
        assert cfg.sample_rate == 22050

    def test_save_without_path_is_noop(self):
        mgr = ConfigManager()
        mgr.load()
        mgr.set("sample_rate", 22050)
        mgr.save()  # should not raise


# ---------------------------------------------------------------------------
# diff()
# ---------------------------------------------------------------------------


class TestDiff:
    def test_no_changes_empty_diff(self):
        mgr = ConfigManager()
        mgr.load()
        assert mgr.diff() == {}

    def test_changed_key_in_diff(self):
        mgr = ConfigManager()
        mgr.load()
        mgr.set("sample_rate", 22050)
        d = mgr.diff()
        assert "sample_rate" in d
        assert d["sample_rate"] == 22050

    def test_reset_removes_from_diff(self):
        mgr = ConfigManager()
        mgr.load()
        mgr.set("sample_rate", 22050)
        mgr.reset("sample_rate")
        assert "sample_rate" not in mgr.diff()
