"""
Centralised configuration manager for Hey-Claude.

Hey-Claude has configuration values scattered across individual modules
(sample_rate, VAD threshold, model names, budget limits, etc.).  This module
unifies them into a single typed config object that can be:

  - Loaded from a JSON file on disk
  - Overridden by environment variables (``HEY_CLAUDE_<UPPER_KEY>``)
  - Accessed with dot notation and type safety
  - Saved back to disk with a single call

Layer priority (highest wins):
    1. Environment variables
    2. File-based config
    3. Built-in defaults

Usage in app.py
---------------
    from hey_claude_features.config_manager import Config, ConfigManager

    mgr = ConfigManager(path=Path("~/.hey-claude/config.json").expanduser())
    cfg = mgr.load()

    print(cfg.sample_rate)          # 16000
    print(cfg.haiku_model)          # "claude-haiku-4-5-20251001"
    print(cfg.vad_energy_threshold) # 0.015

    # Override one value and persist:
    mgr.set("vad_energy_threshold", 0.025)
    mgr.save()

    # Reset a key to its default:
    mgr.reset("vad_energy_threshold")

Environment variable override example:
    HEY_CLAUDE_SAMPLE_RATE=22050 python app.py
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_PREFIX = "HEY_CLAUDE_"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """All Hey-Claude configuration values with their defaults."""

    # Audio
    sample_rate: int = 16_000
    frame_size: int = 512
    vad_energy_threshold: float = 0.015
    calibration_duration: float = 2.0
    calibration_margin: float = 3.0

    # Models
    haiku_model: str = "claude-haiku-4-5-20251001"
    opus_model: str = "claude-opus-4-8"

    # Pipeline
    wake_phrases: list[str] = field(default_factory=lambda: ["hey claude"])
    wake_fuzzy_threshold: float = 0.66
    follow_up_threshold: float = 0.80
    intent_confidence_threshold: float = 0.75

    # Budget
    hourly_usd_limit: float = 0.0      # 0 = unlimited
    daily_usd_limit: float = 0.0       # 0 = unlimited

    # Response cache
    cache_max_size: int = 256
    cache_ttl_seconds: int = 300

    # Latency profiler
    profiler_max_history: int = 1000

    # TTS / audio feedback
    tts_volume: float = 1.0
    feedback_volume: float = 0.5
    feedback_enabled: bool = True

    # Reminders
    reminder_check_interval: int = 10   # seconds

    # Paths (empty string = use default in each module)
    notes_dir: str = ""
    personas_dir: str = ""
    plugins_dir: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------


def _coerce(value: str, target_type: type) -> Any:
    """Convert a string (from env var or JSON) to *target_type*."""
    if target_type == bool:
        return value.lower() in ("1", "true", "yes", "on")
    if target_type == int:
        return int(value)
    if target_type == float:
        return float(value)
    if target_type == list:
        # Comma-separated strings for env vars
        return [v.strip() for v in value.split(",") if v.strip()]
    return value  # str passthrough


def _field_type(f: Any) -> type:
    """Best-effort extraction of the plain Python type for a dataclass field."""
    t = f.type if isinstance(f.type, type) else type(None)
    # Handle 'list[str]' annotation stored as string at runtime
    if not isinstance(f.type, type):
        ann = str(f.type)
        if ann.startswith("list"):
            return list
        try:
            return eval(f.type)  # noqa: S307 — internal config types only
        except Exception:
            return str
    return t


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------


class ConfigManager:
    """
    Loads, merges, and persists Hey-Claude configuration.

    Parameters
    ----------
    path:
        JSON config file path.  Created on first ``save()``.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._cfg = Config()
        self._file_overrides: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Config:
        """
        Load config from disk (if the file exists) then apply env overrides.

        Returns the merged Config.  Safe to call multiple times.
        """
        self._file_overrides = {}

        if self._path and self._path.exists():
            try:
                data: dict[str, Any] = json.loads(self._path.read_text())
                self._file_overrides = data
                logger.debug("Loaded config from %s", self._path)
            except Exception:
                logger.exception("Failed to load config from %s; using defaults", self._path)

        self._cfg = self._build()
        return self._cfg

    @property
    def config(self) -> Config:
        """Return the currently loaded Config (call load() first)."""
        return self._cfg

    def get(self, key: str, default: Any = None) -> Any:
        """Return a single config value by key name."""
        return getattr(self._cfg, key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Override a config key in memory and in the file-override layer.

        Does NOT persist to disk — call save() for that.
        """
        if not hasattr(self._cfg, key):
            raise KeyError(f"Unknown config key: {key!r}")
        self._file_overrides[key] = value
        self._cfg = self._build()

    def reset(self, key: str) -> None:
        """Remove a file-override for *key*, reverting to default (or env var)."""
        self._file_overrides.pop(key, None)
        self._cfg = self._build()

    def save(self) -> None:
        """Persist the current file-override layer to disk."""
        if not self._path:
            logger.warning("No persist path configured; save() is a no-op")
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._file_overrides, indent=2))
            logger.debug("Saved config to %s", self._path)
        except Exception:
            logger.exception("Failed to save config to %s", self._path)

    def reset_all(self) -> None:
        """Clear all file overrides and revert to defaults + env vars."""
        self._file_overrides = {}
        self._cfg = self._build()

    def diff(self) -> dict[str, Any]:
        """Return keys that differ from the built-in default."""
        default = Config()
        return {
            k: getattr(self._cfg, k)
            for k in (f.name for f in fields(Config))
            if getattr(self._cfg, k) != getattr(default, k)
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self) -> Config:
        """Merge defaults ← file overrides ← env vars → Config."""
        merged: dict[str, Any] = {}

        field_map = {f.name: f for f in fields(Config)}

        # Start with file overrides
        for key, value in self._file_overrides.items():
            if key in field_map:
                try:
                    ft = _field_type(field_map[key])
                    merged[key] = _coerce(str(value), ft) if isinstance(value, str) else value
                except Exception:
                    logger.warning("Could not coerce config key %r; skipping", key)

        # Apply env vars (highest priority)
        for fname, fobj in field_map.items():
            env_key = _ENV_PREFIX + fname.upper()
            env_val = os.environ.get(env_key)
            if env_val is not None:
                try:
                    ft = _field_type(fobj)
                    merged[fname] = _coerce(env_val, ft)
                    logger.debug("Config %r overridden by env var %s", fname, env_key)
                except Exception:
                    logger.warning("Invalid env var %s=%r; ignoring", env_key, env_val)

        try:
            return Config(**merged)
        except TypeError as exc:
            logger.error("Config construction failed: %s; falling back to defaults", exc)
            return Config()
