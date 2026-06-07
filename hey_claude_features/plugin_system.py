"""
Drop-in plugin loader for Hey-Claude tools.

Any .py file placed in ~/.hey-claude/plugins/ (or a custom directory) that
exports a module-level `TOOLS` list of Tool-compatible objects is auto-loaded
on startup — no edits to core code required.

Usage in tools.py / app.py:
    from hey_claude_features.plugin_system import PluginLoader

    loader = PluginLoader()          # scans ~/.hey-claude/plugins/ by default
    for tool in loader.load():
        registry.register(tool)

Plugin file example (~/.hey-claude/plugins/weather.py):
    from dataclasses import dataclass
    from hey_claude.tools import Tool   # or any compatible type

    def _get_weather(location: str) -> str:
        return f"Sunny in {location}"

    TOOLS = [
        Tool(
            name="get_weather",
            description="Get the current weather for a location.",
            input_schema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
            function=_get_weather,
            requires_confirmation=False,
        )
    ]
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_PLUGIN_DIR = Path.home() / ".hey-claude" / "plugins"


@runtime_checkable
class ToolLike(Protocol):
    """Structural protocol matching Hey-Claude's Tool dataclass."""

    name: str
    description: str
    input_schema: dict[str, Any]


class PluginLoadError(Exception):
    """Raised when a plugin file cannot be loaded."""


class PluginLoader:
    """
    Discovers and loads Tool objects from Python plugin files.

    Each plugin file must export a module-level `TOOLS` list whose items
    satisfy the ToolLike protocol.  Malformed plugins are logged and skipped
    so one bad plugin never kills the assistant.
    """

    def __init__(self, plugin_dir: Path | str | None = None) -> None:
        self.plugin_dir = Path(plugin_dir) if plugin_dir else DEFAULT_PLUGIN_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[Any]:
        """
        Scan the plugin directory and return all discovered Tool objects.

        Returns an empty list if the directory does not exist.
        """
        if not self.plugin_dir.exists():
            logger.debug("Plugin directory %s does not exist; skipping", self.plugin_dir)
            return []

        tools: list[Any] = []
        for path in sorted(self.plugin_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                plugin_tools = self._load_file(path)
                tools.extend(plugin_tools)
                logger.info("Loaded %d tool(s) from %s", len(plugin_tools), path.name)
            except PluginLoadError as exc:
                logger.warning("Skipping plugin %s: %s", path.name, exc)

        return tools

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> list[Any]:
        """Import a single plugin file and extract its TOOLS list."""
        module_name = f"_hey_claude_plugin_{path.stem}"

        # Avoid double-loading if the module is already imported.
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Cannot create module spec from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)  # type: ignore[union-attr]
            except Exception as exc:
                del sys.modules[module_name]
                raise PluginLoadError(f"Execution error: {exc}") from exc

        tools_attr = getattr(module, "TOOLS", None)
        if tools_attr is None:
            raise PluginLoadError("Missing module-level 'TOOLS' list")
        if not isinstance(tools_attr, list):
            raise PluginLoadError(f"'TOOLS' must be a list, got {type(tools_attr).__name__}")

        valid: list[Any] = []
        for i, tool in enumerate(tools_attr):
            if not isinstance(tool, ToolLike):
                logger.warning(
                    "Item %d in %s does not satisfy ToolLike protocol; skipping", i, path.name
                )
                continue
            valid.append(tool)

        return valid

    def list_plugins(self) -> list[Path]:
        """Return paths of all discoverable plugin files (without loading them)."""
        if not self.plugin_dir.exists():
            return []
        return sorted(p for p in self.plugin_dir.glob("*.py") if not p.name.startswith("_"))
