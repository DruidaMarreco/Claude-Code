"""Unit tests for PluginLoader — no API key or audio required."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hey_claude_features.plugin_system import PluginLoadError, PluginLoader, ToolLike


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeTool:
    name: str
    description: str
    input_schema: dict[str, Any]


def _write_plugin(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# ToolLike protocol
# ---------------------------------------------------------------------------


class TestToolLikeProtocol:
    def test_fake_tool_satisfies_protocol(self):
        t = FakeTool(name="x", description="y", input_schema={})
        assert isinstance(t, ToolLike)

    def test_missing_field_fails_protocol(self):
        class Bad:
            name = "x"
            description = "y"
            # no input_schema

        assert not isinstance(Bad(), ToolLike)


# ---------------------------------------------------------------------------
# PluginLoader.load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        loader = PluginLoader(plugin_dir=tmp_path)
        assert loader.load() == []

    def test_nonexistent_dir_returns_empty_list(self, tmp_path):
        loader = PluginLoader(plugin_dir=tmp_path / "does_not_exist")
        assert loader.load() == []

    def test_loads_valid_plugin(self, tmp_path):
        _write_plugin(
            tmp_path,
            "hello.py",
            """
            from dataclasses import dataclass
            from typing import Any

            @dataclass
            class T:
                name: str
                description: str
                input_schema: dict

            TOOLS = [T(name="greet", description="Says hello", input_schema={})]
            """,
        )
        loader = PluginLoader(plugin_dir=tmp_path)
        tools = loader.load()
        assert len(tools) == 1
        assert tools[0].name == "greet"

    def test_loads_multiple_plugins(self, tmp_path):
        for i in range(3):
            _write_plugin(
                tmp_path,
                f"plugin_{i}.py",
                f"""
                from dataclasses import dataclass

                @dataclass
                class T:
                    name: str
                    description: str
                    input_schema: dict

                TOOLS = [T(name="tool_{i}", description="desc", input_schema={{}})]
                """,
            )
        loader = PluginLoader(plugin_dir=tmp_path)
        tools = loader.load()
        assert len(tools) == 3

    def test_skips_underscore_files(self, tmp_path):
        _write_plugin(tmp_path, "_private.py", "TOOLS = []")
        loader = PluginLoader(plugin_dir=tmp_path)
        assert loader.load() == []

    def test_skips_plugin_missing_tools(self, tmp_path):
        _write_plugin(tmp_path, "no_tools.py", "x = 1")
        loader = PluginLoader(plugin_dir=tmp_path)
        # Should not raise; bad plugin is skipped
        assert loader.load() == []

    def test_skips_plugin_with_syntax_error(self, tmp_path):
        _write_plugin(tmp_path, "broken.py", "TOOLS = [invalid syntax!!!]")
        loader = PluginLoader(plugin_dir=tmp_path)
        assert loader.load() == []

    def test_skips_non_toollike_items(self, tmp_path):
        _write_plugin(
            tmp_path,
            "bad_item.py",
            """
            TOOLS = ["not a tool", 42]
            """,
        )
        loader = PluginLoader(plugin_dir=tmp_path)
        tools = loader.load()
        assert tools == []

    def test_partial_load_skips_bad_item(self, tmp_path):
        _write_plugin(
            tmp_path,
            "mixed.py",
            """
            from dataclasses import dataclass

            @dataclass
            class T:
                name: str
                description: str
                input_schema: dict

            TOOLS = [T(name="good", description="ok", input_schema={}), "bad"]
            """,
        )
        loader = PluginLoader(plugin_dir=tmp_path)
        tools = loader.load()
        assert len(tools) == 1
        assert tools[0].name == "good"


# ---------------------------------------------------------------------------
# PluginLoader.list_plugins
# ---------------------------------------------------------------------------


class TestListPlugins:
    def test_lists_plugin_files(self, tmp_path):
        (tmp_path / "a.py").write_text("TOOLS = []")
        (tmp_path / "b.py").write_text("TOOLS = []")
        (tmp_path / "_skip.py").write_text("")
        loader = PluginLoader(plugin_dir=tmp_path)
        names = [p.name for p in loader.list_plugins()]
        assert "a.py" in names
        assert "b.py" in names
        assert "_skip.py" not in names

    def test_empty_when_dir_missing(self, tmp_path):
        loader = PluginLoader(plugin_dir=tmp_path / "nope")
        assert loader.list_plugins() == []
