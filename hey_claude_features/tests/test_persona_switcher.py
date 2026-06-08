"""Unit tests for PersonaSwitcher — no API key required."""

from __future__ import annotations

from pathlib import Path

import pytest

from hey_claude_features.persona_switcher import PersonaSwitcher, _BUILTIN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ps(tmp_path: Path, initial: str = "default") -> PersonaSwitcher:
    return PersonaSwitcher(personas_dir=tmp_path / "personas", initial=initial)


# ---------------------------------------------------------------------------
# Built-in personas
# ---------------------------------------------------------------------------


class TestBuiltins:
    def test_all_builtins_available(self, tmp_path):
        ps = _ps(tmp_path)
        for name in _BUILTIN:
            assert name in ps.list_names()

    def test_default_active_on_init(self, tmp_path):
        ps = _ps(tmp_path)
        assert ps.active_name == "default"

    def test_current_prompt_matches_builtin(self, tmp_path):
        ps = _ps(tmp_path)
        assert ps.current_prompt == _BUILTIN["default"]

    def test_initial_persona_set(self, tmp_path):
        ps = _ps(tmp_path, initial="concise")
        assert ps.active_name == "concise"

    def test_each_builtin_has_non_empty_prompt(self, tmp_path):
        ps = _ps(tmp_path)
        for name, prompt in ps:
            assert len(prompt) > 20, f"Persona '{name}' prompt is too short"


# ---------------------------------------------------------------------------
# switch()
# ---------------------------------------------------------------------------


class TestSwitch:
    def test_switch_to_valid_persona(self, tmp_path):
        ps = _ps(tmp_path)
        result = ps.switch("concise")
        assert ps.active_name == "concise"
        assert "concise" in result.lower()

    def test_switch_changes_prompt(self, tmp_path):
        ps = _ps(tmp_path)
        default_prompt = ps.current_prompt
        ps.switch("professional")
        assert ps.current_prompt != default_prompt

    def test_switch_unknown_raises(self, tmp_path):
        ps = _ps(tmp_path)
        with pytest.raises(ValueError, match="Unknown persona"):
            ps.switch("nonexistent")

    def test_switch_case_insensitive(self, tmp_path):
        ps = _ps(tmp_path)
        ps.switch("CONCISE")
        assert ps.active_name == "concise"

    def test_switch_strips_whitespace(self, tmp_path):
        ps = _ps(tmp_path)
        ps.switch("  tutor  ")
        assert ps.active_name == "tutor"

    def test_error_lists_available_personas(self, tmp_path):
        ps = _ps(tmp_path)
        with pytest.raises(ValueError) as exc_info:
            ps.switch("unknown")
        assert "default" in str(exc_info.value)


# ---------------------------------------------------------------------------
# add() / remove()
# ---------------------------------------------------------------------------


class TestAddRemove:
    def test_add_custom_persona(self, tmp_path):
        ps = _ps(tmp_path)
        ps.add("pirate", "Ye shall respond like a pirate, arrr!")
        assert "pirate" in ps.list_names()

    def test_added_persona_switchable(self, tmp_path):
        ps = _ps(tmp_path)
        ps.add("robot", "Respond in a robotic, emotionless style.")
        ps.switch("robot")
        assert ps.active_name == "robot"
        assert "robot" in ps.current_prompt.lower()

    def test_add_with_persist_saves_file(self, tmp_path):
        personas_dir = tmp_path / "personas"
        ps = PersonaSwitcher(personas_dir=personas_dir)
        ps.add("test_persona", "A test prompt.", persist=True)
        assert (personas_dir / "test_persona.md").exists()

    def test_persisted_persona_loaded_on_new_instance(self, tmp_path):
        personas_dir = tmp_path / "personas"
        ps1 = PersonaSwitcher(personas_dir=personas_dir)
        ps1.add("saved", "Saved prompt content.", persist=True)

        ps2 = PersonaSwitcher(personas_dir=personas_dir)
        assert "saved" in ps2.list_names()
        assert ps2.describe("saved") == "Saved prompt content."

    def test_remove_custom_persona(self, tmp_path):
        ps = _ps(tmp_path)
        ps.add("temp", "Temp prompt")
        ps.remove("temp")
        assert "temp" not in ps.list_names()

    def test_remove_returns_true_when_existed(self, tmp_path):
        ps = _ps(tmp_path)
        ps.add("temp2", "Prompt")
        assert ps.remove("temp2") is True

    def test_remove_returns_false_when_missing(self, tmp_path):
        ps = _ps(tmp_path)
        assert ps.remove("never_added") is False

    def test_cannot_remove_builtin(self, tmp_path):
        ps = _ps(tmp_path)
        with pytest.raises(ValueError, match="built-in"):
            ps.remove("default")

    def test_overwrite_custom_persona(self, tmp_path):
        ps = _ps(tmp_path)
        ps.add("custom", "First prompt")
        ps.add("custom", "Second prompt")
        assert ps.describe("custom") == "Second prompt"


# ---------------------------------------------------------------------------
# Custom persona files
# ---------------------------------------------------------------------------


class TestCustomFiles:
    def test_loads_md_files_from_dir(self, tmp_path):
        personas_dir = tmp_path / "personas"
        personas_dir.mkdir()
        (personas_dir / "chef.md").write_text("Respond like a gourmet chef.")
        ps = PersonaSwitcher(personas_dir=personas_dir)
        assert "chef" in ps.list_names()
        assert "chef" in ps.describe("chef")

    def test_bad_file_skipped_gracefully(self, tmp_path):
        personas_dir = tmp_path / "personas"
        personas_dir.mkdir()
        # Create an unreadable file (write valid content then make it a dir)
        bad = personas_dir / "bad.md"
        bad.mkdir()  # This will cause read_text to fail
        ps = PersonaSwitcher(personas_dir=personas_dir)
        assert "bad" not in ps.list_names()  # skipped


# ---------------------------------------------------------------------------
# list_names / describe / iteration
# ---------------------------------------------------------------------------


class TestListDescribeIter:
    def test_list_sorted(self, tmp_path):
        ps = _ps(tmp_path)
        names = ps.list_names()
        assert names == sorted(names)

    def test_describe_valid(self, tmp_path):
        ps = _ps(tmp_path)
        prompt = ps.describe("creative")
        assert len(prompt) > 20

    def test_describe_invalid_raises(self, tmp_path):
        ps = _ps(tmp_path)
        with pytest.raises(ValueError):
            ps.describe("nonexistent")

    def test_iteration_yields_all(self, tmp_path):
        ps = _ps(tmp_path)
        ps.add("extra", "Extra prompt")
        items = dict(ps)
        assert "default" in items
        assert "extra" in items
        assert len(items) == len(ps.list_names())
