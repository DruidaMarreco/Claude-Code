"""Unit tests for ReminderSystem — no API key or audio required."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hey_claude_features.reminder_system import Reminder, ReminderSystem, parse_time


# ---------------------------------------------------------------------------
# parse_time
# ---------------------------------------------------------------------------


class TestParseTime:
    def test_iso_datetime(self):
        dt = parse_time("2030-01-15T09:30")
        assert dt is not None
        assert dt.year == 2030
        assert dt.hour == 9
        assert dt.minute == 30

    def test_iso_datetime_with_seconds(self):
        dt = parse_time("2030-01-15T09:30:45")
        assert dt is not None
        assert dt.second == 45

    def test_iso_date_only(self):
        dt = parse_time("2030-06-01")
        assert dt is not None
        assert dt.year == 2030

    def test_delta_in_30_minutes(self):
        before = datetime.now()
        dt = parse_time("in 30 minutes")
        assert dt is not None
        diff = (dt - before).total_seconds()
        assert 29 * 60 <= diff <= 31 * 60

    def test_delta_in_1_hour(self):
        before = datetime.now()
        dt = parse_time("in 1 hour")
        assert dt is not None
        diff = (dt - before).total_seconds()
        assert 3500 <= diff <= 3700

    def test_delta_generic_n_minutes(self):
        before = datetime.now()
        dt = parse_time("in 45 minutes")
        assert dt is not None
        diff = (dt - before).total_seconds()
        assert 44 * 60 <= diff <= 46 * 60

    def test_delta_days(self):
        before = datetime.now()
        dt = parse_time("in 2 days")
        assert dt is not None
        diff = (dt - before).total_seconds()
        assert 2 * 86400 - 10 <= diff <= 2 * 86400 + 10

    def test_unparseable_returns_none(self):
        assert parse_time("next tuesday afternoon") is None
        assert parse_time("soon") is None
        assert parse_time("") is None


# ---------------------------------------------------------------------------
# Reminder dataclass
# ---------------------------------------------------------------------------


class TestReminderModel:
    def test_is_due_when_past(self):
        r = Reminder(id="x", message="test", fire_at=time.time() - 1)
        assert r.is_due()

    def test_not_due_when_future(self):
        r = Reminder(id="x", message="test", fire_at=time.time() + 3600)
        assert not r.is_due()

    def test_not_due_when_already_fired(self):
        r = Reminder(id="x", message="test", fire_at=time.time() - 1, fired=True)
        assert not r.is_due()


# ---------------------------------------------------------------------------
# set_reminder
# ---------------------------------------------------------------------------


def _rs(tmp_path: Path, on_fire=None) -> ReminderSystem:
    return ReminderSystem(
        on_fire=on_fire or MagicMock(),
        store_path=tmp_path / "reminders.json",
    )


class TestSetReminder:
    def test_valid_delta_returns_confirmation(self, tmp_path):
        rs = _rs(tmp_path)
        result = rs.set_reminder("take pills", "in 30 minutes")
        assert "remind" in result.lower()
        assert "take pills" in result

    def test_valid_iso_returns_confirmation(self, tmp_path):
        rs = _rs(tmp_path)
        future = (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
        result = rs.set_reminder("check oven", future)
        assert "check oven" in result

    def test_bad_time_returns_error_message(self, tmp_path):
        rs = _rs(tmp_path)
        result = rs.set_reminder("do thing", "whenever")
        assert "couldn't understand" in result.lower()

    def test_past_time_returns_error(self, tmp_path):
        rs = _rs(tmp_path)
        past = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
        result = rs.set_reminder("too late", past)
        assert "past" in result.lower()

    def test_reminder_stored(self, tmp_path):
        rs = _rs(tmp_path)
        rs.set_reminder("meeting", "in 1 hour")
        assert len(rs._reminders) == 1


# ---------------------------------------------------------------------------
# list_reminders
# ---------------------------------------------------------------------------


class TestListReminders:
    def test_empty(self, tmp_path):
        rs = _rs(tmp_path)
        result = rs.list_reminders()
        assert "no pending" in result.lower()

    def test_lists_pending(self, tmp_path):
        rs = _rs(tmp_path)
        rs.set_reminder("lunch", "in 1 hour")
        rs.set_reminder("walk dog", "in 2 hours")
        result = rs.list_reminders()
        assert "lunch" in result
        assert "walk dog" in result
        assert "2 reminder" in result

    def test_does_not_show_fired(self, tmp_path):
        rs = _rs(tmp_path)
        rs.set_reminder("old", "in 1 hour")
        # Mark it as fired manually
        for r in rs._reminders.values():
            r.fired = True
        result = rs.list_reminders()
        assert "no pending" in result.lower()


# ---------------------------------------------------------------------------
# Background firing
# ---------------------------------------------------------------------------


class TestFiring:
    def test_fires_due_reminder(self, tmp_path):
        fired = []
        rs = _rs(tmp_path, on_fire=lambda msg: fired.append(msg))

        # Insert an already-due reminder directly
        r = Reminder(id="abc", message="hello!", fire_at=time.time() - 1)
        rs._reminders["abc"] = r

        rs._check_due()  # trigger manually without background thread

        assert fired == ["hello!"]
        assert rs._reminders["abc"].fired is True

    def test_does_not_fire_future_reminder(self, tmp_path):
        fired = []
        rs = _rs(tmp_path, on_fire=lambda msg: fired.append(msg))
        rs.set_reminder("future", "in 1 hour")
        rs._check_due()
        assert fired == []

    def test_callback_exception_does_not_crash(self, tmp_path):
        def bad_callback(msg: str) -> None:
            raise RuntimeError("speaker offline")

        rs = _rs(tmp_path, on_fire=bad_callback)
        r = Reminder(id="x", message="test", fire_at=time.time() - 1)
        rs._reminders["x"] = r
        rs._check_due()  # should not raise


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "reminders.json"
        rs1 = ReminderSystem(on_fire=MagicMock(), store_path=path)
        rs1.set_reminder("dentist", "in 1 hour")

        rs2 = ReminderSystem(on_fire=MagicMock(), store_path=path)
        assert len(rs2._reminders) == 1
        r = next(iter(rs2._reminders.values()))
        assert r.message == "dentist"

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "reminders.json"
        path.write_text("{{bad json")
        rs = ReminderSystem(on_fire=MagicMock(), store_path=path)
        assert len(rs._reminders) == 0


# ---------------------------------------------------------------------------
# tools() descriptor
# ---------------------------------------------------------------------------


class TestToolDescriptors:
    def test_returns_two_tools(self, tmp_path):
        rs = _rs(tmp_path)
        tools = rs.tools()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"set_reminder", "list_reminders"}

    def test_tool_has_required_schema_fields(self, tmp_path):
        rs = _rs(tmp_path)
        for tool in rs.tools():
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
