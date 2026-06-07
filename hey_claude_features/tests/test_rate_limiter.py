"""Unit tests for RateLimiter — no API key required."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from hey_claude_features.rate_limiter import BudgetExceeded, RateLimiter, SpendSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _at(offset: float) -> float:
    """Return a fake timestamp `offset` seconds from now."""
    return time.time() + offset


def _limiter(hourly=None, daily=None) -> RateLimiter:
    return RateLimiter(hourly_usd=hourly, daily_usd=daily)


# ---------------------------------------------------------------------------
# No limits configured
# ---------------------------------------------------------------------------


class TestNoLimits:
    def test_check_passes_with_no_limits(self):
        rl = _limiter()
        rl.record(999.0)
        rl.check()  # should not raise

    def test_snapshot_limits_are_none(self):
        rl = _limiter()
        s = rl.snapshot()
        assert s.hourly_limit is None
        assert s.daily_limit is None
        assert s.hourly_remaining is None
        assert s.daily_remaining is None


# ---------------------------------------------------------------------------
# Hourly limit
# ---------------------------------------------------------------------------


class TestHourlyLimit:
    def test_under_limit_passes(self):
        rl = _limiter(hourly=1.00)
        rl.record(0.50)
        rl.check()

    def test_at_limit_raises(self):
        rl = _limiter(hourly=1.00)
        rl.record(1.00)
        with pytest.raises(BudgetExceeded, match="Hourly budget"):
            rl.check()

    def test_over_limit_raises(self):
        rl = _limiter(hourly=0.10)
        rl.record(0.05)
        rl.record(0.06)
        with pytest.raises(BudgetExceeded):
            rl.check()

    def test_error_message_mentions_budget(self):
        rl = _limiter(hourly=0.50)
        rl.record(0.50)
        with pytest.raises(BudgetExceeded) as exc_info:
            rl.check()
        assert "0.50" in str(exc_info.value)

    def test_old_spend_evicted_after_hour(self):
        rl = _limiter(hourly=0.50)
        # Inject an old entry directly, older than 1 hour
        old_ts = time.time() - 3700
        rl._hourly_log.append((old_ts, 0.49))
        rl.check()  # should not raise — old entry evicted

    def test_snapshot_remaining(self):
        rl = _limiter(hourly=1.00)
        rl.record(0.30)
        s = rl.snapshot()
        assert abs(s.hourly_remaining - 0.70) < 0.001


# ---------------------------------------------------------------------------
# Daily limit
# ---------------------------------------------------------------------------


class TestDailyLimit:
    def test_under_daily_limit_passes(self):
        rl = _limiter(daily=5.00)
        rl.record(4.99)
        rl.check()

    def test_at_daily_limit_raises(self):
        rl = _limiter(daily=2.00)
        rl.record(2.00)
        with pytest.raises(BudgetExceeded, match="Daily budget"):
            rl.check()

    def test_old_daily_spend_evicted_after_day(self):
        rl = _limiter(daily=1.00)
        old_ts = time.time() - (86400 + 10)
        rl._daily_log.append((old_ts, 0.99))
        rl.check()  # should not raise

    def test_daily_remaining_at_zero(self):
        rl = _limiter(daily=1.00)
        rl.record(1.00)
        s = rl.snapshot()
        assert s.daily_remaining == 0.0


# ---------------------------------------------------------------------------
# Both limits
# ---------------------------------------------------------------------------


class TestBothLimits:
    def test_hourly_triggers_before_daily(self):
        rl = _limiter(hourly=0.10, daily=10.00)
        rl.record(0.10)
        with pytest.raises(BudgetExceeded, match="Hourly"):
            rl.check()

    def test_daily_triggers_when_hourly_ok(self):
        rl = _limiter(hourly=10.00, daily=0.10)
        rl.record(0.10)
        with pytest.raises(BudgetExceeded, match="Daily"):
            rl.check()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_spend(self):
        rl = _limiter(hourly=0.10, daily=1.00)
        rl.record(0.10)
        with pytest.raises(BudgetExceeded):
            rl.check()
        rl.reset()
        rl.check()  # should not raise after reset

    def test_snapshot_after_reset(self):
        rl = _limiter(hourly=1.00)
        rl.record(0.50)
        rl.reset()
        s = rl.snapshot()
        assert s.hourly_usd == 0.0


# ---------------------------------------------------------------------------
# Zero / negative spend ignored
# ---------------------------------------------------------------------------


class TestZeroSpend:
    def test_zero_usd_not_recorded(self):
        rl = _limiter(hourly=0.01)
        rl.record(0.0)
        rl.check()  # should not raise
        assert len(rl._hourly_log) == 0

    def test_negative_usd_not_recorded(self):
        rl = _limiter(daily=0.01)
        rl.record(-1.0)
        rl.check()
