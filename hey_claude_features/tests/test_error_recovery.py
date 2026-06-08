"""Unit tests for error recovery utilities — no API key required."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from hey_claude_features.error_recovery import (
    CircuitBreaker,
    CircuitOpen,
    CircuitState,
    PipelineGuard,
    RetryExhausted,
    RetryPolicy,
)


# ---------------------------------------------------------------------------
# RetryPolicy.delay_for
# ---------------------------------------------------------------------------


class TestRetryPolicyDelay:
    def test_first_attempt_uses_base(self):
        p = RetryPolicy(base_delay=1.0, jitter=False)
        assert p.delay_for(0) == pytest.approx(1.0)

    def test_backoff_doubles(self):
        p = RetryPolicy(base_delay=1.0, multiplier=2.0, jitter=False)
        assert p.delay_for(1) == pytest.approx(2.0)
        assert p.delay_for(2) == pytest.approx(4.0)

    def test_capped_at_max(self):
        p = RetryPolicy(base_delay=1.0, multiplier=10.0, max_delay=5.0, jitter=False)
        assert p.delay_for(3) == pytest.approx(5.0)

    def test_jitter_positive(self):
        p = RetryPolicy(base_delay=1.0, jitter=True)
        delay = p.delay_for(0)
        assert delay >= 1.0


# ---------------------------------------------------------------------------
# RetryPolicy.call
# ---------------------------------------------------------------------------


class TestRetryPolicyCall:
    def test_succeeds_first_attempt(self):
        p = RetryPolicy(max_attempts=3)
        result = p.call(lambda: 42)
        assert result == 42

    def test_retries_on_failure(self):
        attempts = [0]

        def flaky():
            attempts[0] += 1
            if attempts[0] < 3:
                raise RuntimeError("not yet")
            return "ok"

        p = RetryPolicy(max_attempts=3, base_delay=0)
        assert p.call(flaky) == "ok"
        assert attempts[0] == 3

    def test_exhausted_raises_retry_exhausted(self):
        p = RetryPolicy(max_attempts=2, base_delay=0)
        with pytest.raises(RetryExhausted) as exc_info:
            p.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert exc_info.value.attempts == 2

    def test_non_retryable_exception_not_retried(self):
        attempts = [0]

        def fn():
            attempts[0] += 1
            raise ValueError("not retryable")

        p = RetryPolicy(max_attempts=3, base_delay=0, retryable=(RuntimeError,))
        with pytest.raises(ValueError):
            p.call(fn)
        assert attempts[0] == 1

    def test_passes_args_kwargs(self):
        p = RetryPolicy(max_attempts=1)
        result = p.call(lambda x, y=0: x + y, 3, y=4)
        assert result == 7

    def test_retry_decorator(self):
        calls = [0]

        @RetryPolicy(max_attempts=2, base_delay=0).retry
        def fn():
            calls[0] += 1
            if calls[0] < 2:
                raise RuntimeError("retry me")
            return "done"

        assert fn() == "done"
        assert calls[0] == 2

    def test_sleep_called_between_retries(self):
        p = RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False)
        attempt = [0]

        def flaky():
            attempt[0] += 1
            if attempt[0] < 3:
                raise RuntimeError("fail")
            return "ok"

        with patch("hey_claude_features.error_recovery.time.sleep") as mock_sleep:
            p.call(flaky)
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(open_after=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.OPEN

    def test_open_raises_circuit_open(self):
        cb = CircuitBreaker(open_after=1)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        with pytest.raises(CircuitOpen):
            cb.call(lambda: "ok")

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(open_after=3)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        cb.call(lambda: "ok")
        assert cb.failure_count == 0

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(open_after=1, reset_timeout=0.01)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_recovery_from_half_open(self):
        cb = CircuitBreaker(open_after=1, reset_timeout=0.01)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        time.sleep(0.02)
        cb.call(lambda: "ok")
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(open_after=1, reset_timeout=0.01)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        time.sleep(0.02)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail again")))
        assert cb.state == CircuitState.OPEN

    def test_manual_reset(self):
        cb = CircuitBreaker(open_after=1)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_call_returns_value(self):
        cb = CircuitBreaker()
        assert cb.call(lambda: 99) == 99


# ---------------------------------------------------------------------------
# PipelineGuard
# ---------------------------------------------------------------------------


class TestPipelineGuard:
    def test_success_passes_through(self):
        g = PipelineGuard("test", max_attempts=1)
        assert g.call(lambda: "hi") == "hi"

    def test_retries_then_raises(self):
        g = PipelineGuard("test", max_attempts=2, base_delay=0, open_after=10)
        with pytest.raises(RetryExhausted):
            g.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

    def test_fallback_called_on_exhaustion(self):
        fallback = MagicMock(return_value="safe default")
        g = PipelineGuard("test", max_attempts=1, base_delay=0, fallback=fallback)
        result = g.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert result == "safe default"
        fallback.assert_called_once()

    def test_circuit_state_exposed(self):
        g = PipelineGuard("test")
        assert g.circuit_state == CircuitState.CLOSED

    def test_failure_count_exposed(self):
        g = PipelineGuard("test", max_attempts=1, base_delay=0, open_after=10)
        with pytest.raises(RetryExhausted):
            g.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert g.failure_count >= 1

    def test_reset_clears_circuit(self):
        g = PipelineGuard("test", max_attempts=1, base_delay=0, open_after=1)
        with pytest.raises((RetryExhausted, CircuitOpen)):
            g.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        g.reset()
        assert g.circuit_state == CircuitState.CLOSED

    def test_circuit_open_uses_fallback(self):
        fallback = MagicMock(return_value="fallback")
        g = PipelineGuard("test", max_attempts=1, base_delay=0, open_after=1, fallback=fallback)
        # First call opens circuit
        g.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        # Second call hits open circuit → fallback
        result = g.call(lambda: "would succeed")
        assert result == "fallback"


# ---------------------------------------------------------------------------
# RetryExhausted / CircuitOpen exceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_retry_exhausted_message(self):
        exc = RetryExhausted(3, RuntimeError("root"))
        assert "3" in str(exc)

    def test_circuit_open_message(self):
        exc = CircuitOpen("my_stage", 5)
        assert "my_stage" in str(exc)
        assert "5" in str(exc)
