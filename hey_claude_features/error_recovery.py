"""
Error recovery utilities for Hey-Claude's pipeline.

The pipeline has three external failure points:
  - Anthropic API (Haiku gate, Opus responder)
  - Audio hardware (microphone, speaker)
  - Local I/O (file persistence for cache, notes, config)

This module provides:

  RetryPolicy       — exponential backoff with jitter for API calls
  CircuitBreaker    — stops hammering a failing service
  with_retry()      — decorator / context helper using a RetryPolicy
  PipelineGuard     — composes retry + circuit breaker for a named stage

Usage in app.py
---------------
    from hey_claude_features.error_recovery import PipelineGuard, RetryPolicy

    # Wrap the Haiku gate:
    haiku_guard = PipelineGuard("haiku_gate", max_attempts=3, open_after=5)

    result = haiku_guard.call(wake_detector.detect, transcript)

    # Or as a decorator:
    policy = RetryPolicy(max_attempts=3, base_delay=0.5)

    @policy.retry
    def call_api():
        return client.messages.create(...)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(f"Failed after {attempts} attempts: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


class CircuitOpen(Exception):
    """Raised when the circuit breaker is open (service marked as down)."""

    def __init__(self, name: str, failures: int) -> None:
        super().__init__(f"Circuit '{name}' is open after {failures} failures")
        self.name = name
        self.failures = failures


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


@dataclass
class RetryPolicy:
    """
    Exponential backoff retry policy.

    Parameters
    ----------
    max_attempts:
        Total attempts (1 = no retry).
    base_delay:
        Initial delay in seconds.
    max_delay:
        Cap on delay growth.
    multiplier:
        Backoff multiplier per attempt.
    jitter:
        If True, add uniform(0, delay) jitter to each sleep.
    retryable:
        Tuple of exception types that trigger a retry.  Default: Exception.
    """
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: bool = True
    retryable: tuple[type[BaseException], ...] = field(
        default_factory=lambda: (Exception,)
    )

    def delay_for(self, attempt: int) -> float:
        """Return the sleep duration for the given attempt (0-indexed)."""
        raw = self.base_delay * (self.multiplier ** attempt)
        capped = min(raw, self.max_delay)
        if self.jitter:
            capped += random.uniform(0, capped * 0.1)
        return capped

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call *fn* with retries according to this policy."""
        last_exc: BaseException = RuntimeError("No attempts made")
        for attempt in range(self.max_attempts):
            try:
                return fn(*args, **kwargs)
            except self.retryable as exc:
                last_exc = exc
                if attempt < self.max_attempts - 1:
                    delay = self.delay_for(attempt)
                    logger.warning(
                        "Attempt %d/%d failed (%s); retrying in %.2fs",
                        attempt + 1, self.max_attempts, type(exc).__name__, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "All %d attempts exhausted: %s", self.max_attempts, exc
                    )
        raise RetryExhausted(self.max_attempts, last_exc)

    def retry(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Decorator version of call()."""
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call(fn, *args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    CLOSED = "closed"     # normal operation
    OPEN = "open"         # blocking calls
    HALF_OPEN = "half_open"  # testing recovery


class CircuitBreaker:
    """
    Circuit breaker — stops calling a failing service after *open_after*
    consecutive failures, then probes recovery after *reset_timeout* seconds.

    Parameters
    ----------
    name:
        Human-readable name for logging.
    open_after:
        Number of consecutive failures before opening.
    reset_timeout:
        Seconds to wait before moving from OPEN → HALF_OPEN.
    """

    def __init__(
        self,
        name: str = "circuit",
        open_after: int = 5,
        reset_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self._open_after = open_after
        self._reset_timeout = reset_timeout
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self._reset_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call *fn*, tracking successes and failures."""
        if self.state == CircuitState.OPEN:
            raise CircuitOpen(self.name, self._failures)

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise exc

    def _on_success(self) -> None:
        self._failures = 0
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit '%s' recovered → CLOSED", self.name)
        self._state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        self._failures += 1
        if self._state == CircuitState.HALF_OPEN or self._failures >= self._open_after:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "Circuit '%s' OPEN after %d failures", self.name, self._failures
            )

    def reset(self) -> None:
        """Manually reset to CLOSED (useful in tests)."""
        self._failures = 0
        self._state = CircuitState.CLOSED

    @property
    def failure_count(self) -> int:
        return self._failures


# ---------------------------------------------------------------------------
# PipelineGuard — retry + circuit breaker composed
# ---------------------------------------------------------------------------


class PipelineGuard:
    """
    Composes RetryPolicy and CircuitBreaker for a named pipeline stage.

    Parameters
    ----------
    name:
        Stage name used in logs and circuit breaker.
    max_attempts:
        Retry attempts (passed to RetryPolicy).
    base_delay:
        Initial retry delay in seconds.
    open_after:
        Circuit opens after this many consecutive failures.
    reset_timeout:
        Seconds before circuit moves to HALF_OPEN.
    fallback:
        Optional callable called with the exception when all retries are
        exhausted and the circuit opens.  Should return a safe default.
    """

    def __init__(
        self,
        name: str,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        open_after: int = 5,
        reset_timeout: float = 30.0,
        fallback: Callable[[Exception], Any] | None = None,
    ) -> None:
        self.name = name
        self._policy = RetryPolicy(max_attempts=max_attempts, base_delay=base_delay)
        self._circuit = CircuitBreaker(
            name=name, open_after=open_after, reset_timeout=reset_timeout
        )
        self._fallback = fallback

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Call *fn* through the retry policy and circuit breaker.

        If the circuit is open OR all retries are exhausted, calls the
        fallback (if configured) and returns its result, otherwise re-raises.
        """
        try:
            return self._circuit.call(self._policy.call, fn, *args, **kwargs)
        except (CircuitOpen, RetryExhausted) as exc:
            logger.error("PipelineGuard '%s' gave up: %s", self.name, exc)
            if self._fallback:
                return self._fallback(exc)  # type: ignore[return-value]
            raise

    @property
    def circuit_state(self) -> CircuitState:
        return self._circuit.state

    @property
    def failure_count(self) -> int:
        return self._circuit.failure_count

    def reset(self) -> None:
        self._circuit.reset()
