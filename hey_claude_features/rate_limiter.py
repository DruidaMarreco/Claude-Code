"""
Hourly and daily USD spend budget guard for Hey-Claude.

Prevents runaway API costs by raising BudgetExceeded before an API call
is made if the rolling spend would exceed a configured limit.

Works best alongside CostTracker: record() every response, then check()
before every new request.

Usage in responder.py
---------------------
    from hey_claude_features.rate_limiter import RateLimiter, BudgetExceeded

    limiter = RateLimiter(hourly_usd=0.50, daily_usd=2.00)

    def respond(query: str) -> str:
        try:
            limiter.check()
        except BudgetExceeded as e:
            return str(e)          # speak the error aloud
        reply, usage = _call_opus(query)
        limiter.record(usage.usd_cost)
        return reply

Pairing with CostTracker
-------------------------
    tracker = CostTracker()
    limiter = RateLimiter(daily_usd=2.00)

    response = client.messages.create(...)
    qc = tracker.record(response.usage, model=config.responder_model)
    limiter.record(qc.usd)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_HOUR = 3600.0
_DAY = 86400.0


class BudgetExceeded(Exception):
    """Raised by RateLimiter.check() when a spend budget would be exceeded."""


@dataclass
class SpendSnapshot:
    hourly_usd: float
    daily_usd: float
    hourly_limit: float | None
    daily_limit: float | None

    @property
    def hourly_remaining(self) -> float | None:
        if self.hourly_limit is None:
            return None
        return max(0.0, self.hourly_limit - self.hourly_usd)

    @property
    def daily_remaining(self) -> float | None:
        if self.daily_limit is None:
            return None
        return max(0.0, self.daily_limit - self.daily_usd)


class RateLimiter:
    """
    Rolling-window USD spend limiter.

    Tracks spend in two independent sliding windows (1 hour, 24 hours).
    Call `check()` before every API call; it raises `BudgetExceeded` with
    a voice-friendly message if either budget is exhausted.

    Parameters
    ----------
    hourly_usd:
        Max USD to spend per rolling hour.  None = no limit.
    daily_usd:
        Max USD to spend per rolling 24-hour window.  None = no limit.
    """

    def __init__(
        self,
        hourly_usd: float | None = None,
        daily_usd: float | None = None,
    ) -> None:
        self._hourly_limit = hourly_usd
        self._daily_limit = daily_usd
        self._lock = threading.Lock()
        # Deques of (timestamp, usd) pairs — one per window
        self._hourly_log: deque[tuple[float, float]] = deque()
        self._daily_log: deque[tuple[float, float]] = deque()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> None:
        """
        Raise BudgetExceeded if either spend limit has been reached.

        Call this BEFORE making an API request.
        """
        with self._lock:
            now = time.time()
            self._evict(now)
            hourly = self._sum(self._hourly_log)
            daily = self._sum(self._daily_log)

        if self._hourly_limit is not None and hourly >= self._hourly_limit:
            msg = (
                f"Hourly budget of ${self._hourly_limit:.2f} reached "
                f"(spent ${hourly:.2f}). I'll be available again shortly."
            )
            logger.warning(msg)
            raise BudgetExceeded(msg)

        if self._daily_limit is not None and daily >= self._daily_limit:
            msg = (
                f"Daily budget of ${self._daily_limit:.2f} reached "
                f"(spent ${daily:.2f}). I'll reset at midnight."
            )
            logger.warning(msg)
            raise BudgetExceeded(msg)

    def record(self, usd: float) -> None:
        """Record a spend amount (USD) after a completed API call."""
        if usd <= 0:
            return
        now = time.time()
        with self._lock:
            self._hourly_log.append((now, usd))
            self._daily_log.append((now, usd))
        logger.debug("Rate limiter recorded $%.5f", usd)

    def snapshot(self) -> SpendSnapshot:
        """Return current rolling spend totals for monitoring."""
        with self._lock:
            now = time.time()
            self._evict(now)
            return SpendSnapshot(
                hourly_usd=self._sum(self._hourly_log),
                daily_usd=self._sum(self._daily_log),
                hourly_limit=self._hourly_limit,
                daily_limit=self._daily_limit,
            )

    def reset(self) -> None:
        """Clear all recorded spend (useful for testing or manual override)."""
        with self._lock:
            self._hourly_log.clear()
            self._daily_log.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict(self, now: float) -> None:
        """Remove entries older than their respective window."""
        hour_cutoff = now - _HOUR
        day_cutoff = now - _DAY
        while self._hourly_log and self._hourly_log[0][0] < hour_cutoff:
            self._hourly_log.popleft()
        while self._daily_log and self._daily_log[0][0] < day_cutoff:
            self._daily_log.popleft()

    @staticmethod
    def _sum(log: deque[tuple[float, float]]) -> float:
        return sum(usd for _, usd in log)
