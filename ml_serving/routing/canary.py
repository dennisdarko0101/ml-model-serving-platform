"""Canary deployment — gradual traffic shift with automatic rollback."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

PROMOTION_STEPS = [5, 25, 50, 100]


@dataclass
class CanaryMetrics:
    """Per-model metrics tracked during canary deployment."""

    error_count: int = 0
    request_count: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.error_count / self.request_count if self.request_count > 0 else 0.0

    @property
    def latency_p50(self) -> float:
        return self._percentile(50)

    @property
    def latency_p99(self) -> float:
        return self._percentile(99)

    def _percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p / 100)
        idx = min(idx, len(sorted_lat) - 1)
        return sorted_lat[idx]


class CanaryDeployment:
    """Manages a canary deployment between current and canary models."""

    def __init__(
        self,
        current_model: str,
        canary_model: str,
        initial_percentage: int = 5,
        error_threshold: float = 0.1,
    ) -> None:
        self.current_model = current_model
        self.canary_model = canary_model
        self._canary_percentage = initial_percentage
        self._error_threshold = error_threshold
        self._current_metrics = CanaryMetrics()
        self._canary_metrics = CanaryMetrics()
        self._lock = threading.Lock()
        self._active = True
        self._created_at = time.time()

    @property
    def canary_percentage(self) -> int:
        return self._canary_percentage

    @property
    def is_active(self) -> bool:
        return self._active

    def route_request(self) -> str:
        """Route request to current or canary based on percentage."""
        if not self._active:
            return self.current_model
        if random.randint(1, 100) <= self._canary_percentage:
            return self.canary_model
        return self.current_model

    def record_request(
        self, model_used: str, latency_ms: float, is_error: bool = False
    ) -> None:
        """Record a request outcome for metrics tracking."""
        with self._lock:
            if model_used == self.canary_model:
                metrics = self._canary_metrics
            else:
                metrics = self._current_metrics
            metrics.request_count += 1
            metrics.latencies.append(latency_ms)
            if is_error:
                metrics.error_count += 1

    def promote(self) -> int:
        """Advance canary to next promotion step.

        Returns the new canary percentage.
        """
        with self._lock:
            current_idx = -1
            for i, step in enumerate(PROMOTION_STEPS):
                if step == self._canary_percentage:
                    current_idx = i
                    break

            if current_idx == -1:
                # Find next step above current percentage
                for step in PROMOTION_STEPS:
                    if step > self._canary_percentage:
                        self._canary_percentage = step
                        break
            elif current_idx < len(PROMOTION_STEPS) - 1:
                self._canary_percentage = PROMOTION_STEPS[current_idx + 1]

            if self._canary_percentage >= 100:
                self._active = False

            logger.info(
                "canary_promoted",
                canary=self.canary_model,
                percentage=self._canary_percentage,
            )
            return self._canary_percentage

    def rollback(self) -> None:
        """Revert to the current model, deactivating canary."""
        with self._lock:
            self._canary_percentage = 0
            self._active = False
        logger.info("canary_rolled_back", canary=self.canary_model)

    def auto_promote(self) -> bool:
        """Promote if canary error rate is below threshold, rollback if above.

        Returns True if promoted, False if rolled back.
        """
        with self._lock:
            canary_error_rate = self._canary_metrics.error_rate

        if canary_error_rate > self._error_threshold:
            self.rollback()
            logger.warning(
                "canary_auto_rollback",
                error_rate=canary_error_rate,
                threshold=self._error_threshold,
            )
            return False

        self.promote()
        return True

    def get_metrics(self) -> dict:
        """Return current metrics for both models."""
        with self._lock:
            return {
                "current_model": self.current_model,
                "canary_model": self.canary_model,
                "canary_percentage": self._canary_percentage,
                "active": self._active,
                "current": {
                    "error_rate": self._current_metrics.error_rate,
                    "latency_p50": self._current_metrics.latency_p50,
                    "latency_p99": self._current_metrics.latency_p99,
                    "request_count": self._current_metrics.request_count,
                },
                "canary": {
                    "error_rate": self._canary_metrics.error_rate,
                    "latency_p50": self._canary_metrics.latency_p50,
                    "latency_p99": self._canary_metrics.latency_p99,
                    "request_count": self._canary_metrics.request_count,
                },
            }
