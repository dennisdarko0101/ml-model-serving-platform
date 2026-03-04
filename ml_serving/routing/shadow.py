"""Shadow mode — run a shadow model alongside the primary without affecting responses."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


@dataclass
class ShadowComparison:
    """A single comparison between primary and shadow predictions."""

    primary_prediction: Any
    shadow_prediction: Any
    agreed: bool
    primary_latency_ms: float
    shadow_latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ShadowReport:
    """Aggregated report comparing primary and shadow models."""

    primary_model: str
    shadow_model: str
    total_comparisons: int
    agreement_count: int
    agreement_rate: float
    divergences: list[ShadowComparison]
    avg_primary_latency_ms: float
    avg_shadow_latency_ms: float


class ShadowMode:
    """Run a shadow model alongside the primary model.

    The primary model serves the actual response. The shadow model runs
    asynchronously (fire-and-forget) and predictions are logged for comparison.
    """

    def __init__(
        self,
        primary_model: str,
        shadow_model: str,
        predict_fn: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self.primary_model = primary_model
        self.shadow_model = shadow_model
        self._predict_fn = predict_fn
        self._comparisons: list[ShadowComparison] = []
        self._lock = threading.Lock()

    def set_predict_fn(self, fn: Callable[[str, Any], Any]) -> None:
        """Set the prediction function used for both models."""
        self._predict_fn = fn

    def predict(self, input_data: Any) -> Any:
        """Run primary prediction and fire-and-forget shadow prediction.

        Returns the primary model's prediction result only.
        """
        if self._predict_fn is None:
            raise RuntimeError("predict_fn not set — call set_predict_fn() first")

        # Primary prediction (synchronous)
        start = time.monotonic()
        primary_result = self._predict_fn(self.primary_model, input_data)
        primary_latency = (time.monotonic() - start) * 1000

        # Shadow prediction (fire-and-forget in background thread)
        threading.Thread(
            target=self._run_shadow,
            args=(input_data, primary_result, primary_latency),
            daemon=True,
        ).start()

        return primary_result

    async def predict_async(self, input_data: Any) -> Any:
        """Async version — runs shadow prediction as a background task."""
        if self._predict_fn is None:
            raise RuntimeError("predict_fn not set — call set_predict_fn() first")

        loop = asyncio.get_running_loop()

        start = time.monotonic()
        primary_result = await loop.run_in_executor(
            None, self._predict_fn, self.primary_model, input_data
        )
        primary_latency = (time.monotonic() - start) * 1000

        # Fire and forget shadow
        asyncio.ensure_future(
            self._run_shadow_async(input_data, primary_result, primary_latency)
        )

        return primary_result

    def compare_predictions(self) -> ShadowReport:
        """Generate a comparison report between primary and shadow predictions."""
        with self._lock:
            comparisons = list(self._comparisons)

        total = len(comparisons)
        agreements = sum(1 for c in comparisons if c.agreed)
        divergences = [c for c in comparisons if not c.agreed]

        avg_primary = (
            sum(c.primary_latency_ms for c in comparisons) / total if total > 0 else 0.0
        )
        avg_shadow = (
            sum(c.shadow_latency_ms for c in comparisons) / total if total > 0 else 0.0
        )

        return ShadowReport(
            primary_model=self.primary_model,
            shadow_model=self.shadow_model,
            total_comparisons=total,
            agreement_count=agreements,
            agreement_rate=agreements / total if total > 0 else 0.0,
            divergences=divergences,
            avg_primary_latency_ms=avg_primary,
            avg_shadow_latency_ms=avg_shadow,
        )

    def _run_shadow(
        self, input_data: Any, primary_result: Any, primary_latency: float
    ) -> None:
        """Run shadow prediction synchronously (for background thread)."""
        try:
            start = time.monotonic()
            shadow_result = self._predict_fn(self.shadow_model, input_data)
            shadow_latency = (time.monotonic() - start) * 1000

            self._record_comparison(
                primary_result, shadow_result, primary_latency, shadow_latency
            )
        except Exception:
            logger.warning("shadow_prediction_failed", shadow=self.shadow_model, exc_info=True)

    async def _run_shadow_async(
        self, input_data: Any, primary_result: Any, primary_latency: float
    ) -> None:
        """Run shadow prediction asynchronously."""
        try:
            loop = asyncio.get_running_loop()
            start = time.monotonic()
            shadow_result = await loop.run_in_executor(
                None, self._predict_fn, self.shadow_model, input_data
            )
            shadow_latency = (time.monotonic() - start) * 1000

            self._record_comparison(
                primary_result, shadow_result, primary_latency, shadow_latency
            )
        except Exception:
            logger.warning("shadow_prediction_failed", shadow=self.shadow_model, exc_info=True)

    def _record_comparison(
        self,
        primary_result: Any,
        shadow_result: Any,
        primary_latency: float,
        shadow_latency: float,
    ) -> None:
        """Record a comparison between primary and shadow results."""
        # Compare predictions — handle PredictionResult objects
        primary_pred = getattr(primary_result, "prediction", primary_result)
        shadow_pred = getattr(shadow_result, "prediction", shadow_result)

        try:
            agreed = primary_pred == shadow_pred
        except Exception:
            agreed = False

        comparison = ShadowComparison(
            primary_prediction=primary_pred,
            shadow_prediction=shadow_pred,
            agreed=agreed,
            primary_latency_ms=primary_latency,
            shadow_latency_ms=shadow_latency,
        )

        with self._lock:
            self._comparisons.append(comparison)
