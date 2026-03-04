"""Prometheus metrics collection for model serving."""

from __future__ import annotations

import threading
import time
from typing import Any

import structlog

logger = structlog.get_logger()

# Lazy imports to allow mocking in tests
_prom = None


def _get_prom():
    global _prom
    if _prom is None:
        import prometheus_client

        _prom = prometheus_client
    return _prom


class MetricsCollector:
    """Collects and exposes Prometheus metrics for the serving platform."""

    def __init__(self, registry: Any | None = None) -> None:
        prom = _get_prom()

        # Use a custom registry if provided (for testing), else default
        self._registry = registry or prom.REGISTRY

        kwargs: dict[str, Any] = {}
        if registry is not None:
            kwargs["registry"] = registry

        self.prediction_count = prom.Counter(
            "ml_prediction_total",
            "Total number of predictions",
            ["model", "version", "status"],
            **kwargs,
        )
        self.prediction_latency = prom.Histogram(
            "ml_prediction_latency_seconds",
            "Prediction latency in seconds",
            ["model", "version"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            **kwargs,
        )
        self.model_load_time = prom.Histogram(
            "ml_model_load_seconds",
            "Time to load a model in seconds",
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            **kwargs,
        )
        self.active_models = prom.Gauge(
            "ml_active_models",
            "Number of currently loaded models",
            **kwargs,
        )
        self.batch_size = prom.Histogram(
            "ml_batch_size",
            "Batch sizes for predictions",
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
            **kwargs,
        )
        self.request_queue_size = prom.Gauge(
            "ml_request_queue_size",
            "Number of requests waiting in queue",
            **kwargs,
        )
        self.drift_score = prom.Gauge(
            "ml_drift_score",
            "Data drift score per model and feature",
            ["model", "feature"],
            **kwargs,
        )

        self._lock = threading.Lock()
        self._prediction_log: list[dict] = []

    def record_prediction(
        self,
        model: str,
        version: str,
        latency_seconds: float,
        status: str = "success",
    ) -> None:
        """Record a single prediction's metrics."""
        self.prediction_count.labels(model=model, version=version, status=status).inc()
        self.prediction_latency.labels(model=model, version=version).observe(latency_seconds)

        with self._lock:
            self._prediction_log.append(
                {
                    "model": model,
                    "version": version,
                    "latency_seconds": latency_seconds,
                    "status": status,
                    "timestamp": time.time(),
                }
            )

    def record_batch(self, size: int) -> None:
        """Record a batch prediction size."""
        self.batch_size.observe(size)

    def record_model_load(self, duration_seconds: float) -> None:
        """Record model load time."""
        self.model_load_time.observe(duration_seconds)

    def set_active_models(self, count: int) -> None:
        """Update the active models gauge."""
        self.active_models.set(count)

    def set_queue_size(self, size: int) -> None:
        """Update the request queue size gauge."""
        self.request_queue_size.set(size)

    def set_drift_score(self, model: str, feature: str, score: float) -> None:
        """Update a drift score for a model feature."""
        self.drift_score.labels(model=model, feature=feature).set(score)

    def get_metrics_summary(self) -> dict:
        """Return a summary of recent metrics."""
        with self._lock:
            log = list(self._prediction_log)

        if not log:
            return {
                "total_predictions": 0,
                "error_rate": 0.0,
                "avg_latency_seconds": 0.0,
                "models": {},
            }

        total = len(log)
        errors = sum(1 for e in log if e["status"] != "success")
        avg_latency = sum(e["latency_seconds"] for e in log) / total

        models: dict[str, dict] = {}
        for entry in log:
            key = f"{entry['model']}:{entry['version']}"
            if key not in models:
                models[key] = {"count": 0, "errors": 0, "latencies": []}
            models[key]["count"] += 1
            if entry["status"] != "success":
                models[key]["errors"] += 1
            models[key]["latencies"].append(entry["latency_seconds"])

        model_summaries = {}
        for key, data in models.items():
            lats = sorted(data["latencies"])
            model_summaries[key] = {
                "count": data["count"],
                "error_rate": data["errors"] / data["count"] if data["count"] else 0,
                "avg_latency": sum(lats) / len(lats),
                "p50_latency": lats[len(lats) // 2] if lats else 0,
                "p99_latency": lats[int(len(lats) * 0.99)] if lats else 0,
            }

        return {
            "total_predictions": total,
            "error_rate": errors / total,
            "avg_latency_seconds": avg_latency,
            "models": model_summaries,
        }

    def generate_latest(self) -> bytes:
        """Generate Prometheus exposition format output."""
        prom = _get_prom()
        return prom.generate_latest(self._registry)
