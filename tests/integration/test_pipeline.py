"""Integration tests for the full predict → route → monitor pipeline."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import prometheus_client
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from ml_serving.config.settings import Settings
from ml_serving.monitoring.drift_detector import DriftDetector
from ml_serving.monitoring.metrics import MetricsCollector
from ml_serving.registry.model_registry import ModelRegistry
from ml_serving.registry.model_store import ModelStore
from ml_serving.registry.schemas import Framework
from ml_serving.routing.ab_testing import ABTestManager
from ml_serving.routing.canary import CanaryDeployment
from ml_serving.serving.model_server import ModelServer


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture()
def settings(tmp_dir: Path) -> Settings:
    return Settings(
        MODEL_STORE_PATH=tmp_dir / "artifacts",
        REGISTRY_PATH=tmp_dir / "registry",
        MAX_LOADED_MODELS=5,
    )


@pytest.fixture()
def iris_data():
    return load_iris(return_X_y=True)


@pytest.fixture()
def iris_clf(iris_data) -> RandomForestClassifier:
    X, y = iris_data
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture()
def server(settings: Settings) -> ModelServer:
    registry = ModelRegistry(settings)
    store = ModelStore(settings)
    return ModelServer(registry=registry, store=store, settings=settings)


@pytest.fixture()
def metrics() -> MetricsCollector:
    return MetricsCollector(registry=prometheus_client.CollectorRegistry())


class TestPredictRouteMonitorPipeline:
    def test_predict_and_record_metrics(self, server: ModelServer, iris_clf, metrics):
        """Prediction should flow through to metrics recording."""
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)

        start = time.monotonic()
        result = server.predict("iris", [5.1, 3.5, 1.4, 0.2], version="v1")
        latency = time.monotonic() - start

        metrics.record_prediction("iris", "v1", latency, "success")

        summary = metrics.get_metrics_summary()
        assert summary["total_predictions"] == 1
        assert summary["models"]["iris:v1"]["count"] == 1

    def test_ab_route_then_predict(self, server: ModelServer, iris_clf, iris_data, metrics):
        """A/B test routes request, prediction is made, metrics recorded."""
        X, y = iris_data
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)

        # Create A/B test (both point to same model for simplicity)
        mgr = ABTestManager()
        test = mgr.create_test("ab", "iris:v1", "iris:v1", traffic_split=0.5)

        for i in range(20):
            routed = test.route_request(f"req-{i}")
            name, version = routed.split(":")
            result = server.predict(name, X[i].tolist(), version=version)
            metrics.record_prediction(name, version, result.latency_ms / 1000)
            mgr.record_result("ab", routed, int(result.prediction), int(y[i]))

        summary = metrics.get_metrics_summary()
        assert summary["total_predictions"] == 20

        results = mgr.get_results("ab")
        assert results.sample_size_a + results.sample_size_b == 20

    def test_canary_route_and_metrics(self, server: ModelServer, iris_clf, metrics):
        """Canary routes traffic and records per-model metrics."""
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)

        canary = CanaryDeployment("iris:v1", "iris:v1", initial_percentage=50)

        for i in range(20):
            routed = canary.route_request()
            name, version = routed.split(":")
            result = server.predict(name, [5.1, 3.5, 1.4, 0.2], version=version)
            latency_ms = result.latency_ms
            canary.record_request(routed, latency_ms)
            metrics.record_prediction(name, version, latency_ms / 1000)

        canary_metrics = canary.get_metrics()
        assert canary_metrics["current"]["request_count"] + canary_metrics["canary"]["request_count"] == 20

    def test_drift_detection_pipeline(self, server: ModelServer, iris_clf, iris_data):
        """Collect predictions and detect drift."""
        X, y = iris_data
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)

        detector = DriftDetector(psi_threshold=0.2)
        detector.set_reference("iris", X[:100])

        # Feed samples from a shifted distribution to trigger drift
        rng = np.random.RandomState(42)
        shifted_data = rng.randn(100, 4) + 10  # Very shifted

        for sample in shifted_data:
            detector.add_sample("iris", sample)

        report = detector.check_model_drift("iris")
        assert report is not None
        assert report.is_drifted

    def test_end_to_end_predict_route_monitor(
        self, server: ModelServer, iris_clf, iris_data, metrics
    ):
        """Full pipeline: predict, route via A/B test, record metrics, check drift."""
        X, y = iris_data
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)

        mgr = ABTestManager()
        mgr.create_test("e2e", "iris:v1", "iris:v1", traffic_split=0.5)

        detector = DriftDetector()
        detector.set_reference("iris", X)

        # Sample uniformly across all classes to match reference distribution
        indices = list(range(len(X)))
        rng = np.random.RandomState(42)
        rng.shuffle(indices)

        for idx in indices[:50]:
            # Route
            test = mgr.get_test("e2e")
            routed = test.route_request(f"req-{idx}")
            name, version = routed.split(":")

            # Predict
            result = server.predict(name, X[idx].tolist(), version=version)

            # Record metrics
            metrics.record_prediction(name, version, result.latency_ms / 1000)
            mgr.record_result("e2e", routed, int(result.prediction), int(y[idx]))

            # Feed drift detector
            detector.add_sample("iris", X[idx])

        # Verify all systems recorded data
        assert metrics.get_metrics_summary()["total_predictions"] == 50
        results = mgr.get_results("e2e")
        assert results.sample_size_a + results.sample_size_b == 50

        report = detector.check_model_drift("iris")
        assert report is not None
        assert not report.is_drifted  # Same distribution should not drift
