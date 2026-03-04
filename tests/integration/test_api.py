"""Integration tests for the FastAPI REST API."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import prometheus_client
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from ml_serving.api.main import AppState, app, set_app_state
from ml_serving.config.settings import Settings
from ml_serving.monitoring.alerting import AlertManager
from ml_serving.monitoring.drift_detector import DriftDetector
from ml_serving.monitoring.metrics import MetricsCollector
from ml_serving.registry.model_registry import ModelRegistry
from ml_serving.registry.model_store import ModelStore
from ml_serving.registry.schemas import Framework
from ml_serving.routing.ab_testing import ABTestManager
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
def iris_clf() -> RandomForestClassifier:
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture()
def client(settings: Settings, iris_clf) -> TestClient:
    registry = ModelRegistry(settings)
    store = ModelStore(settings)
    server = ModelServer(registry=registry, store=store, settings=settings)
    prom_registry = prometheus_client.CollectorRegistry()
    metrics = MetricsCollector(registry=prom_registry)

    state = AppState(
        settings=settings,
        registry=registry,
        store=store,
        model_server=server,
        metrics=metrics,
        drift_detector=DriftDetector(),
        alert_manager=AlertManager(),
        ab_manager=ABTestManager(),
    )
    set_app_state(state)

    # Register and load a model for prediction tests
    registry.register("iris", "v1", Framework.SKLEARN, description="Iris classifier")
    server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)

    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc

    set_app_state(None)


# ===================================================================
# Health
# ===================================================================


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "iris"


# ===================================================================
# Models
# ===================================================================


class TestModelEndpoints:
    def test_register_model(self, client: TestClient):
        resp = client.post(
            "/api/v1/models",
            json={"name": "new_model", "version": "v1", "framework": "sklearn"},
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "new_model"

    def test_list_models(self, client: TestClient):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        assert len(resp.json()["models"]) >= 1

    def test_get_model(self, client: TestClient):
        resp = client.get("/api/v1/models/iris?version=v1")
        assert resp.status_code == 200
        assert resp.json()["name"] == "iris"

    def test_get_model_not_found(self, client: TestClient):
        resp = client.get("/api/v1/models/nonexistent")
        assert resp.status_code == 404

    def test_promote_model(self, client: TestClient):
        resp = client.put(
            "/api/v1/models/iris/promote",
            json={"version": "v1", "stage": "production"},
        )
        assert resp.status_code == 200
        assert resp.json()["stage"] == "production"

    def test_archive_model(self, client: TestClient):
        # Register a model to archive
        client.post(
            "/api/v1/models",
            json={"name": "to_archive", "version": "v1", "framework": "sklearn"},
        )
        resp = client.delete("/api/v1/models/to_archive/v1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"


# ===================================================================
# Predict
# ===================================================================


class TestPredictEndpoints:
    def test_single_prediction(self, client: TestClient):
        resp = client.post(
            "/api/v1/predict",
            json={
                "model_name": "iris",
                "version": "v1",
                "input_data": [5.1, 3.5, 1.4, 0.2],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in [0, 1, 2]
        assert data["model_used"] == "iris"

    def test_prediction_not_loaded(self, client: TestClient):
        resp = client.post(
            "/api/v1/predict",
            json={"model_name": "nope", "input_data": [1.0]},
        )
        assert resp.status_code == 404

    def test_batch_prediction(self, client: TestClient):
        resp = client.post(
            "/api/v1/predict/batch",
            json={
                "model_name": "iris",
                "version": "v1",
                "inputs": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5]],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_size"] == 2
        assert len(data["predictions"]) == 2


# ===================================================================
# Experiments
# ===================================================================


class TestExperimentEndpoints:
    def test_create_ab_test(self, client: TestClient):
        resp = client.post(
            "/api/v1/experiments/ab",
            json={
                "name": "test1",
                "model_a": "iris:v1",
                "model_b": "iris:v2",
                "traffic_split": 0.5,
            },
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "test1"

    def test_get_ab_results(self, client: TestClient):
        client.post(
            "/api/v1/experiments/ab",
            json={"name": "t1", "model_a": "a:v1", "model_b": "b:v1"},
        )
        resp = client.get("/api/v1/experiments/ab/t1")
        assert resp.status_code == 200
        assert resp.json()["test_name"] == "t1"

    def test_ab_not_found(self, client: TestClient):
        resp = client.get("/api/v1/experiments/ab/nope")
        assert resp.status_code == 404

    def test_canary_lifecycle(self, client: TestClient):
        # Create canary
        resp = client.post(
            "/api/v1/experiments/canary",
            json={"current_model": "iris:v1", "canary_model": "iris:v2"},
        )
        assert resp.status_code == 201
        assert resp.json()["active"]

        # Promote
        resp = client.post("/api/v1/experiments/canary/promote")
        assert resp.status_code == 200
        assert resp.json()["canary_percentage"] == 25

        # Rollback
        resp = client.post("/api/v1/experiments/canary/rollback")
        assert resp.status_code == 200
        assert not resp.json()["active"]


# ===================================================================
# Monitoring
# ===================================================================


class TestMonitoringEndpoints:
    def test_get_metrics(self, client: TestClient):
        resp = client.get("/api/v1/monitoring/metrics")
        assert resp.status_code == 200
        assert "total_predictions" in resp.json()

    def test_prometheus_endpoint(self, client: TestClient):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_drift_not_found(self, client: TestClient):
        resp = client.get("/api/v1/monitoring/drift/unknown_model")
        assert resp.status_code == 404

    def test_alerts_empty(self, client: TestClient):
        resp = client.get("/api/v1/monitoring/alerts")
        assert resp.status_code == 200
        assert resp.json() == []
