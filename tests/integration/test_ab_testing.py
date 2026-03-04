"""Integration tests for full A/B testing lifecycle."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from ml_serving.config.settings import Settings
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
def iris_data():
    return load_iris(return_X_y=True)


@pytest.fixture()
def model_a(iris_data) -> RandomForestClassifier:
    X, y = iris_data
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture()
def model_b(iris_data) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    X, y = iris_data
    clf.fit(X, y)
    return clf


@pytest.fixture()
def server(settings: Settings) -> ModelServer:
    registry = ModelRegistry(settings)
    store = ModelStore(settings)
    return ModelServer(registry=registry, store=store, settings=settings)


class TestABTestingLifecycle:
    def test_full_lifecycle(self, server: ModelServer, model_a, model_b, iris_data):
        """Full A/B test: create, route, record, conclude."""
        X, y = iris_data

        # Load both models
        server.load_model("clf_a", "v1", Framework.SKLEARN, model_object=model_a)
        server.load_model("clf_b", "v1", Framework.SKLEARN, model_object=model_b)

        # Create A/B test
        mgr = ABTestManager()
        test = mgr.create_test("iris_ab", "clf_a:v1", "clf_b:v1", traffic_split=0.5)

        # Run predictions and record results
        for i in range(100):
            request_id = f"req-{i}"
            routed = test.route_request(request_id)
            model_name = routed.split(":")[0]
            version = routed.split(":")[1]

            result = server.predict(model_name, X[i % len(X)].tolist(), version=version)
            ground_truth = int(y[i % len(y)])
            mgr.record_result("iris_ab", routed, int(result.prediction), ground_truth)

        # Get results
        results = mgr.get_results("iris_ab")
        assert results.sample_size_a + results.sample_size_b == 100
        assert results.sample_size_a > 0
        assert results.sample_size_b > 0

        # Conclude
        outcome = mgr.conclude_test("iris_ab")
        assert outcome["winner"] in ("clf_a:v1", "clf_b:v1")

    def test_sticky_sessions(self, server: ModelServer, model_a, model_b):
        """Same request ID always routes to the same model."""
        server.load_model("clf_a", "v1", Framework.SKLEARN, model_object=model_a)
        server.load_model("clf_b", "v1", Framework.SKLEARN, model_object=model_b)

        mgr = ABTestManager()
        test = mgr.create_test("sticky", "clf_a:v1", "clf_b:v1", traffic_split=0.5)

        routes = set()
        for _ in range(50):
            routes.add(test.route_request("same-request-id"))
        assert len(routes) == 1

    def test_traffic_split_distribution(self):
        """Verify approximate traffic split over many requests."""
        mgr = ABTestManager()
        test = mgr.create_test("split", "a:v1", "b:v1", traffic_split=0.3)

        b_count = sum(1 for i in range(1000) if test.route_request(f"r-{i}") == "b:v1")
        # 30% split should give roughly 300 +/- 50
        assert 200 < b_count < 400

    def test_multiple_concurrent_tests(self, server: ModelServer, model_a, model_b):
        """Run multiple A/B tests simultaneously."""
        server.load_model("clf_a", "v1", Framework.SKLEARN, model_object=model_a)
        server.load_model("clf_b", "v1", Framework.SKLEARN, model_object=model_b)

        mgr = ABTestManager()
        mgr.create_test("test1", "clf_a:v1", "clf_b:v1", traffic_split=0.5)
        mgr.create_test("test2", "clf_a:v1", "clf_b:v1", traffic_split=0.2)

        assert len(mgr.list_tests()) == 2

        # Both tests independently route
        r1 = mgr.get_test("test1").route_request("x")
        r2 = mgr.get_test("test2").route_request("x")
        assert r1 in ("clf_a:v1", "clf_b:v1")
        assert r2 in ("clf_a:v1", "clf_b:v1")

    def test_ab_with_accuracy_comparison(self, server: ModelServer, model_a, model_b, iris_data):
        """Record results and verify accuracy is calculated correctly."""
        X, y = iris_data
        server.load_model("clf_a", "v1", Framework.SKLEARN, model_object=model_a)

        mgr = ABTestManager()
        mgr.create_test("acc_test", "clf_a:v1", "clf_b:v1")

        # All to model A, all correct
        for i in range(20):
            result = server.predict("clf_a", X[i].tolist(), version="v1")
            mgr.record_result("acc_test", "clf_a:v1", int(result.prediction), int(y[i]))

        results = mgr.get_results("acc_test")
        assert results.sample_size_a == 20
        assert results.model_a_metrics["accuracy"] > 0.8  # RF on iris should be high

    def test_statistical_significance_calculation(self):
        """With enough samples and clear difference, p-value should be small."""
        mgr = ABTestManager()
        mgr.create_test("sig", "a:v1", "b:v1")

        # A: 100% correct
        for _ in range(100):
            mgr.record_result("sig", "a:v1", 1, 1)
        # B: 50% correct
        for i in range(100):
            mgr.record_result("sig", "b:v1", i % 2, 1)

        results = mgr.get_results("sig")
        assert results.is_significant
        assert results.p_value < 0.01

    def test_conclude_removes_test(self):
        mgr = ABTestManager()
        mgr.create_test("temp", "a:v1", "b:v1")
        mgr.record_result("temp", "a:v1", 1, 1)
        mgr.conclude_test("temp")

        with pytest.raises(KeyError):
            mgr.get_test("temp")

    def test_empty_test_conclude(self):
        """Concluding a test with no results should still work."""
        mgr = ABTestManager()
        mgr.create_test("empty", "a:v1", "b:v1")
        result = mgr.conclude_test("empty")
        assert result["p_value"] == 1.0

    def test_unequal_sample_sizes(self):
        """Test works with very unequal sample sizes."""
        mgr = ABTestManager()
        mgr.create_test("unequal", "a:v1", "b:v1")

        for _ in range(100):
            mgr.record_result("unequal", "a:v1", 1, 1)
        for _ in range(5):
            mgr.record_result("unequal", "b:v1", 1, 1)

        results = mgr.get_results("unequal")
        assert results.sample_size_a == 100
        assert results.sample_size_b == 5

    def test_delete_cleans_up(self):
        mgr = ABTestManager()
        mgr.create_test("del", "a:v1", "b:v1")
        mgr.record_result("del", "a:v1", 1, 1)
        mgr.delete_test("del")

        with pytest.raises(KeyError):
            mgr.get_results("del")
