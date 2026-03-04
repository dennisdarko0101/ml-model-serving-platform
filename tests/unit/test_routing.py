"""Tests for routing — A/B testing, canary deployments, and shadow mode."""

from __future__ import annotations

import time
import threading

import pytest

from ml_serving.routing.ab_testing import ABTest, ABTestManager
from ml_serving.routing.canary import CanaryDeployment, CanaryMetrics, PROMOTION_STEPS
from ml_serving.routing.shadow import ShadowMode


# ===================================================================
# ABTest
# ===================================================================


class TestABTest:
    def test_create_ab_test(self):
        test = ABTest(name="test1", model_a="m1:v1", model_b="m2:v1", traffic_split=0.5)
        assert test.name == "test1"
        assert test.model_a == "m1:v1"
        assert test.model_b == "m2:v1"

    def test_invalid_traffic_split(self):
        with pytest.raises(ValueError, match="traffic_split"):
            ABTest(name="bad", model_a="a:v1", model_b="b:v1", traffic_split=1.5)

    def test_route_request_consistent(self):
        """Same request_id should always route to the same model."""
        test = ABTest(name="t", model_a="a:v1", model_b="b:v1", traffic_split=0.5)
        first = test.route_request("req-123")
        for _ in range(100):
            assert test.route_request("req-123") == first

    def test_route_request_splits_traffic(self):
        """With 50/50 split, both models should get traffic."""
        test = ABTest(name="t", model_a="a:v1", model_b="b:v1", traffic_split=0.5)
        results = {test.route_request(f"req-{i}") for i in range(200)}
        assert "a:v1" in results
        assert "b:v1" in results

    def test_route_all_to_a(self):
        """0% traffic to B means all go to A."""
        test = ABTest(name="t", model_a="a:v1", model_b="b:v1", traffic_split=0.0)
        for i in range(50):
            assert test.route_request(f"r-{i}") == "a:v1"

    def test_route_all_to_b(self):
        """100% traffic to B."""
        test = ABTest(name="t", model_a="a:v1", model_b="b:v1", traffic_split=1.0)
        for i in range(50):
            assert test.route_request(f"r-{i}") == "b:v1"


# ===================================================================
# ABTestManager
# ===================================================================


class TestABTestManager:
    def test_create_and_get_test(self):
        mgr = ABTestManager()
        test = mgr.create_test("exp1", "a:v1", "b:v1", 0.3)
        retrieved = mgr.get_test("exp1")
        assert retrieved.name == "exp1"
        assert retrieved.traffic_split == 0.3

    def test_get_nonexistent_raises(self):
        mgr = ABTestManager()
        with pytest.raises(KeyError, match="not found"):
            mgr.get_test("nope")

    def test_list_tests(self):
        mgr = ABTestManager()
        mgr.create_test("t1", "a:v1", "b:v1")
        mgr.create_test("t2", "c:v1", "d:v1")
        tests = mgr.list_tests()
        assert len(tests) == 2

    def test_record_and_get_results(self):
        mgr = ABTestManager()
        mgr.create_test("exp", "a:v1", "b:v1")

        # Model A: 8/10 correct
        for i in range(10):
            mgr.record_result("exp", "a:v1", i < 8, True)

        # Model B: 9/10 correct
        for i in range(10):
            mgr.record_result("exp", "b:v1", i < 9, True)

        results = mgr.get_results("exp")
        assert results.sample_size_a == 10
        assert results.sample_size_b == 10
        assert results.model_a_metrics["accuracy"] == pytest.approx(0.8)
        assert results.model_b_metrics["accuracy"] == pytest.approx(0.9)

    def test_record_nonexistent_raises(self):
        mgr = ABTestManager()
        with pytest.raises(KeyError):
            mgr.record_result("nope", "a:v1", 1, 1)

    def test_conclude_test(self):
        mgr = ABTestManager()
        mgr.create_test("exp", "a:v1", "b:v1")

        for i in range(50):
            mgr.record_result("exp", "a:v1", 1, 1)
        for i in range(50):
            mgr.record_result("exp", "b:v1", 0, 1)

        result = mgr.conclude_test("exp")
        assert result["winner"] == "a:v1"
        assert "p_value" in result

        # Test should be removed
        with pytest.raises(KeyError):
            mgr.get_test("exp")

    def test_conclude_with_significance(self):
        """Large sample with clear difference should be significant."""
        mgr = ABTestManager()
        mgr.create_test("exp", "a:v1", "b:v1")

        # A: 95% accuracy, B: 50% accuracy, 200 samples each
        for i in range(200):
            mgr.record_result("exp", "a:v1", i < 190, True)
        for i in range(200):
            mgr.record_result("exp", "b:v1", i < 100, True)

        result = mgr.conclude_test("exp")
        assert result["is_significant"]
        assert result["winner"] == "a:v1"

    def test_delete_test(self):
        mgr = ABTestManager()
        mgr.create_test("exp", "a:v1", "b:v1")
        mgr.delete_test("exp")
        with pytest.raises(KeyError):
            mgr.get_test("exp")

    def test_results_no_samples(self):
        mgr = ABTestManager()
        mgr.create_test("exp", "a:v1", "b:v1")
        results = mgr.get_results("exp")
        assert results.sample_size_a == 0
        assert results.sample_size_b == 0
        assert results.p_value == 1.0


# ===================================================================
# Canary
# ===================================================================


class TestCanaryDeployment:
    def test_create_canary(self):
        canary = CanaryDeployment("current:v1", "canary:v2")
        assert canary.canary_percentage == 5
        assert canary.is_active

    def test_route_request(self):
        """At 5%, most requests should go to current."""
        canary = CanaryDeployment("current:v1", "canary:v2", initial_percentage=50)
        results = [canary.route_request() for _ in range(200)]
        assert "current:v1" in results
        assert "canary:v2" in results

    def test_promote_steps(self):
        canary = CanaryDeployment("current:v1", "canary:v2", initial_percentage=5)
        assert canary.promote() == 25
        assert canary.promote() == 50
        assert canary.promote() == 100
        assert not canary.is_active  # Fully promoted

    def test_rollback(self):
        canary = CanaryDeployment("current:v1", "canary:v2", initial_percentage=25)
        canary.rollback()
        assert canary.canary_percentage == 0
        assert not canary.is_active

    def test_rollback_routes_to_current(self):
        canary = CanaryDeployment("current:v1", "canary:v2")
        canary.rollback()
        for _ in range(50):
            assert canary.route_request() == "current:v1"

    def test_record_and_get_metrics(self):
        canary = CanaryDeployment("current:v1", "canary:v2")
        canary.record_request("current:v1", 10.0)
        canary.record_request("canary:v2", 15.0, is_error=True)

        metrics = canary.get_metrics()
        assert metrics["current"]["request_count"] == 1
        assert metrics["canary"]["request_count"] == 1
        assert metrics["canary"]["error_rate"] == 1.0

    def test_auto_promote_success(self):
        canary = CanaryDeployment("current:v1", "canary:v2", error_threshold=0.1)
        # Record successful requests for canary
        for _ in range(10):
            canary.record_request("canary:v2", 5.0, is_error=False)
        assert canary.auto_promote()  # Should promote
        assert canary.canary_percentage == 25

    def test_auto_promote_rollback(self):
        canary = CanaryDeployment("current:v1", "canary:v2", error_threshold=0.1)
        # Record erroring requests for canary
        for _ in range(10):
            canary.record_request("canary:v2", 5.0, is_error=True)
        assert not canary.auto_promote()  # Should rollback
        assert not canary.is_active


class TestCanaryMetrics:
    def test_empty_metrics(self):
        m = CanaryMetrics()
        assert m.error_rate == 0.0
        assert m.latency_p50 == 0.0
        assert m.latency_p99 == 0.0

    def test_metrics_calculation(self):
        m = CanaryMetrics(error_count=2, request_count=10, latencies=[1, 2, 3, 4, 5])
        assert m.error_rate == pytest.approx(0.2)
        assert m.latency_p50 == 3.0


# ===================================================================
# Shadow Mode
# ===================================================================


class TestShadowMode:
    def test_predict_returns_primary(self):
        def predict_fn(model, data):
            return f"{model}_result"

        shadow = ShadowMode("primary:v1", "shadow:v1", predict_fn=predict_fn)
        result = shadow.predict("input_data")
        assert result == "primary:v1_result"

    def test_shadow_records_comparison(self):
        def predict_fn(model, data):
            return 1 if model == "primary:v1" else 1

        shadow = ShadowMode("primary:v1", "shadow:v1", predict_fn=predict_fn)
        shadow.predict("data")

        # Give the background thread time to run
        time.sleep(0.2)

        report = shadow.compare_predictions()
        assert report.total_comparisons == 1
        assert report.agreement_rate == 1.0

    def test_shadow_detects_divergence(self):
        def predict_fn(model, data):
            return 1 if model == "primary:v1" else 2

        shadow = ShadowMode("primary:v1", "shadow:v1", predict_fn=predict_fn)
        shadow.predict("data")

        time.sleep(0.2)

        report = shadow.compare_predictions()
        assert report.total_comparisons == 1
        assert report.agreement_rate == 0.0
        assert len(report.divergences) == 1

    def test_no_predict_fn_raises(self):
        shadow = ShadowMode("primary:v1", "shadow:v1")
        with pytest.raises(RuntimeError, match="predict_fn not set"):
            shadow.predict("data")

    def test_set_predict_fn(self):
        shadow = ShadowMode("primary:v1", "shadow:v1")
        shadow.set_predict_fn(lambda model, data: "ok")
        assert shadow.predict("data") == "ok"

    def test_empty_report(self):
        shadow = ShadowMode("primary:v1", "shadow:v1")
        report = shadow.compare_predictions()
        assert report.total_comparisons == 0
        assert report.agreement_rate == 0.0
