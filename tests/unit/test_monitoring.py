"""Tests for monitoring — metrics, drift detection, and alerting."""

from __future__ import annotations

import numpy as np
import pytest

from ml_serving.monitoring.alerting import AlertManager, Condition, Severity
from ml_serving.monitoring.drift_detector import DriftDetector
from ml_serving.monitoring.metrics import MetricsCollector


# ===================================================================
# MetricsCollector (mocked Prometheus registry)
# ===================================================================


class TestMetricsCollector:
    @pytest.fixture()
    def collector(self):
        import prometheus_client

        registry = prometheus_client.CollectorRegistry()
        return MetricsCollector(registry=registry)

    def test_record_prediction(self, collector: MetricsCollector):
        collector.record_prediction("iris", "v1", 0.05, status="success")
        summary = collector.get_metrics_summary()
        assert summary["total_predictions"] == 1
        assert summary["error_rate"] == 0.0

    def test_record_multiple_predictions(self, collector: MetricsCollector):
        collector.record_prediction("iris", "v1", 0.05, status="success")
        collector.record_prediction("iris", "v1", 0.10, status="success")
        collector.record_prediction("iris", "v1", 0.20, status="error")
        summary = collector.get_metrics_summary()
        assert summary["total_predictions"] == 3
        assert summary["error_rate"] == pytest.approx(1 / 3)

    def test_per_model_metrics(self, collector: MetricsCollector):
        collector.record_prediction("iris", "v1", 0.05)
        collector.record_prediction("reg", "v1", 0.10)
        summary = collector.get_metrics_summary()
        assert "iris:v1" in summary["models"]
        assert "reg:v1" in summary["models"]

    def test_empty_summary(self, collector: MetricsCollector):
        summary = collector.get_metrics_summary()
        assert summary["total_predictions"] == 0
        assert summary["error_rate"] == 0.0

    def test_record_batch(self, collector: MetricsCollector):
        collector.record_batch(16)
        # Just ensure no error — batch size is in histogram

    def test_set_active_models(self, collector: MetricsCollector):
        collector.set_active_models(3)
        # Gauge should be set

    def test_set_drift_score(self, collector: MetricsCollector):
        collector.set_drift_score("iris", "feature_0", 0.15)

    def test_generate_latest(self, collector: MetricsCollector):
        collector.record_prediction("iris", "v1", 0.05)
        output = collector.generate_latest()
        assert b"ml_prediction_total" in output

    def test_model_load_time(self, collector: MetricsCollector):
        collector.record_model_load(2.5)

    def test_queue_size(self, collector: MetricsCollector):
        collector.set_queue_size(42)


# ===================================================================
# DriftDetector
# ===================================================================


class TestDriftDetector:
    @pytest.fixture()
    def detector(self):
        return DriftDetector(psi_threshold=0.2, ks_threshold=0.1)

    def test_no_drift_same_distribution(self, detector: DriftDetector):
        rng = np.random.RandomState(42)
        ref = rng.randn(500, 3)
        cur = rng.randn(500, 3)
        report = detector.detect_data_drift(ref, cur, method="psi")
        assert not report.is_drifted
        assert report.method == "psi"

    def test_drift_shifted_distribution(self, detector: DriftDetector):
        rng = np.random.RandomState(42)
        ref = rng.randn(500, 3)
        cur = rng.randn(500, 3) + 5  # Large shift
        report = detector.detect_data_drift(ref, cur, method="psi")
        assert report.is_drifted
        assert len(report.drifted_features) > 0

    def test_ks_no_drift(self, detector: DriftDetector):
        rng = np.random.RandomState(42)
        ref = rng.randn(500, 2)
        cur = rng.randn(500, 2)
        report = detector.detect_data_drift(ref, cur, method="ks")
        assert report.method == "ks"
        assert not report.is_drifted

    def test_ks_drift_detected(self, detector: DriftDetector):
        rng = np.random.RandomState(42)
        ref = rng.randn(500, 2)
        cur = rng.randn(500, 2) + 3
        report = detector.detect_data_drift(ref, cur, method="ks")
        assert report.is_drifted

    def test_prediction_drift(self, detector: DriftDetector):
        ref_preds = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 50)
        cur_preds = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2] * 50)
        report = detector.detect_prediction_drift(ref_preds, cur_preds)
        assert report.is_drifted

    def test_prediction_no_drift(self, detector: DriftDetector):
        rng = np.random.RandomState(42)
        preds = rng.randint(0, 3, 500)
        report = detector.detect_prediction_drift(preds, preds)
        assert not report.is_drifted

    def test_categorical_drift(self, detector: DriftDetector):
        ref = {"cat": 100, "dog": 100, "bird": 100}
        cur = {"cat": 10, "dog": 10, "bird": 280}
        report = detector.detect_categorical_drift(ref, cur)
        assert report.method == "chi_squared"
        assert report.is_drifted

    def test_categorical_no_drift(self, detector: DriftDetector):
        ref = {"cat": 100, "dog": 100}
        cur = {"cat": 105, "dog": 95}
        report = detector.detect_categorical_drift(ref, cur)
        assert not report.is_drifted

    def test_1d_data_drift(self, detector: DriftDetector):
        rng = np.random.RandomState(42)
        ref = rng.randn(500)
        cur = rng.randn(500) + 5
        report = detector.detect_data_drift(ref, cur, method="psi")
        assert report.is_drifted

    def test_unknown_method_raises(self, detector: DriftDetector):
        with pytest.raises(ValueError, match="Unknown method"):
            detector.detect_data_drift(np.array([1]), np.array([1]), method="bad")

    def test_set_reference_and_check(self, detector: DriftDetector):
        rng = np.random.RandomState(42)
        ref = rng.randn(200, 3)
        detector.set_reference("iris", ref)

        # Add samples from a shifted distribution
        for sample in rng.randn(100, 3) + 5:
            detector.add_sample("iris", sample)

        report = detector.check_model_drift("iris")
        assert report is not None
        assert report.is_drifted

    def test_check_drift_no_reference(self, detector: DriftDetector):
        assert detector.check_model_drift("unknown") is None

    def test_check_drift_too_few_samples(self, detector: DriftDetector):
        detector.set_reference("m", np.random.randn(100, 2))
        detector.add_sample("m", np.array([1, 2]))
        assert detector.check_model_drift("m") is None

    def test_empty_categorical(self, detector: DriftDetector):
        report = detector.detect_categorical_drift({}, {})
        assert not report.is_drifted


# ===================================================================
# AlertManager
# ===================================================================


class TestAlertManager:
    def test_add_and_check_rule(self):
        mgr = AlertManager()
        mgr.add_rule("high_latency", "p99_latency", "gt", 1.0)
        alerts = mgr.check_rules({"p99_latency": 2.0})
        assert len(alerts) == 1
        assert alerts[0].rule_name == "high_latency"

    def test_no_alert_below_threshold(self):
        mgr = AlertManager()
        mgr.add_rule("high_latency", "p99_latency", "gt", 1.0)
        alerts = mgr.check_rules({"p99_latency": 0.5})
        assert len(alerts) == 0

    def test_multiple_rules(self):
        mgr = AlertManager()
        mgr.add_rule("high_latency", "p99_latency", "gt", 1.0, cooldown_seconds=0)
        mgr.add_rule("high_errors", "error_rate", "gt", 0.05, cooldown_seconds=0)
        alerts = mgr.check_rules({"p99_latency": 2.0, "error_rate": 0.1})
        assert len(alerts) == 2

    def test_condition_types(self):
        mgr = AlertManager()
        mgr.add_rule("lt", "x", "lt", 5.0, cooldown_seconds=0)
        mgr.add_rule("gte", "x", "gte", 10.0, cooldown_seconds=0)
        mgr.add_rule("lte", "x", "lte", 3.0, cooldown_seconds=0)
        mgr.add_rule("eq", "x", "eq", 3.0, cooldown_seconds=0)

        alerts = mgr.check_rules({"x": 3.0})
        names = {a.rule_name for a in alerts}
        assert "lt" in names
        assert "lte" in names
        assert "eq" in names
        assert "gte" not in names

    def test_cooldown(self):
        mgr = AlertManager()
        mgr.add_rule("test", "x", "gt", 1.0, cooldown_seconds=60)

        alerts1 = mgr.check_rules({"x": 2.0})
        assert len(alerts1) == 1

        # Should be in cooldown
        alerts2 = mgr.check_rules({"x": 2.0})
        assert len(alerts2) == 0

    def test_get_alerts(self):
        mgr = AlertManager()
        mgr.add_rule("test", "x", "gt", 1.0, cooldown_seconds=0)
        mgr.check_rules({"x": 2.0})
        alerts = mgr.get_alerts()
        assert len(alerts) == 1

    def test_clear_alerts(self):
        mgr = AlertManager()
        mgr.add_rule("test", "x", "gt", 1.0, cooldown_seconds=0)
        mgr.check_rules({"x": 2.0})
        mgr.clear_alerts()
        assert len(mgr.get_alerts()) == 0

    def test_remove_rule(self):
        mgr = AlertManager()
        mgr.add_rule("test", "x", "gt", 1.0)
        mgr.remove_rule("test")
        alerts = mgr.check_rules({"x": 2.0})
        assert len(alerts) == 0

    def test_webhook_action(self):
        mgr = AlertManager()
        webhook_calls = []
        mgr.set_webhook_handler(lambda url, alert: webhook_calls.append((url, alert)))
        mgr.add_rule("test", "x", "gt", 1.0, action="webhook", webhook_url="http://example.com")
        mgr.check_rules({"x": 2.0})
        assert len(webhook_calls) == 1
        assert webhook_calls[0][0] == "http://example.com"

    def test_metric_provider(self):
        mgr = AlertManager()
        mgr.register_metric_provider("x", lambda: 5.0)
        mgr.add_rule("test", "x", "gt", 1.0, cooldown_seconds=0)
        alerts = mgr.check_rules()
        assert len(alerts) == 1

    def test_severity_levels(self):
        mgr = AlertManager()
        mgr.add_rule("info", "x", "gt", 1.0, severity="info", cooldown_seconds=0)
        mgr.add_rule("crit", "x", "gt", 1.0, severity="critical", cooldown_seconds=0)
        alerts = mgr.check_rules({"x": 2.0})
        severities = {a.severity for a in alerts}
        assert Severity.INFO in severities
        assert Severity.CRITICAL in severities
