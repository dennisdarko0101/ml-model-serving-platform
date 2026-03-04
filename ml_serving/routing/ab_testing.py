"""A/B testing for model comparison with statistical significance."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class ABTest:
    """A single A/B test comparing two models."""

    name: str
    model_a: str  # "model_name:version"
    model_b: str  # "model_name:version"
    traffic_split: float  # 0.0-1.0, fraction routed to model_b
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not 0.0 <= self.traffic_split <= 1.0:
            raise ValueError(f"traffic_split must be in [0, 1], got {self.traffic_split}")

    def route_request(self, request_id: str) -> str:
        """Route a request using consistent hashing for sticky sessions."""
        hash_val = int(hashlib.sha256(request_id.encode()).hexdigest(), 16)
        fraction = (hash_val % 10000) / 10000.0
        return self.model_b if fraction < self.traffic_split else self.model_a


@dataclass
class _ResultRecord:
    """A single recorded prediction result."""

    model_used: str
    prediction: Any
    ground_truth: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class ABTestResults:
    """Aggregated results for an A/B test."""

    test_name: str
    model_a: str
    model_b: str
    model_a_metrics: dict[str, float]
    model_b_metrics: dict[str, float]
    sample_size_a: int
    sample_size_b: int
    p_value: float
    is_significant: bool


class ABTestManager:
    """Manage multiple concurrent A/B tests."""

    def __init__(self, significance_level: float = 0.05) -> None:
        self._tests: dict[str, ABTest] = {}
        self._results: dict[str, list[_ResultRecord]] = {}
        self._lock = threading.Lock()
        self._significance_level = significance_level

    def create_test(
        self,
        name: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
    ) -> ABTest:
        """Create a new A/B test."""
        test = ABTest(
            name=name,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
        )
        with self._lock:
            self._tests[name] = test
            self._results[name] = []
        logger.info("ab_test_created", name=name, model_a=model_a, model_b=model_b)
        return test

    def get_test(self, name: str) -> ABTest:
        """Retrieve an A/B test by name."""
        with self._lock:
            if name not in self._tests:
                raise KeyError(f"A/B test '{name}' not found")
            return self._tests[name]

    def list_tests(self) -> list[ABTest]:
        """List all active A/B tests."""
        with self._lock:
            return list(self._tests.values())

    def record_result(
        self,
        test_name: str,
        model_used: str,
        prediction: Any,
        ground_truth: Any,
    ) -> None:
        """Record a prediction result for an A/B test."""
        with self._lock:
            if test_name not in self._tests:
                raise KeyError(f"A/B test '{test_name}' not found")
            self._results[test_name].append(
                _ResultRecord(
                    model_used=model_used,
                    prediction=prediction,
                    ground_truth=ground_truth,
                )
            )

    def get_results(self, test_name: str) -> ABTestResults:
        """Calculate and return results for an A/B test."""
        with self._lock:
            if test_name not in self._tests:
                raise KeyError(f"A/B test '{test_name}' not found")
            test = self._tests[test_name]
            records = list(self._results[test_name])

        a_correct = 0
        a_total = 0
        b_correct = 0
        b_total = 0

        for rec in records:
            if rec.model_used == test.model_a:
                a_total += 1
                if rec.prediction == rec.ground_truth:
                    a_correct += 1
            elif rec.model_used == test.model_b:
                b_total += 1
                if rec.prediction == rec.ground_truth:
                    b_correct += 1

        a_accuracy = a_correct / a_total if a_total > 0 else 0.0
        b_accuracy = b_correct / b_total if b_total > 0 else 0.0

        p_value = self._chi_squared_test(a_correct, a_total, b_correct, b_total)

        return ABTestResults(
            test_name=test_name,
            model_a=test.model_a,
            model_b=test.model_b,
            model_a_metrics={"accuracy": a_accuracy, "correct": a_correct, "total": a_total},
            model_b_metrics={"accuracy": b_accuracy, "correct": b_correct, "total": b_total},
            sample_size_a=a_total,
            sample_size_b=b_total,
            p_value=p_value,
            is_significant=p_value < self._significance_level,
        )

    def conclude_test(self, test_name: str) -> dict[str, Any]:
        """Conclude an A/B test and declare a winner."""
        results = self.get_results(test_name)
        test = self.get_test(test_name)

        if results.model_a_metrics["accuracy"] >= results.model_b_metrics["accuracy"]:
            winner = test.model_a
        else:
            winner = test.model_b

        with self._lock:
            del self._tests[test_name]
            del self._results[test_name]

        logger.info(
            "ab_test_concluded",
            name=test_name,
            winner=winner,
            significant=results.is_significant,
        )
        return {
            "test_name": test_name,
            "winner": winner,
            "is_significant": results.is_significant,
            "p_value": results.p_value,
            "model_a_accuracy": results.model_a_metrics["accuracy"],
            "model_b_accuracy": results.model_b_metrics["accuracy"],
        }

    def delete_test(self, name: str) -> None:
        """Remove an A/B test without concluding."""
        with self._lock:
            self._tests.pop(name, None)
            self._results.pop(name, None)

    @staticmethod
    def _chi_squared_test(
        a_success: int, a_total: int, b_success: int, b_total: int
    ) -> float:
        """Compute p-value using chi-squared test for two proportions."""
        if a_total == 0 or b_total == 0:
            return 1.0

        total = a_total + b_total
        total_success = a_success + b_success
        total_fail = total - total_success

        if total_success == 0 or total_fail == 0:
            return 1.0

        # Expected values under null hypothesis
        e_a_success = a_total * total_success / total
        e_a_fail = a_total * total_fail / total
        e_b_success = b_total * total_success / total
        e_b_fail = b_total * total_fail / total

        chi2 = 0.0
        for observed, expected in [
            (a_success, e_a_success),
            (a_total - a_success, e_a_fail),
            (b_success, e_b_success),
            (b_total - b_success, e_b_fail),
        ]:
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

        # Approximate p-value from chi-squared with 1 df
        # Using survival function approximation
        return _chi2_sf(chi2, df=1)


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function for chi-squared distribution (no scipy dependency).

    Uses the incomplete gamma function approximation for df=1.
    """
    if x <= 0:
        return 1.0
    if df != 1:
        # Fallback: for df=1 only
        return 1.0

    import math

    # For df=1, chi2 SF = 2 * (1 - Phi(sqrt(x))) where Phi is normal CDF
    z = math.sqrt(x)
    # Approximation of normal SF using error function
    return math.erfc(z / math.sqrt(2))
