"""Drift detection using PSI, KS test, and chi-squared for categorical features."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class DriftReport:
    """Report from a drift detection check."""

    feature_scores: dict[str, float]
    overall_score: float
    is_drifted: bool
    drifted_features: list[str]
    method: str
    threshold: float


class DriftDetector:
    """Detect data and prediction drift using statistical tests.

    Supports:
      - PSI (Population Stability Index) for distribution shift
      - KS test (Kolmogorov-Smirnov) for feature drift
      - Chi-squared for categorical feature drift
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1,
        chi2_threshold: float = 0.05,
        window_size: int = 1000,
    ) -> None:
        self._psi_threshold = psi_threshold
        self._ks_threshold = ks_threshold
        self._chi2_threshold = chi2_threshold
        self._window_size = window_size

        # Reference distributions stored per model
        self._reference_data: dict[str, np.ndarray] = {}
        self._current_windows: dict[str, deque] = {}
        self._lock = threading.Lock()

    def set_reference(self, model_name: str, data: np.ndarray) -> None:
        """Store reference data (training distribution) for a model."""
        with self._lock:
            self._reference_data[model_name] = np.asarray(data)
            self._current_windows[model_name] = deque(maxlen=self._window_size)

    def add_sample(self, model_name: str, sample: np.ndarray) -> None:
        """Add a sample to the current data sliding window."""
        with self._lock:
            if model_name not in self._current_windows:
                self._current_windows[model_name] = deque(maxlen=self._window_size)
            self._current_windows[model_name].append(np.asarray(sample))

    def detect_data_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        method: str = "psi",
    ) -> DriftReport:
        """Detect drift between reference and current data distributions.

        Args:
            reference_data: shape (n_samples, n_features)
            current_data: shape (n_samples, n_features)
            method: "psi" or "ks"
        """
        ref = np.asarray(reference_data)
        cur = np.asarray(current_data)

        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if cur.ndim == 1:
            cur = cur.reshape(-1, 1)

        n_features = ref.shape[1]

        if method == "psi":
            return self._detect_psi(ref, cur, n_features)
        elif method == "ks":
            return self._detect_ks(ref, cur, n_features)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'psi' or 'ks'.")

    def detect_prediction_drift(
        self,
        reference_preds: np.ndarray,
        current_preds: np.ndarray,
    ) -> DriftReport:
        """Detect drift in model predictions using PSI."""
        ref = np.asarray(reference_preds).ravel()
        cur = np.asarray(current_preds).ravel()
        return self._detect_psi(
            ref.reshape(-1, 1), cur.reshape(-1, 1), n_features=1
        )

    def detect_categorical_drift(
        self,
        reference_counts: dict[str, int],
        current_counts: dict[str, int],
    ) -> DriftReport:
        """Detect drift in categorical feature using chi-squared test."""
        categories = sorted(set(reference_counts) | set(current_counts))

        ref_total = sum(reference_counts.values())
        cur_total = sum(current_counts.values())

        if ref_total == 0 or cur_total == 0:
            return DriftReport(
                feature_scores={},
                overall_score=0.0,
                is_drifted=False,
                drifted_features=[],
                method="chi_squared",
                threshold=self._chi2_threshold,
            )

        chi2 = 0.0
        for cat in categories:
            observed = current_counts.get(cat, 0)
            expected = (reference_counts.get(cat, 0) / ref_total) * cur_total
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

        df = max(len(categories) - 1, 1)
        # Approximate p-value
        p_value = _chi2_p_value(chi2, df)

        return DriftReport(
            feature_scores={"chi2_statistic": chi2, "p_value": p_value},
            overall_score=chi2,
            is_drifted=p_value < self._chi2_threshold,
            drifted_features=["categorical"] if p_value < self._chi2_threshold else [],
            method="chi_squared",
            threshold=self._chi2_threshold,
        )

    def check_model_drift(self, model_name: str) -> DriftReport | None:
        """Check drift for a model using stored reference data and sliding window."""
        with self._lock:
            if model_name not in self._reference_data:
                return None
            ref = self._reference_data[model_name]
            window = self._current_windows.get(model_name)
            if not window or len(window) < 10:
                return None
            current = np.array(list(window))

        return self.detect_data_drift(ref, current, method="psi")

    # ------------------------------------------------------------------
    # PSI
    # ------------------------------------------------------------------

    def _detect_psi(
        self, ref: np.ndarray, cur: np.ndarray, n_features: int
    ) -> DriftReport:
        feature_scores: dict[str, float] = {}
        drifted: list[str] = []

        for i in range(n_features):
            name = f"feature_{i}"
            psi = _compute_psi(ref[:, i], cur[:, i])
            feature_scores[name] = psi
            if psi > self._psi_threshold:
                drifted.append(name)

        overall = sum(feature_scores.values()) / n_features if n_features > 0 else 0.0

        return DriftReport(
            feature_scores=feature_scores,
            overall_score=overall,
            is_drifted=len(drifted) > 0,
            drifted_features=drifted,
            method="psi",
            threshold=self._psi_threshold,
        )

    # ------------------------------------------------------------------
    # KS test
    # ------------------------------------------------------------------

    def _detect_ks(
        self, ref: np.ndarray, cur: np.ndarray, n_features: int
    ) -> DriftReport:
        feature_scores: dict[str, float] = {}
        drifted: list[str] = []

        for i in range(n_features):
            name = f"feature_{i}"
            ks_stat = _ks_statistic(ref[:, i], cur[:, i])
            feature_scores[name] = ks_stat
            if ks_stat > self._ks_threshold:
                drifted.append(name)

        overall = sum(feature_scores.values()) / n_features if n_features > 0 else 0.0

        return DriftReport(
            feature_scores=feature_scores,
            overall_score=overall,
            is_drifted=len(drifted) > 0,
            drifted_features=drifted,
            method="ks",
            threshold=self._ks_threshold,
        )


# ---------------------------------------------------------------------------
# Statistical helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    eps = 1e-4

    # Use reference distribution to define bin edges
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)

    ref_counts = np.histogram(reference, bins=bins)[0].astype(float)
    cur_counts = np.histogram(current, bins=bins)[0].astype(float)

    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic between two samples."""
    all_values = np.sort(np.concatenate([reference, current]))
    n_ref = len(reference)
    n_cur = len(current)

    cdf_ref = np.searchsorted(np.sort(reference), all_values, side="right") / n_ref
    cdf_cur = np.searchsorted(np.sort(current), all_values, side="right") / n_cur

    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def _chi2_p_value(chi2: float, df: int) -> float:
    """Approximate chi-squared p-value without scipy."""
    if chi2 <= 0:
        return 1.0
    # Use Wilson-Hilferty approximation for chi-squared CDF
    z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    # Normal SF approximation
    return 0.5 * math.erfc(z / math.sqrt(2))
