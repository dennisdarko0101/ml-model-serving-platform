"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    model_name: str
    version: str | None = None
    input_data: Any
    request_id: str | None = None


class PredictResponse(BaseModel):
    prediction: Any
    probabilities: list[float] | None = None
    model_used: str
    version: str
    latency_ms: float
    request_id: str | None = None


class BatchPredictRequest(BaseModel):
    model_name: str
    version: str | None = None
    inputs: list[Any]
    request_id: str | None = None


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total_latency_ms: float
    batch_size: int


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ModelCreateRequest(BaseModel):
    name: str
    version: str
    framework: str = "sklearn"
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class ModelCreateResponse(BaseModel):
    name: str
    version: str
    framework: str
    stage: str
    created_at: datetime


class ModelPromoteRequest(BaseModel):
    version: str
    stage: str = "production"


class ModelDetailResponse(BaseModel):
    name: str
    version: str
    framework: str
    stage: str
    status: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime | None = None


class ModelListResponse(BaseModel):
    models: list[ModelDetailResponse]


# ---------------------------------------------------------------------------
# A/B Testing
# ---------------------------------------------------------------------------


class ABTestCreateRequest(BaseModel):
    name: str
    model_a: str
    model_b: str
    traffic_split: float = 0.5


class ABTestResponse(BaseModel):
    name: str
    model_a: str
    model_b: str
    traffic_split: float


class ABTestResultsResponse(BaseModel):
    test_name: str
    model_a: str
    model_b: str
    model_a_metrics: dict[str, Any]
    model_b_metrics: dict[str, Any]
    sample_size_a: int
    sample_size_b: int
    p_value: float
    is_significant: bool


class ABTestConcludeResponse(BaseModel):
    test_name: str
    winner: str
    is_significant: bool
    p_value: float
    model_a_accuracy: float
    model_b_accuracy: float


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


class CanaryCreateRequest(BaseModel):
    current_model: str
    canary_model: str
    initial_percentage: int = 5
    error_threshold: float = 0.1


class CanaryStatusResponse(BaseModel):
    current_model: str
    canary_model: str
    canary_percentage: int
    active: bool
    current_metrics: dict[str, Any]
    canary_metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


class DriftReportResponse(BaseModel):
    feature_scores: dict[str, float]
    overall_score: float
    is_drifted: bool
    drifted_features: list[str]
    method: str
    threshold: float


class MetricsSummaryResponse(BaseModel):
    total_predictions: int
    error_rate: float
    avg_latency_seconds: float
    models: dict[str, Any]


class AlertResponse(BaseModel):
    rule_name: str
    metric: str
    metric_value: float
    threshold: float
    severity: str
    timestamp: float
    message: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ModelHealthStatus(BaseModel):
    name: str
    version: str
    status: str
    healthy: bool


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    models: list[ModelHealthStatus] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
