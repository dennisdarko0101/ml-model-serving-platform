"""Monitoring endpoints — metrics, drift, and alerts."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from ml_serving.api.schemas import (
    AlertResponse,
    DriftReportResponse,
    MetricsSummaryResponse,
)

router = APIRouter(tags=["monitoring"])


@router.get("/api/v1/monitoring/metrics", response_model=MetricsSummaryResponse)
async def get_metrics_summary():
    """Return current metrics summary."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    summary = state.metrics.get_metrics_summary()
    return MetricsSummaryResponse(**summary)


@router.get("/api/v1/monitoring/drift/{model_name}", response_model=DriftReportResponse)
async def get_drift_report(model_name: str):
    """Get drift report for a specific model."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    report = state.drift_detector.check_model_drift(model_name)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No drift data available for model '{model_name}'",
        )
    return DriftReportResponse(
        feature_scores=report.feature_scores,
        overall_score=report.overall_score,
        is_drifted=report.is_drifted,
        drifted_features=report.drifted_features,
        method=report.method,
        threshold=report.threshold,
    )


@router.get("/api/v1/monitoring/alerts", response_model=list[AlertResponse])
async def get_alerts():
    """Return active alerts."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    alerts = state.alert_manager.get_alerts()
    return [
        AlertResponse(
            rule_name=a.rule_name,
            metric=a.metric,
            metric_value=a.metric_value,
            threshold=a.threshold,
            severity=a.severity.value,
            timestamp=a.timestamp,
            message=a.message,
        )
        for a in alerts
    ]


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    data = state.metrics.generate_latest()
    return Response(content=data, media_type="text/plain; charset=utf-8")
