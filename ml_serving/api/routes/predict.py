"""Prediction endpoints."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException

from ml_serving.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Single prediction — routes through A/B test or canary if active."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    request_id = request.request_id or str(uuid.uuid4())

    model_name = request.model_name
    version = request.version

    # Check if there's an active A/B test for this model
    ab_manager = state.ab_manager
    for test in ab_manager.list_tests():
        model_a_name = test.model_a.split(":")[0]
        model_b_name = test.model_b.split(":")[0]
        if model_name in (model_a_name, model_b_name):
            routed = test.route_request(request_id)
            parts = routed.split(":")
            model_name = parts[0]
            version = parts[1] if len(parts) > 1 else version
            break

    # Check canary deployment
    canary = state.canary
    if canary and canary.is_active:
        canary_current = canary.current_model.split(":")[0]
        canary_new = canary.canary_model.split(":")[0]
        if model_name in (canary_current, canary_new):
            routed = canary.route_request()
            parts = routed.split(":")
            model_name = parts[0]
            version = parts[1] if len(parts) > 1 else version

    start = time.monotonic()
    try:
        result = state.model_server.predict(model_name, request.input_data, version=version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        latency = time.monotonic() - start
        state.metrics.record_prediction(model_name, version or "latest", latency, status="error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = time.monotonic() - start
    state.metrics.record_prediction(
        model_name, result.model_version, latency, status="success"
    )

    return PredictResponse(
        prediction=result.prediction,
        probabilities=result.probabilities,
        model_used=result.model_name,
        version=result.model_version,
        latency_ms=result.latency_ms,
        request_id=request_id,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Batch prediction endpoint."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()

    start = time.monotonic()
    try:
        results = state.model_server.predict_batch(
            request.model_name, request.inputs, version=request.version
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total_latency = (time.monotonic() - start) * 1000
    state.metrics.record_batch(len(request.inputs))

    predictions = [
        PredictResponse(
            prediction=r.prediction,
            probabilities=r.probabilities,
            model_used=r.model_name,
            version=r.model_version,
            latency_ms=r.latency_ms,
            request_id=request.request_id,
        )
        for r in results
    ]

    return BatchPredictResponse(
        predictions=predictions,
        total_latency_ms=total_latency,
        batch_size=len(results),
    )
