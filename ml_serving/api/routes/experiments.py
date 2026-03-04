"""A/B testing and canary deployment endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ml_serving.api.schemas import (
    ABTestConcludeResponse,
    ABTestCreateRequest,
    ABTestResponse,
    ABTestResultsResponse,
    CanaryCreateRequest,
    CanaryStatusResponse,
)
from ml_serving.routing.canary import CanaryDeployment

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


# ---------------------------------------------------------------------------
# A/B Testing
# ---------------------------------------------------------------------------


@router.post("/ab", response_model=ABTestResponse, status_code=201)
async def create_ab_test(request: ABTestCreateRequest):
    """Create a new A/B test."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    test = state.ab_manager.create_test(
        name=request.name,
        model_a=request.model_a,
        model_b=request.model_b,
        traffic_split=request.traffic_split,
    )
    return ABTestResponse(
        name=test.name,
        model_a=test.model_a,
        model_b=test.model_b,
        traffic_split=test.traffic_split,
    )


@router.get("/ab/{name}", response_model=ABTestResultsResponse)
async def get_ab_results(name: str):
    """Get A/B test results."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    try:
        results = state.ab_manager.get_results(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ABTestResultsResponse(
        test_name=results.test_name,
        model_a=results.model_a,
        model_b=results.model_b,
        model_a_metrics=results.model_a_metrics,
        model_b_metrics=results.model_b_metrics,
        sample_size_a=results.sample_size_a,
        sample_size_b=results.sample_size_b,
        p_value=results.p_value,
        is_significant=results.is_significant,
    )


@router.post("/ab/{name}/conclude", response_model=ABTestConcludeResponse)
async def conclude_ab_test(name: str):
    """Conclude an A/B test and declare a winner."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    try:
        result = state.ab_manager.conclude_test(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ABTestConcludeResponse(**result)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@router.post("/canary", response_model=CanaryStatusResponse, status_code=201)
async def create_canary(request: CanaryCreateRequest):
    """Start a canary deployment."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()

    state.canary = CanaryDeployment(
        current_model=request.current_model,
        canary_model=request.canary_model,
        initial_percentage=request.initial_percentage,
        error_threshold=request.error_threshold,
    )

    metrics = state.canary.get_metrics()
    return CanaryStatusResponse(
        current_model=metrics["current_model"],
        canary_model=metrics["canary_model"],
        canary_percentage=metrics["canary_percentage"],
        active=metrics["active"],
        current_metrics=metrics["current"],
        canary_metrics=metrics["canary"],
    )


@router.post("/canary/promote", response_model=CanaryStatusResponse)
async def promote_canary():
    """Promote the canary to the next traffic level."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    if state.canary is None:
        raise HTTPException(status_code=404, detail="No active canary deployment")

    state.canary.promote()
    metrics = state.canary.get_metrics()

    return CanaryStatusResponse(
        current_model=metrics["current_model"],
        canary_model=metrics["canary_model"],
        canary_percentage=metrics["canary_percentage"],
        active=metrics["active"],
        current_metrics=metrics["current"],
        canary_metrics=metrics["canary"],
    )


@router.post("/canary/rollback", response_model=CanaryStatusResponse)
async def rollback_canary():
    """Rollback the canary deployment."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    if state.canary is None:
        raise HTTPException(status_code=404, detail="No active canary deployment")

    state.canary.rollback()
    metrics = state.canary.get_metrics()

    return CanaryStatusResponse(
        current_model=metrics["current_model"],
        canary_model=metrics["canary_model"],
        canary_percentage=metrics["canary_percentage"],
        active=metrics["active"],
        current_metrics=metrics["current"],
        canary_metrics=metrics["canary"],
    )
