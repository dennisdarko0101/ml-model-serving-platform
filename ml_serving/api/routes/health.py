"""Health check endpoint with per-component status."""

from __future__ import annotations

import time

from fastapi import APIRouter

from ml_serving.api.schemas import HealthResponse, ModelHealthStatus

router = APIRouter(tags=["health"])

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check including per-model status."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()

    model_statuses: list[ModelHealthStatus] = []
    for key in state.model_server.get_loaded_models():
        parts = key.split(":")
        name = parts[0]
        version = parts[1] if len(parts) > 1 else "unknown"
        status = state.model_server.get_model_status(name, version)
        healthy = state.model_server.is_healthy(name, version)
        model_statuses.append(
            ModelHealthStatus(
                name=name,
                version=version,
                status=status.value,
                healthy=healthy,
            )
        )

    overall = "healthy" if all(m.healthy for m in model_statuses) or not model_statuses else "degraded"

    return HealthResponse(
        status=overall,
        uptime_seconds=time.time() - _start_time,
        models=model_statuses,
    )
