"""Model management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ml_serving.api.schemas import (
    ModelCreateRequest,
    ModelCreateResponse,
    ModelDetailResponse,
    ModelListResponse,
    ModelPromoteRequest,
)
from ml_serving.registry.schemas import Framework, ModelStage

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.post("", response_model=ModelCreateResponse, status_code=201)
async def register_model(request: ModelCreateRequest):
    """Register a new model version."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()

    try:
        fw = Framework(request.framework)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown framework: {request.framework}") from exc

    meta = state.registry.register(
        name=request.name,
        version=request.version,
        framework=fw,
        description=request.description,
        tags=request.tags,
        metrics=request.metrics,
    )

    return ModelCreateResponse(
        name=meta.name,
        version=meta.version,
        framework=meta.framework.value,
        stage=meta.stage.value,
        created_at=meta.created_at,
    )


@router.get("", response_model=ModelListResponse)
async def list_models():
    """List all registered models."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    models = state.registry.list_models()

    return ModelListResponse(
        models=[
            ModelDetailResponse(
                name=m.name,
                version=m.version,
                framework=m.framework.value,
                stage=m.stage.value,
                status=m.status.value,
                description=m.description,
                tags=m.tags,
                metrics=m.metrics,
                created_at=m.created_at,
            )
            for m in models
        ]
    )


@router.get("/{name}", response_model=ModelDetailResponse)
async def get_model(name: str, version: str | None = None):
    """Get model details."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()

    try:
        if version:
            meta = state.registry.get(name, version)
        else:
            meta = state.registry.get_latest(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ModelDetailResponse(
        name=meta.name,
        version=meta.version,
        framework=meta.framework.value,
        stage=meta.stage.value,
        status=meta.status.value,
        description=meta.description,
        tags=meta.tags,
        metrics=meta.metrics,
        created_at=meta.created_at,
    )


@router.put("/{name}/promote")
async def promote_model(name: str, request: ModelPromoteRequest):
    """Promote a model version to a new stage."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()

    try:
        stage = ModelStage(request.stage)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown stage: {request.stage}") from exc

    try:
        meta = state.registry.promote(name, request.version, stage)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"name": meta.name, "version": meta.version, "stage": meta.stage.value}


@router.delete("/{name}/{version}")
async def archive_model(name: str, version: str):
    """Archive a model version."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()

    try:
        state.registry.promote(name, version, ModelStage.ARCHIVED)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"status": "archived", "name": name, "version": version}


@router.post("/{name}/load")
async def load_model(name: str, version: str | None = None):
    """Load a model into the serving server."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()

    try:
        if version is None:
            meta = state.registry.get_latest(name)
            version = meta.version
        state.model_server.load_model(name, version)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state.metrics.set_active_models(len(state.model_server.get_loaded_models()))

    return {"status": "loaded", "name": name, "version": version}


@router.post("/{name}/unload")
async def unload_model(name: str, version: str):
    """Unload a model from the serving server."""
    from ml_serving.api.main import get_app_state

    state = get_app_state()
    state.model_server.unload_model(name, version)
    state.metrics.set_active_models(len(state.model_server.get_loaded_models()))

    return {"status": "unloaded", "name": name, "version": version}
