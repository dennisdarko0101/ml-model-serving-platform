"""FastAPI application — startup, shutdown, CORS, and router mounting."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml_serving.api.middleware import RequestLoggingMiddleware
from ml_serving.api.routes import experiments, health, models, monitoring, predict
from ml_serving.config.settings import Settings, get_settings
from ml_serving.monitoring.alerting import AlertManager
from ml_serving.monitoring.drift_detector import DriftDetector
from ml_serving.monitoring.metrics import MetricsCollector
from ml_serving.registry.model_registry import ModelRegistry
from ml_serving.registry.model_store import ModelStore
from ml_serving.routing.ab_testing import ABTestManager
from ml_serving.routing.canary import CanaryDeployment
from ml_serving.serving.model_server import ModelServer

logger = structlog.get_logger()


@dataclass
class AppState:
    """Shared application state available to all route handlers."""

    settings: Settings
    registry: ModelRegistry
    store: ModelStore
    model_server: ModelServer
    metrics: MetricsCollector
    drift_detector: DriftDetector
    alert_manager: AlertManager
    ab_manager: ABTestManager
    canary: CanaryDeployment | None = None


_app_state: AppState | None = None


def get_app_state() -> AppState:
    """Return the current application state (set during startup)."""
    if _app_state is None:
        raise RuntimeError("Application state not initialized")
    return _app_state


def set_app_state(state: AppState) -> None:
    """Override application state (used in tests)."""
    global _app_state
    _app_state = state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global _app_state

    # If state was pre-set (e.g. by tests), don't override it
    if _app_state is None:
        settings = get_settings()
        registry = ModelRegistry(settings)
        store = ModelStore(settings)
        server = ModelServer(registry=registry, store=store, settings=settings)
        metrics = MetricsCollector()

        _app_state = AppState(
            settings=settings,
            registry=registry,
            store=store,
            model_server=server,
            metrics=metrics,
            drift_detector=DriftDetector(),
            alert_manager=AlertManager(),
            ab_manager=ABTestManager(),
        )

    logger.info("app_started", host=_app_state.settings.API_HOST, port=_app_state.settings.API_PORT)

    yield

    logger.info("app_shutdown")
    _app_state = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ML Model Serving Platform",
        description="Production MLOps platform with A/B testing, drift detection, and monitoring",
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Mount routers
    app.include_router(predict.router)
    app.include_router(models.router)
    app.include_router(experiments.router)
    app.include_router(monitoring.router)
    app.include_router(health.router)

    return app


app = create_app()
