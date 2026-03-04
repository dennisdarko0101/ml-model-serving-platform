"""Data models for the model registry."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Framework(str, enum.Enum):
    """Supported ML frameworks."""

    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    ONNX = "onnx"


class ModelStage(str, enum.Enum):
    """Lifecycle stage for a model version."""

    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(str, enum.Enum):
    """Runtime status of a model."""

    REGISTERED = "registered"
    LOADING = "loading"
    LOADED = "loaded"
    SERVING = "serving"
    FAILED = "failed"
    ARCHIVED = "archived"


class ModelVersion(BaseModel):
    """A specific version of a registered model."""

    version: str
    stage: ModelStage = ModelStage.STAGING
    artifact_path: str = ""
    framework: Framework = Framework.SKLEARN
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, float] = Field(default_factory=dict)
    description: str = ""


class ModelMetadata(BaseModel):
    """Top-level metadata for a registered model."""

    name: str
    version: str
    framework: Framework
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    stage: ModelStage = ModelStage.STAGING
    artifact_path: str = ""
    status: ModelStatus = ModelStatus.REGISTERED


class PredictionResult(BaseModel):
    """Result returned from a model prediction."""

    prediction: Any
    probabilities: list[float] | None = None
    latency_ms: float = 0.0
    model_name: str = ""
    model_version: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
