"""Application settings using pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Platform configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model storage
    MODEL_STORE_PATH: Path = Path("./model_artifacts")
    REGISTRY_PATH: Path = Path("./model_registry")
    MAX_LOADED_MODELS: int = 10

    # Batching
    BATCH_TIMEOUT_MS: int = 50
    BATCH_MAX_SIZE: int = 32

    # Monitoring
    PROMETHEUS_PORT: int = 9090

    # Cloud (optional)
    GCP_PROJECT: str = ""
    GCP_REGION: str = "us-central1"
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = ""

    # Logging
    LOG_LEVEL: str = "INFO"

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
