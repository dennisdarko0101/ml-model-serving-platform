"""Model registry — tracks model metadata, versions, and lifecycle stages."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from ml_serving.config.settings import Settings, get_settings
from ml_serving.registry.schemas import (
    Framework,
    ModelMetadata,
    ModelStage,
    ModelStatus,
    ModelVersion,
)

logger = structlog.get_logger()


class ModelRegistry:
    """JSON-file-backed model registry.

    Layout on disk:
        <registry_root>/<model_name>/metadata.json
        <registry_root>/<model_name>/versions/<version>.json
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._root = Path(self._settings.REGISTRY_PATH)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        version: str,
        framework: Framework | str,
        description: str = "",
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        artifact_path: str = "",
    ) -> ModelMetadata:
        """Register a new model version (creates the model entry if needed)."""
        framework = Framework(framework) if isinstance(framework, str) else framework
        now = datetime.now(timezone.utc)

        metadata = ModelMetadata(
            name=name,
            version=version,
            framework=framework,
            description=description,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            metrics=metrics or {},
            stage=ModelStage.STAGING,
            artifact_path=artifact_path,
            status=ModelStatus.REGISTERED,
        )

        model_version = ModelVersion(
            version=version,
            stage=ModelStage.STAGING,
            artifact_path=artifact_path,
            framework=framework,
            registered_at=now,
            metrics=metrics or {},
            description=description,
        )

        # Persist
        model_dir = self._model_dir(name)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "versions").mkdir(exist_ok=True)

        self._write_metadata(name, metadata)
        self._write_version(name, version, model_version)

        logger.info("model_registered", name=name, version=version)
        return metadata

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str, version: str) -> ModelMetadata:
        """Get metadata for a specific model version."""
        ver = self._read_version(name, version)
        base = self._read_metadata(name)
        return base.model_copy(
            update={
                "version": ver.version,
                "stage": ver.stage,
                "artifact_path": ver.artifact_path,
                "metrics": ver.metrics,
                "framework": ver.framework,
            }
        )

    def get_latest(self, name: str) -> ModelMetadata:
        """Return the most recently registered version."""
        versions = self.list_versions(name)
        if not versions:
            raise KeyError(f"No versions found for model '{name}'")
        latest = max(versions, key=lambda v: v.registered_at)
        return self.get(name, latest.version)

    def get_production(self, name: str) -> ModelMetadata:
        """Return the version currently in the production stage."""
        versions = self.list_versions(name)
        prod = [v for v in versions if v.stage == ModelStage.PRODUCTION]
        if not prod:
            raise KeyError(f"No production version for model '{name}'")
        return self.get(name, prod[0].version)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def promote(self, name: str, version: str, stage: ModelStage | str) -> ModelMetadata:
        """Move a model version to a new lifecycle stage.

        If promoting to PRODUCTION, any existing production version is
        automatically demoted to ARCHIVED.
        """
        stage = ModelStage(stage) if isinstance(stage, str) else stage

        if stage == ModelStage.PRODUCTION:
            self._demote_current_production(name)

        ver = self._read_version(name, version)
        ver.stage = stage
        self._write_version(name, version, ver)

        meta = self._read_metadata(name)
        meta.updated_at = datetime.now(timezone.utc)
        self._write_metadata(name, meta)

        logger.info("model_promoted", name=name, version=version, stage=stage.value)
        return self.get(name, version)

    # ------------------------------------------------------------------
    # Listing / comparison
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelMetadata]:
        """List all registered model names with their latest metadata."""
        results: list[ModelMetadata] = []
        if not self._root.exists():
            return results
        for model_dir in sorted(self._root.iterdir()):
            if model_dir.is_dir() and (model_dir / "metadata.json").exists():
                try:
                    results.append(self._read_metadata(model_dir.name))
                except Exception:
                    continue
        return results

    def list_versions(self, name: str) -> list[ModelVersion]:
        """List all versions for a given model."""
        versions_dir = self._model_dir(name) / "versions"
        if not versions_dir.exists():
            return []
        results: list[ModelVersion] = []
        for path in sorted(versions_dir.glob("*.json")):
            results.append(ModelVersion.model_validate_json(path.read_text()))
        return results

    def compare_versions(self, name: str, v1: str, v2: str) -> dict[str, Any]:
        """Return a side-by-side comparison of metrics for two versions."""
        meta1 = self.get(name, v1)
        meta2 = self.get(name, v2)

        all_keys = sorted(set(meta1.metrics) | set(meta2.metrics))
        comparison: dict[str, Any] = {
            "model": name,
            "v1": v1,
            "v2": v2,
            "metrics": {},
        }
        for key in all_keys:
            val1 = meta1.metrics.get(key)
            val2 = meta2.metrics.get(key)
            diff = None
            if val1 is not None and val2 is not None:
                diff = round(val2 - val1, 6)
            comparison["metrics"][key] = {"v1": val1, "v2": val2, "diff": diff}
        return comparison

    def delete(self, name: str, version: str) -> None:
        """Remove a version entry from the registry (does not touch artifacts)."""
        path = self._model_dir(name) / "versions" / f"{version}.json"
        if path.exists():
            path.unlink()
            logger.info("version_deleted", name=name, version=version)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_dir(self, name: str) -> Path:
        return self._root / name

    def _write_metadata(self, name: str, meta: ModelMetadata) -> None:
        path = self._model_dir(name) / "metadata.json"
        path.write_text(meta.model_dump_json(indent=2))

    def _read_metadata(self, name: str) -> ModelMetadata:
        path = self._model_dir(name) / "metadata.json"
        if not path.exists():
            raise KeyError(f"Model '{name}' not found in registry")
        return ModelMetadata.model_validate_json(path.read_text())

    def _write_version(self, name: str, version: str, ver: ModelVersion) -> None:
        path = self._model_dir(name) / "versions" / f"{version}.json"
        path.write_text(ver.model_dump_json(indent=2))

    def _read_version(self, name: str, version: str) -> ModelVersion:
        path = self._model_dir(name) / "versions" / f"{version}.json"
        if not path.exists():
            raise KeyError(f"Version '{version}' not found for model '{name}'")
        return ModelVersion.model_validate_json(path.read_text())

    def _demote_current_production(self, name: str) -> None:
        """Demote any current production version to archived."""
        for ver in self.list_versions(name):
            if ver.stage == ModelStage.PRODUCTION:
                ver.stage = ModelStage.ARCHIVED
                self._write_version(name, ver.version, ver)
