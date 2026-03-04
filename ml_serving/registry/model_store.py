"""Model artifact storage — save/load models across frameworks."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import joblib
import structlog

from ml_serving.config.settings import Settings, get_settings
from ml_serving.registry.schemas import Framework

logger = structlog.get_logger()


class ModelStore:
    """Manages serialisation and storage of model artifacts on local disk."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._root = Path(self._settings.MODEL_STORE_PATH)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_model(
        self,
        model: Any,
        name: str,
        version: str,
        framework: Framework | str,
    ) -> str:
        """Persist a model artifact and return the artifact path."""
        framework = Framework(framework) if isinstance(framework, str) else framework
        model_dir = self._model_dir(name, version)
        model_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = self._artifact_path(model_dir, framework)
        self._save_by_framework(model, artifact_path, framework)

        logger.info(
            "model_saved",
            name=name,
            version=version,
            framework=framework.value,
            path=str(artifact_path),
        )
        return str(artifact_path)

    def load_model(self, name: str, version: str, framework: Framework | str) -> Any:
        """Load a model artifact from disk."""
        framework = Framework(framework) if isinstance(framework, str) else framework
        model_dir = self._model_dir(name, version)
        artifact_path = self._artifact_path(model_dir, framework)

        if not artifact_path.exists():
            raise FileNotFoundError(
                f"No artifact found for {name}:{version} at {artifact_path}"
            )

        model = self._load_by_framework(artifact_path, framework)
        logger.info(
            "model_loaded",
            name=name,
            version=version,
            framework=framework.value,
        )
        return model

    def delete_model(self, name: str, version: str) -> None:
        """Remove a model artifact from disk."""
        model_dir = self._model_dir(name, version)
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info("model_deleted", name=name, version=version)

    def list_artifacts(self, name: str) -> list[str]:
        """Return a list of artifact paths for every version of *name*."""
        base = self._root / name
        if not base.exists():
            return []
        return sorted(str(p) for p in base.rglob("*") if p.is_file())

    def artifact_exists(self, name: str, version: str, framework: Framework | str) -> bool:
        """Check whether an artifact exists on disk."""
        framework = Framework(framework) if isinstance(framework, str) else framework
        path = self._artifact_path(self._model_dir(name, version), framework)
        return path.exists()

    # ------------------------------------------------------------------
    # Framework-specific serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _save_by_framework(model: Any, path: Path, framework: Framework) -> None:
        if framework == Framework.SKLEARN:
            joblib.dump(model, path)
        elif framework == Framework.PYTORCH:
            import torch

            torch.save(model.state_dict() if hasattr(model, "state_dict") else model, path)
        elif framework == Framework.ONNX:
            # ONNX models are already serialised bytes / a file path.
            if isinstance(model, bytes):
                path.write_bytes(model)
            elif isinstance(model, (str, Path)):
                shutil.copy2(str(model), path)
            else:
                raise TypeError(f"Cannot save ONNX model of type {type(model)}")
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def _load_by_framework(path: Path, framework: Framework) -> Any:
        if framework == Framework.SKLEARN:
            return joblib.load(path)
        elif framework == Framework.PYTORCH:
            import torch

            return torch.load(path, map_location="cpu", weights_only=False)
        elif framework == Framework.ONNX:
            import onnxruntime as ort

            return ort.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _model_dir(self, name: str, version: str) -> Path:
        return self._root / name / version

    @staticmethod
    def _artifact_path(model_dir: Path, framework: Framework) -> Path:
        extensions = {
            Framework.SKLEARN: "joblib",
            Framework.PYTORCH: "pt",
            Framework.ONNX: "onnx",
        }
        return model_dir / f"model.{extensions[framework]}"

    @staticmethod
    def detect_framework(model: Any) -> Framework:
        """Best-effort detection of the framework a model belongs to."""
        type_name = type(model).__module__

        if "sklearn" in type_name or "joblib" in type_name:
            return Framework.SKLEARN

        try:
            import torch

            if isinstance(model, torch.nn.Module):
                return Framework.PYTORCH
        except ImportError:
            pass

        if isinstance(model, bytes) or (
            isinstance(model, (str, Path)) and str(model).endswith(".onnx")
        ):
            return Framework.ONNX

        raise ValueError(f"Cannot detect framework for model type {type(model)}")
