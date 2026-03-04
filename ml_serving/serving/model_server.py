"""Model server — loads models into memory and serves predictions."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

import structlog

from ml_serving.config.settings import Settings, get_settings
from ml_serving.registry.model_registry import ModelRegistry
from ml_serving.registry.model_store import ModelStore
from ml_serving.registry.schemas import Framework, ModelStatus, PredictionResult
from ml_serving.serving.predictor import BasePredictor, PredictorFactory
from ml_serving.serving.preprocessor import PreprocessingPipeline

logger = structlog.get_logger()


class _LoadedModel:
    """Internal bookkeeping for a model loaded into memory."""

    __slots__ = ("predictor", "pipeline", "status", "loaded_at")

    def __init__(
        self,
        predictor: BasePredictor,
        pipeline: PreprocessingPipeline | None,
        status: ModelStatus,
    ) -> None:
        self.predictor = predictor
        self.pipeline = pipeline
        self.status = status
        self.loaded_at = time.time()


class ModelServer:
    """Thread-safe, multi-model prediction server with LRU eviction."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        store: ModelStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._registry = registry or ModelRegistry(self._settings)
        self._store = store or ModelStore(self._settings)

        self._max_loaded = self._settings.MAX_LOADED_MODELS
        self._models: OrderedDict[str, _LoadedModel] = OrderedDict()
        self._lock = threading.Lock()
        self._pipelines: dict[str, PreprocessingPipeline] = {}

    # ------------------------------------------------------------------
    # Pipeline registration
    # ------------------------------------------------------------------

    def register_pipeline(
        self, model_name: str, pipeline: PreprocessingPipeline
    ) -> None:
        """Associate a preprocessing pipeline with a model name."""
        self._pipelines[model_name] = pipeline

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(
        self,
        name: str,
        version: str,
        framework: Framework | str | None = None,
        model_object: Any | None = None,
    ) -> None:
        """Load a model into memory for serving.

        If *model_object* is supplied it is used directly (useful for tests);
        otherwise the artifact is loaded via ModelStore.
        """
        key = self._key(name, version)

        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                logger.info("model_already_loaded", key=key)
                return

            self._evict_if_needed()

            # Determine framework
            if framework is None:
                meta = self._registry.get(name, version)
                framework = meta.framework

            fw = Framework(framework) if isinstance(framework, str) else framework

            status = ModelStatus.LOADING
            # Placeholder while loading
            placeholder = _LoadedModel(
                predictor=None,  # type: ignore[arg-type]
                pipeline=self._pipelines.get(name),
                status=status,
            )
            self._models[key] = placeholder

        try:
            if model_object is None:
                model_object = self._store.load_model(name, version, fw)

            predictor = PredictorFactory.create(
                model_object, fw, name=name, version=version
            )
            predictor.warmup()

            with self._lock:
                entry = self._models.get(key)
                if entry is not None:
                    entry.predictor = predictor
                    entry.status = ModelStatus.LOADED
                    self._models.move_to_end(key)

            logger.info("model_loaded", key=key)

        except Exception as exc:
            with self._lock:
                entry = self._models.get(key)
                if entry is not None:
                    entry.status = ModelStatus.FAILED
            logger.error("model_load_failed", key=key, error=str(exc))
            raise

    def unload_model(self, name: str, version: str) -> None:
        """Remove a model from memory."""
        key = self._key(name, version)
        with self._lock:
            if key in self._models:
                del self._models[key]
                logger.info("model_unloaded", key=key)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, model_name: str, input_data: Any, version: str | None = None) -> PredictionResult:
        """Run inference on a loaded model."""
        entry = self._get_entry(model_name, version)

        # Preprocess
        data = input_data
        if entry.pipeline is not None:
            data = entry.pipeline.process(data)

        # Mark serving
        entry.status = ModelStatus.SERVING
        try:
            result = entry.predictor.predict(data)
        finally:
            entry.status = ModelStatus.LOADED

        with self._lock:
            key = self._key(model_name, entry.predictor._version)
            if key in self._models:
                self._models.move_to_end(key)

        return result

    def predict_batch(
        self, model_name: str, inputs: list[Any], version: str | None = None
    ) -> list[PredictionResult]:
        """Batch prediction on a loaded model."""
        entry = self._get_entry(model_name, version)

        data = inputs
        if entry.pipeline is not None:
            data = [entry.pipeline.process(inp) for inp in inputs]

        entry.status = ModelStatus.SERVING
        try:
            results = entry.predictor.predict_batch(data)
        finally:
            entry.status = ModelStatus.LOADED

        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_loaded_models(self) -> list[str]:
        with self._lock:
            return list(self._models.keys())

    def get_model_status(self, name: str, version: str | None = None) -> ModelStatus:
        entry = self._get_entry(name, version, strict=False)
        if entry is None:
            return ModelStatus.REGISTERED
        return entry.status

    def is_healthy(self, name: str, version: str | None = None) -> bool:
        entry = self._get_entry(name, version, strict=False)
        if entry is None:
            return False
        return entry.status in (ModelStatus.LOADED, ModelStatus.SERVING)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _key(name: str, version: str) -> str:
        return f"{name}:{version}"

    def _get_entry(
        self, name: str, version: str | None = None, *, strict: bool = True
    ) -> _LoadedModel | None:
        with self._lock:
            if version:
                key = self._key(name, version)
                entry = self._models.get(key)
            else:
                # Find the first loaded version for this model name
                entry = None
                for k, v in self._models.items():
                    if k.startswith(f"{name}:"):
                        entry = v
                        break

        if entry is None and strict:
            raise KeyError(f"Model '{name}' (version={version}) is not loaded")
        return entry

    def _evict_if_needed(self) -> None:
        """Evict the least-recently-used model if at capacity. Must hold _lock."""
        while len(self._models) >= self._max_loaded:
            evicted_key, _ = self._models.popitem(last=False)
            logger.info("model_evicted", key=evicted_key)
