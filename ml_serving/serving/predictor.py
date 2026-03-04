"""Predictor abstractions for each supported ML framework."""

from __future__ import annotations

import abc
import time
from typing import Any

import numpy as np
import structlog

from ml_serving.registry.schemas import Framework, PredictionResult

logger = structlog.get_logger()


class BasePredictor(abc.ABC):
    """Interface every predictor must implement."""

    def __init__(self, model: Any, name: str = "", version: str = "") -> None:
        self._model = model
        self._name = name
        self._version = version

    @abc.abstractmethod
    def predict(self, input_data: Any) -> PredictionResult:
        """Run inference on a single input."""

    def predict_batch(self, inputs: list[Any]) -> list[PredictionResult]:
        """Run inference on a batch.  Default: loop over predict()."""
        return [self.predict(inp) for inp in inputs]

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "framework": self.framework.value,
            "type": type(self._model).__name__,
        }

    @property
    @abc.abstractmethod
    def framework(self) -> Framework: ...

    def warmup(self) -> None:
        """Run a dummy prediction to warm JIT / caches."""
        try:
            dummy = self._make_dummy_input()
            if dummy is not None:
                self.predict(dummy)
                logger.info("predictor_warmup_done", name=self._name, version=self._version)
        except Exception as exc:
            logger.warning("predictor_warmup_failed", error=str(exc))

    def _make_dummy_input(self) -> Any | None:
        """Override to provide a framework-specific dummy input."""
        return None


# -----------------------------------------------------------------------
# Sklearn
# -----------------------------------------------------------------------


class SklearnPredictor(BasePredictor):
    """Wraps a scikit-learn estimator."""

    @property
    def framework(self) -> Framework:
        return Framework.SKLEARN

    def predict(self, input_data: Any) -> PredictionResult:
        arr = np.asarray(input_data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        start = time.perf_counter()
        preds = self._model.predict(arr)
        latency = (time.perf_counter() - start) * 1000

        probas = None
        if hasattr(self._model, "predict_proba"):
            try:
                probas = self._model.predict_proba(arr)[0].tolist()
            except Exception:
                pass

        return PredictionResult(
            prediction=preds[0].item() if hasattr(preds[0], "item") else preds[0],
            probabilities=probas,
            latency_ms=round(latency, 3),
            model_name=self._name,
            model_version=self._version,
        )

    def predict_batch(self, inputs: list[Any]) -> list[PredictionResult]:
        arr = np.asarray(inputs)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        start = time.perf_counter()
        preds = self._model.predict(arr)
        latency = (time.perf_counter() - start) * 1000

        probas_batch: list[list[float] | None] = [None] * len(preds)
        if hasattr(self._model, "predict_proba"):
            try:
                probas_all = self._model.predict_proba(arr)
                probas_batch = [row.tolist() for row in probas_all]
            except Exception:
                pass

        per_item_latency = round(latency / len(preds), 3)
        return [
            PredictionResult(
                prediction=p.item() if hasattr(p, "item") else p,
                probabilities=probas_batch[i],
                latency_ms=per_item_latency,
                model_name=self._name,
                model_version=self._version,
            )
            for i, p in enumerate(preds)
        ]

    def _make_dummy_input(self) -> Any | None:
        if hasattr(self._model, "n_features_in_"):
            return np.zeros((1, self._model.n_features_in_))
        return None


# -----------------------------------------------------------------------
# PyTorch
# -----------------------------------------------------------------------


class PyTorchPredictor(BasePredictor):
    """Wraps a PyTorch nn.Module."""

    def __init__(self, model: Any, name: str = "", version: str = "", device: str = "cpu") -> None:
        super().__init__(model, name, version)
        import torch

        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

    @property
    def framework(self) -> Framework:
        return Framework.PYTORCH

    def predict(self, input_data: Any) -> PredictionResult:
        import torch

        tensor = torch.as_tensor(np.asarray(input_data), dtype=torch.float32).to(self._device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        start = time.perf_counter()
        with torch.no_grad():
            output = self._model(tensor)
        latency = (time.perf_counter() - start) * 1000

        result = output.cpu().numpy()
        probas = None
        if result.shape[-1] > 1:
            import torch.nn.functional as F

            probas = F.softmax(output, dim=-1).cpu().numpy()[0].tolist()

        pred_val = result[0].tolist()
        if isinstance(pred_val, list) and len(pred_val) == 1:
            pred_val = pred_val[0]
        elif isinstance(pred_val, list):
            pred_val = int(np.argmax(pred_val))

        return PredictionResult(
            prediction=pred_val,
            probabilities=probas,
            latency_ms=round(latency, 3),
            model_name=self._name,
            model_version=self._version,
        )

    def predict_batch(self, inputs: list[Any]) -> list[PredictionResult]:
        import torch

        tensor = torch.as_tensor(np.asarray(inputs), dtype=torch.float32).to(self._device)

        start = time.perf_counter()
        with torch.no_grad():
            output = self._model(tensor)
        latency = (time.perf_counter() - start) * 1000

        results_np = output.cpu().numpy()
        per_item = round(latency / len(results_np), 3)
        out: list[PredictionResult] = []
        for row in results_np:
            pred_val = row.tolist()
            if isinstance(pred_val, list) and len(pred_val) == 1:
                pred_val = pred_val[0]
            elif isinstance(pred_val, list):
                pred_val = int(np.argmax(pred_val))
            out.append(
                PredictionResult(
                    prediction=pred_val,
                    latency_ms=per_item,
                    model_name=self._name,
                    model_version=self._version,
                )
            )
        return out


# -----------------------------------------------------------------------
# ONNX
# -----------------------------------------------------------------------


class ONNXPredictor(BasePredictor):
    """Wraps an ONNX Runtime InferenceSession."""

    @property
    def framework(self) -> Framework:
        return Framework.ONNX

    def predict(self, input_data: Any) -> PredictionResult:
        arr = np.asarray(input_data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        input_name = self._model.get_inputs()[0].name

        start = time.perf_counter()
        outputs = self._model.run(None, {input_name: arr})
        latency = (time.perf_counter() - start) * 1000

        pred = outputs[0][0]
        pred_val = pred.item() if hasattr(pred, "item") else pred

        probas = None
        if len(outputs) > 1:
            probas = outputs[1][0].tolist() if hasattr(outputs[1][0], "tolist") else None

        return PredictionResult(
            prediction=pred_val,
            probabilities=probas,
            latency_ms=round(latency, 3),
            model_name=self._name,
            model_version=self._version,
        )

    def predict_batch(self, inputs: list[Any]) -> list[PredictionResult]:
        arr = np.asarray(inputs, dtype=np.float32)
        input_name = self._model.get_inputs()[0].name

        start = time.perf_counter()
        outputs = self._model.run(None, {input_name: arr})
        latency = (time.perf_counter() - start) * 1000

        per_item = round(latency / len(outputs[0]), 3)
        return [
            PredictionResult(
                prediction=row.item() if hasattr(row, "item") else row,
                latency_ms=per_item,
                model_name=self._name,
                model_version=self._version,
            )
            for row in outputs[0]
        ]


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------


class PredictorFactory:
    """Instantiate the correct predictor based on the framework."""

    _MAP: dict[Framework, type[BasePredictor]] = {
        Framework.SKLEARN: SklearnPredictor,
        Framework.PYTORCH: PyTorchPredictor,
        Framework.ONNX: ONNXPredictor,
    }

    @classmethod
    def create(
        cls,
        model: Any,
        framework: Framework | str,
        name: str = "",
        version: str = "",
        **kwargs: Any,
    ) -> BasePredictor:
        fw = Framework(framework) if isinstance(framework, str) else framework
        predictor_cls = cls._MAP.get(fw)
        if predictor_cls is None:
            raise ValueError(f"Unsupported framework: {fw}")
        return predictor_cls(model=model, name=name, version=version, **kwargs)
