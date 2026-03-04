"""Tests for serving layer — predictors, preprocessor, and model server."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from ml_serving.config.settings import Settings
from ml_serving.registry.model_registry import ModelRegistry
from ml_serving.registry.model_store import ModelStore
from ml_serving.registry.schemas import Framework, ModelStatus
from ml_serving.serving.model_server import ModelServer
from ml_serving.serving.predictor import (
    ONNXPredictor,
    PredictorFactory,
    PyTorchPredictor,
    SklearnPredictor,
)
from ml_serving.serving.preprocessor import (
    PreprocessingPipeline,
    encode_categorical,
    handle_missing,
    normalize_numeric,
    to_numpy,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture()
def settings(tmp_dir: Path) -> Settings:
    return Settings(
        MODEL_STORE_PATH=tmp_dir / "artifacts",
        REGISTRY_PATH=tmp_dir / "registry",
        MAX_LOADED_MODELS=3,
    )


@pytest.fixture()
def iris_clf() -> RandomForestClassifier:
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture()
def reg_model() -> LinearRegression:
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    m = LinearRegression()
    m.fit(X, y)
    return m


@pytest.fixture()
def server(settings: Settings) -> ModelServer:
    registry = ModelRegistry(settings)
    store = ModelStore(settings)
    return ModelServer(registry=registry, store=store, settings=settings)


# ===================================================================
# SklearnPredictor
# ===================================================================


class TestSklearnPredictor:
    def test_predict_single(self, iris_clf):
        pred = SklearnPredictor(iris_clf, name="iris", version="v1")
        result = pred.predict([5.1, 3.5, 1.4, 0.2])
        assert result.prediction in (0, 1, 2)
        assert result.latency_ms > 0
        assert result.model_name == "iris"

    def test_predict_with_probabilities(self, iris_clf):
        pred = SklearnPredictor(iris_clf, name="iris", version="v1")
        result = pred.predict([5.1, 3.5, 1.4, 0.2])
        assert result.probabilities is not None
        assert len(result.probabilities) == 3
        assert abs(sum(result.probabilities) - 1.0) < 1e-6

    def test_predict_batch(self, iris_clf):
        pred = SklearnPredictor(iris_clf, name="iris", version="v1")
        inputs = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5]]
        results = pred.predict_batch(inputs)
        assert len(results) == 2
        for r in results:
            assert r.prediction in (0, 1, 2)

    def test_predict_regression(self, reg_model):
        pred = SklearnPredictor(reg_model, name="reg", version="v1")
        result = pred.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(result.prediction, float)
        assert result.probabilities is None

    def test_get_model_info(self, iris_clf):
        pred = SklearnPredictor(iris_clf, name="iris", version="v1")
        info = pred.get_model_info()
        assert info["name"] == "iris"
        assert info["framework"] == "sklearn"

    def test_warmup(self, iris_clf):
        pred = SklearnPredictor(iris_clf, name="iris", version="v1")
        pred.warmup()  # Should not raise

    def test_factory_creates_sklearn(self, iris_clf):
        pred = PredictorFactory.create(iris_clf, Framework.SKLEARN, name="iris")
        assert isinstance(pred, SklearnPredictor)


# ===================================================================
# PyTorchPredictor (mocked)
# ===================================================================


class TestPyTorchPredictor:
    def test_predict_mocked(self):
        import torch

        model = MagicMock(spec=torch.nn.Module)
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        output_tensor = torch.tensor([[0.1, 0.9, 0.0]])
        model.__call__ = MagicMock(return_value=output_tensor)
        model.return_value = output_tensor

        pred = PyTorchPredictor(model, name="torch_model", version="v1", device="cpu")
        result = pred.predict([1.0, 2.0, 3.0])
        assert result.model_name == "torch_model"

    def test_factory_creates_pytorch(self):
        import torch

        model = MagicMock(spec=torch.nn.Module)
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        pred = PredictorFactory.create(model, Framework.PYTORCH, name="pt")
        assert isinstance(pred, PyTorchPredictor)


# ===================================================================
# ONNXPredictor (mocked)
# ===================================================================


class TestONNXPredictor:
    def test_predict_mocked(self):
        session = MagicMock()
        inp = MagicMock()
        inp.name = "input"
        session.get_inputs.return_value = [inp]
        session.run.return_value = [np.array([1.0])]

        pred = ONNXPredictor(session, name="onnx_model", version="v1")
        result = pred.predict([1.0, 2.0])
        assert result.prediction == 1.0
        assert result.model_name == "onnx_model"

    def test_predict_batch_mocked(self):
        session = MagicMock()
        inp = MagicMock()
        inp.name = "input"
        session.get_inputs.return_value = [inp]
        session.run.return_value = [np.array([1.0, 2.0])]

        pred = ONNXPredictor(session, name="onnx_model", version="v1")
        results = pred.predict_batch([[1.0], [2.0]])
        assert len(results) == 2

    def test_factory_creates_onnx(self):
        session = MagicMock()
        pred = PredictorFactory.create(session, Framework.ONNX, name="ox")
        assert isinstance(pred, ONNXPredictor)


# ===================================================================
# PredictorFactory
# ===================================================================


class TestPredictorFactory:
    def test_unsupported_framework_raises(self):
        with pytest.raises(ValueError):
            PredictorFactory.create(None, "unknown")


# ===================================================================
# PreprocessingPipeline
# ===================================================================


class TestPreprocessingPipeline:
    def test_add_and_process(self):
        pipe = PreprocessingPipeline()
        pipe.add_step("double", lambda x: [v * 2 for v in x])
        result = pipe.process([1, 2, 3])
        assert result == [2, 4, 6]

    def test_chain_steps(self):
        pipe = PreprocessingPipeline()
        pipe.add_step("add_one", lambda x: [v + 1 for v in x])
        pipe.add_step("double", lambda x: [v * 2 for v in x])
        result = pipe.process([1, 2, 3])
        assert result == [4, 6, 8]

    def test_step_names(self):
        pipe = PreprocessingPipeline()
        pipe.add_step("a", lambda x: x).add_step("b", lambda x: x)
        assert pipe.steps == ["a", "b"]
        assert len(pipe) == 2

    def test_validate_schema_pass(self):
        fn = validate_schema(["x", "y"])
        result = fn({"x": 1, "y": 2, "z": 3})
        assert result == {"x": 1, "y": 2, "z": 3}

    def test_validate_schema_fail(self):
        fn = validate_schema(["x", "y"])
        with pytest.raises(ValueError, match="Missing"):
            fn({"x": 1})

    def test_normalize_numeric(self):
        fn = normalize_numeric(means=[10.0, 20.0], stds=[2.0, 5.0])
        result = fn([12.0, 25.0])
        assert result == pytest.approx([1.0, 1.0])

    def test_normalize_zero_std(self):
        fn = normalize_numeric(means=[5.0], stds=[0.0])
        result = fn([10.0])
        assert result == pytest.approx([5.0])  # (10-5)/1 = 5

    def test_encode_categorical(self):
        fn = encode_categorical({"color": {"red": 0, "blue": 1}})
        result = fn({"color": "red", "size": 10})
        assert result["color"] == 0
        assert result["size"] == 10

    def test_encode_categorical_unknown(self):
        fn = encode_categorical({"color": {"red": 0}})
        with pytest.raises(ValueError, match="Unknown category"):
            fn({"color": "green"})

    def test_handle_missing_mean(self):
        fn = handle_missing(strategy="mean")
        result = fn([1.0, float("nan"), 3.0])
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_handle_missing_zero(self):
        fn = handle_missing(strategy="zero")
        result = fn([1.0, float("nan")])
        assert result == pytest.approx([1.0, 0.0])

    def test_handle_missing_value(self):
        fn = handle_missing(strategy="value", fill_value=-1.0)
        result = fn([float("nan")])
        assert result == pytest.approx([-1.0])

    def test_handle_missing_no_nans(self):
        fn = handle_missing(strategy="mean")
        result = fn([1.0, 2.0])
        assert result == pytest.approx([1.0, 2.0])

    def test_to_numpy(self):
        fn = to_numpy(dtype="float32")
        result = fn([1, 2, 3])
        assert result.shape == (1, 3)
        assert result.dtype == np.float32

    def test_pipeline_error_propagation(self):
        pipe = PreprocessingPipeline()
        pipe.add_step("fail", lambda x: 1 / 0)
        with pytest.raises(ValueError, match="Preprocessing failed at step 'fail'"):
            pipe.process([1])


# ===================================================================
# ModelServer
# ===================================================================


class TestModelServer:
    def test_load_and_predict(self, server: ModelServer, iris_clf):
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
        result = server.predict("iris", [5.1, 3.5, 1.4, 0.2], version="v1")
        assert result.prediction in (0, 1, 2)

    def test_load_multiple_models(self, server: ModelServer, iris_clf, reg_model):
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
        server.load_model("reg", "v1", Framework.SKLEARN, model_object=reg_model)
        loaded = server.get_loaded_models()
        assert "iris:v1" in loaded
        assert "reg:v1" in loaded

    def test_unload_model(self, server: ModelServer, iris_clf):
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
        server.unload_model("iris", "v1")
        assert "iris:v1" not in server.get_loaded_models()

    def test_lru_eviction(self, server: ModelServer, iris_clf, reg_model):
        """MAX_LOADED_MODELS=3 in fixture, so loading a 4th evicts the LRU."""
        clf2 = RandomForestClassifier(n_estimators=5, random_state=0)
        clf2.fit(*load_iris(return_X_y=True))
        clf3 = RandomForestClassifier(n_estimators=5, random_state=1)
        clf3.fit(*load_iris(return_X_y=True))

        server.load_model("m1", "v1", Framework.SKLEARN, model_object=iris_clf)
        server.load_model("m2", "v1", Framework.SKLEARN, model_object=reg_model)
        server.load_model("m3", "v1", Framework.SKLEARN, model_object=clf2)

        assert len(server.get_loaded_models()) == 3

        # Loading a 4th should evict m1 (LRU)
        server.load_model("m4", "v1", Framework.SKLEARN, model_object=clf3)
        loaded = server.get_loaded_models()
        assert "m1:v1" not in loaded
        assert "m4:v1" in loaded
        assert len(loaded) == 3

    def test_lru_access_refreshes(self, server: ModelServer, iris_clf, reg_model):
        clf2 = RandomForestClassifier(n_estimators=5, random_state=0)
        clf2.fit(*load_iris(return_X_y=True))
        clf3 = RandomForestClassifier(n_estimators=5, random_state=1)
        clf3.fit(*load_iris(return_X_y=True))

        server.load_model("m1", "v1", Framework.SKLEARN, model_object=iris_clf)
        server.load_model("m2", "v1", Framework.SKLEARN, model_object=reg_model)
        server.load_model("m3", "v1", Framework.SKLEARN, model_object=clf2)

        # Access m1 to refresh it
        server.predict("m1", [5.1, 3.5, 1.4, 0.2], version="v1")

        # Now m2 is the LRU
        server.load_model("m4", "v1", Framework.SKLEARN, model_object=clf3)
        loaded = server.get_loaded_models()
        assert "m1:v1" in loaded
        assert "m2:v1" not in loaded

    def test_predict_unloaded_raises(self, server: ModelServer):
        with pytest.raises(KeyError, match="not loaded"):
            server.predict("nope", [1.0])

    def test_model_status(self, server: ModelServer, iris_clf):
        assert server.get_model_status("iris", "v1") == ModelStatus.REGISTERED
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
        assert server.get_model_status("iris", "v1") == ModelStatus.LOADED

    def test_is_healthy(self, server: ModelServer, iris_clf):
        assert not server.is_healthy("iris", "v1")
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
        assert server.is_healthy("iris", "v1")

    def test_predict_batch(self, server: ModelServer, iris_clf):
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
        inputs = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5]]
        results = server.predict_batch("iris", inputs, version="v1")
        assert len(results) == 2

    def test_load_same_model_twice_is_idempotent(self, server: ModelServer, iris_clf):
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
        assert server.get_loaded_models().count("iris:v1") == 1

    def test_predict_without_version(self, server: ModelServer, iris_clf):
        """If no version given, find any loaded version for the model."""
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
        result = server.predict("iris", [5.1, 3.5, 1.4, 0.2])
        assert result.prediction in (0, 1, 2)

    def test_register_pipeline(self, server: ModelServer, iris_clf):
        pipe = PreprocessingPipeline()
        pipe.add_step("double", lambda x: [v * 2 for v in x])
        server.register_pipeline("iris", pipe)
        server.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
        # Pipeline doubles the input features before prediction
        result = server.predict("iris", [2.5, 1.75, 0.7, 0.1], version="v1")
        assert result.prediction in (0, 1, 2)
