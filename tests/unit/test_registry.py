"""Tests for model registry and model store."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from ml_serving.config.settings import Settings
from ml_serving.registry.model_registry import ModelRegistry
from ml_serving.registry.model_store import ModelStore
from ml_serving.registry.schemas import Framework, ModelStage, ModelStatus


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
    )


@pytest.fixture()
def store(settings: Settings) -> ModelStore:
    return ModelStore(settings)


@pytest.fixture()
def registry(settings: Settings) -> ModelRegistry:
    return ModelRegistry(settings)


@pytest.fixture()
def iris_clf() -> RandomForestClassifier:
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture()
def regression_model() -> LinearRegression:
    X = np.random.RandomState(42).randn(100, 3)
    y = X @ [1.5, -2.0, 0.5] + 1.0
    reg = LinearRegression()
    reg.fit(X, y)
    return reg


# ===================================================================
# ModelStore tests
# ===================================================================


class TestModelStore:
    def test_save_and_load_sklearn(self, store: ModelStore, iris_clf):
        path = store.save_model(iris_clf, "iris", "v1", Framework.SKLEARN)
        assert Path(path).exists()

        loaded = store.load_model("iris", "v1", Framework.SKLEARN)
        X = np.array([[5.1, 3.5, 1.4, 0.2]])
        np.testing.assert_array_equal(loaded.predict(X), iris_clf.predict(X))

    def test_save_and_load_regression(self, store: ModelStore, regression_model):
        store.save_model(regression_model, "regressor", "v1", Framework.SKLEARN)
        loaded = store.load_model("regressor", "v1", Framework.SKLEARN)

        X = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(loaded.predict(X), regression_model.predict(X))

    def test_delete_model(self, store: ModelStore, iris_clf):
        store.save_model(iris_clf, "iris", "v1", Framework.SKLEARN)
        assert store.artifact_exists("iris", "v1", Framework.SKLEARN)

        store.delete_model("iris", "v1")
        assert not store.artifact_exists("iris", "v1", Framework.SKLEARN)

    def test_list_artifacts(self, store: ModelStore, iris_clf, regression_model):
        store.save_model(iris_clf, "mymodel", "v1", Framework.SKLEARN)
        store.save_model(regression_model, "mymodel", "v2", Framework.SKLEARN)
        artifacts = store.list_artifacts("mymodel")
        assert len(artifacts) == 2

    def test_list_artifacts_empty(self, store: ModelStore):
        assert store.list_artifacts("nonexistent") == []

    def test_load_nonexistent_raises(self, store: ModelStore):
        with pytest.raises(FileNotFoundError):
            store.load_model("nope", "v0", Framework.SKLEARN)

    def test_detect_framework_sklearn(self, store: ModelStore, iris_clf):
        assert ModelStore.detect_framework(iris_clf) == Framework.SKLEARN

    def test_detect_framework_unknown_raises(self, store: ModelStore):
        with pytest.raises(ValueError, match="Cannot detect framework"):
            ModelStore.detect_framework(42)

    def test_artifact_exists(self, store: ModelStore, iris_clf):
        assert not store.artifact_exists("iris", "v1", Framework.SKLEARN)
        store.save_model(iris_clf, "iris", "v1", Framework.SKLEARN)
        assert store.artifact_exists("iris", "v1", Framework.SKLEARN)

    def test_overwrite_model(self, store: ModelStore, iris_clf, regression_model):
        store.save_model(iris_clf, "m", "v1", Framework.SKLEARN)
        store.save_model(regression_model, "m", "v1", Framework.SKLEARN)
        loaded = store.load_model("m", "v1", Framework.SKLEARN)
        # Should be the regression model now
        assert hasattr(loaded, "coef_")


# ===================================================================
# ModelRegistry tests
# ===================================================================


class TestModelRegistry:
    def test_register_model(self, registry: ModelRegistry):
        meta = registry.register("iris", "v1", Framework.SKLEARN, description="Iris clf")
        assert meta.name == "iris"
        assert meta.version == "v1"
        assert meta.stage == ModelStage.STAGING

    def test_get_model(self, registry: ModelRegistry):
        registry.register("iris", "v1", Framework.SKLEARN)
        meta = registry.get("iris", "v1")
        assert meta.name == "iris"
        assert meta.version == "v1"

    def test_get_nonexistent_raises(self, registry: ModelRegistry):
        with pytest.raises(KeyError):
            registry.get("nope", "v0")

    def test_get_latest(self, registry: ModelRegistry):
        registry.register("m", "v1", Framework.SKLEARN)
        registry.register("m", "v2", Framework.SKLEARN)
        latest = registry.get_latest("m")
        assert latest.version == "v2"

    def test_get_latest_no_versions_raises(self, registry: ModelRegistry):
        with pytest.raises(KeyError, match="No versions"):
            registry.get_latest("empty")

    def test_get_production(self, registry: ModelRegistry):
        registry.register("m", "v1", Framework.SKLEARN)
        registry.promote("m", "v1", ModelStage.PRODUCTION)
        prod = registry.get_production("m")
        assert prod.version == "v1"
        assert prod.stage == ModelStage.PRODUCTION

    def test_get_production_none_raises(self, registry: ModelRegistry):
        registry.register("m", "v1", Framework.SKLEARN)
        with pytest.raises(KeyError, match="No production"):
            registry.get_production("m")

    def test_promote_to_production(self, registry: ModelRegistry):
        registry.register("m", "v1", Framework.SKLEARN)
        meta = registry.promote("m", "v1", ModelStage.PRODUCTION)
        assert meta.stage == ModelStage.PRODUCTION

    def test_promote_demotes_old_production(self, registry: ModelRegistry):
        registry.register("m", "v1", Framework.SKLEARN)
        registry.register("m", "v2", Framework.SKLEARN)
        registry.promote("m", "v1", ModelStage.PRODUCTION)
        registry.promote("m", "v2", ModelStage.PRODUCTION)

        v1 = registry.get("m", "v1")
        v2 = registry.get("m", "v2")
        assert v1.stage == ModelStage.ARCHIVED
        assert v2.stage == ModelStage.PRODUCTION

    def test_promote_to_archived(self, registry: ModelRegistry):
        registry.register("m", "v1", Framework.SKLEARN)
        meta = registry.promote("m", "v1", ModelStage.ARCHIVED)
        assert meta.stage == ModelStage.ARCHIVED

    def test_list_models(self, registry: ModelRegistry):
        registry.register("a", "v1", Framework.SKLEARN)
        registry.register("b", "v1", Framework.PYTORCH)
        models = registry.list_models()
        names = [m.name for m in models]
        assert "a" in names
        assert "b" in names

    def test_list_models_empty(self, registry: ModelRegistry):
        assert registry.list_models() == []

    def test_list_versions(self, registry: ModelRegistry):
        registry.register("m", "v1", Framework.SKLEARN)
        registry.register("m", "v2", Framework.SKLEARN)
        registry.register("m", "v3", Framework.SKLEARN)
        versions = registry.list_versions("m")
        assert len(versions) == 3

    def test_list_versions_empty(self, registry: ModelRegistry):
        assert registry.list_versions("nope") == []

    def test_compare_versions(self, registry: ModelRegistry):
        registry.register(
            "m", "v1", Framework.SKLEARN, metrics={"accuracy": 0.85, "f1": 0.80}
        )
        registry.register(
            "m", "v2", Framework.SKLEARN, metrics={"accuracy": 0.90, "f1": 0.88}
        )
        cmp = registry.compare_versions("m", "v1", "v2")
        assert cmp["model"] == "m"
        assert cmp["metrics"]["accuracy"]["diff"] == pytest.approx(0.05)
        assert cmp["metrics"]["f1"]["diff"] == pytest.approx(0.08)

    def test_compare_versions_missing_metric(self, registry: ModelRegistry):
        registry.register("m", "v1", Framework.SKLEARN, metrics={"accuracy": 0.9})
        registry.register("m", "v2", Framework.SKLEARN, metrics={"f1": 0.8})
        cmp = registry.compare_versions("m", "v1", "v2")
        assert cmp["metrics"]["accuracy"]["v2"] is None
        assert cmp["metrics"]["f1"]["v1"] is None

    def test_register_with_tags_and_metrics(self, registry: ModelRegistry):
        meta = registry.register(
            "m",
            "v1",
            Framework.SKLEARN,
            tags=["prod-ready", "classification"],
            metrics={"accuracy": 0.95},
        )
        assert "prod-ready" in meta.tags
        assert meta.metrics["accuracy"] == 0.95

    def test_delete_version(self, registry: ModelRegistry):
        registry.register("m", "v1", Framework.SKLEARN)
        registry.delete("m", "v1")
        with pytest.raises(KeyError):
            registry.get("m", "v1")

    def test_register_with_artifact_path(self, registry: ModelRegistry):
        meta = registry.register(
            "m", "v1", Framework.SKLEARN, artifact_path="/data/model.joblib"
        )
        assert meta.artifact_path == "/data/model.joblib"

    def test_version_stage_lifecycle(self, registry: ModelRegistry):
        registry.register("m", "v1", Framework.SKLEARN)

        meta = registry.get("m", "v1")
        assert meta.stage == ModelStage.STAGING

        registry.promote("m", "v1", ModelStage.PRODUCTION)
        meta = registry.get("m", "v1")
        assert meta.stage == ModelStage.PRODUCTION

        registry.promote("m", "v1", ModelStage.ARCHIVED)
        meta = registry.get("m", "v1")
        assert meta.stage == ModelStage.ARCHIVED
