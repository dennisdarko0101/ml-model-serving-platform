"""Tests for the dynamic batching system."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from ml_serving.config.settings import Settings
from ml_serving.registry.model_registry import ModelRegistry
from ml_serving.registry.model_store import ModelStore
from ml_serving.registry.schemas import Framework
from ml_serving.serving.batching import DynamicBatcher
from ml_serving.serving.model_server import ModelServer


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
        MAX_LOADED_MODELS=5,
    )


@pytest.fixture()
def iris_clf() -> RandomForestClassifier:
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture()
def server(settings: Settings, iris_clf) -> ModelServer:
    registry = ModelRegistry(settings)
    store = ModelStore(settings)
    srv = ModelServer(registry=registry, store=store, settings=settings)
    srv.load_model("iris", "v1", Framework.SKLEARN, model_object=iris_clf)
    return srv


@pytest.fixture()
def batcher(server: ModelServer) -> DynamicBatcher:
    return DynamicBatcher(
        model_server=server,
        model_name="iris",
        model_version="v1",
        max_batch_size=4,
        timeout_ms=100,
    )


# ===================================================================
# Tests
# ===================================================================


class TestDynamicBatcher:
    @pytest.mark.asyncio
    async def test_single_request(self, batcher: DynamicBatcher):
        await batcher.start()
        try:
            result = await batcher.add_request([5.1, 3.5, 1.4, 0.2])
            assert result.prediction in (0, 1, 2)
            assert result.model_name == "iris"
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_batch_collects_multiple(self, batcher: DynamicBatcher):
        """Send multiple requests concurrently; they should be batched."""
        await batcher.start()
        try:
            inputs = [
                [5.1, 3.5, 1.4, 0.2],
                [6.7, 3.1, 4.7, 1.5],
                [5.9, 3.0, 5.1, 1.8],
            ]
            results = await asyncio.gather(
                *[batcher.add_request(inp) for inp in inputs]
            )
            assert len(results) == 3
            for r in results:
                assert r.prediction in (0, 1, 2)
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_max_batch_size_flush(self, server: ModelServer):
        """Batch should flush when max_batch_size is reached."""
        batcher = DynamicBatcher(
            model_server=server,
            model_name="iris",
            model_version="v1",
            max_batch_size=2,
            timeout_ms=5000,  # Long timeout — should flush by size first
        )
        await batcher.start()
        try:
            inputs = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5]]
            results = await asyncio.gather(
                *[batcher.add_request(inp) for inp in inputs]
            )
            assert len(results) == 2
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_timeout_flush(self, server: ModelServer):
        """A single request should still be flushed after timeout."""
        batcher = DynamicBatcher(
            model_server=server,
            model_name="iris",
            model_version="v1",
            max_batch_size=100,  # Very large — should flush by timeout
            timeout_ms=50,
        )
        await batcher.start()
        try:
            result = await batcher.add_request([5.1, 3.5, 1.4, 0.2])
            assert result.prediction in (0, 1, 2)
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, batcher: DynamicBatcher):
        """Many concurrent requests should all resolve."""
        await batcher.start()
        try:
            inputs = [[5.1, 3.5, 1.4, 0.2]] * 10
            results = await asyncio.gather(
                *[batcher.add_request(inp) for inp in inputs]
            )
            assert len(results) == 10
            for r in results:
                assert r.prediction in (0, 1, 2)
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_start_stop(self, batcher: DynamicBatcher):
        assert not batcher.is_running
        await batcher.start()
        assert batcher.is_running
        await batcher.stop()
        assert not batcher.is_running

    @pytest.mark.asyncio
    async def test_double_start(self, batcher: DynamicBatcher):
        await batcher.start()
        await batcher.start()  # Should be idempotent
        assert batcher.is_running
        await batcher.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, batcher: DynamicBatcher):
        await batcher.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_results_match_inputs(self, batcher: DynamicBatcher):
        """Each caller gets back the correct result for their input."""
        await batcher.start()
        try:
            # Class 0 sample and class 2 sample
            r0 = await batcher.add_request([5.1, 3.5, 1.4, 0.2])
            r2 = await batcher.add_request([6.3, 3.3, 6.0, 2.5])
            assert r0.prediction == 0
            assert r2.prediction == 2
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_different_batch_sizes(self, server: ModelServer):
        """Verify batcher works with batch_size=1."""
        batcher = DynamicBatcher(
            model_server=server,
            model_name="iris",
            model_version="v1",
            max_batch_size=1,
            timeout_ms=100,
        )
        await batcher.start()
        try:
            result = await batcher.add_request([5.1, 3.5, 1.4, 0.2])
            assert result.prediction in (0, 1, 2)
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_sequential_batches(self, batcher: DynamicBatcher):
        """Multiple sequential requests each get valid results."""
        await batcher.start()
        try:
            for _ in range(5):
                result = await batcher.add_request([5.1, 3.5, 1.4, 0.2])
                assert result.prediction in (0, 1, 2)
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_prediction_has_metadata(self, batcher: DynamicBatcher):
        await batcher.start()
        try:
            result = await batcher.add_request([5.1, 3.5, 1.4, 0.2])
            assert result.model_name == "iris"
            assert result.model_version == "v1"
            assert result.latency_ms >= 0
        finally:
            await batcher.stop()
