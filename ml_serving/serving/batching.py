"""Dynamic request batching for high-throughput model serving."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from ml_serving.registry.schemas import PredictionResult
from ml_serving.serving.model_server import ModelServer

logger = structlog.get_logger()


class _BatchItem:
    """A single request waiting to be batched."""

    __slots__ = ("input_data", "future", "submitted_at")

    def __init__(self, input_data: Any, future: asyncio.Future[PredictionResult]) -> None:
        self.input_data = input_data
        self.future = future
        self.submitted_at = time.monotonic()


class DynamicBatcher:
    """Collect individual prediction requests into batches.

    Requests are flushed when either:
      - the batch reaches *max_batch_size*, or
      - *timeout_ms* milliseconds have elapsed since the first item arrived.

    Each caller receives its individual result via an asyncio.Future.
    """

    def __init__(
        self,
        model_server: ModelServer,
        model_name: str,
        model_version: str | None = None,
        max_batch_size: int = 32,
        timeout_ms: int = 50,
    ) -> None:
        self._server = model_server
        self._model_name = model_name
        self._model_version = model_version
        self._max_batch_size = max_batch_size
        self._timeout_ms = timeout_ms

        self._queue: asyncio.Queue[_BatchItem] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background batch-processing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info(
            "batcher_started",
            model=self._model_name,
            max_batch=self._max_batch_size,
            timeout_ms=self._timeout_ms,
        )

    async def stop(self) -> None:
        """Gracefully stop the batcher, flushing remaining items."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def add_request(self, input_data: Any) -> PredictionResult:
        """Submit a single prediction request and wait for the result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PredictionResult] = loop.create_future()
        await self._queue.put(_BatchItem(input_data, future))
        return await future

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal batch loop
    # ------------------------------------------------------------------

    async def _process_loop(self) -> None:
        """Continuously collect and flush batches."""
        while self._running:
            batch = await self._collect_batch()
            if batch:
                await self._flush_batch(batch)

    async def _collect_batch(self) -> list[_BatchItem]:
        """Wait for items up to max_batch_size or timeout."""
        batch: list[_BatchItem] = []

        try:
            # Block until at least one item arrives
            first = await asyncio.wait_for(
                self._queue.get(), timeout=0.1
            )
            batch.append(first)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return batch

        # Collect more items up to batch size or timeout
        deadline = time.monotonic() + self._timeout_ms / 1000.0

        while len(batch) < self._max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break
            except asyncio.CancelledError:
                break

        return batch

    async def _flush_batch(self, batch: list[_BatchItem]) -> None:
        """Run a batched prediction and resolve each caller's future."""
        inputs = [item.input_data for item in batch]

        try:
            results = await asyncio.get_running_loop().run_in_executor(
                None,
                self._server.predict_batch,
                self._model_name,
                inputs,
                self._model_version,
            )

            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)

            logger.debug("batch_flushed", size=len(batch), model=self._model_name)

        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            logger.error("batch_failed", size=len(batch), error=str(exc))
