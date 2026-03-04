# Handoff — ML Model Serving Platform

## What was built (Phase 1, Steps 1-3)

### Step 1: Scaffolding + Model Registry

- **`pyproject.toml`** — Python 3.11+, all production and dev dependencies
- **`ml_serving/config/settings.py`** — Pydantic Settings with env-var loading for model store paths, batching config, cloud settings, and server config
- **`ml_serving/registry/schemas.py`** — Core data models: `ModelMetadata`, `ModelVersion`, `ModelStatus` enum, `ModelStage` enum, `Framework` enum, `PredictionResult`
- **`ml_serving/registry/model_store.py`** — `ModelStore` class for saving/loading model artifacts across sklearn (joblib), PyTorch (state_dict), and ONNX (onnxruntime). Local file storage, framework auto-detection
- **`ml_serving/registry/model_registry.py`** — `ModelRegistry` class with JSON file-based storage. Supports register, get, get_latest, get_production, promote (with auto-demotion), list_models, list_versions, compare_versions

### Step 2: Model Server + Prediction Pipeline

- **`ml_serving/serving/predictor.py`** — `BasePredictor` ABC with `SklearnPredictor`, `PyTorchPredictor`, `ONNXPredictor` implementations. `PredictorFactory` for framework-based instantiation. Each predictor handles framework-specific conversions (numpy, tensors, ONNX sessions)
- **`ml_serving/serving/preprocessor.py`** — `PreprocessingPipeline` with fluent chaining. Built-in steps: `validate_schema`, `normalize_numeric`, `encode_categorical`, `handle_missing`, `to_numpy`
- **`ml_serving/serving/model_server.py`** — `ModelServer` class: multi-model serving, LRU eviction (configurable `MAX_LOADED_MODELS`), thread-safe loading/unloading, per-model preprocessing pipelines, warmup on load, health checking

### Step 3: Dynamic Batching

- **`ml_serving/serving/batching.py`** — `DynamicBatcher` class: asyncio-based request collection with configurable `max_batch_size` and `timeout_ms`. Flushes on size or timeout, returns individual `PredictionResult` futures to each caller

### Tests

- **`tests/unit/test_registry.py`** — 20 tests covering ModelStore (save/load/delete/list/detect) and ModelRegistry (register/get/promote/compare/lifecycle)
- **`tests/unit/test_serving.py`** — 24 tests covering all three predictor types, preprocessing pipeline and built-in steps, model server with LRU eviction
- **`tests/unit/test_batching.py`** — 12 async tests covering single/batch/concurrent requests, size and timeout flush triggers, start/stop lifecycle

### Supporting Files

- **`models/sample/train_sample_models.py`** — Creates iris classifier + regression model
- **`.env.example`**, **`Makefile`**, **`.gitignore`**, **`LICENSE`**, **`CONTRIBUTING.md`**
- **`.github/workflows/ci.yml`** — Lint + test matrix (3.11, 3.12)

## Key Design Decisions

1. **JSON file-based registry** — No database dependency for Phase 1. Registry data is human-readable and easy to debug. Can be swapped for SQLite/Postgres later.
2. **LRU eviction** — `OrderedDict`-based LRU keeps memory bounded. Every prediction refreshes the model's position.
3. **Framework abstraction** — `BasePredictor` + `PredictorFactory` lets the server be framework-agnostic. Adding a new framework requires one new predictor class.
4. **Async batching** — `DynamicBatcher` uses `asyncio.Queue` and `Future`s so callers don't need to know about batching. Transparent throughput improvement for GPU models.
5. **Composable preprocessing** — Each model can have its own pipeline. Steps are plain functions, easy to test independently.

## What's Next (Phase 2)

- **A/B Testing / Traffic Routing** — Route requests across model versions by weight, sticky sessions, gradual rollout
- **Drift Detection** — Statistical tests (PSI, KS, chi-squared) on input features and prediction distributions
- **Prometheus Monitoring** — Latency histograms, prediction counters, model health gauges
- **FastAPI Endpoints** — REST API for predictions, model management, health checks
