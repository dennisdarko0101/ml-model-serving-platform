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

## What was built (Phase 2-3, Steps 4-8)

### Step 4: A/B Testing + Canary Deployments

- **`ml_serving/routing/ab_testing.py`** — `ABTest` with consistent hashing for sticky sessions, `ABTestManager` for creating/managing tests, recording results, computing statistical significance (chi-squared), and concluding tests with winner declaration
- **`ml_serving/routing/canary.py`** — `CanaryDeployment` with configurable promotion steps (5% → 25% → 50% → 100%), automatic rollback on error threshold, per-model `CanaryMetrics` (error rate, latency percentiles)
- **`ml_serving/routing/shadow.py`** — `ShadowMode` runs a shadow model in a background thread (fire-and-forget), compares predictions via `ShadowReport` (agreement rate, divergences, latencies)

### Steps 5-6: Monitoring & Drift Detection

- **`ml_serving/monitoring/metrics.py`** — `MetricsCollector` wrapping prometheus_client: prediction count/latency/status counters, model load time, active models gauge, batch size histogram, drift score gauge, request queue size. Supports custom registries for testing
- **`ml_serving/monitoring/drift_detector.py`** — `DriftDetector` with PSI (Population Stability Index), KS test (Kolmogorov-Smirnov), and chi-squared for categorical features. Configurable thresholds, sliding window for current data, reference distribution storage per model
- **`ml_serving/monitoring/alerting.py`** — `AlertManager` with configurable rules (gt/lt/gte/lte/eq conditions), severity levels (info/warning/critical), cooldown periods, log and webhook actions, metric provider registration

### Step 7: FastAPI Server

- **`ml_serving/api/schemas.py`** — Pydantic request/response models for all endpoints
- **`ml_serving/api/middleware.py`** — Request logging middleware with correlation ID injection
- **`ml_serving/api/main.py`** — FastAPI app with lifespan management, CORS, shared `AppState` dataclass, test-friendly `set_app_state()` override
- **`ml_serving/api/routes/predict.py`** — Single and batch prediction with A/B test and canary routing integration
- **`ml_serving/api/routes/models.py`** — Model CRUD: register, list, get, promote, archive, load, unload
- **`ml_serving/api/routes/experiments.py`** — A/B test and canary lifecycle endpoints
- **`ml_serving/api/routes/monitoring.py`** — Metrics summary, drift reports, alerts, Prometheus scrape endpoint
- **`ml_serving/api/routes/health.py`** — Detailed health check with per-model status

### Step 8: Docker + Grafana

- **`docker/Dockerfile`** — Multi-stage build, non-root user, healthcheck
- **`docker/docker-compose.yml`** — API + Prometheus + Grafana (3 services) with volumes
- **`docker/prometheus.yml`** — Prometheus scrape config targeting the API service
- **`grafana/dashboards/model_serving.json`** — Pre-built dashboard: prediction latency (p50/p95/p99), request throughput, error rate, active models, drift scores

### Tests (99 new, 181 total)

- **`tests/unit/test_routing.py`** — 31 tests: A/B routing, consistent hashing, canary promotion/rollback/auto-promote, shadow mode
- **`tests/unit/test_monitoring.py`** — 35 tests: metrics recording, drift detection (PSI, KS, chi-squared), alerting rules/cooldown/webhooks
- **`tests/integration/test_api.py`** — 18 tests: all API endpoints (health, models, predict, experiments, monitoring)
- **`tests/integration/test_ab_testing.py`** — 10 tests: full A/B test lifecycle, sticky sessions, traffic distribution, statistical significance
- **`tests/integration/test_pipeline.py`** — 5 tests: predict → route → monitor end-to-end flow

## Key Design Decisions

1. **JSON file-based registry** — No database dependency for Phase 1. Registry data is human-readable and easy to debug. Can be swapped for SQLite/Postgres later.
2. **LRU eviction** — `OrderedDict`-based LRU keeps memory bounded. Every prediction refreshes the model's position.
3. **Framework abstraction** — `BasePredictor` + `PredictorFactory` lets the server be framework-agnostic. Adding a new framework requires one new predictor class.
4. **Async batching** — `DynamicBatcher` uses `asyncio.Queue` and `Future`s so callers don't need to know about batching. Transparent throughput improvement for GPU models.
5. **Composable preprocessing** — Each model can have its own pipeline. Steps are plain functions, easy to test independently.
6. **Consistent hashing for A/B tests** — SHA-256 based routing ensures the same request_id always hits the same model (sticky sessions) without server-side session state.
7. **No scipy dependency** — PSI, KS, and chi-squared tests use numpy + math.erfc approximations. Keeps the dependency tree lean.
8. **Test-friendly AppState** — `set_app_state()` lets tests inject custom state without hitting the lifespan. Custom Prometheus registries prevent metric name collisions across tests.
9. **Shadow mode fire-and-forget** — Background thread for shadow predictions ensures zero impact on primary response time.
10. **Canary promotion steps** — Fixed steps (5% → 25% → 50% → 100%) prevent accidental full rollout. Auto-promote checks error rate before advancing.

## What's Next (Phase 4+)

- **Database-backed registry** — Replace JSON files with SQLite or Postgres for concurrent access
- **Cloud storage** — S3/GCS backends for model artifacts (boto3 and google-cloud-storage already in deps)
- **Authentication** — API key or OAuth2 for the REST API
- **Model warm-up benchmarks** — Track inference latency during warmup and expose as metrics
- **Kubernetes deployment** — Helm charts, HPA based on queue size, model-specific resource limits
- **Streaming predictions** — WebSocket endpoint for real-time inference
