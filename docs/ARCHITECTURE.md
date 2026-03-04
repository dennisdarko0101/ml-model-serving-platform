# Architecture — ML Model Serving Platform

## System Overview

```
                                    ┌─────────────────────────────────────────────────────────┐
                                    │                     FastAPI Server                       │
                                    │                                                         │
  Client ──▶ POST /api/v1/predict ──▶  Middleware (logging, correlation ID)                   │
                                    │      │                                                  │
                                    │      ▼                                                  │
                                    │  Routing Layer                                          │
                                    │  ┌──────────┐  ┌────────┐  ┌────────┐                  │
                                    │  │ A/B Test  │  │ Canary │  │ Shadow │                  │
                                    │  └────┬─────┘  └───┬────┘  └───┬────┘                  │
                                    │       └────────────┼───────────┘                        │
                                    │                    ▼                                     │
                                    │  Model Server (LRU cache, thread-safe)                  │
                                    │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
                                    │  │ sklearn  │  │ PyTorch  │  │   ONNX   │              │
                                    │  │ Predictor│  │ Predictor│  │ Predictor│              │
                                    │  └──────────┘  └──────────┘  └──────────┘              │
                                    │       │              │              │                    │
                                    │       └──────────────┼──────────────┘                   │
                                    │                      ▼                                  │
                                    │  Monitoring Layer                                       │
                                    │  ┌──────────┐  ┌───────────┐  ┌──────────┐             │
                                    │  │ Metrics  │  │   Drift   │  │ Alerting │             │
                                    │  │Collector │  │ Detector  │  │ Manager  │             │
                                    │  └────┬─────┘  └───────────┘  └──────────┘             │
                                    │       │                                                 │
                                    └───────┼─────────────────────────────────────────────────┘
                                            ▼
                                    Prometheus ──▶ Grafana
```

## Module Architecture

### Registry Layer (`ml_serving/registry/`)

- **ModelStore** — Persists model artifacts to disk (joblib, state_dict, .onnx)
- **ModelRegistry** — JSON-file metadata store with version management, stage promotion
- **Schemas** — Pydantic models: `Framework`, `ModelStage`, `ModelStatus`, `ModelMetadata`, `PredictionResult`

### Serving Layer (`ml_serving/serving/`)

- **BasePredictor** (ABC) — Framework-agnostic prediction interface
  - `SklearnPredictor` — numpy-based sklearn inference
  - `PyTorchPredictor` — tensor-based with device management
  - `ONNXPredictor` — onnxruntime session
- **PredictorFactory** — Maps `Framework` enum → concrete predictor class
- **PreprocessingPipeline** — Composable chain of named transform functions
- **ModelServer** — Thread-safe multi-model server with LRU eviction (OrderedDict)
- **DynamicBatcher** — asyncio queue-based request batching with size/timeout flush

### Routing Layer (`ml_serving/routing/`)

- **ABTest / ABTestManager** — Consistent-hash routing, result recording, chi-squared significance testing
- **CanaryDeployment** — Percentage-based routing with fixed promotion steps (5→25→50→100%), auto-rollback on error threshold
- **ShadowMode** — Fire-and-forget background thread for shadow predictions, agreement tracking

### Monitoring Layer (`ml_serving/monitoring/`)

- **MetricsCollector** — Prometheus counters, histograms, gauges for predictions, latency, drift, models
- **DriftDetector** — PSI, KS test, chi-squared with configurable thresholds and sliding windows
- **AlertManager** — Rule-based alerting with conditions, severity levels, cooldown, log/webhook actions

### API Layer (`ml_serving/api/`)

- **main.py** — FastAPI app with lifespan, CORS, shared `AppState` dataclass
- **middleware.py** — Correlation ID injection, request/response logging
- **schemas.py** — Pydantic request/response models for all endpoints
- **routes/** — Modular routers: predict, models, experiments, monitoring, health

### Infrastructure (`docker/`, `grafana/`)

- **Dockerfile** — Multi-stage build, non-root user, healthcheck
- **docker-compose.yml** — API + Prometheus + Grafana (3 services)
- **Grafana dashboard** — Latency percentiles, throughput, error rate, active models, drift scores

## Data Flow: Prediction Request

1. Client sends `POST /api/v1/predict` with `model_name`, `input_data`, optional `request_id`
2. Middleware assigns correlation ID, logs request start
3. Route handler checks for active A/B test or canary → determines target model
4. `ModelServer.predict()` acquires lock, finds loaded model, runs preprocessing pipeline
5. Framework-specific predictor runs inference, returns `PredictionResult`
6. `MetricsCollector.record_prediction()` updates Prometheus counters/histograms
7. Response returned with prediction, probabilities, latency, model info

## Key Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| Factory | `PredictorFactory` | Framework-agnostic model instantiation |
| Strategy | `BasePredictor` subclasses | Framework-specific inference logic |
| Observer | `MetricsCollector` | Decoupled metrics recording |
| Decorator | `PreprocessingPipeline` | Composable data transforms |
| LRU Cache | `ModelServer._models` | Memory-bounded multi-model serving |
| Singleton | `AppState` | Shared state across route handlers |

## Testing Strategy

- **Unit tests** — Individual modules with mocked dependencies (custom Prometheus registries, in-memory stores)
- **Integration tests** — FastAPI TestClient with real sklearn models, full A/B lifecycle, end-to-end pipeline
- **Test fixtures** — Shared `tmp_dir`, `settings`, `iris_clf` fixtures across test files
- **181 total tests** — 82 from Phase 1 + 99 new from Phases 2-3
