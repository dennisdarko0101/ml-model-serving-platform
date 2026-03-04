# Architecture — ML Model Serving Platform

## System Design

```
  Client Request
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI Server                                              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Middleware Layer                                       │  │
│  │  • Correlation ID injection (X-Correlation-ID)         │  │
│  │  • Request/response logging with structlog             │  │
│  │  • Latency measurement                                 │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          ▼                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Routing Layer                                         │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────┐   │  │
│  │  │ A/B Test │  │    Canary     │  │  Shadow Mode   │   │  │
│  │  │ (hash)   │  │ (percentage) │  │ (fire-forget)  │   │  │
│  │  └────┬─────┘  └──────┬───────┘  └───────┬────────┘   │  │
│  │       └───────────────┼───────────────────┘            │  │
│  └───────────────────────┼────────────────────────────────┘  │
│                          ▼                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Serving Layer                                         │  │
│  │                                                        │  │
│  │  ModelServer (OrderedDict LRU, threading.Lock)         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │  │
│  │  │ sklearn  │  │ PyTorch  │  │   ONNX   │             │  │
│  │  │Predictor │  │Predictor │  │Predictor │             │  │
│  │  └──────────┘  └──────────┘  └──────────┘             │  │
│  │                                                        │  │
│  │  PreprocessingPipeline ─── DynamicBatcher (asyncio)    │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          ▼                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Monitoring Layer                                      │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │ Metrics     │  │    Drift     │  │    Alert     │  │  │
│  │  │ Collector   │  │  Detector    │  │   Manager    │  │  │
│  │  │ (prometheus)│  │ (PSI/KS/χ²) │  │ (rules)      │  │  │
│  │  └──────┬──────┘  └──────────────┘  └──────────────┘  │  │
│  └─────────┼──────────────────────────────────────────────┘  │
└────────────┼─────────────────────────────────────────────────┘
             ▼
     Prometheus ──▶ Grafana Dashboard

     ┌────────────────────────────────────────────────┐
     │  Deployment Layer                              │
     │  CloudRunDeployer ─── RollbackManager          │
     │  DockerDeployer ─── deploy.py CLI              │
     └────────────────────────────────────────────────┘
```

## Component Details

### Registry Layer (`ml_serving/registry/`)

| Component | Responsibility |
|-----------|---------------|
| `ModelStore` | Persist model artifacts to disk (joblib, state_dict, .onnx) |
| `ModelRegistry` | JSON-file metadata: versions, stages, metrics, promotion |
| `schemas.py` | Pydantic models: Framework, ModelStage, ModelStatus, ModelMetadata, PredictionResult |

### Serving Layer (`ml_serving/serving/`)

| Component | Responsibility |
|-----------|---------------|
| `BasePredictor` | Abstract interface for all predictors |
| `SklearnPredictor` | numpy-based sklearn inference |
| `PyTorchPredictor` | Tensor-based with device management |
| `ONNXPredictor` | onnxruntime session inference |
| `PredictorFactory` | Maps Framework enum → concrete predictor |
| `PreprocessingPipeline` | Composable chain of named transform functions |
| `ModelServer` | Thread-safe multi-model server with OrderedDict LRU |
| `DynamicBatcher` | asyncio queue-based request batching |

### Routing Layer (`ml_serving/routing/`)

| Component | Responsibility |
|-----------|---------------|
| `ABTest` | SHA-256 consistent hash routing for sticky sessions |
| `ABTestManager` | Create/manage tests, record results, chi-squared significance |
| `CanaryDeployment` | Percentage routing, fixed promotion steps, auto-rollback |
| `ShadowMode` | Background thread shadow predictions, agreement tracking |

### Monitoring Layer (`ml_serving/monitoring/`)

| Component | Responsibility |
|-----------|---------------|
| `MetricsCollector` | Prometheus counters, histograms, gauges |
| `DriftDetector` | PSI, KS test, chi-squared with sliding windows |
| `AlertManager` | Rule evaluation, cooldown, log/webhook actions |

### API Layer (`ml_serving/api/`)

| Component | Responsibility |
|-----------|---------------|
| `main.py` | FastAPI app, lifespan, CORS, shared AppState |
| `middleware.py` | Correlation ID, request/response logging |
| `schemas.py` | Pydantic request/response models |
| `routes/predict.py` | Prediction endpoints with routing integration |
| `routes/models.py` | Model CRUD: register, list, promote, load/unload |
| `routes/experiments.py` | A/B test and canary lifecycle |
| `routes/monitoring.py` | Metrics, drift, alerts, Prometheus endpoint |
| `routes/health.py` | Per-model health check |

### Deployment Layer (`ml_serving/deployment/`)

| Component | Responsibility |
|-----------|---------------|
| `CloudRunDeployer` | Build images, deploy/update/delete Cloud Run services |
| `RollbackManager` | Checkpoint creation, manual/auto rollback |
| `DockerDeployer` | Local Docker build, run, stop, health check |

## Request Flow: Prediction

```
1. POST /api/v1/predict
       │
2. Middleware assigns correlation ID, logs start
       │
3. Check active A/B tests → consistent hash on request_id → pick model
   Check canary deployment → random % → pick model
       │
4. ModelServer.predict()
   ├── Acquire lock, find model in OrderedDict
   ├── Run PreprocessingPipeline (if registered)
   ├── Framework-specific predictor.predict()
   └── Refresh LRU position
       │
5. MetricsCollector.record_prediction()
   ├── Increment prediction_count counter
   └── Observe latency histogram
       │
6. Return PredictResponse with prediction, probabilities, latency, model info
```

## Design Decisions

| Decision | Rationale | Tradeoff |
|----------|-----------|----------|
| JSON file registry | Zero dependencies, human-readable, debuggable | Not suitable for concurrent writes at scale |
| OrderedDict LRU | Simple, built-in, O(1) access refresh | Limited to single-process; use Redis for distributed |
| No scipy dependency | PSI/KS/chi-squared via numpy + math.erfc | Slightly less precise p-values for edge cases |
| Consistent hash routing | Deterministic sticky sessions, no state needed | Hash collisions can create slight traffic imbalance |
| Fixed canary steps | Prevents accidental full rollout | Less flexible than arbitrary percentages |
| Protocol-based mocking | CloudRunClient/DockerClient protocols | Requires test doubles instead of patching |
| Fire-and-forget shadow | Zero latency impact on primary | Shadow failures are silently logged |
| AppState dataclass | Single source of truth for all components | Global mutable state (mitigated by test override) |
| Background thread for shadow | Simple, no async required | Thread overhead per shadow prediction |
| Prometheus registry injection | Tests use isolated registries, no metric collisions | Slightly more setup in test fixtures |

## Testing Strategy

| Layer | Tests | Approach |
|-------|-------|----------|
| Registry | 30 | Real sklearn models, temp directories |
| Serving | 40 | Real Iris classifier, mocked PyTorch/ONNX |
| Batching | 12 | Async tests with real predictions |
| Routing | 31 | Statistical verification of traffic splits |
| Monitoring | 35 | Isolated Prometheus registries, numpy-generated distributions |
| Deployment | 19 | Protocol mocks for Cloud Run and Docker |
| Rollback | 13 | Mocked HTTP for health checks |
| API | 18 | FastAPI TestClient with real models |
| A/B Integration | 10 | Full lifecycle: route → predict → record → conclude |
| Pipeline | 5 | End-to-end: predict → route → metrics → drift |
| **Total** | **213** | |
