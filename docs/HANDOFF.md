# Handoff — ML Model Serving Platform

## Summary

A production-grade MLOps platform for serving machine learning models with A/B testing, canary deployments, drift detection, Prometheus monitoring, and Cloud Run deployment. 213 tests passing. Built with FastAPI, scikit-learn, PyTorch, ONNX Runtime, and Prometheus.

## Phase 1 (Steps 1-3): Foundation

### Step 1: Scaffolding + Model Registry

- **`pyproject.toml`** — Python 3.11+, 15 production + 5 dev dependencies, Hatchling build
- **`ml_serving/config/settings.py`** — Pydantic Settings with env-var loading
- **`ml_serving/registry/schemas.py`** — Core enums (Framework, ModelStage, ModelStatus) and Pydantic models (ModelMetadata, ModelVersion, PredictionResult)
- **`ml_serving/registry/model_store.py`** — Artifact persistence: sklearn (joblib), PyTorch (state_dict), ONNX (.onnx). Framework auto-detection
- **`ml_serving/registry/model_registry.py`** — JSON file-based metadata registry: register, get, promote (with auto-demotion), list, compare versions

### Step 2: Model Server + Prediction Pipeline

- **`ml_serving/serving/predictor.py`** — `BasePredictor` ABC with sklearn/PyTorch/ONNX implementations. `PredictorFactory` for framework-agnostic instantiation
- **`ml_serving/serving/preprocessor.py`** — Composable `PreprocessingPipeline`. Built-in steps: validate_schema, normalize_numeric, encode_categorical, handle_missing, to_numpy
- **`ml_serving/serving/model_server.py`** — Thread-safe `ModelServer` with OrderedDict LRU eviction, per-model preprocessing, warmup on load

### Step 3: Dynamic Batching

- **`ml_serving/serving/batching.py`** — asyncio `DynamicBatcher`: queue-based collection, size/timeout flush triggers, individual Future resolution

## Phase 2-3 (Steps 4-8): Routing, Monitoring & API

### Step 4: A/B Testing + Canary Deployments

- **`ml_serving/routing/ab_testing.py`** — `ABTest` (consistent SHA-256 hashing for sticky sessions), `ABTestManager` (create/record/conclude with chi-squared significance), `ABTestResults`
- **`ml_serving/routing/canary.py`** — `CanaryDeployment` (5→25→50→100% promotion steps, auto-rollback on error threshold), `CanaryMetrics` (error rate, latency percentiles)
- **`ml_serving/routing/shadow.py`** — `ShadowMode` (fire-and-forget background thread, `ShadowReport` with agreement rate and divergences)

### Steps 5-6: Monitoring & Drift Detection

- **`ml_serving/monitoring/metrics.py`** — `MetricsCollector` wrapping prometheus_client: prediction counter, latency histogram, model load time, active models gauge, batch size, queue size, drift score
- **`ml_serving/monitoring/drift_detector.py`** — `DriftDetector` with PSI, KS test, chi-squared for categorical features. Configurable thresholds, sliding window, per-model reference distributions
- **`ml_serving/monitoring/alerting.py`** — `AlertManager` with rule-based evaluation (gt/lt/gte/lte/eq), severity levels, cooldown periods, log and webhook actions

### Step 7: FastAPI Server

- **`ml_serving/api/schemas.py`** — 20 Pydantic request/response models
- **`ml_serving/api/middleware.py`** — Correlation ID injection, request/response logging
- **`ml_serving/api/main.py`** — FastAPI app with lifespan, CORS, `AppState` dataclass, `set_app_state()` for testing
- **`ml_serving/api/routes/`** — 5 route modules: predict (single + batch with A/B/canary routing), models (CRUD + load/unload), experiments (A/B + canary lifecycle), monitoring (metrics/drift/alerts/prometheus), health (per-model status)

### Step 8: Docker + Grafana

- **`docker/Dockerfile`** — Multi-stage build, non-root user, healthcheck
- **`docker/docker-compose.yml`** — API + Prometheus + Grafana (3 services)
- **`docker/prometheus.yml`** — Scrape config
- **`grafana/dashboards/model_serving.json`** — Latency (p50/p95/p99), throughput, error rate, active models, drift scores

## Phase 4 (Steps 9-10): Deployment & Polish

### Step 9: GCP Cloud Run + CI/CD

- **`ml_serving/deployment/cloud_run.py`** — `CloudRunDeployer`: build_image, deploy, update, get_status, get_url, delete, estimate_cost. Protocol-based `CloudRunClient` for test mocking
- **`ml_serving/deployment/rollback.py`** — `RollbackManager`: create_checkpoint, rollback, auto_rollback (health check with retry), list/clear checkpoints
- **`ml_serving/deployment/docker_deploy.py`** — `DockerDeployer`: build, run, stop, health_check, compose_up/down
- **`scripts/deploy.py`** — CLI: `--target local|cloud-run`, `--action deploy|update|rollback|status|delete`, cost estimate, confirmation prompt
- **`.github/workflows/ci.yml`** — Lint + test (Python 3.11/3.12) + coverage (80%+ threshold) + artifact upload
- **`.github/workflows/cd.yml`** — Build → push GHCR → deploy staging → smoke test → manual approval → production
- **`.github/workflows/model-test.yml`** — Triggered on model file changes: train, validate outputs, verify accuracy

### Step 10: Documentation

- **`README.md`** — Portfolio-ready: badges, architecture diagram, feature table, API reference, usage guides, configuration reference, tech stack
- **`docs/ARCHITECTURE.md`** — System design diagram, component details, request flow, design decisions with tradeoffs, testing strategy
- **`docs/DEPLOYMENT.md`** — Local (Python/Docker/Compose), Cloud Run step-by-step, deploy script usage, CI/CD pipeline, monitoring setup
- **`docs/MONITORING_GUIDE.md`** — Prometheus metrics reference, drift detection methodology (PSI/KS/chi-squared explained), alerting configuration, Grafana dashboard guide
- **`docs/AB_TESTING_GUIDE.md`** — A/B testing workflow, consistent hashing, statistical significance, canary deployment guide, shadow mode, best practices

## Tests: 213 Total

| File | Count | Coverage |
|------|-------|----------|
| `tests/unit/test_serving.py` | 40 | Predictors, preprocessing, model server |
| `tests/unit/test_monitoring.py` | 35 | Metrics, drift (PSI/KS/chi-sq), alerting |
| `tests/unit/test_routing.py` | 31 | A/B routing, canary, shadow mode |
| `tests/unit/test_registry.py` | 30 | ModelStore, ModelRegistry |
| `tests/unit/test_deployment.py` | 19 | Cloud Run deploy, Docker deploy (mocked) |
| `tests/unit/test_rollback.py` | 13 | Checkpoints, auto-rollback |
| `tests/unit/test_batching.py` | 12 | Async batching |
| `tests/integration/test_api.py` | 18 | All API endpoints |
| `tests/integration/test_ab_testing.py` | 10 | Full A/B lifecycle |
| `tests/integration/test_pipeline.py` | 5 | End-to-end pipeline |

## Key Design Decisions

1. **JSON file-based registry** — Zero dependencies, human-readable. Swappable for SQL later
2. **OrderedDict LRU** — Built-in O(1) eviction. Every predict refreshes position
3. **Framework abstraction** — BasePredictor + PredictorFactory. One class per new framework
4. **Async batching** — Transparent to callers via Futures. Size/timeout flush triggers
5. **Consistent hash routing** — SHA-256 sticky sessions without server state
6. **No scipy dependency** — PSI/KS/chi-squared via numpy + math.erfc
7. **Protocol-based mocking** — CloudRunClient/DockerClient protocols for clean test doubles
8. **Test-friendly AppState** — set_app_state() injection, isolated Prometheus registries
9. **Fire-and-forget shadow** — Zero latency impact, background thread
10. **Fixed canary steps** — Safety guardrail against accidental full rollout
11. **Cost estimation** — CLI shows estimated Cloud Run monthly cost before deploying
12. **Auto-rollback** — Health check with configurable retry threshold

## File Inventory

```
ml_serving/
├── __init__.py (if needed)
├── config/
│   └── settings.py
├── registry/
│   ├── schemas.py
│   ├── model_store.py
│   └── model_registry.py
├── serving/
│   ├── predictor.py
│   ├── preprocessor.py
│   ├── model_server.py
│   └── batching.py
├── routing/
│   ├── __init__.py
│   ├── ab_testing.py
│   ├── canary.py
│   └── shadow.py
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py
│   ├── drift_detector.py
│   └── alerting.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── middleware.py
│   ├── schemas.py
│   └── routes/
│       ├── __init__.py
│       ├── predict.py
│       ├── models.py
│       ├── experiments.py
│       ├── monitoring.py
│       └── health.py
└── deployment/
    ├── __init__.py
    ├── cloud_run.py
    ├── rollback.py
    └── docker_deploy.py

scripts/
└── deploy.py

tests/
├── unit/
│   ├── test_registry.py
│   ├── test_serving.py
│   ├── test_batching.py
│   ├── test_routing.py
│   ├── test_monitoring.py
│   ├── test_deployment.py
│   └── test_rollback.py
└── integration/
    ├── __init__.py
    ├── test_api.py
    ├── test_ab_testing.py
    └── test_pipeline.py

docker/
├── Dockerfile
├── docker-compose.yml
└── prometheus.yml

grafana/dashboards/
└── model_serving.json

.github/workflows/
├── ci.yml
├── cd.yml
└── model-test.yml

docs/
├── ARCHITECTURE.md
├── DEPLOYMENT.md
├── MONITORING_GUIDE.md
├── AB_TESTING_GUIDE.md
└── HANDOFF.md
```
