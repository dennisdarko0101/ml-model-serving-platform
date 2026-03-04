# ML Model Serving Platform

Production MLOps platform for serving ML models with A/B testing, drift detection, and monitoring.

## Features

- **Model Registry** — Version, stage-promote, and compare ML models (sklearn, PyTorch, ONNX)
- **Model Store** — Local artifact storage with framework-aware serialisation
- **Multi-Model Server** — Serve multiple models concurrently with LRU eviction
- **Dynamic Batching** — Collect individual requests into efficient GPU-friendly batches
- **Preprocessing Pipelines** — Composable, per-model data transformations
- **A/B Testing** — Statistical comparison of models with consistent hashing for sticky sessions
- **Canary Deployments** — Gradual traffic shifting with automatic rollback on error spikes
- **Shadow Mode** — Run shadow models alongside primary without affecting response times
- **Drift Detection** — PSI, KS test, and chi-squared for data/prediction/categorical drift
- **Prometheus Monitoring** — Latency histograms, prediction counters, model health gauges
- **Alerting** — Configurable threshold rules with log and webhook actions
- **FastAPI REST API** — Full model management, prediction, experiment, and monitoring endpoints
- **Docker + Grafana** — Multi-stage Docker build, Prometheus + Grafana dashboards

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train sample models
make train-samples

# Run tests (181 tests)
make test

# Start the API server
make serve

# Docker (API + Prometheus + Grafana)
cd docker && docker-compose up
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/predict` | Single prediction (routes via A/B test or canary) |
| POST | `/api/v1/predict/batch` | Batch prediction |
| POST | `/api/v1/models` | Register a model |
| GET | `/api/v1/models` | List all models |
| GET | `/api/v1/models/{name}` | Get model details |
| PUT | `/api/v1/models/{name}/promote` | Promote to production |
| DELETE | `/api/v1/models/{name}/{version}` | Archive model |
| POST | `/api/v1/models/{name}/load` | Load model into server |
| POST | `/api/v1/models/{name}/unload` | Unload from server |
| POST | `/api/v1/experiments/ab` | Create A/B test |
| GET | `/api/v1/experiments/ab/{name}` | Get A/B test results |
| POST | `/api/v1/experiments/ab/{name}/conclude` | Conclude A/B test |
| POST | `/api/v1/experiments/canary` | Start canary deployment |
| POST | `/api/v1/experiments/canary/promote` | Promote canary |
| POST | `/api/v1/experiments/canary/rollback` | Rollback canary |
| GET | `/api/v1/monitoring/metrics` | Metrics summary |
| GET | `/api/v1/monitoring/drift/{model}` | Drift report |
| GET | `/api/v1/monitoring/alerts` | Active alerts |
| GET | `/metrics` | Prometheus scrape endpoint |
| GET | `/health` | Health check with per-model status |

## Project Structure

```
ml_serving/
  config/       — Pydantic settings & environment config
  registry/     — Model registry, metadata store, artifact storage
  serving/      — Predictors, preprocessing, model server, dynamic batching
  routing/      — A/B testing, canary deployments, shadow mode
  monitoring/   — Metrics collection, drift detection, alerting
  api/          — FastAPI endpoints, middleware, schemas
tests/
  unit/         — Unit tests (routing, monitoring, serving, registry, batching)
  integration/  — Integration tests (API, A/B lifecycle, end-to-end pipeline)
models/sample/  — Sample model training scripts
docker/         — Dockerfile, docker-compose, Prometheus config
grafana/        — Grafana dashboard definitions
docs/           — Architecture & handoff documentation
```

## Supported Frameworks

| Framework    | Save/Load | Prediction | Batch |
|-------------|-----------|------------|-------|
| scikit-learn | joblib    | numpy      | numpy |
| PyTorch      | state_dict| tensor     | tensor|
| ONNX Runtime | .onnx     | numpy      | numpy |

## Development

```bash
make lint       # ruff check
make format     # ruff format + fix
make typecheck  # mypy
make test-cov   # pytest with coverage
```

## License

MIT
