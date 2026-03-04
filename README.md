# ML Model Serving Platform

[![CI](https://github.com/ml-platform-team/ml-model-serving-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/ml-platform-team/ml-model-serving-platform/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-213%20passing-brightgreen.svg)]()

Production-grade MLOps platform for serving machine learning models at scale. Built with FastAPI, Prometheus, and Grafana — featuring A/B testing, canary deployments, drift detection, and automated rollback.

## Architecture

```
  Client Request
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Server          Middleware: Logging + Correlation ID    │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐     │
│  │  A/B Test   │    │    Canary     │    │  Shadow Mode   │     │
│  │  Router     │    │   Deployer    │    │  (async)       │     │
│  └──────┬──────┘    └──────┬───────┘    └───────┬────────┘     │
│         └──────────────────┼────────────────────┘              │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Model Server  (LRU cache, thread-safe)          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│  │  │ sklearn  │  │ PyTorch  │  │   ONNX   │              │   │
│  │  │Predictor │  │Predictor │  │Predictor │              │   │
│  │  └──────────┘  └──────────┘  └──────────┘              │   │
│  │  Preprocessing Pipeline ─── Dynamic Batching            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│  ┌─────────────┐  ┌───────┴──────┐  ┌─────────────────────┐   │
│  │  Metrics    │  │    Drift     │  │    Alert Manager    │   │
│  │  Collector  │  │   Detector   │  │  (rules + webhook)  │   │
│  └──────┬──────┘  └──────────────┘  └─────────────────────┘   │
└─────────┼──────────────────────────────────────────────────────┘
          ▼
  Prometheus ──▶ Grafana Dashboard
```

## Features

| Category | Capability |
|----------|-----------|
| **Model Serving** | Multi-model inference with LRU eviction, preprocessing pipelines, dynamic batching |
| **Frameworks** | scikit-learn, PyTorch, ONNX Runtime — extensible via `BasePredictor` |
| **A/B Testing** | Consistent-hash routing, sticky sessions, chi-squared significance testing |
| **Canary Deployments** | Gradual rollout (5% → 25% → 50% → 100%), automatic rollback on error threshold |
| **Shadow Mode** | Fire-and-forget comparison of primary vs shadow model predictions |
| **Drift Detection** | PSI, Kolmogorov-Smirnov, chi-squared tests with configurable thresholds |
| **Monitoring** | Prometheus counters/histograms/gauges, prediction latency, error rates |
| **Alerting** | Rule-based alerts with cooldown, severity levels, log and webhook actions |
| **API** | FastAPI REST with 20 endpoints, correlation IDs, batch predictions |
| **Deployment** | Docker multi-stage build, Cloud Run, Prometheus + Grafana stack |
| **CI/CD** | GitHub Actions: lint, test (3.11/3.12), coverage, staging → production pipeline |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train sample models
make train-samples

# Run all 213 tests
make test

# Start the API server (development)
make serve
```

### Docker Compose (API + Prometheus + Grafana)

```bash
docker-compose -f docker/docker-compose.yml up -d
```

- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## API Reference

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | Single prediction (routes through A/B test or canary if active) |
| `POST` | `/api/v1/predict/batch` | Batch prediction |

### Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/models` | Register a new model |
| `GET` | `/api/v1/models` | List all models |
| `GET` | `/api/v1/models/{name}` | Get model details |
| `PUT` | `/api/v1/models/{name}/promote` | Promote version to production |
| `DELETE` | `/api/v1/models/{name}/{version}` | Archive model version |
| `POST` | `/api/v1/models/{name}/load` | Load model into serving server |
| `POST` | `/api/v1/models/{name}/unload` | Unload model from memory |

### Experiments

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/experiments/ab` | Create A/B test |
| `GET` | `/api/v1/experiments/ab/{name}` | Get A/B test results with p-value |
| `POST` | `/api/v1/experiments/ab/{name}/conclude` | Conclude test, declare winner |
| `POST` | `/api/v1/experiments/canary` | Start canary deployment |
| `POST` | `/api/v1/experiments/canary/promote` | Promote canary to next traffic level |
| `POST` | `/api/v1/experiments/canary/rollback` | Rollback canary deployment |

### Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/monitoring/metrics` | Metrics summary (latency, error rate, per-model) |
| `GET` | `/api/v1/monitoring/drift/{model}` | Drift report for a model |
| `GET` | `/api/v1/monitoring/alerts` | Active alerts |
| `GET` | `/metrics` | Prometheus scrape endpoint |
| `GET` | `/health` | Health check with per-model status |

## Usage Guide

### A/B Testing

```bash
# Create an A/B test (50% traffic split)
curl -X POST http://localhost:8000/api/v1/experiments/ab \
  -H "Content-Type: application/json" \
  -d '{"name": "v1_vs_v2", "model_a": "iris:v1", "model_b": "iris:v2", "traffic_split": 0.5}'

# Predictions are automatically routed based on request_id hash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "iris", "input_data": [5.1, 3.5, 1.4, 0.2], "request_id": "user-123"}'

# Check results (includes p-value for statistical significance)
curl http://localhost:8000/api/v1/experiments/ab/v1_vs_v2

# Conclude the test
curl -X POST http://localhost:8000/api/v1/experiments/ab/v1_vs_v2/conclude
```

### Canary Deployment

```bash
# Start canary at 5% traffic
curl -X POST http://localhost:8000/api/v1/experiments/canary \
  -H "Content-Type: application/json" \
  -d '{"current_model": "iris:v1", "canary_model": "iris:v2"}'

# Promote: 5% → 25% → 50% → 100%
curl -X POST http://localhost:8000/api/v1/experiments/canary/promote

# Rollback if issues detected
curl -X POST http://localhost:8000/api/v1/experiments/canary/rollback
```

### Cloud Run Deployment

```bash
# Deploy to Cloud Run with cost estimate
python scripts/deploy.py --target cloud-run --action deploy \
  --service-name ml-serving --memory 1Gi --cpu 2
```

## Grafana Dashboard

The pre-built dashboard (`grafana/dashboards/model_serving.json`) provides:

- **Prediction Latency** — p50, p95, p99 time series
- **Request Throughput** — Requests per second
- **Error Rate** — Percentage of failed predictions
- **Active Models** — Number of loaded models
- **Drift Scores** — Per-model, per-feature drift values

## Configuration

All settings are loaded from environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_STORE_PATH` | `./model_artifacts` | Model artifact storage directory |
| `REGISTRY_PATH` | `./model_registry` | Model metadata registry directory |
| `MAX_LOADED_MODELS` | `10` | Maximum models in memory (LRU eviction) |
| `BATCH_TIMEOUT_MS` | `50` | Dynamic batching timeout |
| `BATCH_MAX_SIZE` | `32` | Maximum batch size |
| `PROMETHEUS_PORT` | `9090` | Prometheus port |
| `GCP_PROJECT` | — | GCP project ID (for Cloud Run) |
| `GCP_REGION` | `us-central1` | GCP region |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `LOG_LEVEL` | `INFO` | Logging level |

## Project Structure

```
ml_serving/
  config/        — Pydantic settings from environment
  registry/      — Model registry, store, and schemas
  serving/       — Predictors, preprocessing, model server, dynamic batching
  routing/       — A/B testing, canary deployments, shadow mode
  monitoring/    — Prometheus metrics, drift detection, alerting
  api/           — FastAPI endpoints, middleware, request/response schemas
  deployment/    — Cloud Run, Docker, and rollback management
scripts/         — CLI deployment script
tests/
  unit/          — 180 unit tests
  integration/   — 33 integration tests
models/sample/   — Sample model training scripts
docker/          — Dockerfile, docker-compose, Prometheus config
grafana/         — Grafana dashboard definitions
docs/            — Architecture, deployment, monitoring, and A/B testing guides
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| ML Frameworks | scikit-learn, PyTorch, ONNX Runtime |
| Data | NumPy, Pandas |
| Monitoring | Prometheus Client, Grafana |
| Configuration | Pydantic Settings |
| Logging | structlog |
| Serialization | joblib |
| HTTP Client | httpx |
| Deployment | Docker, Google Cloud Run |
| CI/CD | GitHub Actions |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Linting | Ruff, mypy |

## Development

```bash
make lint        # Ruff linting
make format      # Auto-format code
make typecheck   # mypy type checking
make test-cov    # Tests with coverage report
make docker-up   # Start Docker Compose stack
make docker-down # Stop Docker Compose stack
```

## License

MIT
