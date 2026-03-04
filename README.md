# ML Model Serving Platform

Production MLOps platform for serving ML models with A/B testing, drift detection, and monitoring.

## Features

- **Model Registry** — Version, stage-promote, and compare ML models (sklearn, PyTorch, ONNX)
- **Model Store** — Local artifact storage with framework-aware serialisation
- **Multi-Model Server** — Serve multiple models concurrently with LRU eviction
- **Dynamic Batching** — Collect individual requests into efficient GPU-friendly batches
- **Preprocessing Pipelines** — Composable, per-model data transformations
- **A/B Testing & Traffic Routing** — *(coming soon)*
- **Drift Detection** — *(coming soon)*
- **Prometheus Monitoring** — *(coming soon)*

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train sample models
make train-samples

# Run tests
make test

# Start the API server
make serve
```

## Project Structure

```
ml_serving/
  config/       — Pydantic settings & environment config
  registry/     — Model registry, metadata store, artifact storage
  serving/      — Predictors, preprocessing, model server, dynamic batching
  routing/      — A/B testing & traffic routing (Phase 2)
  monitoring/   — Drift detection & Prometheus metrics (Phase 2)
  api/          — FastAPI endpoints (Phase 2)
  deployment/   — Docker & cloud deployment helpers (Phase 3)
  utils/        — Shared utilities
tests/
  unit/         — Unit tests
  integration/  — Integration tests
  e2e/          — End-to-end tests
models/sample/  — Sample model training scripts
docker/         — Dockerfiles & compose
grafana/        — Dashboard definitions
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
