# Deployment Guide

## Local Development

### Direct (Python)

```bash
pip install -e ".[dev]"
make serve
# API available at http://localhost:8000
```

### Docker Compose

Start the full stack (API + Prometheus + Grafana):

```bash
docker-compose -f docker/docker-compose.yml up -d
```

| Service | URL | Purpose |
|---------|-----|---------|
| API | http://localhost:8000 | ML serving REST API |
| Prometheus | http://localhost:9090 | Metrics aggregation |
| Grafana | http://localhost:3000 | Dashboard visualization |

Default Grafana credentials: `admin` / `admin`

Stop services:

```bash
docker-compose -f docker/docker-compose.yml down
```

### Docker (standalone)

```bash
# Build
docker build -f docker/Dockerfile -t ml-serving:latest .

# Run
docker run -d --name ml-serving -p 8000:8000 \
  -e MODEL_STORE_PATH=/app/model_artifacts \
  -e REGISTRY_PATH=/app/model_registry \
  ml-serving:latest

# Verify
curl http://localhost:8000/health
```

## Google Cloud Run

### Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
2. GCP project with Cloud Run API enabled
3. Authenticated: `gcloud auth login`

### Step-by-Step

```bash
# Set your project
export GCP_PROJECT=your-project-id

# Build and push the image
gcloud builds submit --tag gcr.io/$GCP_PROJECT/ml-serving:latest

# Deploy to Cloud Run
gcloud run deploy ml-serving \
  --image gcr.io/$GCP_PROJECT/ml-serving:latest \
  --region us-central1 \
  --memory 1Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 20 \
  --port 8000 \
  --allow-unauthenticated

# Verify
URL=$(gcloud run services describe ml-serving \
  --region us-central1 --format 'value(status.url)')
curl $URL/health
```

### Using the Deploy Script

```bash
# Deploy with cost estimate
python scripts/deploy.py --target cloud-run --action deploy \
  --service-name ml-serving --memory 1Gi --cpu 2

# Check status
python scripts/deploy.py --target cloud-run --action status \
  --service-name ml-serving

# Update image
python scripts/deploy.py --target cloud-run --action update \
  --service-name ml-serving --image gcr.io/$GCP_PROJECT/ml-serving:v2

# Delete
python scripts/deploy.py --target cloud-run --action delete \
  --service-name ml-serving
```

### Environment Variables

Set these via Cloud Run environment configuration:

```bash
gcloud run deploy ml-serving \
  --set-env-vars="MAX_LOADED_MODELS=5,LOG_LEVEL=INFO,BATCH_MAX_SIZE=16"
```

See [Configuration Reference](../README.md#configuration) for all variables.

## CI/CD Pipeline

### Workflow Overview

```
Push to main
    │
    ├─ CI: Lint + Test (Python 3.11/3.12) + Coverage
    │
    ├─ CD: Build Docker → Push to GHCR
    │     │
    │     ├─ Deploy to Cloud Run (staging)
    │     ├─ Smoke test staging
    │     │
    │     └─ Manual approval → Deploy to production
    │
    └─ Model Validation (when model files change)
```

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT` | Google Cloud project ID |
| `GCP_SA_KEY` | Service account JSON key |

### Environments

Configure two GitHub environments:
- **staging** — Auto-deploys on push to main
- **production** — Requires manual approval

## Monitoring Setup

### Prometheus

The API exposes metrics at `GET /metrics` in Prometheus exposition format.

Prometheus is pre-configured in `docker/prometheus.yml` to scrape the API every 15 seconds.

### Grafana

1. Open http://localhost:3000
2. Add Prometheus data source: `http://prometheus:9090`
3. Import dashboard from `grafana/dashboards/model_serving.json`

See [Monitoring Guide](MONITORING_GUIDE.md) for detailed metrics reference.

## Rollback

### Manual Rollback (Cloud Run)

Cloud Run keeps previous revisions. Rollback via:

```bash
# List revisions
gcloud run revisions list --service ml-serving --region us-central1

# Route traffic to previous revision
gcloud run services update-traffic ml-serving \
  --to-revisions=ml-serving-00001=100 \
  --region us-central1
```

### Programmatic Rollback

```python
from ml_serving.deployment.cloud_run import CloudRunDeployer
from ml_serving.deployment.rollback import RollbackManager

deployer = CloudRunDeployer(project_id="my-project")
manager = RollbackManager(deployer)

# Create checkpoint before deployment
manager.create_checkpoint("ml-serving", "rev-001", image_uri="gcr.io/p/ml:v1")

# Deploy new version
deployer.update("ml-serving", "gcr.io/p/ml:v2")

# Auto-rollback if health check fails
healthy = manager.auto_rollback(
    "ml-serving",
    "https://ml-serving-abc.a.run.app/health",
    threshold=3,
)
```
