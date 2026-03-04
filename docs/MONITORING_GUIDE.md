# Monitoring Guide

## Prometheus Metrics Reference

The platform exposes the following Prometheus metrics at `GET /metrics`:

### Counters

| Metric | Labels | Description |
|--------|--------|-------------|
| `ml_prediction_total` | `model`, `version`, `status` | Total predictions (status: success/error) |

### Histograms

| Metric | Labels | Buckets | Description |
|--------|--------|---------|-------------|
| `ml_prediction_latency_seconds` | `model`, `version` | 5ms–5s | Prediction latency |
| `ml_model_load_seconds` | — | 0.1s–30s | Model load time |
| `ml_batch_size` | — | 1–128 | Batch sizes |

### Gauges

| Metric | Labels | Description |
|--------|--------|-------------|
| `ml_active_models` | — | Number of models currently loaded |
| `ml_request_queue_size` | — | Requests waiting in queue |
| `ml_drift_score` | `model`, `feature` | Current drift score per feature |

## Metrics Summary API

`GET /api/v1/monitoring/metrics` returns a JSON summary:

```json
{
  "total_predictions": 1250,
  "error_rate": 0.004,
  "avg_latency_seconds": 0.023,
  "models": {
    "iris:v1": {
      "count": 800,
      "error_rate": 0.002,
      "avg_latency": 0.018,
      "p50_latency": 0.015,
      "p99_latency": 0.085
    }
  }
}
```

## Drift Detection

### Methods

#### PSI (Population Stability Index)

Measures distribution shift between reference (training) and current data.

- **How it works**: Bins both distributions, computes `sum((current% - ref%) * ln(current% / ref%))` per feature
- **Threshold**: PSI > 0.2 indicates significant drift
- **Interpretation**:
  - PSI < 0.1 — No significant change
  - 0.1 < PSI < 0.2 — Moderate change, monitor
  - PSI > 0.2 — Significant drift, investigate

#### KS Test (Kolmogorov-Smirnov)

Compares empirical CDFs of reference and current data.

- **How it works**: Computes maximum absolute difference between two CDFs
- **Threshold**: KS statistic > 0.1 indicates drift
- **Best for**: Continuous features, detecting any kind of distribution change

#### Chi-Squared Test

For categorical features.

- **How it works**: Compares observed vs expected category frequencies
- **Threshold**: p-value < 0.05 indicates significant drift
- **Best for**: Detecting shifts in categorical distributions

### Setting Up Drift Detection

```python
from ml_serving.monitoring.drift_detector import DriftDetector
import numpy as np

detector = DriftDetector(psi_threshold=0.2, ks_threshold=0.1)

# Store reference distribution from training data
detector.set_reference("iris", X_train)

# Feed incoming data samples
for sample in incoming_data:
    detector.add_sample("iris", sample)

# Check for drift
report = detector.check_model_drift("iris")
if report and report.is_drifted:
    print(f"Drift detected in features: {report.drifted_features}")
    print(f"Overall score: {report.overall_score}")
```

### Drift API

```bash
# Get drift report for a model
curl http://localhost:8000/api/v1/monitoring/drift/iris
```

Response:
```json
{
  "feature_scores": {"feature_0": 0.05, "feature_1": 0.32, "feature_2": 0.08},
  "overall_score": 0.15,
  "is_drifted": true,
  "drifted_features": ["feature_1"],
  "method": "psi",
  "threshold": 0.2
}
```

## Alerting

### Configuring Alert Rules

```python
from ml_serving.monitoring.alerting import AlertManager

mgr = AlertManager()

# High latency alert
mgr.add_rule(
    name="high_p99_latency",
    metric="p99_latency",
    condition="gt",
    threshold=1.0,
    severity="warning",
    cooldown_seconds=300,
)

# Error rate spike
mgr.add_rule(
    name="error_rate_spike",
    metric="error_rate",
    condition="gt",
    threshold=0.05,
    severity="critical",
    action="webhook",
    webhook_url="https://hooks.slack.com/...",
)

# Drift detected
mgr.add_rule(
    name="drift_alert",
    metric="drift_score",
    condition="gt",
    threshold=0.2,
    severity="warning",
)
```

### Alert Conditions

| Condition | Code | Example |
|-----------|------|---------|
| Greater than | `gt` | latency > 1.0s |
| Less than | `lt` | accuracy < 0.9 |
| Greater or equal | `gte` | error_rate >= 0.05 |
| Less or equal | `lte` | throughput <= 10 |
| Equal | `eq` | active_models == 0 |

### Alert Severity

- **info** — Informational, no action needed
- **warning** — Investigate soon
- **critical** — Immediate action required

### Alert Actions

- **log** (default) — Log the alert via structlog
- **webhook** — POST alert details to a configurable URL

### Alerts API

```bash
# View active alerts
curl http://localhost:8000/api/v1/monitoring/alerts
```

## Grafana Dashboard

The pre-built dashboard (`grafana/dashboards/model_serving.json`) includes:

### Panel: Prediction Latency
- Type: Time series
- Shows p50, p95, p99 latency over time
- Source: `histogram_quantile` on `ml_prediction_latency_seconds`

### Panel: Request Throughput
- Type: Time series
- Shows requests per second
- Source: `rate(ml_prediction_total[5m])`

### Panel: Error Rate
- Type: Time series
- Shows error percentage
- Source: Ratio of `status="error"` to total predictions

### Panel: Active Models
- Type: Stat
- Shows current number of loaded models
- Source: `ml_active_models`

### Panel: Drift Scores
- Type: Table
- Shows per-model, per-feature drift values
- Source: `ml_drift_score`

### Importing the Dashboard

1. Open Grafana at http://localhost:3000
2. Go to Dashboards > Import
3. Upload `grafana/dashboards/model_serving.json`
4. Select your Prometheus data source
