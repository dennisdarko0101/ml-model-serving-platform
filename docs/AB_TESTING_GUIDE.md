# A/B Testing & Canary Deployment Guide

## Overview

The platform supports two strategies for safely rolling out model changes:

| Strategy | Best For | Traffic Control | Rollback |
|----------|----------|----------------|----------|
| **A/B Testing** | Comparing two models to find the better one | Hash-based split (sticky sessions) | Remove test |
| **Canary Deployment** | Safely releasing a new model version | Percentage-based, gradual increase | Instant rollback |

## A/B Testing

### How It Works

1. **Create a test** — Define model A, model B, and traffic split percentage
2. **Route requests** — Each request is routed using consistent hashing on `request_id`
3. **Record results** — Ground truth is recorded for each prediction
4. **Analyze** — Platform computes accuracy metrics and statistical significance (chi-squared test)
5. **Conclude** — Declare a winner and remove the test

### Consistent Hashing for Sticky Sessions

The same `request_id` always routes to the same model. This ensures:
- Users get consistent experiences during the test
- No server-side session state is needed
- Deterministic and reproducible routing

The hash is computed as: `SHA-256(request_id) mod 10000 / 10000`

### Creating an A/B Test

```bash
curl -X POST http://localhost:8000/api/v1/experiments/ab \
  -H "Content-Type: application/json" \
  -d '{
    "name": "iris_v1_vs_v2",
    "model_a": "iris:v1",
    "model_b": "iris:v2",
    "traffic_split": 0.3
  }'
```

- `traffic_split: 0.3` sends 30% to model B, 70% to model A
- `traffic_split: 0.5` is an even 50/50 split

### Making Predictions During a Test

Predictions are automatically routed when a test is active:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "iris",
    "input_data": [5.1, 3.5, 1.4, 0.2],
    "request_id": "user-session-abc"
  }'
```

The response includes which model was actually used:
```json
{
  "prediction": 0,
  "model_used": "iris",
  "version": "v2",
  "latency_ms": 3.2
}
```

### Checking Results

```bash
curl http://localhost:8000/api/v1/experiments/ab/iris_v1_vs_v2
```

Response:
```json
{
  "test_name": "iris_v1_vs_v2",
  "model_a": "iris:v1",
  "model_b": "iris:v2",
  "model_a_metrics": {"accuracy": 0.94, "correct": 188, "total": 200},
  "model_b_metrics": {"accuracy": 0.97, "correct": 97, "total": 100},
  "sample_size_a": 200,
  "sample_size_b": 100,
  "p_value": 0.023,
  "is_significant": true
}
```

### Statistical Significance

The platform uses a **chi-squared test** to determine if the difference between model A and model B is statistically significant.

- **p-value < 0.05** — Significant difference (safe to conclude)
- **p-value >= 0.05** — Not enough evidence, collect more data

**Rule of thumb**: Aim for at least 100 samples per model before concluding.

### Concluding a Test

```bash
curl -X POST http://localhost:8000/api/v1/experiments/ab/iris_v1_vs_v2/conclude
```

This returns the winner and removes the test:
```json
{
  "test_name": "iris_v1_vs_v2",
  "winner": "iris:v2",
  "is_significant": true,
  "p_value": 0.023,
  "model_a_accuracy": 0.94,
  "model_b_accuracy": 0.97
}
```

## Canary Deployments

### How It Works

1. **Start** — Deploy the canary at 5% traffic
2. **Monitor** — Track error rate and latency for both models
3. **Promote** — If healthy, increase traffic: 5% → 25% → 50% → 100%
4. **Rollback** — If errors spike, instantly revert to the current model

### Promotion Steps

```
5% ──▶ 25% ──▶ 50% ──▶ 100% (fully promoted)
 │       │       │
 └───────┴───────┴──▶ Rollback (0%, deactivated)
```

### Starting a Canary

```bash
curl -X POST http://localhost:8000/api/v1/experiments/canary \
  -H "Content-Type: application/json" \
  -d '{
    "current_model": "iris:v1",
    "canary_model": "iris:v2",
    "initial_percentage": 5,
    "error_threshold": 0.1
  }'
```

### Promoting

```bash
# Each call advances to the next step
curl -X POST http://localhost:8000/api/v1/experiments/canary/promote
# 5% → 25%

curl -X POST http://localhost:8000/api/v1/experiments/canary/promote
# 25% → 50%
```

### Auto-Promotion (Programmatic)

```python
from ml_serving.routing.canary import CanaryDeployment

canary = CanaryDeployment("iris:v1", "iris:v2", error_threshold=0.1)

# After collecting metrics...
if canary.auto_promote():
    print(f"Promoted to {canary.canary_percentage}%")
else:
    print("Rolled back due to high error rate")
```

### Rolling Back

```bash
curl -X POST http://localhost:8000/api/v1/experiments/canary/rollback
```

Instantly sets canary traffic to 0% and deactivates the deployment.

## Shadow Mode

For cases where you want to compare models without any traffic impact:

```python
from ml_serving.routing.shadow import ShadowMode

shadow = ShadowMode("primary:v1", "shadow:v2")
shadow.set_predict_fn(model_server.predict)

# Primary serves the response; shadow runs in background
result = shadow.predict(input_data)

# Check agreement after collecting data
report = shadow.compare_predictions()
print(f"Agreement rate: {report.agreement_rate:.1%}")
print(f"Divergences: {len(report.divergences)}")
```

Shadow predictions run in a background thread and never affect response latency.

## Best Practices

1. **Start with shadow mode** — Compare models without risk before running A/B tests
2. **Use canary for rollouts** — Graduate traffic slowly; A/B tests are for comparison
3. **Collect sufficient samples** — Aim for 100+ per model before concluding A/B tests
4. **Set error thresholds conservatively** — 5-10% error threshold for canary auto-promote
5. **Monitor drift alongside experiments** — Model accuracy can degrade even with a "winning" model
6. **Use request_id consistently** — Same user should see the same model during an A/B test
7. **Don't run multiple experiments on the same model** — One A/B test or canary per model at a time
