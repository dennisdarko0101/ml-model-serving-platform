# Multi-stage build for ML Model Serving Platform
# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml .
COPY ml_serving/ ml_serving/

RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Create non-root user
RUN groupadd -r mlserving && useradd -r -g mlserving -d /app -s /sbin/nologin mlserving

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY ml_serving/ ml_serving/

# Create directories for model artifacts
RUN mkdir -p /app/model_artifacts /app/model_registry && \
    chown -R mlserving:mlserving /app

USER mlserving

EXPOSE 8000

ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MODEL_STORE_PATH=/app/model_artifacts
ENV REGISTRY_PATH=/app/model_registry

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

CMD ["python", "-m", "uvicorn", "ml_serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
