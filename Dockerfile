# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output for real-time logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal OS dependencies including curl for container health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root application user for container security
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copy application source code and artifacts
COPY config.py .
COPY data_pipeline.py .
COPY forecasting.py .
COPY inventory.py .
COPY explainability.py .
COPY api.py .
COPY artifacts/ ./artifacts/
COPY frontend/ ./frontend/
COPY india_retail_sales_dataset.csv .

# Set appropriate directory permissions
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Container healthcheck testing the readiness of both data and model artifacts
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Production server startup
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
