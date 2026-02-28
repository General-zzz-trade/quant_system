# Multi-stage build for quant_system
# Stage 1: Builder - install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime - minimal image
FROM python:3.11-slim AS runtime

LABEL maintainer="quant-team"
LABEL description="Quant trading system"

# Create non-root user
RUN groupadd -r quant && useradd -r -g quant -d /app quant

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . /app/

# Set ownership
RUN chown -R quant:quant /app

USER quant

# Prometheus metrics port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9090/metrics')" || exit 1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

CMD ["python", "-m", "runner.live_runner", "--config", "/app/infra/config/examples/live.yaml"]
