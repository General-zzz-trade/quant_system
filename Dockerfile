# syntax=docker/dockerfile:1
# Source runtime + installed native wheel. No exchange/account access at build time.
FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app

# Build in one layer and retain only the wheel, including on VFS Docker hosts.
FROM scratch AS rust-sources
# Cargo owns native distribution metadata; Hatch owns the Python source wheel.
COPY Cargo.toml Cargo.lock rust_lib.rs /
COPY common_rust/ /common_rust/
COPY event/rust/ /event/rust/
COPY features/rust/ /features/rust/
COPY state/rust/ /state/rust/
COPY decision/rust/ /decision/rust/
COPY execution/rust/ /execution/rust/
COPY risk/rust/ /risk/rust/
COPY strategy/regime/rust/ /strategy/regime/rust/
COPY alpha/rust/ /alpha/rust/
COPY engine/rust/ /engine/rust/
COPY research/rust/ /research/rust/
COPY rust_bin/ /rust_bin/

FROM base AS rust-builder
ARG RUST_VERSION=1.93.0
ENV PATH="/root/.cargo/bin:${PATH}"
COPY --from=rust-sources / /app/
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then \
      export CURL_CA_BUNDLE=/run/secrets/proxy_ca SSL_CERT_FILE=/run/secrets/proxy_ca \
             PIP_CERT=/run/secrets/proxy_ca CARGO_HTTP_CAINFO=/run/secrets/proxy_ca; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl libssl-dev pkg-config ca-certificates && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh && \
    sh /tmp/rustup.sh -y --profile minimal --default-toolchain "$RUST_VERSION" && \
    pip install --no-cache-dir maturin==1.12.6 && \
    maturin build --locked --release --features python --out /tmp/wheels && \
    rm -rf target /root/.cargo /root/.rustup /tmp/rustup.sh /var/lib/apt/lists/*

FROM base AS paper
COPY . .
COPY --from=rust-builder /tmp/wheels/ /tmp/wheels/
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi && \
    apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    pip install --no-cache-dir --require-hashes -r requirements/container.lock && \
    pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels /var/lib/apt/lists/* && \
    python -c "from _quant_hotpath import RustFeatureEngine, RustStateStore, RustInferenceBridge" && \
    python -m runner.alpha_main --check-startup
CMD ["python", "-m", "runner.alpha_main", "--symbols", "ETHUSDT", "--ws", "--dry-run"]

# Existing live arguments are preserved; this task does not enable execution.
FROM paper AS live
CMD ["python", "-m", "scripts.run_bybit_alpha", "--symbols", "BTCUSDT", "ETHUSDT", "ETHUSDT_15m", "SUIUSDT", "AXSUSDT", "--ws"]

# Reuse the same installed wheel/runtime; no maturin develop or ignored failures.
FROM paper AS ci
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi && \
    apt-get update && apt-get install -y --no-install-recommends git && \
    pip install --no-cache-dir pytest==9.0.2 pytest-cov==7.0.0 hatchling==1.33.0 && \
    rm -rf /var/lib/apt/lists/*
CMD ["python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"]
