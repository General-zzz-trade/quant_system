# ---- Build stage: compile Rust extensions ----
FROM python:3.12-slim AS builder

WORKDIR /app

# Build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    make curl gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

# Rust toolchain (cached layer — only re-run when base image changes)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Python deps (cached layer — only re-run when dependency list changes)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    websocket-client python-dotenv pyyaml \
    pandas numpy lightgbm xgboost scikit-learn \
    prometheus-client maturin

# ---- Rust extension (only rebuild when ext/rust/src or Cargo.toml changes) ----
COPY ext/rust/Cargo.toml ext/rust/Cargo.lock* ext/rust/
COPY ext/rust/src/ ext/rust/src/
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/app/ext/rust/target \
    cd ext/rust && RUSTFLAGS="-C target-cpu=skylake-avx512" maturin build --release && \
    pip install --force-reinstall target/wheels/*.whl

# ---- CI stage (bind-mount checkout, keep prod image slim) ----
FROM builder AS ci

RUN pip install --no-cache-dir \
    aiohttp websockets pyarrow \
    pytest pytest-cov pytest-timeout hypothesis ruff

WORKDIR /app
ENV PYTHONPATH=/app

# ---- Runtime stage (slim) ────────────────────────────────
FROM python:3.12-slim AS paper

WORKDIR /app

# Runtime deps: OpenMP (LightGBM), jemalloc (malloc), util-linux (chrt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libjemalloc2 util-linux && \
    rm -rf /var/lib/apt/lists/*

# Copy Python packages + built extensions (from builder)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy ONLY production source code (no data, tests, scripts, models)
COPY alpha/ /app/alpha/
COPY engine/ /app/engine/
COPY event/ /app/event/
COPY execution/ /app/execution/
COPY features/ /app/features/
COPY infra/ /app/infra/
COPY monitoring/ /app/monitoring/
COPY runner/ /app/runner/
COPY state/ /app/state/
COPY strategies/ /app/strategies/
COPY portfolio/ /app/portfolio/
COPY decision/ /app/decision/
COPY risk/ /app/risk/
COPY regime/ /app/regime/
COPY attribution/ /app/attribution/
COPY core/ /app/core/
COPY data/ /app/data/

# Pre-compile .pyc for faster startup
RUN python3 -m compileall -q /app

ENV PYTHONUNBUFFERED=1
CMD ["python3", "-m", "runner.testnet_validation", "--phase", "paper", "--duration", "0"]
