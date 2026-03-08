# ---- Build stage: compile Rust extensions ----
FROM python:3.12-slim AS builder

WORKDIR /app

# Build tools (no g++ needed — Rust only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    make curl gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

# Rust for PyO3
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Python deps (cached layer — only re-run when pyproject.toml changes)
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
    cd ext/rust && maturin build --release && \
    pip install --force-reinstall target/wheels/*.whl

# Copy remaining source (changes here won't trigger recompilation above)
COPY . .

# ---- Runtime stage (slim) ────────────────────────────────
FROM python:3.12-slim AS paper

WORKDIR /app

# Runtime deps for LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy Python packages + built extensions
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app source (without build artifacts)
COPY --from=builder /app /app

ENV PYTHONUNBUFFERED=1
CMD ["python3", "-m", "runner.testnet_validation", "--phase", "paper", "--duration", "0"]
