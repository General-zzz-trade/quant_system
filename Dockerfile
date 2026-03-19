# Multi-stage Dockerfile for quant_system
# Stages: ci (testing), paper (paper trading), live (production)

# ─── Base stage: Python 3.12 + system deps ──────────────────
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libssl-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

# ─── Rust toolchain stage ───────────────────────────────────
FROM base AS rust-builder

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin

COPY ext/rust/ ext/rust/
COPY _quant_hotpath/ _quant_hotpath/

RUN cd ext/rust && maturin build --release --features python --out /tmp/wheels && \
    pip install /tmp/wheels/*.whl && \
    cp $(python3 -c "import _quant_hotpath, os; print(os.path.dirname(_quant_hotpath.__file__))")/*.so /tmp/hotpath.so 2>/dev/null || true

# ─── CI stage: full test environment ────────────────────────
FROM base AS ci

RUN pip install --no-cache-dir \
    "pytest>=7.4" "pytest-cov>=4.1" "pytest-forked>=1.6" \
    "ruff>=0.3" "maturin>=1.4" \
    "websocket-client>=1.6" "python-dotenv>=1.0" \
    "pyyaml>=6.0" \
    "pandas>=2.1,<3.0" "numpy>=1.26,<3.0" "pyarrow>=14.0" \
    "scikit-learn>=1.4" "lightgbm>=4.1" \
    "prometheus-client>=0.19" \
    "aiohttp>=3.9" "websockets>=12.0"

# Install Rust toolchain for CI builds
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin

COPY . .

# Build Rust extension in CI
RUN cd ext/rust && maturin develop --release --features python 2>/dev/null || true
RUN cp $(python3 -c "import _quant_hotpath, os; print(os.path.dirname(_quant_hotpath.__file__))")/*.so _quant_hotpath/ 2>/dev/null || true

ENV PYTHONPATH=/app
CMD ["python", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"]

# ─── Paper trading stage: minimal runtime ───────────────────
FROM base AS paper

RUN pip install --no-cache-dir \
    "websocket-client>=1.6" "python-dotenv>=1.0" \
    "pyyaml>=6.0" \
    "pandas>=2.1,<3.0" "numpy>=1.26,<3.0" \
    "scikit-learn>=1.4" "lightgbm>=4.1" \
    "prometheus-client>=0.19"

COPY --from=rust-builder /tmp/hotpath.so /tmp/hotpath.so
COPY . .
RUN cp /tmp/hotpath.so _quant_hotpath/ 2>/dev/null || true

ENV PYTHONPATH=/app
CMD ["python", "-m", "scripts.run_bybit_alpha", "--symbols", "ETHUSDT", "--ws", "--dry-run"]

# ─── Live trading stage: runtime + monitoring ───────────────
FROM paper AS live

RUN pip install --no-cache-dir \
    "aiohttp>=3.9" "websockets>=12.0"

CMD ["python", "-m", "scripts.run_bybit_alpha", "--symbols", "BTCUSDT", "ETHUSDT", "ETHUSDT_15m", "SUIUSDT", "AXSUSDT", "--ws"]
