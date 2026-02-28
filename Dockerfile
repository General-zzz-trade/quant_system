# ---- Base stage: core system (stdlib only) ----
FROM python:3.12-slim AS base

WORKDIR /app
COPY pyproject.toml ./
COPY . .

RUN pip install --no-cache-dir -e ".[config]"

# ---- ML stage: adds data science + ML deps ----
FROM base AS ml

RUN pip install --no-cache-dir -e ".[data,ml]"

# ---- Live stage: production trading ----
FROM base AS live

RUN pip install --no-cache-dir -e ".[live,data,config,monitoring]"

ENV PYTHONUNBUFFERED=1
CMD ["python3", "-m", "runner.live_runner", "--config", "/app/infra/config/examples/live.yaml"]

# ---- Dev stage: full development environment ----
FROM base AS dev

RUN pip install --no-cache-dir -e ".[live,data,ml,config,monitoring,dev,test]"

CMD ["python3", "-m", "pytest", "tests_unit/", "-x", "-q"]
