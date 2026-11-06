#!/usr/bin/env bash
# Compose paper only. Does not install/restart host systemd units or build images.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

check_only=false
if [[ "${1:-}" == --check ]]; then
    check_only=true
    shift
fi
if [[ "$#" -ne 1 || "$1" != quant-paper ]]; then
    echo 'Usage: bash scripts/deploy.sh [--check] quant-paper (only supported service)' >&2
    exit 2
fi

# Fail before Docker mutations when the checked-in runtime entry is unavailable.
python3 - <<'PY'
from pathlib import Path
for name in ("docker-compose.yml", "scripts/run_bybit_alpha.py"):
    path = Path(name)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"Deployment blocked: required runtime file unavailable: {name}")
PY
docker compose config --quiet
if "$check_only"; then
    echo 'Paper deployment preflight passed; no services changed'
    exit 0
fi
docker compose up -d --no-deps --no-build --force-recreate --wait --wait-timeout 120 quant-paper
