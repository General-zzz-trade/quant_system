#!/usr/bin/env bash
# Runner-side smoke for the default compose/deploy path.
# Uses an isolated Compose project name so CI does not collide with any
# long-lived deployment that may exist on the same host.

set -euo pipefail

SERVICE="${SMOKE_SERVICE:-quant-paper}"
TIMEOUT="${SMOKE_TIMEOUT_SEC:-180}"
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-quant-ci-smoke}"

export COMPOSE_PROJECT_NAME="$PROJECT_NAME"

cleanup() {
    docker compose down --remove-orphans >/dev/null 2>&1 || true
    rm -f .env
}

wait_healthy() {
    local elapsed=0
    local container_id=""
    local status="missing"
    while [ "$elapsed" -lt "$TIMEOUT" ]; do
        container_id="$(docker compose ps -q "$SERVICE" 2>/dev/null || true)"
        if [ -n "$container_id" ]; then
            status="$(docker inspect --format='{{.State.Health.Status}}' "$container_id" 2>/dev/null || echo "missing")"
            if [ "$status" = "healthy" ]; then
                echo "$SERVICE healthy after ${elapsed}s"
                return 0
            fi
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    echo "$SERVICE failed to become healthy; last status=${status}" >&2
    docker compose ps "$SERVICE" || true
    docker compose logs "$SERVICE" || true
    return 1
}

trap cleanup EXIT

cp .env.example .env
mkdir -p logs models_v8 data_files

echo "=== default deploy smoke: rolling deploy ==="
bash scripts/deploy.sh
wait_healthy

echo "=== default deploy smoke: rollback command path ==="
docker compose up -d --no-deps --force-recreate "$SERVICE"
wait_healthy
