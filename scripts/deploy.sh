#!/usr/bin/env bash
# Rolling deploy for production alpha trading services.
#
# Default: deploys alpha-runner only (current production service).
# Pass service names as args to override: ./deploy.sh paper-multi trader-rust
#
# Recreates each service one-by-one so updated images/config are applied.
# Exits non-zero (triggering rollback in CI) if any service fails.

set -euo pipefail

if [ $# -gt 0 ]; then
    SERVICES=("$@")
else
    SERVICES=(alpha-runner)  # Default: only active production service
fi
TIMEOUT=120  # seconds per service

notify() {
    local msg="$1"
    echo "$msg"
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -H "Content-Type: application/json" \
            -d "{\"chat_id\":\"${TELEGRAM_CHAT_ID}\",\"text\":\"$msg\",\"parse_mode\":\"Markdown\"}" >/dev/null 2>&1 || true
    fi
}

wait_healthy() {
    local svc="$1"
    local elapsed=0
    local status="missing"
    while [ $elapsed -lt $TIMEOUT ]; do
        status=$(docker inspect --format='{{.State.Health.Status}}' "$(docker compose ps -q "$svc" 2>/dev/null)" 2>/dev/null || echo "missing")
        if [ "$status" = "healthy" ]; then
            echo "  $svc: healthy (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  $svc: $status (${elapsed}s / ${TIMEOUT}s)"
    done
    echo "  $svc: TIMEOUT after ${TIMEOUT}s (last status: $status)"
    return 1
}

deploy_start=$(date +%s)
notify "*[DEPLOY]* Starting rolling deploy at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

failed=""
for svc in "${SERVICES[@]}"; do
    echo "--- Recreating $svc ---"
    docker compose up -d --no-deps --force-recreate "$svc"

    if wait_healthy "$svc"; then
        echo "$svc OK"
    else
        failed="$svc"
        notify "*[DEPLOY FAILED]* $svc failed healthcheck after ${TIMEOUT}s — aborting"
        exit 1
    fi
done

elapsed=$(( $(date +%s) - deploy_start ))
notify "*[DEPLOY OK]* All services healthy in ${elapsed}s"
