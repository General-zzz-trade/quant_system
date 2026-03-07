#!/usr/bin/env bash
# Rolling deploy for paper trading services.
# Restarts each service one-by-one, waiting for healthcheck between each.
# Exits non-zero (triggering rollback in CI) if any service fails.

set -euo pipefail

SERVICES=(paper-btc paper-sol paper-eth)
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
    echo "--- Restarting $svc ---"
    docker compose restart "$svc"

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
