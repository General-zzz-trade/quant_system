#!/bin/bash
# BTC Strategy F — Paper Trading
# Model: models_v8/BTCUSDT_gate_v2 (LGBM+XGB ensemble + bear regime-switch)
# WF validated: 15/21 positive Sharpe, Avg Sharpe=2.04

set -euo pipefail
cd "$(dirname "$0")/.."

DURATION="${1:-86400}"
CONFIG="infra/config/examples/testnet_v8_gate_v2.yaml"
LOG_DIR="logs"
PID_FILE="$LOG_DIR/paper_trading.pid"
LOG_FILE="$LOG_DIR/paper_trading.log"

mkdir -p "$LOG_DIR"

# Check for existing process
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Paper trading already running (PID: $(cat "$PID_FILE"))"
    echo "Stop it first: kill $(cat "$PID_FILE")"
    exit 1
fi

nohup python3 -m runner.testnet_validation \
    --config "$CONFIG" \
    --phase paper \
    --duration "$DURATION" \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Paper trading started (PID: $(cat "$PID_FILE"), duration: ${DURATION}s)"
echo "Log: $LOG_FILE"
echo "Stop: kill $(cat "$PID_FILE")"
