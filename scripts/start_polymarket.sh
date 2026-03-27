#!/bin/bash
# Start Polymarket maker with default config.
# Usage:
#   ./scripts/start_polymarket.sh                  # foreground
#   ./scripts/start_polymarket.sh --once            # single cycle
#   ./scripts/start_polymarket.sh --systemd         # install + start systemd service
#
# Environment: POLYMARKET_API_KEY, POLYMARKET_API_SECRET required.
# See config/polymarket.yaml for full config.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
ENV_FILE="$PROJECT_DIR/.env"

# -- helpers ------------------------------------------------------------------
red()   { echo -e "\033[0;31m$*\033[0m"; }
green() { echo -e "\033[0;32m$*\033[0m"; }
yellow(){ echo -e "\033[0;33m$*\033[0m"; }

die() { red "ERROR: $*" >&2; exit 1; }

# -- preflight checks --------------------------------------------------------
check_env() {
    if [[ -f "$ENV_FILE" ]]; then
        set -a; source "$ENV_FILE"; set +a
    fi

    if [[ -z "${POLYMARKET_API_KEY:-}" ]]; then
        die "POLYMARKET_API_KEY not set. Export it or add to $ENV_FILE"
    fi
    if [[ -z "${POLYMARKET_API_SECRET:-}" ]]; then
        die "POLYMARKET_API_SECRET not set. Export it or add to $ENV_FILE"
    fi
    green "API credentials found"
}

check_imports() {
    cd "$PROJECT_DIR"
    python3 -c "from polymarket.runner import PolymarketRunner; print('Runner import OK')" \
        || die "Cannot import PolymarketRunner"
    python3 -c "from polymarket.binance_feed import BinanceFeed; print('BinanceFeed import OK')" 2>/dev/null \
        || yellow "WARNING: BinanceFeed import failed (non-fatal)"
    green "Python imports OK"
}

check_collector() {
    if systemctl is-active --quiet polymarket-collector 2>/dev/null; then
        green "Collector service: running"
    elif [[ -f "$PROJECT_DIR/data/polymarket/collector.db" ]]; then
        yellow "Collector service not running, but DB exists"
    else
        yellow "WARNING: No collector DB found. Data collection may be needed first."
    fi
}

# -- systemd ------------------------------------------------------------------
install_systemd() {
    local svc_src="$PROJECT_DIR/infra/systemd/polymarket-maker.service"
    local svc_dest="/etc/systemd/system/polymarket-maker.service"

    if [[ ! -f "$svc_src" ]]; then
        die "Service file not found: $svc_src"
    fi

    echo "Installing systemd service..."
    sudo cp "$svc_src" "$svc_dest"
    sudo systemctl daemon-reload
    sudo systemctl enable polymarket-maker
    sudo systemctl start polymarket-maker
    green "polymarket-maker.service installed and started"
    echo ""
    echo "Useful commands:"
    echo "  sudo systemctl status polymarket-maker"
    echo "  sudo journalctl -u polymarket-maker -f"
    echo "  tail -f $LOG_DIR/polymarket_maker.log"
    exit 0
}

# -- main ---------------------------------------------------------------------
main() {
    echo "========================================="
    echo " Polymarket Maker Launcher"
    echo "========================================="
    echo ""

    # Parse --systemd flag early
    for arg in "$@"; do
        if [[ "$arg" == "--systemd" ]]; then
            check_env
            install_systemd
        fi
    done

    # Preflight
    check_env
    check_imports
    check_collector
    echo ""

    # Ensure log directory
    mkdir -p "$LOG_DIR"

    # Default parameters
    GAMMA="${POLYMARKET_GAMMA:-0.1}"
    KAPPA="${POLYMARKET_KAPPA:-1.5}"
    ORDER_SIZE="${POLYMARKET_ORDER_SIZE:-10}"
    MAX_INVENTORY="${POLYMARKET_MAX_INVENTORY:-100}"
    REFRESH="${POLYMARKET_REFRESH:-30}"
    LOG_LEVEL="${POLYMARKET_LOG_LEVEL:-INFO}"

    green "Starting Polymarket maker..."
    echo "  gamma=$GAMMA kappa=$KAPPA order_size=$ORDER_SIZE"
    echo "  max_inventory=$MAX_INVENTORY refresh=${REFRESH}s"
    echo "  log_level=$LOG_LEVEL"
    echo ""

    cd "$PROJECT_DIR"
    exec python3 scripts/run_polymarket_maker.py \
        --gamma "$GAMMA" \
        --kappa "$KAPPA" \
        --order-size "$ORDER_SIZE" \
        --max-inventory "$MAX_INVENTORY" \
        --refresh "$REFRESH" \
        --log-level "$LOG_LEVEL" \
        "$@"
}

main "$@"
