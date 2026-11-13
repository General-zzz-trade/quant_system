#!/usr/bin/env bash
# Sync systemd service/timer files from repo to system.
# Usage: sudo bash infra/sync_systemd.sh
set -euo pipefail

REPO_DIR="/quant_system/infra/systemd"
SYS_DIR="/etc/systemd/system"
CHANGED=0

# Units removed from the repository for safety. They ran code outside this
# repository with 20x-50x leverage. Copying new files never removes an old
# installed unit, so stop, disable and delete them explicitly.
RETIRED_UNITS=(
    bybit-mm.service
    liq-sniper.service
    hft-signal.service
    binary-signal.service
    # Pointed at modules that do not exist in this repository:
    quant-runner.service
    depth-recorder.service
    defi-flows.service
    defi-flows.timer
    deribit-options.service
    deribit-options.timer
)

for unit in "${RETIRED_UNITS[@]}"; do
    if [ -e "$SYS_DIR/$unit" ]; then
        echo "Retiring $unit"
        systemctl disable --now "$unit" || true
        rm -f "$SYS_DIR/$unit"
        CHANGED=1
    fi
done

for f in "$REPO_DIR"/*.service "$REPO_DIR"/*.timer; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    if ! diff -q "$f" "$SYS_DIR/$base" >/dev/null 2>&1; then
        echo "Updating $base"
        cp "$f" "$SYS_DIR/$base"
        CHANGED=1
    fi
done

if [ "$CHANGED" -eq 1 ]; then
    echo "Reloading systemd daemon..."
    systemctl daemon-reload
    echo "Done. Restart affected services manually."
else
    echo "All systemd files are in sync."
fi
