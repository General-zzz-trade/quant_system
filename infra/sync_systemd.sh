#!/usr/bin/env bash
# Sync systemd service/timer files from repo to system.
# Usage: sudo bash infra/sync_systemd.sh
set -euo pipefail

REPO_DIR="/quant_system/infra/systemd"
SYS_DIR="/etc/systemd/system"
CHANGED=0

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
