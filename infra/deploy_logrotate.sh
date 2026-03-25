#!/usr/bin/env bash
# Deploy logrotate config for quant system logs.
# Usage: sudo bash infra/deploy_logrotate.sh
set -euo pipefail

SRC="/quant_system/infra/logrotate.d/quant-system"
DST="/etc/logrotate.d/quant-system"

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found"
    exit 1
fi

cp "$SRC" "$DST"
chmod 644 "$DST"

# Validate
logrotate -d "$DST" 2>&1 | head -5
echo "Logrotate config deployed. Test with: sudo logrotate -f $DST"
