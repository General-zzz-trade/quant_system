#!/usr/bin/env bash
# Deploy logrotate config for quant system logs.
# Usage: sudo bash infra/deploy_logrotate.sh
set -euo pipefail

SRC="${QUANT_LOGROTATE_SRC:-/quant_system/infra/logrotate.d/quant-system}"
DST="${QUANT_LOGROTATE_DST:-/etc/logrotate.d/quant-system}"

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found"
    exit 1
fi

cp "$SRC" "$DST"
chmod 644 "$DST"

# Validate. The output is collected first: piping it into `head` killed logrotate with SIGPIPE once it printed more
# than five lines, and with pipefail that ended the whole host install (exit 141, 2026-11-17). A rejected config
# still fails the script.
if ! out=$(logrotate -d "$DST" 2>&1); then
    printf '%s\n' "$out"
    echo "ERROR: logrotate rejected $DST"
    exit 1
fi
sed -n '1,5p' <<<"$out"
echo "Logrotate config deployed. Test with: sudo logrotate -f $DST"
