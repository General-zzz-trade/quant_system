#!/bin/bash
# Quant System Backup — run daily via cron
# Backs up: .env, models, checkpoints, data, logs
# Usage: bash scripts/ops/backup.sh [/path/to/backup/dir]

set -euo pipefail
BACKUP_DIR="${1:-/home/ubuntu/backups}"
DATE=$(date +%Y%m%d_%H%M)
DEST="$BACKUP_DIR/quant_backup_$DATE"

mkdir -p "$DEST"

echo "Backing up quant system to $DEST..."

# Critical config (not in git)
cp .env "$DEST/.env" 2>/dev/null || true

# Model weights
tar -czf "$DEST/models.tar.gz" models_v8/*/config.json models_v8/*/*.pkl 2>/dev/null || true

# Checkpoints (for instant restart)
cp -r data/runtime/checkpoints "$DEST/checkpoints" 2>/dev/null || true

# Trading state
cp -r data/runtime "$DEST/runtime" 2>/dev/null || true

# Options data (accumulating)
cp data/options/*.db "$DEST/" 2>/dev/null || true

# Recent logs (last 7 days)
find logs/ -name "*.log" -mtime -7 -exec cp {} "$DEST/" \; 2>/dev/null || true

# Walk-forward results
cp -r results/walkforward "$DEST/walkforward" 2>/dev/null || true

# Package versions
cp requirements.lock.txt "$DEST/" 2>/dev/null || true

# Cleanup old backups (keep last 7)
ls -dt "$BACKUP_DIR"/quant_backup_* 2>/dev/null | tail -n +8 | xargs rm -rf 2>/dev/null || true

SIZE=$(du -sh "$DEST" | cut -f1)
echo "Backup complete: $DEST ($SIZE)"
