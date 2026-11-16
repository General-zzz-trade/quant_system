#!/usr/bin/env bash
# Install or update this host for the Binance target runtime (ADR-0016 point 6, ADR-0017; S8).
#
#   sudo bash /quant_system/infra/install_host.sh
#
# Idempotent: running it again changes only what differs. It reads no secret (never .env), and it never enables,
# starts or restarts a service: a person does that, after the steps printed at the end. Any failing step stops
# the script (set -e); nothing after it runs.
#
# Host layout (tech lead, 2026-11-16): one network per host; the checkout at /quant_system; state, backups and
# the host lock directory under /var/lib/quant, owned by the service user.
set -euo pipefail

CHECKOUT=/quant_system
SERVICE_USER=ubuntu
STATE_ROOT=/var/lib/quant

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run as root (sudo bash $CHECKOUT/infra/install_host.sh)" >&2
    exit 1
fi
if [ ! -f "$CHECKOUT/requirements/container.lock" ]; then
    echo "ERROR: no checkout at $CHECKOUT (the systemd units assume this path)" >&2
    exit 1
fi
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    echo "ERROR: the service user $SERVICE_USER does not exist" >&2
    exit 1
fi

echo "1/5 Python packages, every one checked against its hash in requirements/container.lock"
python3 -m pip install --require-hashes --no-deps --break-system-packages --ignore-installed \
    -r "$CHECKOUT/requirements/container.lock"

echo "2/5 Directories for state, backups and logs (the host lock directory is created by state_dir init)"
install -d -m 0750 -o "$SERVICE_USER" -g "$SERVICE_USER" "$STATE_ROOT" "$STATE_ROOT/state" "$STATE_ROOT/backups"
install -d -m 0750 -o "$SERVICE_USER" -g "$SERVICE_USER" "$CHECKOUT/logs"

echo "3/5 systemd units: copied and reloaded; nothing is enabled or started"
bash "$CHECKOUT/infra/sync_systemd.sh"

echo "4/5 Log rotation"
bash "$CHECKOUT/infra/deploy_logrotate.sh"

echo "5/5 Offline safety checks"
(cd "$CHECKOUT" && sudo -u "$SERVICE_USER" python3 scripts/check_local_safety.py)

cat <<EOF

Installed. What a person does next (docs/binance_runbook.md):
  1. Write $CHECKOUT/.env from .env.example (chmod 600, owner $SERVICE_USER).
  2. Initialize the state directory once, as $SERVICE_USER:
       python3 -m trading.infrastructure.state_dir --state-dir $STATE_ROOT/state/testnet init
  3. Enable the hourly backup:   systemctl enable --now state-backup@testnet.timer
  4. Start execution by hand:    systemctl start binance-target-testnet.service
EOF
