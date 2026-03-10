#!/bin/bash
# Trading OS tuning — run at boot or before starting trading containers.
# Install: cp deploy/tune-os.sh /usr/local/bin/ && chmod +x /usr/local/bin/tune-os.sh
# Systemd: cp deploy/trading-tune.service /etc/systemd/system/ && systemctl enable trading-tune

set -euo pipefail

echo "[tune-os] Applying trading OS optimizations..."

# --- Memory ---
sysctl -w vm.swappiness=0
echo always > /sys/kernel/mm/transparent_hugepage/enabled

# --- Disk scheduler (virtio: none is optimal) ---
for dev in /sys/block/vd*/queue/scheduler; do
    [ -f "$dev" ] && echo none > "$dev"
done

# --- Network IRQ affinity: pin to CPU0 ---
# Parse /proc/interrupts for virtio net (input/output) IRQs
awk '/virtio.*-input|virtio.*-output/ {gsub(/:/, "", $1); print $1}' /proc/interrupts | while read -r irq; do
    echo 1 > "/proc/irq/${irq}/smp_affinity" 2>/dev/null || true
done

# --- Network tuning ---
sysctl -p /etc/sysctl.d/99-trading.conf 2>/dev/null || true
ethtool -K eth0 gro off 2>/dev/null || true
tc qdisc replace dev eth0 root pfifo_fast 2>/dev/null || true

echo "[tune-os] Done."
