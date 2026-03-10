#!/bin/bash
# Install lowlatency kernel + configure GRUB for HFT trading.
# WARNING: Requires reboot to take effect.
#
# Usage: bash deploy/install-lowlatency.sh
# Then: reboot

set -euo pipefail

echo "=== Installing lowlatency kernel ==="
apt-get update
apt-get install -y linux-image-lowlatency-hwe-24.04 linux-headers-lowlatency-hwe-24.04

echo "=== Configuring GRUB ==="
# Backup current GRUB config
cp /etc/default/grub /etc/default/grub.bak.$(date +%s)

# Add isolcpus and skew_tick to existing parameters
CURRENT=$(grep '^GRUB_CMDLINE_LINUX=' /etc/default/grub | sed 's/GRUB_CMDLINE_LINUX="//' | sed 's/"$//')

# Add new params if not present
for param in "isolcpus=1" "skew_tick=1"; do
    if ! echo "$CURRENT" | grep -q "$param"; then
        CURRENT="$CURRENT $param"
    fi
done

sed -i "s|^GRUB_CMDLINE_LINUX=.*|GRUB_CMDLINE_LINUX=\"$CURRENT\"|" /etc/default/grub

echo "=== Updating GRUB ==="
update-grub

echo "=== Installing sysctl config ==="
cp /opt/quant_system/deploy/sysctl-trading.conf /etc/sysctl.d/99-trading.conf

echo "=== Installing tune-os service ==="
cp /opt/quant_system/deploy/tune-os.sh /usr/local/bin/tune-os.sh
chmod +x /usr/local/bin/tune-os.sh
cp /opt/quant_system/deploy/trading-tune.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable trading-tune

echo ""
echo "=== DONE ==="
echo "New kernel: $(dpkg -l | grep linux-image-.*lowlatency | grep ii | awk '{print $3}' | tail -1)"
echo "GRUB_CMDLINE_LINUX: $(grep '^GRUB_CMDLINE_LINUX=' /etc/default/grub)"
echo ""
echo ">>> REBOOT REQUIRED: run 'reboot' to activate lowlatency kernel <<<"
