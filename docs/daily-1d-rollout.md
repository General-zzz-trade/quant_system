# 1D BTC live rollout playbook

This is the runbook for flipping `daily-live-runner.timer` from
prepared (current state) → live trading.

## Current state (prepared, NOT trading)

- `runner/daily_live_runner.py` — ready, but `--live` flag NOT set in
  the service ExecStart, so even if the timer fires the runner is in
  dry-run mode. Audits are written to
  `data/runtime/decision_audit_okx.jsonl` for parity tracking.
- `runner/daily_paper_runner.py` — running daily at UTC 00:30 via
  `daily-paper-runner.timer` (already enabled, low risk).
- `infra/systemd/daily-live-runner.{service,timer}` — installed but the
  timer is `disabled`. It will NOT fire until manually enabled.
- ETHUSDT_1d is intentionally NOT included in `--symbols` (ETH model
  has a Phase 2A holdout-IC concern; needs longer paper window).

## Pre-flight checks (run all before flipping)

```bash
# 1. paper runner has produced ≥3 days of records
tail -10 data/runtime/daily_paper_audit.jsonl

# 2. paper runner state is sane
cat data/runtime/daily_paper_state.json

# 3. live IC kill switch still healthy for BTCUSDT
python3 -m monitoring.live_ic_killswitch | grep BTCUSDT

# 4. dry-run live runner one last time, verify equity + sizing
python3 -m runner.daily_live_runner --symbols BTCUSDT
# ↑ should print equity, EQUITY_PCT_CAP=25%, action=FLAT/HOLD/etc.
# DO NOT pass --live yet.

# 5. parity check still passes
python3 -m scripts.parity_check_1d

# 6. okx-alpha service is healthy (no recent OOM)
journalctl -u okx-alpha.service --since "3 days ago" | grep -i oom
# expect: no output
```

## Day 4 go-live (one-time)

```bash
# Edit the service file to add --live flag.
sudo sed -i \
  's|--symbols BTCUSDT$|--symbols BTCUSDT --live|' \
  /etc/systemd/system/daily-live-runner.service

# Verify the edit
grep ExecStart /etc/systemd/system/daily-live-runner.service
# expect: ExecStart=...runner.daily_live_runner --symbols BTCUSDT --live

# Reload + enable
sudo systemctl daemon-reload
sudo systemctl enable --now daily-live-runner.timer

# Confirm timer is active
systemctl list-timers daily-live-runner.timer --no-pager
```

## What happens next

- The timer fires at UTC 00:35 each day (5 minutes after the paper run).
- The live runner reads the latest 1D bar, computes z, and:
  - If `BTCUSDT_1d` is paused by the live IC kill switch → skips.
  - If exchange already holds any BTC position (1h/4h runner active) →
    skips with "exchange already holds position" — manual trade needed.
  - If z > 1.75 and no position → opens long (25% of equity notional).
  - If z < -1.75 and no position → opens short.
  - If position exists and z reverses past −0.3 sign-corrected → closes.
- Sizing cap: 25% of OKX equity per position (~$94 at $377 equity).
  Override with `DAILY_LIVE_EQUITY_PCT=0.50` env var if needed.
- Audit lands in `data/runtime/decision_audit_okx.jsonl` with type
  `daily_live_signal`, picked up by the existing PnL + IC monitors.

## Emergency stop

```bash
# Disable the timer (the running unit, if any, completes its single shot)
sudo systemctl disable --now daily-live-runner.timer

# Manually pause via kill switch (also affects future runs)
python3 -m monitoring.live_ic_killswitch --force-pause BTCUSDT_1d

# Close any 1D-opened position via the OKX adapter
python3 -c "
from runner.daily_live_runner import _make_okx_adapter, _existing_position_qty
a = _make_okx_adapter()
qty = _existing_position_qty(a, 'BTCUSDT')
if abs(qty) > 1e-8:
    side = 'sell' if qty > 0 else 'buy'
    print(a.send_market_order('BTCUSDT', side, abs(qty)))
"
```

## Rollback to dry-run

```bash
# Re-edit the service file to remove --live
sudo sed -i 's| --live$||' /etc/systemd/system/daily-live-runner.service
sudo systemctl daemon-reload
# (Timer can stay enabled — runner is harmless without --live)
```

## ETH 1D activation (later, separate decision)

Add ETHUSDT to `--symbols` only after:
- 30-day rolling live IC for ETHUSDT_1d > +0.05 (per kill switch monitor)
- At least 2 weeks of ETH 1D paper signals in
  `data/runtime/daily_paper_audit.jsonl` showing positive cumulative PnL.
