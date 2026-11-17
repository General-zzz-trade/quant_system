# Deployment Truth

> Rewritten 2026-11-14 from `infra/systemd/` and `infra/sync_systemd.sh`; updated 2026-11-16 evening (R9).
> When this file and those files disagree, the files win; fix this one.

Status: nothing is approved for live trading (see `CLAUDE.md`). The paper unit trades nothing real. The two
execution units trade only when a person starts them; mainnet execution is refused while the signed spec has
not passed. The owner cancelled the testnet stage on 2026-11-16: the first execution is a small-capital mainnet
pilot (`docs/m5_checklist.md`).

## 1. Units in the repository (`infra/systemd/`)

| Unit | Kind / schedule | Runs | Note |
|---|---|---|---|
| `binance-target` | service | `runner.binance_target --spec models_binance/strategy/final_spec.json --dry-run` | paper; state in `data/runtime/paper` |
| `binance-target-mainnet` | service | the same with `--enable-execution --state-dir /var/lib/quant/state/mainnet`, network fixed by `/usr/bin/env BINANCE_TESTNET=0` | the pilot; `Restart=no`, no `[Install]`: a person starts it; no `--stop-pct`, so the stop distance comes from the environment |
| `binance-target-testnet` | service | the same with `--state-dir /var/lib/quant/state/testnet`, network fixed by `/usr/bin/env BINANCE_TESTNET=1` | `Restart=on-failure`, at most 3 starts an hour; used only when a person wants it |
| `quant-alert@` | template, oneshot | `trading.infrastructure.unit_alert %i` | `OnFailure=` of the execution and backup units: one message through the configured alert channel (the alert file, or Telegram), confirmed |
| `state-backup@<network>` | timer, hourly | `trading.infrastructure.state_backup` | checked snapshot of `state.sqlite` into `/var/lib/quant/backups/<network>`, 14 days kept |
| `health-watchdog` | timer, every 5 min | `monitoring.watchdog` | observes the paper unit's status and alerts; never restarts |
| `log-archive` | timer, Sun 05:00 | `scripts/archive_logs.py` | |

All units use `WorkingDirectory=/quant_system`; the runtime units, `quant-alert@` and `health-watchdog` load
`EnvironmentFile=/quant_system/.env`, `log-archive` and `state-backup@` do not. One execution network per host;
state, backups, the alert file directory and the host lock directory are under `/var/lib/quant` (tech lead,
2026-11-16).

## 2. Retired units

`RETIRED_UNITS` in `infra/sync_systemd.sh` lists every removed unit with the reason. Retired on
2026-11-14: `binance-runtime` (closed ML runtime) and the models_v8 analysis timers
(`feature-ablation`, `feature-auto-update`, `feature-ic-analysis`, `ic-decay-monitor`,
`oos-sharpe-monitor`, `daily-paper-runner`, `daily-check`, `live-vs-backtest`). Retired on
2026-11-15 with their code (frozen lines deleted, ADR-0008 3b): `polymarket-collector`,
`polymarket-maker`, `audit-reconcile`, `cross-venue-reconcile`, `okx-ramp-up`, `okx-status-probe`.
Retired on 2026-11-15 with the rest of the frozen lines (M4): `bybit-alpha`, `okx-alpha`,
`daily-live-runner`, `live-ic-killswitch`, `daily-pnl-alert`, `portfolio-risk-monitor`,
`prometheus-exporter`, `quant-backup`. State-directory backup was redone in S8 (`state-backup@`).
CI never deploys; deploys are manual.

## 3. Installing and syncing a host

```bash
sudo bash infra/install_host.sh     # idempotent: hash-checked packages, /var/lib/quant dirs, units, log rotation,
                                    # offline safety checks; reads no secret, enables or starts nothing
sudo bash infra/sync_systemd.sh     # units only: disables + removes retired units, copies changed ones, daemon-reload
```

The script never restarts a service. Restart what changed by hand, after reading the diff:

```bash
sudo systemctl restart binance-target.service
```

## 4. Checks

```bash
systemctl list-units 'binance-target*' 'health-watchdog*' 'state-backup*' --no-pager
ls -l /var/lib/quant/backups/mainnet/                          # an hourly snapshot once state-backup@mainnet runs
cat /var/lib/quant/state/mainnet/target_status.json            # execution (the pilot)
cat /quant_system/data/runtime/paper/target_status.json        # paper (the binance-target unit)
cat /quant_system/data/runtime/health_status.json              # the watchdog's own status
tail -f /quant_system/logs/binance_target_mainnet.log
sudo systemctl list-timers --all --no-pager
```

## 5. Logs

`sudo bash infra/deploy_logrotate.sh` installs `infra/logrotate.d/quant-system` (daily rotation,
14 days kept, 50 MB max size, copytruncate).
