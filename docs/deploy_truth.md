# Deployment Truth

> Rewritten 2026-11-14 from `infra/systemd/`, `infra/sync_systemd.sh` and `docker-compose.yml`.
> The earlier version (2026-03-22) listed retired timers and unverified Sharpe figures.
> When this file and those files disagree, the files win; fix this one.

Status: nothing is approved for live trading (see `CLAUDE.md`). Every unit below either runs in
paper/dry-run or only observes.

## 1. Units in the repository (`infra/systemd/`)

| Unit | Kind / schedule | Runs | Note |
|---|---|---|---|
| `binance-target` | service | `runner.binance_target --spec models_binance/strategy/final_spec.json --dry-run` | product runtime, paper |
| `binance-target-testnet` | service | the same with `--enable-execution --state-dir /var/lib/quant/state/testnet`, network fixed by `/usr/bin/env BINANCE_TESTNET=1` | execution on the testnet (M5); `Restart=on-failure`, at most 3 starts an hour |
| `binance-target-mainnet` | service | the same with `BINANCE_TESTNET=0` and `/var/lib/quant/state/mainnet` | `Restart=no`, no `[Install]`: a person starts it; refused anyway while the spec has not passed |
| `quant-alert@` | template, oneshot | `trading.infrastructure.unit_alert %i` | `OnFailure=` of the execution and backup units: one confirmed Telegram message |
| `state-backup@<network>` | timer, hourly | `trading.infrastructure.state_backup` | checked snapshot of `state.sqlite` into `/var/lib/quant/backups/<network>`, 14 days kept |
| `health-watchdog` | timer, every 5 min | `monitoring.watchdog` | observes and alerts; never restarts |
| `log-archive` | timer, Sun 05:00 | `scripts/archive_logs.py` | |

All units use `WorkingDirectory=/quant_system`; the runtime units, `quant-alert@` and `health-watchdog` load
`EnvironmentFile=/quant_system/.env`, `log-archive` and `state-backup@` do not. One execution network per host;
state, backups and the host lock directory are under `/var/lib/quant` (tech lead, 2026-11-16).

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

## 4. Docker

The Docker image, `docker-compose.yml` and the manual deploy workflow were deleted on 2026-11-15 (they ran
the deleted shadow process and Bybit runner). On a host that still runs those containers, a person stops them
(`docker compose down` in the old checkout) before pulling. CI never deploys; deploys are manual.

## 5. Checks

```bash
systemctl list-units 'binance-target*' 'health-watchdog*' 'state-backup*' --no-pager
ls -l /var/lib/quant/backups/testnet/                         # an hourly snapshot once state-backup@testnet runs
cat /quant_system/data/runtime/paper/target_status.json     # the unit runs paper (--dry-run); execution: data/runtime/binance_target
cat /quant_system/data/runtime/health_status.json
tail -f /quant_system/logs/binance_target.log
sudo systemctl list-timers --all --no-pager
```

## 6. Logs

`sudo bash infra/deploy_logrotate.sh` installs `infra/logrotate.d/quant-system` (daily rotation,
14 days kept, 50 MB max size, copytruncate).
