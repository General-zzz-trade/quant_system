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
| `health-watchdog` | timer, every 5 min | `monitoring.watchdog` | observes and alerts; never restarts |
| `audit-reconcile` | timer, hourly | `scripts.audit_fill_reconcile --days 7 --alert` | |
| `daily-pnl-alert` | timer, 23:00 UTC | `monitoring.daily_pnl_alert` | |
| `portfolio-risk-monitor` | timer, every 60 s | `scripts.portfolio_risk_monitor` | |
| `prometheus-exporter` | service | `monitoring.prometheus_exporter --port 9102 --bind 127.0.0.1` | |
| `log-archive` | timer, Sun 05:00 | `scripts/archive_logs.py` | |
| `quant-backup` | timer, 04:00 UTC | `scripts/backup_remote.py` | |
| `bybit-alpha` | service | `runner.alpha_main --venue bybit ...` | frozen venue, dry-run |
| `okx-alpha` | service | `runner.alpha_main --venue okx ...` | frozen venue, dry-run |
| `okx-ramp-up` | timer, 6-hourly | `scripts.okx_ramp_up` | report-only |
| `okx-status-probe` | timer, hourly | `monitoring.okx_status_probe` | |
| `cross-venue-reconcile` | timer, 6-hourly | `scripts.cross_venue_reconcile` | |
| `daily-live-runner` | timer, 00:35 UTC | `runner.daily_live_runner --symbols BTCUSDT` | OKX, `--live` refused in code |
| `live-ic-killswitch` | timer, hourly | `monitoring.live_ic_killswitch --update --alert` | |
| `polymarket-collector` | service | `polymarket.collector` | frozen venue |
| `polymarket-maker` | service | `scripts/run_polymarket_maker.py` | frozen venue |

All units use `WorkingDirectory=/quant_system`; all but `daily-pnl-alert`, `log-archive`,
`polymarket-collector` and `quant-backup` load `EnvironmentFile=/quant_system/.env`.

## 2. Retired units

`RETIRED_UNITS` in `infra/sync_systemd.sh` lists every removed unit with the reason. Retired on
2026-11-14: `binance-runtime` (closed ML runtime) and the models_v8 analysis timers
(`feature-ablation`, `feature-auto-update`, `feature-ic-analysis`, `ic-decay-monitor`,
`oos-sharpe-monitor`, `daily-paper-runner`, `daily-check`, `live-vs-backtest`).

## 3. Syncing a host

```bash
sudo bash infra/sync_systemd.sh     # disables + removes retired units, copies changed ones, daemon-reload
```

The script never restarts a service. Restart what changed by hand, after reading the diff:

```bash
sudo systemctl restart binance-target.service
```

## 4. Compose (`docker-compose.yml`)

| Service | Command | Note |
|---|---|---|
| `quant-paper` | `runner.shadow_main --manifest config/shadow.json --dry-run` | signal observation only |
| `quant-live` | `scripts.run_bybit_alpha ...` | profile `live`; frozen venue |
| `quant-framework` | `runner.live_runner ...` | profile `framework`; **broken**: `runner/live_runner.py` does not exist |

CI never deploys; deploys are manual.

## 5. Checks

```bash
systemctl list-units 'binance-target*' 'health-watchdog*' --no-pager
cat /quant_system/data/runtime/binance_target/target_status.json
cat /quant_system/data/runtime/health_status.json
tail -f /quant_system/logs/binance_target.log
sudo systemctl list-timers --all --no-pager
```

## 6. Logs

`sudo bash infra/deploy_logrotate.sh` installs `infra/logrotate.d/quant-system` (daily rotation,
14 days kept, 50 MB max size, copytruncate).
