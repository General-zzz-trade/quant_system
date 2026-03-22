# Operations Manual (2026-03-22)

Current operator guide. When this conflicts with other docs, priority:

1. [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md)
2. [`docs/deploy_truth.md`](/quant_system/docs/deploy_truth.md)
3. [`docs/production_runbook.md`](/quant_system/docs/production_runbook.md)
4. [`docs/execution_contracts.md`](/quant_system/docs/execution_contracts.md)

---

## 1. Daily Operations

### Automated (no action needed)

| Timer/Cron | Schedule | What |
|---|---|---|
| `health-watchdog.timer` | every 5 min | Service health + data freshness + Telegram alerts + auto-restart stale |
| `ic-decay-monitor.timer` | daily 3am UTC | IC decay check per symbol; RED = retrain needed |
| `data-refresh.timer` | every 6h | Kline + funding + OI sync from Binance |
| `daily-retrain.timer` | daily 2am UTC | 4h models retrain + SIGHUP hot-reload |
| `auto-retrain.timer` | Sunday 2am UTC | 1h walk-forward retrain with IC/Sharpe gates |
| cron: `demo_tracker` | hourly | Track record update from logs |
| cron: `weekly_report` | Sunday 3am | Weekly performance report |
| cron: `auto_bug_scan` | Sunday 1am | Static bug scan (30 patterns) |
| cron: OI download | every 6h | Binance OI history (28d retention, must accumulate) |

### Manual checks

```bash
# Signal reconciliation (live vs backtest)
python3 -m scripts.ops.signal_reconcile --hours 24

# Heartbeat check
tail -f logs/bybit_alpha.log | grep HEARTBEAT

# Timer status
sudo systemctl list-timers --all | grep -E "health|retrain|refresh|decay"

# Health watchdog (manual trigger)
python3 -m scripts.ops.health_watchdog
python3 -m scripts.ops.health_watchdog --json
```

---

## 2. Service Management

### Active services

| Service | Entry | Status |
|---|---|---|
| `bybit-alpha.service` | `python3 -m scripts.ops.run_bybit_alpha --symbols BTCUSDT ETHUSDT ETHUSDT_15m --ws` | BTC+ETH Alpha (BTC Sharpe 4.37, ETH 4.67) |
| polymarket collector | `python3 -m polymarket.collector --mode intra --db data/polymarket/collector.db` | Background process (PID in `data/polymarket/collector.pid`) |
| polymarket dryrun | `python3 -m scripts.run_polymarket_dryrun --bet-size 10 --rsi-low 30 --rsi-high 70` | RSI taker validation |

### Common operations

```bash
# Start / restart / stop
sudo systemctl start bybit-alpha.service
sudo systemctl restart bybit-alpha.service    # checkpoints ensure instant recovery
sudo systemctl stop bybit-alpha.service

# Status and logs
sudo systemctl status bybit-alpha.service
journalctl -u bybit-alpha.service -f

# Hot-reload models (no restart needed)
kill -HUP $(systemctl show -p MainPID bybit-alpha.service | cut -d= -f2)
```

### Inactive services (validated but not recommended)

- `hft-signal.service` -- Sharpe -5, NOT recommended
- `bybit-mm.service` -- BTC spread < fee, mathematically unprofitable

---

## 3. Model Management

### Automated retraining

- **Daily (2am UTC)**: `daily-retrain.timer` -- 4h models + SIGHUP hot-reload to running service
- **Weekly (Sunday 2am)**: `auto-retrain.timer` -- 1h walk-forward retrain with IC/Sharpe validation gates

### Manual retraining

```bash
# Retrain 1h + 4h models with hot-reload
python3 -m scripts.auto_retrain --include-4h --sighup

# Retrain 1h + 15m models
python3 -m scripts.auto_retrain --include-15m --force

# Retrain 15m only
python3 -m scripts.auto_retrain --only-15m --force

# Dry-run (preview without saving)
python3 -m scripts.auto_retrain --dry-run

# Validate all production models
python3 -m scripts.training.train_all_production --dry-run

# Force retrain all production models
python3 -m scripts.training.train_all_production --force
```

### IC decay monitoring

```bash
# Automated via ic-decay-monitor.timer (daily 3am UTC)
# Manual check:
python3 -m monitoring.ic_decay_monitor
# RED = retrain needed, triggers Telegram alert
```

---

## 4. Data Pipeline

```bash
# Cross-market data (also updates ETF volume)
python3 -m scripts.data.download_cross_market

# Stablecoin supply
python3 -m scripts.data.download_stablecoin_supply

# On-chain data
python3 -m scripts.data.download_onchain

# Deribit implied volatility
python3 -m scripts.data.download_deribit_iv --all

# Deribit options put/call ratio
python3 -m scripts.data.download_deribit_pcr --all

# Standard data (handled by data-refresh.timer, but can run manually)
python3 -m scripts.data.download_15m_klines
python3 -m scripts.data.download_5m_klines --symbols ETHUSDT
python3 -m scripts.data.download_funding_rates --symbols ETHUSDT SOLUSDT
python3 -m scripts.data.download_oi_data --symbols ETHUSDT BTCUSDT
python3 -m scripts.data.download_multi_exchange_funding --symbols ETHUSDT
```

---

## 5. Troubleshooting

### 4h runner produces no signals

Check if watchdog is restarting the service. The 4h timeframe has `max_silent_s=18000` (5 hours) -- long silence is normal between bars.

### Model not loading

Check `model_loader.py` for `IsADirectoryError`. The `xgb_path.is_file()` guard prevents loading directories as model files.

### Signal mismatch (live vs backtest)

```bash
python3 -m scripts.ops.signal_reconcile --hours 24
python3 -m scripts.ops.compare_live_backtest --log-file logs/bybit_alpha.log
```

### IC decay detected

```bash
python3 -m monitoring.ic_decay_monitor
# RED status = retrain needed
# Trigger manual retrain:
python3 -m scripts.auto_retrain --force --sighup
```

### Service keeps crashing

```bash
# Check recent logs
journalctl -u bybit-alpha.service --since "1 hour ago"

# Health watchdog status
cat data/runtime/health_status.json
cat data/runtime/alert_history.jsonl | tail -5

# Full health check
python3 -m scripts.ops.health_watchdog --json
```

### Bybit order rejections

- `Qty invalid`: `_round_to_step()` not applied -- check all order paths
- `ab not enough`: margin pre-flight check failed -- check available balance
- `MAX_ORDER_NOTIONAL = $5,000` hard limit enforced in both AlphaRunner and PortfolioCombiner

---

## 6. Monitoring and Diagnostics

```bash
# Ops dashboard (unified status)
python3 -m scripts.ops.ops_dashboard

# Pre-live readiness check
python3 -m scripts.ops.pre_live_checklist

# Security audit
python3 -m scripts.ops.security_scan

# Static bug scan
python3 -m scripts.ops.auto_bug_scan --severity warning

# Test Telegram notifications
python3 -m monitoring.notify

# Weekly performance report
python3 -m scripts.ops.weekly_report

# Track record update
python3 -m scripts.ops.demo_tracker
```

---

## 7. Current Limits

- Production scope: **BTC + ETH only** (altcoins removed 2026-03-21 due to poor liquidity)
- Kelly optimal leverage: **1.4x** (half-Kelly 0.7x; demo uses 10x)
- `MAX_ORDER_NOTIONAL = $5,000` hard cap
- Active systemd services do not expose framework control plane (`/control`, `/ops-audit`)
- Deploy workflow does not auto-update host trading services (systemd managed separately)
- Binance OI API retains only ~28 days -- must accumulate via cron
