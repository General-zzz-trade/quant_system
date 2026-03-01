# Operations Manual

Production deployment, monitoring, and incident response for the quant trading system.

## Deployment

### Docker Build

Multi-stage Dockerfile with three targets:

```bash
# Production live trading (minimal deps)
docker build --target live -t quant-system:latest .

# ML training/inference
docker build --target ml -t quant-system:ml .

# Development (all deps + test tools)
docker build --target dev -t quant-system:dev .
```

### Kubernetes

```bash
# Create namespace
kubectl create namespace quant

# Apply all manifests
kubectl apply -f deploy/k8s/priority-class.yaml
kubectl apply -f deploy/k8s/network-policy.yaml
kubectl apply -f deploy/k8s/external-secret.yaml
kubectl apply -f deploy/k8s/pdb.yaml
kubectl apply -f deploy/k8s/deployment.yaml

# Verify
kubectl -n quant get pods
kubectl -n quant logs -f deploy/quant-engine
```

K8s deployment runs as single replica with `Recreate` strategy (no dual-trading). Security hardening:
- Non-root user (UID 1000)
- Read-only root filesystem
- All capabilities dropped
- Seccomp RuntimeDefault profile
- NetworkPolicy restricts egress to exchange APIs and DNS only

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `BINANCE_API_KEY` | Exchange API key | Yes (live) |
| `BINANCE_API_SECRET` | Exchange API secret | Yes (live) |
| `QUANT_ENV` | Environment: `production`, `staging`, `dev` | No (default: `production`) |
| `PYTHONUNBUFFERED` | Disable output buffering | No (set in Dockerfile) |
| `SECRET_CREATED_AT` | ISO timestamp for secret age tracking | No |

## Configuration

Primary config file: YAML loaded by `infra/config/loader.py`. Example at `infra/config/examples/live.yaml`.

### Key Parameters

```yaml
trading:
  symbol: BTCUSDT              # Primary trading pair
  symbols: [BTCUSDT, ETHUSDT]  # Multi-symbol mode
  exchange: binance             # Venue: binance

risk:
  max_position_notional: 25000  # Max USD notional per symbol
  max_leverage: 5.0             # Max leverage multiplier
  max_drawdown_pct: 10.0        # Kill switch trigger (%)
  max_orders_per_minute: 20     # Rate limit

execution:
  fee_bps: 2.0                 # Maker/taker fee in basis points
  slippage_bps: 1.5            # Expected slippage

monitoring:
  health_check_interval: 10.0  # Stale data detection (seconds)
  metrics_port: 9090           # Prometheus metrics port
```

### Risk Gate Defaults (RiskGateConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_position_notional` | 100,000 | Max notional per symbol (USD) |
| `max_order_notional` | 50,000 | Max single order notional (USD) |
| `max_open_orders` | 20 | Max concurrent open orders |
| `max_portfolio_notional` | 500,000 | Max total portfolio notional (USD) |

### Correlation Gate Defaults (CorrelationGateConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_avg_correlation` | 0.70 | Max avg pairwise correlation |
| `max_position_correlation` | 0.85 | Max correlation for new position |
| `min_data_points` | 20 | Min observations before gate activates |

### LiveRunner Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reconcile_interval_sec` | 60 | Position reconciliation interval |
| `margin_warning_ratio` | 0.15 | Margin warning threshold |
| `margin_critical_ratio` | 0.08 | Margin critical / kill switch trigger |
| `pending_order_timeout_sec` | 30 | Timeout for unfilled orders on shutdown |
| `latency_p99_threshold_ms` | 5000 | SLA breach threshold for signal-to-fill |
| `shadow_mode` | false | Simulate orders without executing |
| `health_port` | null | HTTP health endpoint port |

## Monitoring

### Prometheus Metrics

The `PrometheusExporter` exposes metrics on port 9090 at `/metrics`:

```
quant_equity_usd           # Current equity (gauge)
quant_fills_total          # Total fills (counter)
quant_position_{symbol}    # Position size per symbol (gauge)
quant_pipeline_latency_ms  # Pipeline processing latency (histogram)
quant_drawdown_pct         # Current drawdown percentage (gauge)
```

### Grafana Dashboards

Pre-configured dashboard panels in `monitoring/dashboards/panels.py`:
- Equity curve and drawdown
- Position sizes by symbol
- Pipeline latency (P50, P95, P99)
- Fill rate and order flow
- Data freshness and error budget

### SLO Definitions (SLOConfig)

| SLO | Target | Metric |
|-----|--------|--------|
| Pipeline latency | P99 < 5s | `pipeline_latency_p99_ms` |
| System availability | 99.9% | `uptime_fraction` |
| Data freshness | < 30s | `data_freshness_sec` |
| Order fill rate | > 95% | `fill_rate` |
| Error budget window | 1 hour | Rolling window |

### Built-in Alert Rules

LiveRunner registers these alerts automatically:

| Alert | Severity | Condition | Cooldown |
|-------|----------|-----------|----------|
| `stale_data` | WARNING | Data age > `health_stale_data_sec` | 120s |
| `high_drawdown` | ERROR | Drawdown > 15% | 300s |
| `kill_switch_triggered` | CRITICAL | Kill switch active | 60s |
| `latency_sla_breach` | WARNING | Signal-to-fill P99 > threshold | 300s |
| `high_correlation` | WARNING | Avg portfolio correlation > threshold | 300s |

## Incident Response

### Kill Switch Activation

The KillSwitch supports scoped circuit breaking:

```python
from risk.kill_switch import KillSwitch, KillMode, KillScope

kill_switch = KillSwitch()

# Hard kill: block all new orders globally
kill_switch.trigger(
    scope=KillScope.GLOBAL, key="*",
    mode=KillMode.HARD_KILL,
    reason="manual_intervention", source="operator",
)

# Reduce-only: allow position closing only
kill_switch.trigger(
    scope=KillScope.SYMBOL, key="ETHUSDT",
    mode=KillMode.REDUCE_ONLY,
    reason="high_volatility", source="risk_monitor",
)

# Check status
record = kill_switch.is_killed()  # Returns KillRecord or None

# Reset
kill_switch.reset(scope=KillScope.GLOBAL, key="*")
```

### Position Unwinding

Graceful shutdown sequence (handled by `GracefulShutdown`):
1. Kill switch triggered (blocks new orders)
2. Wait for pending orders to fill or timeout (default 30s)
3. Save state snapshot to SQLite (if persistent stores enabled)
4. Stop all subsystems (margin monitor, reconciler, health, runtime)

### Manual Order Cancellation

```python
# Via venue client
venue_client.cancel_all_orders()  # Cancel all open orders
venue_client.cancel_order(symbol="BTCUSDT", order_id="12345")
```

## Runbooks

### High Latency (P99 > 1s)

1. Check `quant_pipeline_latency_ms` histogram in Grafana
2. Identify which pipeline stage is slow (feature compute, decision, execution)
3. Check C++ extension status: `python -c "from features._quant_rolling import *; print('OK')"`
4. If C++ not loaded, rebuild: `make rolling`
5. Check exchange API latency (network, rate limiting)
6. Review open order count — reduce if > 15 concurrent orders

### Stale Market Data (> 60s)

1. Check WebSocket connection status in logs
2. `BinanceMarketDataRuntime` has automatic reconnection with REST fallback
3. Verify exchange status: `curl -s https://fapi.binance.com/fapi/v1/ping`
4. Check NetworkPolicy if running in K8s — ensure egress to `*.binance.com:443`
5. If persistent, restart the pod: `kubectl -n quant rollout restart deploy/quant-engine`

### Model Drift Detected

1. Check drift monitoring logs: `grep "drift" logs/live_trading.log`
2. Review alpha model performance: `python -m tools.alpha_diagnostics --model lgbm_v1`
3. If concept drift confirmed:
   - Switch to shadow mode: add `--shadow` flag
   - Retrain: `python -m alpha.training.train_lgbm --config infra/config/training.yaml`
   - Validate on holdout before redeploying
4. Experimental models (LSTM/Transformer) have automatic OOD detection — check `ood_score` metric

### Margin Utilization High (> 80%)

1. MarginMonitor triggers automatically at `warning_margin_ratio` (15% free margin)
2. At `critical_margin_ratio` (8% free margin), kill switch activates
3. Manual response:
   - Review positions: largest notional exposures
   - Reduce position sizes or close unprofitable trades
   - Check funding rate impact (perpetual contracts settle 3x daily)

### Exchange Connection Lost

1. WebSocket client implements automatic reconnect with exponential backoff
2. REST fallback fetches latest kline data during WS outage
3. If both fail:
   - Check API key validity and IP whitelist
   - Verify exchange maintenance schedule
   - Kill switch activates on stale data if configured
4. After reconnection, startup reconciliation compares local vs exchange state

## Backup & Recovery

### State Persistence

When `enable_persistent_stores: true`:
- `data/live/state.db` — Engine state snapshots (SQLite)
- `data/live/ack_store.db` — Order acknowledgment tracking
- `data/live/event_log.db` — Full event audit log

### Snapshot Restore

On startup, LiveRunner automatically:
1. Loads latest state checkpoint from SQLite
2. Restores `EngineCoordinator` state (positions, account, event index)
3. Runs startup reconciliation against exchange state
4. Logs any mismatches (local vs exchange position/balance)

### K8s Persistent Volume

The deployment mounts a PVC at `/app/data/live` for state persistence:

```yaml
volumes:
  - name: data
    persistentVolumeClaim:
      claimName: quant-engine-data
```

## Secret Management

### External Secrets Operator

Exchange credentials are managed via External Secrets (`deploy/k8s/external-secret.yaml`):

```
SecretStore (quant-secret-store)
  -> ExternalSecret (quant-engine-secrets)
    -> K8s Secret (auto-created)
      -> Pod envFrom
```

Supports AWS Secrets Manager (production) or K8s native secrets (dev/staging).
Refresh interval: 1 hour.

### Secret Rotation

Weekly CronJob (`deploy/k8s/secret-rotation-cronjob.yaml`) checks API key age:
- Schedule: Monday 06:00 UTC
- Max age: 90 days
- Exits with error if key is expired (triggers alert)

### Config Security

`infra/config/loader.py` enforces:
- Credentials resolved from environment variables (`api_key_env`, `api_secret_env`)
- Plaintext secrets in config files are rejected
- Schema validation before runner construction

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

| Job | Trigger | Description |
|-----|---------|-------------|
| `test` | push/PR | Unit + integration tests (Python 3.11, 3.12) |
| `lint` | push/PR | Ruff + Mypy strict checks |
| `security` | push/PR | pip-audit dependency vulnerability scan |
| `docker` | main only | Docker build + Trivy container scan (CRITICAL/HIGH) |

Coverage threshold: 60% minimum.
