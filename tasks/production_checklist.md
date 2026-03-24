# Production Deployment Checklist

> **Status**: COMPLETED (2026-03-24) — 历史 testnet 检查表，面向旧版 `testnet_v8_gate_v2` 路径.
> 当前生产入口: `runner/alpha_main.py` (Strategy H: BTC+ETH x 1h/4h, Bybit demo).
> 当前架构请参考 [`CLAUDE.md`](/quant_system/CLAUDE.md).
> 部署文档: [`docs/deploy_truth.md`](/quant_system/docs/deploy_truth.md),
> [`docs/operations.md`](/quant_system/docs/operations.md),
> [`docs/production_runbook.md`](/quant_system/docs/production_runbook.md)

## Pre-deployment

- [ ] Model files exist: `models_v8/BTCUSDT_gate_v2/config.json`, `lgbm_v8.pkl`, `xgb_v8.pkl`
- [ ] Bear model exists: `models_v8/BTCUSDT_bear_c/config.json`, `lgbm_bear_c.pkl`
- [ ] Config validated: `python3 -c "from infra.config.loader import load_config_secure; load_config_secure('infra/config/examples/testnet_v8_gate_v2.yaml')"`
- [ ] Env vars set: `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET`
- [ ] Docker build: `docker compose build testnet-trader`
- [ ] Docker config valid: `docker compose config`

## Phase 1: Paper (5 min)

```bash
python3 -m runner.testnet_validation \
  --config infra/config/examples/testnet_v8_gate_v2.yaml \
  --phase paper --duration 300
```

- [ ] WS connection established (check logs for "WebSocket connected")
- [ ] 85 features computing (check LONGRUN STATUS log for features=N/85)
- [ ] No crashes or unhandled exceptions
- [ ] Pipeline latency < 5s (check logs)

## Phase 2: Shadow (5 min)

```bash
python3 -m runner.testnet_validation \
  --config infra/config/examples/testnet_v8_gate_v2.yaml \
  --phase shadow --duration 300
```

- [ ] ML signals generating (shadow_events.json has entries)
- [ ] ML score non-zero in at least some events
- [ ] No execution errors in logs

## Phase 3: Live Testnet (5 min)

```bash
python3 -m runner.testnet_validation \
  --config infra/config/examples/testnet_v8_gate_v2.yaml \
  --phase live --duration 300
```

- [ ] Orders submitted to testnet
- [ ] Fills received (live_equity.csv has entries)
- [ ] Position reconciliation passes

## Phase 4: Long-run (24h+)

```bash
python3 -m runner.testnet_validation \
  --config infra/config/examples/testnet_v8_gate_v2.yaml \
  --phase longrun
```

- [ ] WS reconnection works (disconnect and verify auto-reconnect)
- [ ] State persisted to SQLite (check validation_output/state.db)
- [ ] SIGHUP model reload works: `kill -HUP <pid>`
- [ ] Memory stable after 24h (no leaks)
- [ ] Feature completeness stays at 85/85

## Phase 5: Compare

```bash
python3 -m runner.testnet_validation \
  --config infra/config/examples/testnet_v8_gate_v2.yaml \
  --phase compare
```

- [ ] Paper vs live equity correlation > 0.9
- [ ] Return divergence < 5%

## Mainnet Go-Live

- [ ] Switch config to mainnet: `trading.testnet: false`
- [ ] Real API keys: `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- [ ] Start with minimum position: `order_qty: 0.001`
- [ ] Monitor Grafana dashboard for 1h
- [ ] Gradually increase position size after 24h stable operation
