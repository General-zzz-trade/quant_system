# Deployment Checklist

## Pre-deployment

- [ ] Verify models exist in `models_v8/` (LightGBM .pkl files)
- [ ] Run full test suite: `pytest tests/ -x -q -m ""`
- [ ] Run Rust tests: `cd ext/rust && cargo test`
- [ ] Build Rust crate: `make rust`
- [ ] Copy .so and verify: `cp $(python3 -c "import _quant_hotpath, os; print(os.path.dirname(_quant_hotpath.__file__))")/*.so _quant_hotpath/ && python3 -c "import _quant_hotpath; print(_quant_hotpath.rust_version())"`
- [ ] Verify config: `python3 -c "import yaml; yaml.safe_load(open('config/production.yaml'))"`
- [ ] Create `config/production.local.yaml` with API keys and site-specific overrides
- [ ] Verify log directory exists: `mkdir -p /quant_system/logs`
- [ ] Verify data directory exists: `mkdir -p /quant_system/data/live`
- [ ] Check system clock sync: `timedatectl status`
- [ ] Verify network connectivity to Bybit: `curl -s https://api.bybit.com/v5/market/time`
- [ ] Run pre-live checklist: `python3 -m scripts.ops.pre_live_checklist`

## Initial deployment (testnet)

1. Set `testnet: true` and `shadow_mode: true` in config
2. Install systemd service:
   ```bash
   sudo cp infra/systemd/bybit-alpha.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable bybit-alpha
   ```
3. Start runner:
   ```bash
   sudo systemctl start bybit-alpha.service
   sudo journalctl -u bybit-alpha.service -f
   ```
4. Verify health endpoint: `curl localhost:9090/metrics`
5. Monitor logs: `tail -f /quant_system/logs/bybit_alpha.log`
6. Run for 24h minimum, verify:
   - [ ] No crashes (check `systemctl status bybit-alpha`)
   - [ ] Signals generated (check logs for `ml_score`)
   - [ ] Reconciliation passes (check logs for reconcile reports)
   - [ ] Health endpoint responsive

## Production deployment

1. Switch config: `testnet: false`, `shadow_mode: false`
2. Set `preflight_min_balance` to required minimum
3. Restart: `sudo systemctl restart bybit-alpha`
4. Monitor first 4 hours closely

## Monitoring

- Health endpoint: `curl localhost:9090/metrics`
- Service status: `systemctl status bybit-alpha`
- Recent logs: `journalctl -u bybit-alpha --since "1 hour ago"`
- Kill switch: `curl -X POST localhost:9090/kill`

## Model hot-reload

Send SIGHUP to reload models without restarting:
```bash
sudo systemctl kill -s SIGHUP bybit-alpha
```
Or use auto-retrain timer:
```bash
sudo systemctl enable --now auto-retrain.timer
```

## Rollback

1. Emergency stop:
   ```bash
   curl -X POST localhost:9090/kill  # Kill switch (fastest)
   sudo systemctl stop bybit-alpha  # Full stop
   ```
2. Revert config:
   ```bash
   git checkout config/production.yaml
   sudo systemctl restart bybit-alpha
   ```
3. Revert models:
   ```bash
   # Models are in models_v8/, restore from backup
   cp /backup/models_v8/*.pkl models_v8/
   sudo systemctl kill -s SIGHUP bybit-alpha
   ```

## Auto-retrain setup

```bash
sudo cp infra/systemd/auto-retrain.service /etc/systemd/system/
sudo cp infra/systemd/auto-retrain.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now auto-retrain.timer
# Verify: systemctl list-timers auto-retrain.timer
```
