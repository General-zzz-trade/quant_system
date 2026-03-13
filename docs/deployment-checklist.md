# Deployment Checklist

## Pre-deployment

- [ ] Verify models exist in `models_v8/` (LightGBM .pkl files)
- [ ] Run full test suite: `pytest tests/ -x -q -m ""`
- [ ] Run Rust tests: `cd ext/rust && cargo test`
- [ ] Build Rust crate: `make rust`
- [ ] Copy .so: `cp /usr/local/lib/python3.12/dist-packages/_quant_hotpath/*.so /opt/quant_system/_quant_hotpath/`
- [ ] Verify config: `python3 -c "import yaml; yaml.safe_load(open('config/production.yaml'))"`
- [ ] Create `config/production.local.yaml` with API keys and site-specific overrides
- [ ] Verify log directory exists: `mkdir -p /quant_system/logs`
- [ ] Verify data directory exists: `mkdir -p /quant_system/data/live`
- [ ] Check system clock sync: `timedatectl status`
- [ ] Verify network connectivity to Binance: `curl -s https://fapi.binance.com/fapi/v1/ping`

## Initial deployment (testnet)

1. Set `testnet: true` and `shadow_mode: true` in config
2. Install systemd service:
   ```bash
   sudo cp infra/systemd/quant-runner.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable quant-runner
   ```
3. Start runner:
   ```bash
   sudo systemctl start quant-runner
   sudo journalctl -u quant-runner -f
   ```
4. Verify health endpoint: `curl localhost:9090/metrics`
5. Monitor logs: `tail -f /quant_system/logs/runner.log`
6. Run for 24h minimum, verify:
   - [ ] No crashes (check `systemctl status quant-runner`)
   - [ ] Signals generated (check logs for `ml_score`)
   - [ ] Reconciliation passes (check logs for reconcile reports)
   - [ ] Health endpoint responsive

## Production deployment

1. Switch config: `testnet: false`, `shadow_mode: false`
2. Set `preflight_min_balance` to required minimum
3. Restart: `sudo systemctl restart quant-runner`
4. Monitor first 4 hours closely

## Monitoring

- Health endpoint: `curl localhost:9090/metrics`
- Service status: `systemctl status quant-runner`
- Recent logs: `journalctl -u quant-runner --since "1 hour ago"`
- Kill switch: `curl -X POST localhost:9090/kill`

## Model hot-reload

Send SIGHUP to reload models without restarting:
```bash
sudo systemctl kill -s SIGHUP quant-runner
```
Or use auto-retrain timer:
```bash
sudo systemctl enable --now auto-retrain.timer
```

## Rollback

1. Emergency stop:
   ```bash
   curl -X POST localhost:9090/kill  # Kill switch (fastest)
   sudo systemctl stop quant-runner  # Full stop
   ```
2. Revert config:
   ```bash
   git checkout config/production.yaml
   sudo systemctl restart quant-runner
   ```
3. Revert models:
   ```bash
   # Models are in models_v8/, restore from backup
   cp /backup/models_v8/*.pkl models_v8/
   sudo systemctl kill -s SIGHUP quant-runner
   ```

## Auto-retrain setup

```bash
sudo cp infra/systemd/auto-retrain.service /etc/systemd/system/
sudo cp infra/systemd/auto-retrain.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now auto-retrain.timer
# Verify: systemctl list-timers auto-retrain.timer
```
