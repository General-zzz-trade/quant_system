# Grafana + Prometheus setup

Lightweight observability stack for the quant-system.  The Prometheus
exporter is **already built-in** (Python, no extra deps).  Prometheus
and Grafana themselves must be installed separately.

## 1. Start the exporter

```bash
sudo cp /quant_system/infra/systemd/prometheus-exporter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now prometheus-exporter.service

# Sanity check:
curl -s http://127.0.0.1:9102/metrics | head -30
```

The exporter:
- Listens on `127.0.0.1:9102/metrics` (not externally exposed by default)
- Polls `data/runtime/*.json` + `decision_audit_*.jsonl` + `systemctl show`
- Zero extra Python deps (pure stdlib `http.server`)
- ~30 metric families covering services / signals / positions / IC / PnL

## 2. Install Prometheus

```bash
sudo apt install prometheus
sudo cp /quant_system/infra/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
sudo cp /quant_system/infra/prometheus/alerts.yml /etc/prometheus/alerts.yml
sudo systemctl restart prometheus
# UI: http://localhost:9090
```

Test scrape:
```
http://localhost:9090/targets  → should show quant-system UP
```

## 3. Install Grafana

```bash
sudo apt install -y grafana
sudo systemctl enable --now grafana-server
# UI: http://localhost:3000 (default admin/admin)
```

Add Prometheus data source:
- Configuration → Data sources → Add → Prometheus
- URL: `http://localhost:9090`
- Save & test

## 4. Import the dashboard

Dashboards → Import → Upload JSON file →
`/quant_system/infra/grafana/dashboards/quant_system.json`

The dashboard has 11 panels covering:
- Service status + memory
- Model IC health
- Signal z-score per venue/symbol
- Signal direction over time
- Position notional
- Portfolio cap usage (fractional)
- 24h trades + PnL per venue
- OKX ramp level + cap

## 5. (Optional) Alertmanager → Telegram

Prometheus alerts fire from `alerts.yml`:
- `ServiceDown`      (CRITICAL, 2min)
- `ExporterStale`    (WARNING, 3min)
- `PortfolioCapExceeded` (WARNING)
- `ICHealthRed`      (WARNING, 30min)
- `Large24hLoss`     (WARNING, -$200+)

To route to Telegram, install Alertmanager and point at the
`monitoring/notify.py` webhook (not covered here — the existing
Telegram path from `daily_pnl_alert.py` works without Alertmanager).

## Metrics reference

| Metric | Labels | Type | Source |
|---|---|---|---|
| quant_service_up | service | gauge | systemctl show |
| quant_service_memory_bytes | service | gauge | systemctl show |
| quant_signal_zscore | venue, symbol | gauge | decision_audit_{venue}.jsonl |
| quant_signal_direction | venue, symbol | gauge | decision_audit_{venue}.jsonl |
| quant_position_qty | venue, symbol | gauge | portfolio_risk.json |
| quant_position_notional_usd | venue, symbol | gauge | portfolio_risk.json |
| quant_position_is_testnet | venue, symbol | gauge | portfolio_risk.json |
| quant_portfolio_long_usd | symbol | gauge | portfolio_risk.json |
| quant_portfolio_short_usd | symbol | gauge | portfolio_risk.json |
| quant_portfolio_cap_usd | symbol | gauge | strategy/config.py |
| quant_portfolio_cap_usage | symbol, side | gauge | derived |
| quant_model_ic_status | model | gauge | ic_health.json |
| quant_model_ic_training | model, horizon | gauge | ic_health.json |
| quant_model_ic_live | model, horizon, window | gauge | ic_health.json |
| quant_okx_ramp_level | — | gauge | okx_ramp_state.json |
| quant_okx_ramp_cap_usd | — | gauge | okx_ramp_state.json |
| quant_trades_24h | venue | gauge | daily_pnl_alert |
| quant_pnl_24h_usd | venue | gauge | daily_pnl_alert |
| quant_exporter_last_scrape_unixtime | — | gauge | heartbeat |
