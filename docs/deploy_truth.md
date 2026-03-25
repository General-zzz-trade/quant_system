# Deployment Truth

> 更新时间: 2026-03-22
> 上位真相源: [`runtime_truth.md`](/quant_system/docs/runtime_truth.md)

---

## 1. 当前活跃服务

### bybit-alpha.service — ACTIVE

```
ExecStart: python3 -m runner.alpha_main --symbols BTCUSDT BTCUSDT_4h ETHUSDT ETHUSDT_4h --ws
```

- **4 runners**: BTC+ETH × 1h/4h (Strategy H)
- **2 WS connections**: kline.60, kline.240
- **Demo account**: $35,285 USDT on api-demo.bybit.com
- **Backtest Sharpe**: 2.25 (T-1 corrected, no look-ahead bias)
- **SIGHUP hot-reload**: <200ms model swap without restart
- **Per-runner checkpoints**: instant recovery on restart

### 模型版本 (all trained 2026-03-22)

| Model | Version | Sharpe | Deadzone | Direction |
|-------|---------|--------|----------|-----------|
| BTCUSDT_gate_v2 | v11 | 2.43 | 1.0 | long_only |
| ETHUSDT_gate_v2 | v11 | 3.92 | 0.9 | long_only |
| BTCUSDT_4h | v11-4h | 5.21 | 0.8 | bidir |
| ETHUSDT_4h | v11-4h | 3.63 | 0.5 | bidir |

### 数据源 (7)

cross_market_daily, etf_volume_daily, stablecoin_daily, fear_greed_index, DVOL, funding, on-chain

All features use T-1 shift (no look-ahead bias).

---

## 2. Timers

| Timer | Schedule | Notes |
|-------|----------|-------|
| health-watchdog.timer | every 5min | 5h tolerance for 4h bars |
| data-refresh.timer | every 6h | kline + funding + OI sync |
| auto-retrain.timer | Sunday 2am UTC | 1h models |
| daily-retrain.timer | daily 2am UTC | 4h models + SIGHUP hot-reload |
| ic-decay-monitor.timer | daily 3am UTC | IC decay tracking |

```bash
sudo systemctl list-timers --all | grep -E "health|retrain|refresh|decay"
```

---

## 3. Inactive 服务

| Service | Reason |
|---------|--------|
| hft-signal.service | Sharpe -5 |
| bybit-mm.service | spread < fee |

---

## 4. Systemd 工件

### 4.1 活跃

- `infra/systemd/bybit-alpha.service`
- `infra/systemd/health-watchdog.service` + `.timer`
- `infra/systemd/data-refresh.service` + `.timer`
- `infra/systemd/auto-retrain.service` + `.timer`
- `infra/systemd/daily-retrain.service` + `.timer`
- `infra/systemd/ic-decay-monitor.service` + `.timer`

### 4.2 同步规则

`infra/systemd/*.service` 是仓库真相源。修改后必须同步:

```bash
sudo cp infra/systemd/bybit-alpha.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart bybit-alpha.service
```

凭据通过 `EnvironmentFile=/quant_system/.env` 注入。

---

## 5. Compose 路径 (非主机交易默认)

| Compose 服务 | 命令 | 定位 |
|---|---|---|
| `quant-paper` | `runner.alpha_main ... --dry-run` | CI smoke / paper |
| `quant-live` | `runner.alpha_main ...` | containerized live |
| `quant-framework` | `runner.live_runner --config ...` | framework candidate |

Deploy workflow (`scripts/deploy.sh`) 只管 compose 路径，不会同步 host systemd 服务。

---

## 6. 验证命令

```bash
# 服务状态
sudo systemctl status bybit-alpha.service --no-pager -l
journalctl -u bybit-alpha.service -f
tail -f /quant_system/logs/bybit_alpha.log

# Timers
sudo systemctl list-timers --all | grep -E "health|retrain|refresh|decay"

# Compose
docker compose config >/dev/null
```

---

## 7. 常用操作

```bash
# 重启 alpha 服务
sudo systemctl restart bybit-alpha.service

# 手动触发 retrain
sudo systemctl start auto-retrain.service
sudo systemctl start daily-retrain.service

# Compose
docker compose up -d quant-paper
docker compose --profile live up -d quant-live
```

---

## 8. 日志管理

```bash
# 部署 logrotate
sudo bash infra/deploy_logrotate.sh

# 手动触发轮转
sudo logrotate -f /etc/logrotate.d/quant-system

# 检查日志大小
du -sh /quant_system/logs/*.log | sort -h
```

配置: `infra/logrotate.d/quant-system` — daily rotation, 14 天保留, 50MB maxsize, copytruncate。
