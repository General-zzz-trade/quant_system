# Deployment Checklist

> 当前 checklist 按“host systemd 服务”和“framework path”分开写，避免再把不同运行时的步骤混在一起。

## 1. 通用前置检查

- [ ] `make rust`
- [ ] `python3 -m pytest tests/ -x -q --tb=short --ignore=tests/performance`
- [ ] `python3 -m pytest execution/tests/ -x -q --tb=short`
- [ ] `cargo test --manifest-path rust/Cargo.toml --locked`
- [ ] `ruff check --select E,W,F .`
- [ ] `python3 -m scripts.ops.security_scan`
- [ ] 若本次变更涉及 `infra/systemd/bybit-*.service`、`scripts/run_bybit_mm.py`、`scripts/ops/run_bybit_alpha.py` 或 `execution/market_maker/`，不要把 compose deploy workflow 当成 host trading deploy
- [ ] `docker compose config >/dev/null`
- [ ] `bash -n scripts/deploy.sh`
- [ ] 如果需要重建 Python Rust 扩展，使用 `maturin build --release --features python`
- [ ] `mkdir -p /quant_system/logs /quant_system/data/live`
- [ ] `timedatectl status`
- [ ] `.env` 或 host 环境中已准备好所需凭据

## 2. Directional Alpha (`bybit-alpha.service`)

### 2.1 启动前确认

- [ ] 已同步 [`infra/systemd/bybit-alpha.service`](/quant_system/infra/systemd/bybit-alpha.service) 到 `/etc/systemd/system/`
- [ ] `/quant_system/.env` 已存在，且包含 `BYBIT_API_KEY` / `BYBIT_API_SECRET` / `BYBIT_BASE_URL`
- [ ] 目标账户是 Bybit `demo` 还是 `live` 已确认
- [ ] 模型目录 `models_v8/*/config.json` 存在且可读取

### 2.2 启动

```bash
sudo cp infra/systemd/bybit-alpha.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bybit-alpha.service
sudo systemctl restart bybit-alpha.service
sudo systemctl status bybit-alpha.service --no-pager -l
```

### 2.3 启动后验证

- [ ] `logs/bybit_alpha.log` 出现当前时间戳
- [ ] 已看到 WebSocket 连接日志
- [ ] 已看到 heartbeat 或实际下单 / 成交日志
- [ ] Bybit 账户侧余额、持仓、挂单查询正常

推荐直接执行：

```bash
python3 -m scripts.ops.runtime_health_check --service alpha
```

## 3. Market Maker (`bybit-mm.service`)

### 3.1 启动

```bash
sudo cp infra/systemd/bybit-mm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bybit-mm.service
sudo systemctl restart bybit-mm.service
sudo systemctl status bybit-mm.service --no-pager -l
```

### 3.2 启动后验证

- [ ] `logs/bybit_mm.log` 出现当前时间戳
- [ ] 已看到 WS subscribed / quote / fill / metrics 中至少一种新日志
- [ ] Bybit 账户侧有挂单、持仓或新成交
- [ ] 不把“只有 PID 活着”误判成“做市已恢复”

推荐直接执行：

```bash
python3 -m scripts.ops.runtime_health_check --service mm
```

## 4. Compose Path

### 4.1 默认 compose smoke / paper

```bash
docker compose up -d quant-paper
docker compose ps
docker compose logs -f quant-paper
```

### 4.2 Live compose

```bash
docker compose --profile live up -d quant-live
docker compose --profile framework up -d quant-framework
```

说明：

- `quant-paper` / `quant-live` 运行的是 `scripts.run_bybit_alpha`
- `quant-framework` 运行的是 `runner.live_runner`

## 5. Framework Path Only

以下只对 `runner.live_runner` 成立：

- [ ] `GET /health`
- [ ] `GET /operator`
- [ ] `GET /execution-alerts`
- [ ] `GET /ops-audit`
- [ ] startup reconcile 正常
- [ ] checkpoint / restore 正常

注意：

- health server 当前没有 `/kill`
- `/metrics` 只有在单独启动 `PrometheusExporter` 时才存在

## 6. Rollback

### 6.1 systemd services

```bash
sudo systemctl stop bybit-alpha.service
sudo systemctl stop bybit-mm.service
```

必要时回滚模板后重新：

```bash
sudo cp infra/systemd/bybit-alpha.service /etc/systemd/system/
sudo cp infra/systemd/bybit-mm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart bybit-alpha.service
sudo systemctl restart bybit-mm.service
```

### 6.2 compose

```bash
bash scripts/deploy.sh quant-paper
```

或按 workflow 逻辑恢复 rollback image 后重建 `quant-paper`。

## 7. Auto-Retrain

`auto-retrain.service` / `auto-retrain.timer` 是独立运维工件，不等于 live runtime 自带能力。

```bash
sudo cp infra/systemd/auto-retrain.service /etc/systemd/system/
sudo cp infra/systemd/auto-retrain.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now auto-retrain.timer
```
