# Operations Manual

Current operator guide for the repository as it exists today.

当本文和下列文档冲突时，优先级如下：

1. [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md)
2. [`docs/deploy_truth.md`](/quant_system/docs/deploy_truth.md)
3. [`docs/production_runbook.md`](/quant_system/docs/production_runbook.md)
4. [`docs/execution_contracts.md`](/quant_system/docs/execution_contracts.md)

---

## 1. Current Service Matrix

| 服务 | 入口 | 当前定位 | 默认观察面 |
|---|---|---|---|
| `bybit-alpha.service` | `python3 -m scripts.run_bybit_alpha --symbols ... --ws` | 当前活跃的方向性 alpha host service | `systemctl` + `logs/bybit_alpha.log` + Bybit 账户状态 |
| `bybit-mm.service` | `python3 -m scripts.run_bybit_mm --symbol ETHUSDT ...` | 当前活跃的做市 host service | `systemctl` + `logs/bybit_mm.log` + Bybit 账户状态 |
| `quant-framework` / `quant-runner.service` | `runner.live_runner` | framework candidate path | health server + reconcile + checkpoint + operator APIs（仅此路径具备） |

重要边界：

- `bybit-alpha.service` 当前不暴露 framework `/control` / `/ops-audit`
- `bybit-mm.service` 当前也不暴露 framework control plane
- `POST /control`、`GET /operator`、`GET /ops-audit` 只对 `LiveRunner` path 成立

---

## 2. Deployment And Release

当前存在两条发布面：

- host 上真实交易服务：systemd
- CI / GitHub Actions：compose

具体真相：

- 当前 host 上活跃的是 `bybit-alpha.service` 和 `bybit-mm.service`
- [`.github/workflows/deploy.yml`](/quant_system/.github/workflows/deploy.yml) 默认只会跑 [`scripts/deploy.sh`](/quant_system/scripts/deploy.sh)
- [`scripts/deploy.sh`](/quant_system/scripts/deploy.sh) 默认只部署 `quant-paper`
- 如果提交改到了 host-managed trading artifacts，deploy workflow 会直接失败并要求人工同步 systemd 路径

所以：

- CI/CD 不是当前 host trading service 的统一控制面
- compose deploy 成功，不等于 systemd 交易服务已经更新

---

## 3. Monitoring Truth

### 3.1 Active host services

当前最可靠的健康信号是：

- `systemctl status`
- `journalctl`
- 业务日志时间戳是否继续前进
- 交易所账户真实余额 / 持仓 / 挂单 / 成交

当前仓库已经把这套判定固化到 [`scripts/ops/runtime_health_check.py`](/quant_system/scripts/ops/runtime_health_check.py)：

```bash
python3 -m scripts.ops.runtime_health_check
python3 -m scripts.ops.runtime_health_check --service alpha
python3 -m scripts.ops.runtime_health_check --service mm --require-account
python3 -m scripts.ops.runtime_kill_latch --service alpha --json
python3 -m scripts.ops.runtime_kill_latch --service mm --symbol ETHUSDT --json
```

这条命令会同时检查：

- `systemd` 状态
- 日志是否新鲜
- 最近是否有 heartbeat / fill / metrics 等运行证据
- 如果能拿到 Bybit 凭据，再补查账户侧持仓 / 挂单 / 最近成交
- 对 active host services，还会显式暴露持久 kill latch 是否已锁住启动

对 `bybit-mm.service`，当前 [`scripts/run_bybit_mm.py`](/quant_system/scripts/run_bybit_mm.py) 还增加了 market-data stale fail-fast：

- 超过 `market_data_stale_s` 没有新的 WS 消息或 orderbook depth，会直接报错退出
- 依赖 `infra/systemd/bybit-mm.service` 的 `Restart=on-failure` 自动拉起
- 运维上不应再接受“行情停了但进程还活着”的假活状态

### 3.2 Framework path only

如果运行的是 `LiveRunner`，才有以下能力：

- `GET /health`
- `GET /status`
- `GET /operator`
- `GET /control-history`
- `GET /execution-alerts`
- `GET /ops-audit`
- `POST /control`

当前 HTTP health server 不提供：

- `/metrics`
- `/kill`

### 3.3 Prometheus

`/metrics` 来自 [`monitoring/metrics/prometheus.py`](/quant_system/monitoring/metrics/prometheus.py) 的 `PrometheusExporter.start()`，不是 stdlib health server 自带能力。

如果没有显式启动 `PrometheusExporter`，就不应假设有 `/metrics`。

---

## 4. Configuration Truth

当前示例配置主要服务于 framework path：

- [`infra/config/examples/live.yaml`](/quant_system/infra/config/examples/live.yaml)
- [`infra/config/examples/paper_trading.yaml`](/quant_system/infra/config/examples/paper_trading.yaml)
- [`infra/config/examples/testnet_v8_gate_v2.yaml`](/quant_system/infra/config/examples/testnet_v8_gate_v2.yaml)

这些 YAML 的真实定位是：

- 面向 `LiveRunner.from_config()`
- 支持 flat 和 nested 两种 schema
- 当前多为 Binance / testnet / framework 示例
- 不等于 `bybit-alpha.service` 当前的 systemd 运行配置

示例配置说明见 [`infra/config/examples/README.md`](/quant_system/infra/config/examples/README.md)。

---

## 5. CI Truth

当前 CI 真实执行内容见 [`.github/workflows/ci.yml`](/quant_system/.github/workflows/ci.yml)：

### `test`

- 构建 `ci` 镜像
- 构建 `paper` 镜像
- `docker compose config`
- 默认 compose smoke test
- `tests/`（排除 `tests/performance`）
- 单独的 `tests/integration/test_production_integration_e2e.py`
- `execution/tests/`
- Rust tests: `cargo test --manifest-path ext/rust/Cargo.toml --locked`

### `lint`

- `ruff check --select E,W,F .`
- strict `mypy`

### `security`

- `python3 -m scripts.ops.security_scan`

### `model-check`

- 校验 `scripts.ops.config.SYMBOL_CONFIG` 对应的 `models_v8/*/config.json`

补充说明：

- `pytest.ini` 默认排除了 `benchmark` 和 `slow`
- 默认 CI 不会自动跑 `tests/performance`
- 默认 CI 不会跑所有 slow tests
- Rust crate 默认 test feature 现在不再强绑 `pyo3/extension-module`
- 需要构建 Python 扩展时，显式使用 `maturin build --release --features python`

---

## 6. Incident Response

### 6.1 Directional alpha / market maker

排障顺序：

1. 看 systemd
2. 看日志是否继续推进
3. 看交易所账户真相

### 6.2 Framework path

排障顺序：

1. 看 `health` / `status`
2. 看 `operator` / `execution-alerts` / `ops-audit`
3. 看 startup reconcile / periodic reconcile
4. 看 checkpoint / restart 恢复链路

---

## 7. Current Limits

- 当前 active systemd services 还没有统一进 framework control plane
- deploy workflow 还不能代表 host trading service 已更新
- deploy workflow 失败于 scope guard 时，不会再错误触发 compose rollback/recreate
- 示例配置与当前活跃 Bybit directional alpha service 仍是两套装配面
