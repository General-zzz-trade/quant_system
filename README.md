# Quant System

Production quantitative trading system for crypto perpetuals.

当前代码库是混合运行时：

- Python 负责运行时装配、交易所 IO、运维与研究工具
- Rust 负责状态推进、特征热路径、去重、部分执行与风控原语

## Truth Sources

优先阅读这些文档，而不是历史规划或旧脚本注释：

- 运行时真相: [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md)
- 运行时契约: [`docs/runtime_contracts.md`](/quant_system/docs/runtime_contracts.md)
- 执行制度: [`docs/execution_contracts.md`](/quant_system/docs/execution_contracts.md)
- 生产排障 / 恢复: [`docs/production_runbook.md`](/quant_system/docs/production_runbook.md)
- 部署真相: [`docs/deploy_truth.md`](/quant_system/docs/deploy_truth.md)
- 模型治理: [`docs/model_governance.md`](/quant_system/docs/model_governance.md)
- 脚本目录说明: [`scripts/README.md`](/quant_system/scripts/README.md)

## Current Runtime Truth

| 路径 | 入口 | 当前定位 |
|---|---|---|
| 方向性 alpha | [`scripts/ops/run_bybit_alpha.py`](/quant_system/scripts/ops/run_bybit_alpha.py) （由 [`scripts/run_bybit_alpha.py`](/quant_system/scripts/run_bybit_alpha.py) 转发） | 当前活跃的多品种 Bybit demo 交易路径；systemd 服务是 `bybit-alpha.service` |
| 高频做市 | [`scripts/run_bybit_mm.py`](/quant_system/scripts/run_bybit_mm.py) | 当前活跃的专用 Bybit 做市路径；systemd 服务是 `bybit-mm.service` |
| 框架型 live runtime | [`runner/live_runner.py`](/quant_system/runner/live_runner.py) | 完整 engine / decision / execution / recovery 框架路径；当前是候选 / 收敛路径，不是主机上的默认交易服务 |

当前仓库最重要的边界是：

- 事实事件推进状态
- 决策模块只读 snapshot
- 执行结果必须重新进入事件链
- 不允许绕过状态真相源直接改仓位 / 账户状态

## Deployment Truth

当前存在两套部署面，不能混为一谈：

- 主机上实际跑交易的服务是 systemd：[`infra/systemd/bybit-alpha.service`](/quant_system/infra/systemd/bybit-alpha.service) 和 [`infra/systemd/bybit-mm.service`](/quant_system/infra/systemd/bybit-mm.service)
- 仓库里的 Docker / GitHub Actions 路径使用 [`docker-compose.yml`](/quant_system/docker-compose.yml) 和 [`scripts/deploy.sh`](/quant_system/scripts/deploy.sh)；它们当前管理的是 `quant-paper` / `quant-live` / `quant-framework`

重要说明：

- `quant-paper` / `quant-live` 容器运行的是 `scripts.run_bybit_alpha`
- `quant-framework` 运行的是 `runner.live_runner`
- GitHub Actions 的 deploy workflow 当前只会重建 / 重启 compose 服务，不会替你管理 `bybit-alpha.service` 或 `bybit-mm.service`
- 如果提交改到了 host-managed trading artifacts，deploy workflow 现在会直接 fail fast，而不是误报“已经部署当前交易服务”
- compose 通过 `.env` 注入 `BYBIT_*` 环境变量；`infra/systemd/bybit-alpha.service` 本身没有声明 `EnvironmentFile`，systemd 部署要靠主机环境或 drop-in 提供凭据

## Quick Start

### Install

```bash
make rust
pip install -e ".[live,data,ml,config,monitoring,dev,test]" --break-system-packages
cp .env.example .env
```

### Run The Active Directional Alpha Path

```bash
python3 -m scripts.run_bybit_alpha --symbols BTCUSDT ETHUSDT ETHUSDT_15m --ws --dry-run
```

当前这个入口是 AlphaRunner 路径，不是 `LiveRunner`。

### Run The Active Market-Maker Path

```bash
python3 -m scripts.run_bybit_mm --symbol ETHUSDT --leverage 20 --dry-run
```

### Check Active Host Runtime Health

```bash
python3 -m scripts.ops.runtime_health_check
python3 -m scripts.ops.runtime_health_check --service alpha
python3 -m scripts.ops.runtime_health_check --service mm
python3 -m scripts.ops.runtime_kill_latch --service alpha --json
```

这条检查不会把“只有 `systemd active`”误判成“交易已经活着”；如果 active host service 被持久 kill latch 锁住，也会直接显示出来。

### Run The Framework Runtime

```bash
python3 -m runner.live_runner --config infra/config/examples/live.yaml --shadow
```

说明：

- 这个命令跑的是 framework path
- `infra/config/examples/*.yaml` 主要面向 `LiveRunner` / Binance 风格配置，不是 `bybit-alpha.service` 当前使用的 systemd 部署配置

### Compose

```bash
docker compose up -d quant-paper
docker compose --profile live up -d quant-live
docker compose --profile framework up -d quant-framework
```

## Testing And CI

当前 CI 真相见 [`.github/workflows/ci.yml`](/quant_system/.github/workflows/ci.yml)。

本地最接近 CI 的命令是：

```bash
python3 -m pytest tests/ -x -q --tb=short --ignore=tests/performance
python3 -m pytest tests/integration/test_production_integration_e2e.py -x -q --tb=short -k "not test_build_with_data_scheduler"
python3 -m pytest execution/tests/ -x -q --tb=short
cargo test --manifest-path ext/rust/Cargo.toml --locked
ruff check --select E,W,F .
mypy core/ regime/ state/ decision/ engine/ event/ features/ execution/ alpha/ attribution/ runner/backtest/ --strict
python3 -m scripts.ops.security_scan
```

补充说明：

- `pytest.ini` 默认排除了 `benchmark` 和 `slow`
- CI 额外会做 compose config 校验和默认 compose smoke test
- Rust 默认测试入口现在就是 `cargo test --manifest-path ext/rust/Cargo.toml --locked`
- Python wheel / extension 构建需要显式启用 `python` feature，例如 `maturin build --release --features python`
- `make test` 目前只覆盖 Python tests + execution tests + Rust tests + lint，不包含 CI 里的 security、model-check 和单独的 framework integration step

## Project Map

- [`runner/`](/quant_system/runner): framework runtime、builder、恢复、operator control
- [`scripts/`](/quant_system/scripts): 当前活跃的 alpha / 做市入口、研究脚本、运维工具、兼容包装层
- [`execution/`](/quant_system/execution): adapter、canonical model、ingress、状态机、reconcile、store、observability
- [`decision/`](/quant_system/decision): 决策编排、backtest module、regime wiring、组合逻辑
- [`alpha/`](/quant_system/alpha): 在线推理桥、模型加载、模型实现
- [`risk/`](/quant_system/risk): kill switch、相关性、保证金与组合风控
- [`monitoring/`](/quant_system/monitoring): health、alerts、metrics、ops 视图
- [`research/`](/quant_system/research): model registry、artifact store、实验支撑
- [`infra/`](/quant_system/infra): config、logging、systemd 模板与基础设施 glue
- [`tests/`](/quant_system/tests) + [`execution/tests/`](/quant_system/execution/tests): 单元、集成、恢复、契约与执行子系统测试

## License

Proprietary
