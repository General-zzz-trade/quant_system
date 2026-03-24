# Quant System

加密货币永续合约量化交易系统（BTC + ETH）。

- **Python**：编排层、交易所 IO、运维工具、研究
- **Rust**：事件类型、状态管理、特征计算、信号约束、风控、执行安全

## 当前状态（2026-03-24）

**Strategy H**：4h 主导方向 + 1h 仓位调节（框架原生，EngineCoordinator 驱动）
- 回测：**Sharpe 2.25**，$500 → $215万 / 6.5年 @ 10x 杠杆（T-1 修正，无前视偏差）
- 实盘：Bybit Demo 4 个 runner（BTC+ETH × 1h/4h），2 条 WS（kline.60/240）
- 模型：Ridge(60%) + LightGBM(40%) 集成，185 个特征，7 个数据源
- 架构：AlphaDecisionModule → EngineCoordinator → ExecutionBridge → Bybit

| Runner | 角色 | Sharpe |
|--------|------|--------|
| BTC/ETH 4h | 主力方向 | 3.62 / 4.57 |
| BTC/ETH 1h | 仓位调节（4h gate） | 2.43 / 3.92 |

## 系统架构

```
Bybit WS kline → MarketEvent → EngineCoordinator.emit()
  │
  ├─ FeatureComputeHook → RustFeatureEngine → 120+ 特征
  ├─ StatePipeline → RustStateStore（Rust 堆上状态）
  └─ DecisionBridge → AlphaDecisionModule.decide(snapshot)
      ├─ EnsemblePredictor: Ridge(60%)+LGBM(40%)
      ├─ SignalDiscretizer: z-score → deadzone → min-hold
      ├─ RegimeFilter: 自适应 p20/p25 百分位
      ├─ Force exits: ATR 止损, 快速止损, z 反转, 4h 反转
      ├─ Direction alignment: ETH 跟随 BTC
      └─ AdaptivePositionSizer: 资金分层 × IC × vol
  │
  └─ OrderEvent → ExecutionBridge → BybitExecutionAdapter
      └─ FillEvent → StatePipeline（状态更新）
```

## 快速开始

```bash
# 安装
make rust
pip install -e ".[live,data,ml,config,monitoring,dev,test]" --break-system-packages
cp .env.example .env  # 填入 BYBIT_API_KEY, BYBIT_API_SECRET

# 启动 Strategy H（4 个 runner，框架原生）
python3 -m runner.alpha_main --symbols BTCUSDT BTCUSDT_4h ETHUSDT ETHUSDT_4h --ws

# 或通过 systemd
sudo systemctl start bybit-alpha.service

# 模型热加载（无需重启，<200ms）
sudo kill -HUP $(systemctl show -p MainPID bybit-alpha.service | cut -d= -f2)
```

## 核心命令

```bash
# 监控
python3 -m monitoring.watchdog                        # 服务+数据健康检查
python3 -m monitoring.ic_decay_monitor --alert         # IC 衰减检测 + Telegram
python3 -m monitoring.data_quality_check               # 数据质量检查

# 模型训练
python3 -m alpha.auto_retrain --include-4h --sighup    # 重训 + 热加载
python3 -m alpha.auto_retrain --daily --include-4h     # 每日轻量重训

# 数据更新
python3 -m data.downloads.data_refresh                 # 全量数据同步（K线+资金费率+OI）

# 测试
pytest tests/unit/ -x -q                               # 单元测试
cargo test                                             # Rust 测试
ruff check --select E,W,F .                            # 代码检查
```

## 自动化运维

| 定时器 | 频率 | 用途 |
|--------|------|------|
| health-watchdog | 每 5 分钟 | 服务健康 + 数据新鲜度 + Telegram 告警 |
| data-refresh | 每 6 小时 | 同步 K 线、资金费率、持仓量 |
| daily-retrain | 每天凌晨 2 点 UTC | 4h 模型轻量重训 + SIGHUP 热加载 |
| auto-retrain | 每周日凌晨 2 点 UTC | 1h 模型走步验证重训 |
| ic-decay-monitor | 每天凌晨 3 点 UTC | IC 衰减检测 + Telegram 告警 |

## 风控体系

| 机制 | 说明 |
|------|------|
| **ATR 三阶段追踪止损** | 初始(1.2×ATR) → 保本(0.5×ATR) → 追踪(0.2×ATR) |
| **快速止损** | -1% 逆向移动 → 立即平仓 |
| **波动率自适应 deadzone** | deadzone × (实现波动率 / 中位波动率)，范围 [0.5x, 2.0x] |
| **4h 反转止损** | 4h 模型信号反转 → 1h 立即平仓 |
| **方向对齐** | ETH 新开仓必须与 BTC 共识方向一致 |
| **IC 健康仓位缩放** | GREEN=1.2x, YELLOW=0.8x, RED=0.4x |
| **资金分层** | <$500 集中(35%), $500-10K 均衡(25%), >$10K 分散(18%) |

## 项目结构

```
decision/           决策引擎 (AlphaDecisionModule, 信号, 仓位)
  modules/alpha.py  框架原生决策模块 (~300 行)
  signals/          EnsemblePredictor + SignalDiscretizer
  sizing/           AdaptivePositionSizer
  rust/             11 个 .rs (约束管线, 推理桥, ML 预测)

engine/             事件引擎 (Coordinator, Pipeline, Bridge)
  coordinator.py    事件编排中心
  pipeline.py       状态转换管线 (→ RustStateStore)
  feature_hook.py   特征计算桥接 (→ RustFeatureEngine)
  rust/             9 个 .rs (tick_processor ~80μs 热路径)

event/              事件类型 (Rust PyO3 驱动)
  events.py         8 个事件类 (薄 Python 包装)
  rust/             9 个 .rs (EventHeader, 事件类, 验证器)

state/              状态管理 (Rust 类型, 零 Python dataclass)
  snapshot.py       StateSnapshot 容器
  store.py          SQLite 持久化
  rust/             19 个 .rs (类型, reducer, store)

execution/          交易所适配 (Bybit, Hyperliquid, Binance)
  adapters/         3 个交易所 + Polymarket
  safety/           电路断路器, Kill Switch, 去重 (Rust 委托)
  rust/             6 个 .rs (订单状态机, WS 客户端)

features/           特征工程 (185+ 特征, V1-V24)
  enriched_computer 增量特征计算 (Rust 追踪器)
  batch_feature_engine 批量计算
  rust/             15 个 .rs (FeatureEngine, 指标, 增量追踪器)

risk/               风控 (StagedRisk, AdaptiveStop, GateChain)
  rust/             6 个 .rs (gate_chain, 风险引擎, 自适应止损)

runner/             运行时 (alpha_main 入口, 回测, 恢复)
  alpha_main.py     生产入口 (EngineCoordinator + WS)
  strategy_config   SYMBOL_CONFIG (BTC+ETH × 1h/4h)

alpha/              ML 模型 (加载, 在线 Ridge, 重训练)
monitoring/         运维 (健康检查, IC 衰减, 数据质量)
portfolio/          组合管理 (分配器, 组合器, 对冲)
attribution/        PnL 归因 (Rust 驱动)
data/               数据下载 + 质量检查
regime/             市场状态检测 (CompositeRegime)
infra/              基础设施 (日志, 配置, systemd, 异常)
research/           研究脚本 + Rust 工具
```

## Rust 集成

- **102 个 .rs 文件**，分布在各模块的 `rust/` 子目录
- **198 个 PyO3 导出**，100% 被 Python 生产代码调用
- **编译入口**: 项目根 `Cargo.toml` + `rust_lib.rs`
- **热路径**: RustTickProcessor (~80μs/tick), RustFeatureEngine, RustStateStore
- **构建**: `make rust` 或 `maturin develop --release --features python`

## Alpha 研究结论

| 方向 | 结果 | 说明 |
|------|------|------|
| 4h alpha | **Sharpe 3.62-4.57** | 最强时间框架，T-1 修正 |
| 1h alpha | Sharpe 2.43-3.92 | Strategy H 中的调节器 |
| 15m alpha | 失败 | 过拟合，已禁用 |
| 5m/1m HFT | 失败 (Sharpe -5 到 -25) | 成本 > 信号 |
| 做市 | 失败 | 逆向选择 > 价差 |
| 神经网络 | 失败 | Ridge > MLP > LGBM (4h OOS) |
| 跨交易所套利 | 失败 | 价差 < 费用 |

## 文档索引

| 文档 | 内容 |
|------|------|
| [`CLAUDE.md`](CLAUDE.md) | 开发上下文（命令、架构、关键文件） |
| [`docs/wiring_truth.md`](docs/wiring_truth.md) | 模块接线状态 |
| [`docs/deploy_truth.md`](docs/deploy_truth.md) | 部署真相 |
| [`docs/operations.md`](docs/operations.md) | 运维手册 |
| [`docs/production_runbook.md`](docs/production_runbook.md) | 生产操作手册 |

## 许可证

Proprietary
