# Python → Rust 全量替代计划

> 状态: 远期愿景文档，不代表当前默认运行时
> 更新时间: 2026-03-12
> 当前事实请优先参考 [`docs/runtime_truth.md`](/quant_system/docs/runtime_truth.md) 和 [`research.md`](/quant_system/research.md)

## 现状

| 指标 | 值 |
|------|---|
| Python 生产代码 | 86,549 LOC (15个目录) |
| Rust crate | 16,691 LOC (55模块) |
| 迁移比例 | ~19% (按LOC) |
| 异步文件 | 3个 (execution/adapters/binance/async_*) |
| ML依赖文件 | 23个 (alpha/, scripts/) |
| pandas/numpy文件 | 50个 |

## 架构目标

**最终态**: 纯 Rust 二进制 + Python 仅保留离线工具(训练/研究脚本)

```
阶段1-3: PyO3 crate 扩展 (当前模式，Python 调用 Rust)
阶段4-5: Rust binary + Python FFI (Rust 为主，调用 Python ML)
阶段6-7: 纯 Rust binary (ML 推理也用 Rust)
```

## 目录规模与分类

| 目录 | LOC | 分类 | 迁移难度 | 阶段 |
|------|-----|------|----------|------|
| scripts/ | 25,604 | 离线研究/训练 | 保留Python | — |
| execution/ | 13,724 | 网络IO+订单管理 | 高(HTTP/WS) | 4-5 |
| runner/ | 5,360 | 入口+编排 | 中 | 5 |
| portfolio/ | 5,437 | 优化+风险模型 | 中(数学) | 3 |
| features/ | 4,386 | 特征计算 | 低(已60%迁移) | 1 |
| risk/ | 4,170 | 风险规则 | 低(已50%迁移) | 1 |
| event/ | 4,156 | 事件类型+总线 | 中 | 3 |
| engine/ | 3,856 | 管道+协调 | 中 | 2-3 |
| decision/ | 3,825 | 信号+决策 | 中 | 2 |
| alpha/ | 3,098 | ML模型+推理 | 高(ML依赖) | 5-6 |
| monitoring/ | 2,428 | 告警+健康检查 | 中 | 4 |
| strategies/ | 2,064 | 策略示例 | 低(废弃) | — |
| infra/ | 1,457 | 日志+配置 | 低 | 4 |
| state/ | 1,448 | 状态类型 | 低(已70%迁移) | 1 |
| regime/ | 236 | Regime检测 | 已完成 | done |

---

## 阶段1: 完成热路径迁移 (2-3周, +3K LOC Rust)

**目标**: 所有 per-bar 计算完全在 Rust 中，Python 零参与

### 1.1 features/ 清理 (~1,500 LOC → 薄包装)

已有 Rust: `feature_engine.rs`, `cross_asset.rs`, `feature_selector.rs`, `technical.rs`

| 文件 | LOC | 动作 |
|------|-----|------|
| enriched_computer.py | 1,462 | 已委托 RustFeatureEngine，删残余逻辑 |
| live_computer.py | 399 | 简化为纯编排(调用Rust) |
| batch_feature_engine.py | 215 | 已迁移，删 Python 回退 |
| rolling.py | 155 | 已迁移，保留 import shim |
| multi_timeframe.py | 286 | 已迁移，保留薄包装 |
| cross_sectional.py | 162 | 已有 Rust，删 Python 计算 |
| dynamic_selector.py | 455 | 保留 _rankdata/_spearman_ic (scripts用) |
| 其他(types, __init__) | ~200 | 保留(类型定义) |

### 1.2 state/ 消除 Python 适配层 (~800 LOC)

| 文件 | LOC | 动作 |
|------|-----|------|
| rust_adapters.py | 230 | 迁移转换逻辑到 Rust，Python 只剩 import |
| store.py | 317 | 删 Python StateStore，只保留 RustStateStore |
| reducers/*.py | 550 | 删 Python reducers (Rust 已完全接管) |
| snapshot.py, diff.py | 270 | 保留(审计工具，非热路径) |

### 1.3 risk/ 规则完全 Rust 化 (~600 LOC Rust)

| 文件 | LOC | 动作 |
|------|-----|------|
| rules/max_drawdown.py | 287 | 迁移到 risk_engine.rs |
| rules/portfolio_limits.py | 398 | 迁移到 risk_engine.rs |
| rules/leverage_cap.py | 317 | 迁移到 risk_engine.rs |
| margin_monitor.py | 264 | 保留(异步监控，非per-bar) |
| meta_builder_live.py | 180 | 保留(编排) |

### 验证
```bash
cd ext/rust && cargo test
make rust
pytest tests/ -x -q
```

---

## 阶段2: 决策引擎 Rust 化 (3-4周, +4K LOC Rust)

**目标**: 信号生成、集成、决策全链路 Rust

### 2.1 decision/signals/ (~1,800 LOC)

新 Rust 模块: `signal_ensemble.rs`

| 文件 | LOC | 动作 |
|------|-----|------|
| feature_signal.py | 86 | 已部分迁移(rust_compute_feature_signal) |
| factors/*.py (6文件) | 800 | 已迁移到 factor_signals.rs，删Python计算 |
| technical/*.py (5文件) | 600 | 已迁移到 technical.rs/decision_signals.rs |
| adaptive_ensemble.py | 273 | 新: 迁移权重计算+选择逻辑 |

### 2.2 decision/ 核心 (~1,200 LOC)

| 文件 | LOC | 动作 |
|------|-----|------|
| engine.py | 450 | 迁移决策循环到 Rust |
| ml_decision.py | 380 | 迁移推理编排(模型调用留Python) |
| regime_bridge.py | 196 | 已用 RustRegimeBuffer，删残余 |
| multi_strategy.py | 173 | 已迁移数学，删残余 |

### 2.3 decision/ 执行策略 (~600 LOC)

新 Rust 模块: `execution_policy.rs`

| 文件 | LOC | 动作 |
|------|-----|------|
| execution_policy/passive.py | 180 | 迁移限价单逻辑 |
| execution_policy/marketable_limit.py | 150 | 迁移可成交限价单逻辑 |
| intents/target_position.py | 200 | 迁移目标仓位计算 |
| sizing/*.py | 200 | 已部分迁移(fixed_fraction) |

### 验证
```bash
pytest tests/unit/decision/ -x -q
```

---

## 阶段3: 管道+事件+组合 Rust 化 (4-5周, +5K LOC Rust)

**目标**: 引擎内部循环完全 Rust

### 3.1 engine/ (~2,500 LOC)

新 Rust 模块: `engine_loop.rs`, `coordinator.rs`

| 文件 | LOC | 动作 |
|------|-----|------|
| pipeline.py | 680 | 迁移(已有 rust_pipeline_apply 基础) |
| coordinator.py | 850 | 迁移主循环逻辑 |
| loop.py | 420 | 迁移事件分发 |
| dispatcher.py | 350 | 迁移事件路由 |
| feature_hook.py | 200 | 迁移(已是薄包装) |
| tick_engine.py | 180 | 迁移 tick 聚合 |

### 3.2 event/ (~2,000 LOC)

扩展 Rust: `event_store.rs`, `event_types.rs`

| 文件 | LOC | 动作 |
|------|-----|------|
| types.py | 580 | 迁移事件定义到 Rust |
| store.py | 450 | 迁移事件存储 |
| dispatcher.py | 380 | 迁移事件分发 |
| runtime.py | 320 | 迁移运行时 |
| checkpoint.py | 200 | 迁移检查点 |

### 3.3 portfolio/ 核心 (~1,500 LOC)

扩展 Rust: `portfolio_allocator.rs`, `portfolio_optimizer.rs`

| 文件 | LOC | 动作 |
|------|-----|------|
| allocator.py | 632 | 扩展 rust_allocate_portfolio |
| rebalance.py | 214 | 迁移重平衡逻辑 |
| optimizer/black_litterman.py | 340 | 迁移(已有 linalg_math.rs) |
| risk_model/*.py | 1,200 | 保留(非热路径，offline) |

### 验证
```bash
pytest tests/unit/engine/ tests/unit/event/ tests/unit/portfolio/ -x -q
```

---

## 阶段4: 网络/IO/基础设施 Rust 化 (6-8周, +8K LOC Rust)

**目标**: HTTP client, WebSocket, 监控全部 Rust

### 4.1 execution/adapters/ (~8,000 LOC)

新 Rust 依赖: `tokio`, `reqwest`, `tokio-tungstenite`

| 文件 | LOC | 动作 |
|------|-----|------|
| binance/rest.py | 800 | 新: rust_binance_rest (reqwest) |
| binance/ws_trade_stream.py | 450 | 新: rust_ws_stream (tungstenite) |
| binance/kline_processor.py | 350 | 迁移 kline 聚合 |
| binance/funding_poller.py | 200 | 迁移到 Rust 异步任务 |
| binance/oi_poller.py | 180 | 迁移到 Rust 异步任务 |
| binance/depth_processor.py | 300 | 迁移 depth 处理 |
| binance/liquidation_poller.py | 150 | 迁移 |
| binance/market_data_runtime.py | 400 | 迁移运行时编排 |
| deribit_iv_poller.py | 200 | 迁移 |
| fgi_poller.py | 150 | 迁移 |
| macro_poller.py | 180 | 迁移 |
| onchain_poller.py | 200 | 迁移 |
| mempool_poller.py | 180 | 迁移 |
| sentiment_poller.py | 150 | 迁移 |
| async_*.py (3文件) | 1,200 | 替换为 Rust tokio |

### 4.2 execution/ 其他 (~3,000 LOC)

| 文件 | LOC | 动作 |
|------|-----|------|
| bridge/*.py | 400 | 迁移(signer.rs 已有) |
| ingress/*.py | 500 | 迁移路由逻辑 |
| safety/*.py | 600 | 迁移(部分已有 Rust) |
| sim/*.py | 400 | 迁移回测执行 |
| state_machine/machine.py | 350 | 迁移(已有 order_state_machine.rs) |

### 4.3 monitoring/ (~2,400 LOC)

新 Rust 模块: `monitoring.rs`, `health.rs`

| 文件 | LOC | 动作 |
|------|-----|------|
| health.py + health_server.py | 500 | 迁移(Rust HTTP server) |
| metrics.py | 300 | 迁移(Prometheus exposition) |
| alerts/manager.py | 400 | 迁移 |
| slo.py | 250 | 迁移 |
| engine_hook.py | 200 | 迁移 |

### 4.4 infra/ (~1,400 LOC)

| 文件 | LOC | 动作 |
|------|-----|------|
| logging/setup.py | 350 | 迁移(tracing crate) |
| config/load.py | 300 | 迁移(serde_yaml) |
| model_signing.py | 200 | 迁移(已有 signer.rs 基础) |
| runtime/*.py | 200 | 迁移 |

### Cargo 新依赖
```toml
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json", "rustls-tls"] }
tokio-tungstenite = "0.21"
tracing = "0.1"
tracing-subscriber = "0.3"
prometheus = "0.13"
axum = "0.7"       # HTTP health endpoint
serde_yaml = "0.9"
```

### 验证
```bash
pytest tests/integration/ -x -q
```

---

## 阶段5: Runner + Alpha 推理 Rust 化 (4-5周, +5K LOC Rust)

**目标**: main() 入口在 Rust, ML 推理通过 FFI 调用 Python

### 5.1 runner/ (~5,300 LOC)

新 Rust: `main.rs` (Rust binary entry point)

| 文件 | LOC | 动作 |
|------|-----|------|
| live_runner.py | 1,800 | 迁移为 Rust main + config |
| live_paper_runner.py | 1,200 | 迁移 |
| backtest_runner.py | 800 | 迁移 |
| graceful_shutdown.py | 300 | 迁移(tokio signal) |
| testnet_validation.py | 600 | 迁移 |

### 5.2 alpha/ 推理层 (~1,500 LOC)

保留 Python: 训练代码 (training/*.py, models/*.py)
迁移到 Rust: 推理热路径

| 文件 | LOC | 动作 |
|------|-----|------|
| inference/bridge.py | 462 | 迁移推理调度(模型调用留Python FFI) |
| model_loader.py | 250 | 迁移模型加载 |
| signal_transform.py | 200 | 迁移信号变换 |
| strategy_config.py | 150 | 迁移配置 |
| nn_utils.py | 180 | 保留(PyTorch) |

### 5.3 Rust Binary 架构

```rust
// src/bin/quant_trader.rs
#[tokio::main]
async fn main() {
    let config = load_config();
    let engine = Engine::new(config);
    let ws = BinanceWsStream::connect(&config).await;
    let rest = BinanceRestClient::new(&config);

    // ML 推理: 通过 pyo3 调用 Python LightGBM
    let python_bridge = PythonMLBridge::new(&config.model_path);

    engine.run(ws, rest, python_bridge).await;
}
```

### 验证
```bash
cargo build --release --bin quant_trader
./target/release/quant_trader --config config.yaml --mode paper
```

---

## 阶段6: ML 推理原生 Rust (4-6周, +3K LOC Rust)

**目标**: 消除 Python 运行时依赖（生产环境）

### 6.1 LightGBM Rust 原生推理

```rust
// 方案A: lightgbm crate (C API绑定)
use lightgbm::Booster;
let booster = Booster::from_file("model.txt")?;
let prediction = booster.predict(&features)?;

// 方案B: 自定义决策树遍历 (~500 LOC)
// 将 model -> model.json -> Rust struct
// 纯 Rust 树遍历，零外部依赖
```

### 6.2 XGBoost Rust 原生推理

类似方案，使用 `xgboost-rs` 或自定义树遍历

### 6.3 模型格式转换工具

```bash
# 一次性工具: 训练后导出为 Rust 可读格式 (保留在 scripts/)
python3 -m scripts.export_model_to_rust --model models_v8/BTCUSDT_gate_v2/
```

### 验证
```bash
# Rust 推理 vs Python 推理 精度对比
cargo test --test model_parity
# 确保预测值差异 < 1e-6
```

---

## 阶段7: 纯 Rust 生产二进制 (2-3周, 清理)

**目标**: 生产部署零 Python 依赖

### 7.1 消除最后的 Python 依赖

- 替换所有 pyo3 调用
- 静态链接所有依赖
- 单一二进制文件 (~20MB)

### 7.2 Docker 简化

```dockerfile
# FROM python:3.12 -> FROM distroless
FROM gcr.io/distroless/cc
COPY target/release/quant_trader /
ENTRYPOINT ["/quant_trader"]
```

### 7.3 保留的 Python

**永久保留 Python 的部分** (离线工具，不影响生产):
- `scripts/train_*.py` — 训练管道 (LightGBM/XGBoost/sklearn)
- `scripts/walkforward_*.py` — WF 验证
- `scripts/ic_analysis_*.py` — IC 分析研究
- `scripts/download_*.py` — 数据下载
- `scripts/backtest_*.py` — 回测分析
- `alpha/training/*.py` — 训练逻辑
- `portfolio/risk_model/*.py` — 离线风险模型

---

## 总体时间表

| 阶段 | 时间 | 新增Rust LOC | 删除Python LOC | 里程碑 |
|------|------|-------------|---------------|--------|
| 1: 热路径完成 | 2-3周 | +3,000 | -2,500 | per-bar 100% Rust |
| 2: 决策引擎 | 3-4周 | +4,000 | -3,000 | 信号链路 100% Rust |
| 3: 管道+事件+组合 | 4-5周 | +5,000 | -4,000 | 内部循环 100% Rust |
| 4: 网络/IO/基础设施 | 6-8周 | +8,000 | -8,000 | 外部IO 100% Rust |
| 5: Runner+推理 | 4-5周 | +5,000 | -5,000 | Rust binary 入口 |
| 6: ML原生推理 | 4-6周 | +3,000 | -1,500 | 零Python运行时 |
| 7: 纯Rust清理 | 2-3周 | +500 | -2,000 | 单一二进制部署 |
| **总计** | **25-34周** | **+28,500** | **-26,000** | **~45K LOC Rust** |

## 最终态

```
生产部署:
  quant_trader (单一 Rust 二进制, ~45K LOC, ~20MB)
  |-- 市场数据: tokio + tungstenite WebSocket
  |-- 特征计算: 105 features incremental
  |-- ML 推理: 原生决策树遍历
  |-- 决策引擎: 信号集成 + regime switch
  |-- 风险管理: 6 规则实时评估
  |-- 订单执行: reqwest REST + WS
  |-- 监控: axum health + prometheus metrics
  |-- 配置: serde_yaml

离线工具 (保留 Python):
  scripts/train_*.py         — 模型训练
  scripts/walkforward_*.py   — WF 验证
  scripts/backtest_*.py      — 回测分析
  scripts/download_*.py      — 数据下载
  scripts/ic_analysis_*.py   — 特征研究
```

## 关键风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| LightGBM Rust 推理精度 | 预测值偏差 | 阶段6建parity测试，差异<1e-6 |
| tokio + WebSocket 稳定性 | 断线重连 | 参考 binance-rs 成熟实现 |
| 迁移期间回归 | 信号质量下降 | 每阶段 pytest 2654 tests 全过 |
| 模型格式兼容 | 丢精度 | 训练时同时导出 json 格式 |

## 执行原则

1. **每阶段独立可部署** — 不存在"半成品"状态
2. **Python 测试持续通过** — 迁移期间 2654 tests 全绿
3. **Parity 测试** — 每个迁移模块建 Rust vs Python 精度对比
4. **性能回归测试** — 每阶段跑 benchmark，不允许退步
5. **先 PyO3 扩展，后独立 binary** — 渐进式，非大爆炸
