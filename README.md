# Quant System

加密货币永续合约量化交易系统（BTC + ETH）。

- Python：运行时组装、交易所IO、运维工具、研究
- Rust：状态推进、特征热路径、去重、执行原语

## 当前状态（2026-03-22）

**Strategy H**：4h主导方向 + 1h仓位调节 + 15m COMBO确认
- 回测：**Sharpe 2.25**，$500 → $215万 / 6.5年 @ 10x杠杆（T-1修正，无前视偏差）
- 实盘：Bybit Demo 6个runner运行中（$35K权益），3条WS连接（kline.60/15/240）
- 模型：Ridge(60%) + LightGBM(40%) 集成，185个特征，7个数据源

| Runner | 角色 | 仓位比例 | Sharpe |
|--------|------|---------|--------|
| BTC/ETH 4h | 主力方向 | 15% / 10% | 3.62 / 4.57 |
| BTC/ETH 1h | 仓位调节（4h gate） | 8% / 6% | 2.43 / 3.92 |
| BTC/ETH 15m | COMBO辅助确认 | 5% / 5% | — |

## 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/deploy_truth.md`](docs/deploy_truth.md) | 部署真相 |
| [`docs/operations.md`](docs/operations.md) | 运维手册 |
| [`docs/production_runbook.md`](docs/production_runbook.md) | 生产操作手册 |
| [`docs/wiring_truth.md`](docs/wiring_truth.md) | 模块接线状态 |
| [`docs/runtime_contracts.md`](docs/runtime_contracts.md) | 运行时契约 |
| [`CLAUDE.md`](CLAUDE.md) | Claude上下文 |

## 快速开始

```bash
# 安装
make rust
pip install -e ".[live,data,ml,config,monitoring,dev,test]" --break-system-packages
cp .env.example .env  # 填入 BYBIT_API_KEY, BYBIT_API_SECRET

# 启动 Strategy H（6个runner）
python3 -m scripts.run_bybit_alpha --symbols BTCUSDT BTCUSDT_15m BTCUSDT_4h ETHUSDT ETHUSDT_15m ETHUSDT_4h --ws

# 或通过 systemd
sudo systemctl start bybit-alpha.service

# 模型热加载（无需重启，<200ms）
sudo kill -HUP $(systemctl show -p MainPID bybit-alpha.service | cut -d= -f2)
```

## 系统架构

```
信号流（Strategy H）：

  4h Runner（主力）              1h Runner（调节器）           15m Runner（COMBO）
  ┌──────────────────┐          ┌──────────────────┐          ┌──────────────┐
  │ kline.240 → 模型  │──信号──→│ MultiTFConfluence│──AGREE──→│ kline.15      │
  │ 独立交易          │ （gate） │ 4h同向 → 1.3x    │          │ 组合确认       │
  │ BTC 15% ETH 10%  │          │ 4h反向 → 0.3x    │          │ 各5%          │
  └──────────────────┘          │ BTC 8% ETH 6%    │          └──────────────┘
                                └──────────────────┘

数据源（7个，全部T-1日偏移防止前视偏差）：
  传统金融 (SPY/QQQ/国债) → ETF成交量 (IBIT/GBTC) → 稳定币供给 (DeFiLlama)
  → 隐含波动率 (Deribit DVOL) → 恐贪指数 → 资金费率 (Binance) → 链上数据 (Coin Metrics)
```

## 核心命令

```bash
# 交易
python3 -m scripts.run_bybit_alpha --symbols BTCUSDT BTCUSDT_15m BTCUSDT_4h ETHUSDT ETHUSDT_15m ETHUSDT_4h --ws

# 监控
python3 -m scripts.ops.health_watchdog              # 服务+数据健康检查
python3 -m monitoring.ic_decay_monitor --alert       # IC衰减检测 + Telegram告警
python3 -m scripts.ops.signal_reconcile --hours 24   # 实盘vs回测信号一致性

# 模型训练
python3 -m scripts.auto_retrain --include-4h --sighup  # 重训 + 热加载
python3 -m scripts.training.train_4h_daily --all       # 训练4h/日线模型

# 数据更新
python3 -m scripts.data.download_cross_market        # 传统金融 + ETF成交量
python3 -m scripts.data.download_stablecoin_supply   # 稳定币供给
python3 -m scripts.data.download_deribit_iv --all    # DVOL数据
python3 -m scripts.data.download_onchain             # 链上指标

# 研究
python3 -m scripts.research.backtest_small_capital   # $500/10x杠杆回测
python3 -m scripts.research.monte_carlo_risk         # 万次蒙特卡洛风险模拟
python3 -m scripts.ops.shadow_compare --model-a models_v8/BTCUSDT_gate_v2 --model-b models_v8/BTCUSDT_4h --symbol BTCUSDT --days 90  # A/B模型对比

# 测试
pytest tests/unit/ -x -q          # 单元测试 (~18s)
pytest tests/ -x -q -m ""        # 全部测试 (~35s)
cd rust && cargo test         # Rust测试
ruff check --select E,W,F .      # 代码检查
```

## 自动化运维

| 定时器 | 频率 | 用途 |
|--------|------|------|
| health-watchdog | 每5分钟 | 服务健康 + 数据新鲜度 + Telegram告警 |
| data-refresh | 每6小时 | 同步K线、资金费率、持仓量 |
| daily-retrain | 每天凌晨2点 UTC | 4h模型轻量重训 + SIGHUP热加载 |
| auto-retrain | 每周日凌晨2点 UTC | 1h模型走步验证重训 |
| ic-decay-monitor | 每天凌晨3点 UTC | IC衰减检测 + Telegram告警 |

## 风控体系

| 机制 | 说明 |
|------|------|
| **动态杠杆** | 回撤≥10% → 杠杆降至0.75x，≥20% → 0.5x，≥35% → 0.25x |
| **BB入场调节** | 超卖入场 → 仓位1.2x，追涨入场 → 仓位0.3x |
| **波动率自适应deadzone** | deadzone × (实现波动率 / 中位波动率)，范围[0.5x, 2.0x] |
| **4h反转止损** | 当4h模型信号反转时，1h/15m立即平仓 |
| **ATR追踪止损** | 三阶段：初始 → 保本 → 追踪 |
| **Maker限价单** | PostOnly挂单，45秒超时，0费率 |

## 项目结构

| 目录 | 用途 |
|------|------|
| `scripts/ops/` | AlphaRunner核心、配置、模型加载、信号验证 |
| `scripts/training/` | 模型训练（1h、15m、4h/日线） |
| `scripts/data/` | 数据下载（K线、传统金融、稳定币、DVOL、链上） |
| `scripts/research/` | 回测、蒙特卡洛、特征扫描 |
| `features/` | 特征计算（185+特征，V1-V24） |
| `runner/` | 框架运行时、Gate链 |
| `execution/` | 交易所适配器（Bybit、Hyperliquid、Binance） |
| `alpha/` | ML模型、在线Ridge、推理桥 |
| `monitoring/` | IC衰减、健康检查、告警、指标 |
| `rust/` | Rust核心库（_quant_hotpath，77模块，~30K行） |
| `models_v8/` | 已训练模型（{品种}_gate_v2、_15m、_4h） |
| `data_files/` | 数据文件（K线、资金费率、持仓、传统金融、ETF成交量、稳定币） |
| `infra/` | systemd服务单元、配置、日志 |

## Alpha研究结论

| 方向 | 结果 | 说明 |
|------|------|------|
| 4h alpha | **Sharpe 3.62-4.57** | 最强时间框架，T-1修正 |
| 1h alpha | Sharpe 2.43-3.92 | Strategy H中的调节器角色 |
| 15m alpha | 边缘（仅ETH） | COMBO辅助 |
| 5分钟/1分钟HFT | **失败**（Sharpe -5到-25） | 成本 > 信号边际 |
| 做市 | **失败** | 逆向选择 > 价差（即使0%费率） |
| 神经网络 | **失败** | Ridge > MLP > LGBM（4h OOS） |
| 跨交易所套利 | **失败** | 价差 < 费用 |
| 熊市做空 | **失败** | 90+策略全部测试 |
| 波动率交易 | Sharpe 0.79 | 作为特征优于独立交易 |
| 资金费率收割 | 年化11% | 中性对冲，容量有限 |

## 许可证

Proprietary
