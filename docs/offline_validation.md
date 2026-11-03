# 本地回放与样本外验证

此入口不需要本金、VPS 或交易 API 密钥。CLI 主动禁止 socket 连接；不会构造交易所客户端。
需要完整项目环境：Python 3.11+、numpy、pandas 和编译好的 `_quant_hotpath` Rust 扩展。
已有的 `python scripts/check_local_safety.py` 仍是无需 Rust 的轻量检查。

## 一次运行全部研究流程

在仓库根目录运行：

```bash
python -m runner.offline demo --out .cache/offline-demo
```

这会生成 1200 根合成小时 K 线，以及明确假设的资金费数据，用于工程验收。
`report.md` 为中文摘要，`summary.json` 为各阶段指标，`protocol.json` 保存实验协议、
数据/源码/原生扩展指纹与参数，`model.json` 保存可检查的线性模型权重与训练边界。
每个阶段包含 orders、fills、equity、funding、signals、risk_events CSV，以及完整 replay JSON。
空表只保存在 replay JSON 中。输出目录必须为空，避免覆盖已有实验。

**合成行情上的收益数字不提供真实交易优势的证据。** CI 上传同样标注的验收报告供下载检查。

## 四个阶段实现了什么

1. 数据：严格检查 UTC 整点毫秒时间、连续性、重复、乱序、有限正价格、OHLC 关系与非负成交量。
   不自动排序、填洞或补未来值。4h 聚合只保留完整 UTC 分组，信号在收盘时可见；同一收盘时间先更新 4h 再运行 1h。
   同时修复旧 Strategy H 脚本的对齐、末尾不完整分组，以及历史 CSV 游标返回未来首条记录的问题。
2. 回放：复用 EngineCoordinator、DecisionBridge、EnrichedFeatureComputer、AlphaDecisionModule、
   SignalDiscretizer、AdaptivePositionSizer、ExecutionBridge、CompositeRiskGate、OrderLimiter 和 DrawdownGuard。
   离线模式从输入历史暖机，不读取当前监控文件或本地 CSV 尾部；市场时间传入策略。
   4h 仅产生信号；1h 下单。训练期暖机不允许下单。
3. 成交与账务：下一根 K 线开始撮合，post-only 在下一开盘价处会立即成交时拒绝；
   必须严格穿透限价才可成交，以成交量比例限制单根 K 线可成交数量，支持部分成交与过期撤余单。
   平仓下一开盘以不利滑点成交。手续费按正成本计，资金费按持仓方向收付。
   使用 StrategyPosition 维护均价与已实现损益，逐小时核对
   `权益 = 初始资金 + 已实现盈亏 − 手续费 + 资金费净收入 + 浮动盈亏`。
4. 验证：固定新 OHLCV Ridge 研究模型，特征由同一增量特征计算器生成。
   训练 60%、验证 20%、最终测试 20%；训练标签结束时间必须早于切分点，标准化只拟合训练数据。
   开发期前 80% 内另做 3 个扩展窗口，每折重新训练。最终测试使用验证前冻结的模型；无在线更新或参数搜索。
   预先声明手续费/滑点翻倍压力测试，以及相同撮合和成本下的买入持有、50 小时趋势基准。

## 真实历史数据入口

```bash
python -m runner.offline study --bars data_files/BTCUSDT_1h.csv --funding data_files/BTCUSDT_funding.csv --out .cache/btc-study-001
```

小时 K 线 CSV 必需列：

```text
open_time,open,high,low,close,volume
1704067200000,42000,42100,41900,42050,100
```

`open_time` 为 UTC 开盘时间，毫秒；数据在一小时后才可供策略使用。
成交量必须是 BTC 数量，价格为 USDT/BTC。输入必须已经排除未收盘 K 线；CLI 也会检查末尾收盘时间。
需要至少 1200 根连续小时 K 线，真实研究应覆盖更长历史与不同市场阶段；1200 根只是工程最小值。

资金费 CSV 必需列：

```text
funding_time,rate,mark_price
1704067200000,0.0001,42000
```

`funding_time` 为真实结算时刻的 UTC 毫秒，`mark_price` 为该时刻的标记价格。
对于接口数毫秒的发布偏移，可使用名义 UTC 结算时间，并在 `observed_funding_time` 保存原始时间；
导入器只接受 0—1000 毫秒偏移。已附 2025 年 Binance 样本的实际最大偏移为 16 毫秒，见
[数据来源与对齐说明](../research/datasets/btcusdt_2025/README.md)。
本入口仅支持 UTC 00:00/08:00/16:00 的固定 8 小时结算，评估期间缺一条就失败。
正费率意味着多仓付费、空仓收款；结算发生在该小时开盘撮合之前。
不适配历史结算频率变化，不能把缺失资金费默认为零，也不能用当前费率伪造历史。

可以通过 `--config config.json` 覆盖 ReplayConfig 中的字段。默认费率、滑点、最小数量和资金费样例
仅为研究假设，不代表任何交易所当前合约规则。正式研究须记录当时真实规则并检查单位。

## 中断后的恢复验收

`replay` 接受已冻结的 `--model`、`--bars`、`--funding`、`--start-ms`。
`--start-ms` 必须晚于训练标签结束时间，通常取 protocol.json 的 holdout_start_ms。

```bash
python -m runner.offline replay --bars data_files/BTCUSDT_1h.csv --funding data_files/BTCUSDT_funding.csv --model .cache/btc-study-001/model.json --start-ms <holdout_start_ms> --checkpoint .cache/btc-checkpoint.json --stop-after <已处理总根数> --out .cache/btc-partial
python -m runner.offline replay --bars data_files/BTCUSDT_1h.csv --funding data_files/BTCUSDT_funding.csv --model .cache/btc-study-001/model.json --start-ms <holdout_start_ms> --checkpoint .cache/btc-checkpoint.json --resume --out .cache/btc-resumed
```

恢复先从不可变输入重新构造历史状态，并核对已提交前缀的结果指纹，然后完整重放到目标。
恢复检查模型、数据、成本、资金费、代码，以及已记录的 Python、NumPy、pandas、操作系统和 CPU 架构。
Rust 扩展同时支持直接加载和 wheel 包装两种指纹采集方式；这些标识变化会拒绝恢复。
这是离线确定性重建，耗时随历史长度增长；
不是线上订单断点续传，也不声称支持任意机器间逐位相同的浮点结果。
测试在新 Python 子进程恢复，并比较全部订单、成交、权益与连续运行结果。

## 验收与边界

```bash
python -m pytest tests/unit/runner/test_offline_research.py -q
```

覆盖未来数据扰动不影响历史前缀、4h 未收盘不可见、坏数据拒绝、
限价触碰不成交、post-only 拒单、部分成交和撤单、空仓资金费、部分平仓损益、
风险缺失/过期保护、新进程恢复、模型来源约束和最终测试模型冻结。

- 本仓库已附 2025 年 BTC 真实小时行情与资金费，原有模型仍缺完整训练来源，不能由新模型结果证明原有策略盈利。
  新 Ridge 模型用于透明地跑通研究流程；不是复刻现有 Ridge/LGBM 集成模型。
- 新入口使用已有增量特征；未认证原有所有批量特征、外部数据源和线上快速路径与之完全等价。
- OHLC 无法确定真实盘口排队、日内路径、成交延迟或强平。默认仅支持 USDT 线性合约，敞口上限不超过 1 倍；
  没有真实交易所保证金和强平认证。手续费仅 USDT，入出金、跨币种费用不在此账本范围。
- 最终权益按收盘价盯市，期末持仓不强制平仓，待处理订单单独列出。收盘之后的资金费与平仓成本尚未发生。
  报告最大回撤是小时采样结果，可能低于真实日内最大回撤。
- 日收益不足 30 个观测时不报告夏普；基准与策略的平均实际敞口分别显示，不以收益差直接宣称 Alpha。
- 旧脚本 `wf_h` 已改名说明为固定预测的历史分段统计，其简化成交与净值算法仍不是正式研究入口。
- 代码可以阻止单次实验的数据泄漏，但无法证明同一段历史从未被之前的人工研究使用。
  最终测试看过后必须记账，不应不断用同一最终测试集调参。
