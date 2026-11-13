# 币安 USDⓈ-M 运行手册（2026-11-13）

本手册覆盖从数据、研究、纸面运行到测试网执行的完整流程。**本手册不构成实盘批准。**
主网执行在代码层面要求研究门槛通过（`gate.json` 中 `passed=true`），并由人手动开启。

## 0. 范围与硬约束

- 只做币安 USDⓈ-M 永续（BTCUSDT、ETHUSDT）。OKX/Bybit 代码冻结，仅做过安全修复。
- 杠杆硬上限 3 倍：
  - 账户总名义敞口不超过 3 倍权益，由 `shared/exposure_gate.py` 的代码常量保证；
  - 交易所杠杆设置不超过 3，由 `execution/adapters/binance/adapter.py` 的 `MAX_LEVERAGE` 保证；
  - 配置只能调低，不能调高。
- 默认 dry-run（纸面交易）；默认测试网（`BINANCE_TESTNET=1`）。
- 任何脚本都不会自动放宽风险限额，也不会自动重启实盘服务。

## 1. 环境

```bash
pip install --break-system-packages --ignore-installed --require-hashes -r requirements/container.lock
pip install --break-system-packages pytest maturin xgboost
maturin build --release --features python && pip install --break-system-packages --force-reinstall --no-deps target/wheels/*.whl
python3 -m pytest tests/unit/binance/ -q          # 币安执行/运行时测试
python3 scripts/check_local_safety.py             # 离线故障注入检查
```

## 2. 数据（公开接口，无需账户）

```bash
python3 -m data.downloads.binance_public --symbols BTCUSDT ETHUSDT --start 2020-01-01
# 写入新目录 data_files/binance_public_<结束日期>/
```

每次下载都写入一个新的快照目录；目录非空时直接拒绝，绝不覆盖。研究记录的是输入文件的 SHA-256，覆盖之后就无法复现。`data_files/binance_public` 是 2026-11-13 的快照，2026-11-13 的各项研究都引用它，不要再往里下载。

产物：
- `<SYM>_1h.csv`：只含已收盘 K 线，带 taker、trades 字段；
- `<SYM>_funding.csv`：已结算资金费，时间对齐到名义整点；`markPrice` 为空时用该小时 K 线开盘价代替，并在 `mark_source` 中标记；
- `<SYM>_metrics.csv`：5 分钟 OI 和多空比快照，按官方 CHECKSUM 校验；
- `manifest.json`：所有文件的 SHA-256。

OI 和多空比数据大约从 2021-12 才开始有。

## 3. 研究（只用验证集选参，最终测试只看一次）

```bash
python3 -m runner.offline research --data-dir data_files/binance_public --symbol BTCUSDT \
    --start 2021-12-02 --train-end 2024-11-20T04:00 --validation-end 2025-11-16T14:00 \
    --out .cache/research/BTCUSDT
```

流程：
1. **切分：** 用两个固定的 UTC 时间点切分：`--train-end` 之前为训练，`--train-end` 到 `--validation-end` 为验证，之后为最终测试。必须是固定日期而不是行数比例：按比例切分时，数据一追加，验证集就会滑进已经看过的窗口。2026-11-13 的运行用的是 2024-11-20T04:00 和 2025-11-16T14:00，在当时的数据上正好等于 60% 和 80% 的位置。
   **2025-11-16 之后的数据已被查看多次，不能再作为最终测试。**
2. **候选：** 特征集（ohlcv / derivatives）× 标签周期（1/4/24 小时）× Ridge alpha（1/10/100）× deadzone（1.0/1.5/2.0）。
3. **选择：** alpha 按验证集 IC 选；特征集、周期和 deadzone 按验证集回放的日夏普选，且要求成交至少 20 笔。
4. **冻结：** 写出 `protocol.json`（候选清单、选择结果、数据和代码指纹）和 `model.json`（用训练加验证数据重训），然后才运行最终测试。
5. **最终测试：** 包括成本翻倍压力测试、买入持有基准、50 小时趋势基准。

上线门槛预先声明在 `runner/offline/research.py` 的 `passes_gate`，结果写入 `gate.json`。四个条件全部满足才算通过：
- 最终测试日夏普 > 1；
- 成本翻倍后净收益仍为正；
- 成交不少于 100 笔；
- 按单位敞口计，收益同时跑赢买入持有和趋势基准。

**纪律：** 看过最终测试后，再修改任何东西并重跑，该测试即失效，必须换用新的、之后的数据。`protocol.json` 的 `final_test_views` 记录查看次数。

## 4. 发布模型（签名）

```bash
QUANT_MODEL_SIGN_KEY=... python3 -m scripts.publish_research_model --study .cache/research/BTCUSDT --out models_binance
```

- 运行时只接受 `models_binance/<SYM>/` 下的 `model.json`、`protocol.json` 和 `gate.json`。
- 三个文件都要验签；`protocol` 必须描述的正是这个模型。
- 门槛未通过的模型仍可用于纸面运行和测试网，但主网执行会被拒绝。

## 5. 纸面运行（默认）

```bash
python3 -m runner.binance_main --models-dir models_binance --symbols BTCUSDT ETHUSDT
```

- **决策路径：** 和离线回放完全相同，包括冻结模型、增量特征、决策模块，以及同一收盘时刻先 4h 后 1h。
- **成交模型：** 和研究相同，即下一根开盘撮合；挂单必须严格穿透才算成交；计手续费、滑点和资金费。
- **测试保证：** 同一段 K 线下，纸面运行的成交与离线回放逐笔相同（`tests/unit/binance/test_binance_runtime.py`）。
- **行情来源：** 始终用主网公开接口。每 5 秒从上次处理的位置继续拉取已收盘 K 线，断线不会漏 K 线，单线程运行。
- **输出：**
  - `data/runtime/binance/binance_status.json`：每个周期刷新的状态；
  - `data/runtime/binance/binance_events.jsonl`：信号、意图、纸面成交和权益、问题记录。

与同期离线回放逐笔对比（先刷新数据）：

```bash
python3 -m data.downloads.binance_public --start <近期日期> --out data_files/binance_public_compare_<日期>
python3 -m scripts.compare_paper_replay --events data/runtime/binance/binance_events.jsonl \
    --models-dir models_binance --data-dir data_files/binance_public_compare_<日期> --symbol BTCUSDT
```

退出码 0 表示一致。如果不一致，说明线上输入（K 线、资金费、持仓数据）与研究数据有偏差，必须先查清原因。

## 6. 测试网执行

需要的环境变量：

| 变量 | 说明 |
|---|---|
| `BINANCE_TESTNET=1` | 默认值 |
| `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET` | 测试网密钥 |
| `RISK_SCOPE` | 必须含 `binance`；测试网必须含 `testnet`，主网必须不含，例如 `binance-testnet-main` |
| `MAX_DAILY_DRAWDOWN_PCT` / `MAX_WEEKLY_DRAWDOWN_PCT` | 默认 3 / 8，从周期内权益峰值算起 |
| `BINANCE_MAX_ORDER_NOTIONAL` / `BINANCE_MAX_DAILY_NOTIONAL` / `BINANCE_MAX_DAILY_ORDERS` | 默认 1000 / 10000 / 50 |
| `BINANCE_MAX_LEVERAGE` | 只能调低，不能超过 3 |

```bash
python3 -m runner.binance_main --models-dir models_binance --enable-execution
```

账户要求：单向持仓模式（双向持仓会拒绝启动）；同一状态目录同时只允许一个执行进程（文件锁）。

### 下单与恢复

- **入场：** 一笔 GTX 只做 maker 挂单，价格为决策价减 2bp（卖单为加 2bp），最长挂 3 小时，允许部分成交；到期或出场前撤掉剩余部分。这和研究的成交模型一致。
- **出场：** reduce-only 市价单，数量取交易所仓位和本策略已确认仓位中较小的一个。
- **下单前：** 先把订单意图写入 SQLite，客户单号是确定的。超时只查单，不重发，也没有拆单和市价补单。
- **查无此单：** 交易所明确拒单且查单确认不存在，立即结算。状态未知的订单，超过 90 秒（大于币安最大 recvWindow 60 秒）仍查不到时按“未下单”结算，后续平仓不会被永久卡死。
- **仓位来源：** 只认交易所成交记录。成交、策略状态和回执在同一个事务里提交。重启后从日志恢复，不会重复入账。
- **对账：** 每 5 秒核对一次仓位、挂单和未完成订单。只要不一致就禁止开仓，但不会凭空补造成交。

### 风控

- **交易所端止损：** 每个已确认仓位都在币安挂一笔 `STOP_MARKET closePosition` 条件单（Algo 接口），默认距入场价 4%（`--stop-pct`，允许范围 0.5%–15%）。
  - 仓位变化时换新止损单，空仓时撤掉；
  - 止损触发后，产生的订单会被纳入日志，成交只入账一次；
  - 如果价格已经越过止损位，立即市价平仓。
- **开仓门禁：** 按顺序依次检查：
  - 本根 K 线输入新鲜且完整；
  - kill switch 未激活；
  - 回撤未触发；
  - 总敞口不超过 3 倍；
  - 单笔名义、日累计名义和订单数在限额内。

  平仓单不经过这些开仓门禁。
- **回撤：** 触发后锁定到当天或当周结束，期间禁止开仓并主动平仓。
- **平仓重试：** 平仓单以未成交终态结束时，每 30 秒重试一次；5 次都失败后自动激活 kill switch（只拦开仓）。

### kill switch

```bash
python3 -m shared.kill_file activate --reason "手动停止" --flatten   # 拦开仓并平仓
python3 -m shared.kill_file activate --reason "暂停开仓"             # 只拦开仓
python3 -m shared.kill_file clear
python3 -m shared.kill_file status
```

文件损坏或不可读时，按“已激活、不平仓”处理，也就是拦截开仓。

## 7. 主网前检查清单

1. **研究门槛：** 对应模型的 `gate.json` 显示 `passed=true`，而且最终测试只看过一次。
2. **纸面运行：** 连续运行 2–4 周，每周跑一次 `compare_paper_replay`，结果全部一致。
3. **测试网故障演练（逐项记录结果）：**
   - 运行中断网、杀进程后重启：仓位和挂单恢复，没有重复下单；
   - 交易所拒单、超时、限流：开仓锁住，平仓在 90 秒宽限期后可以继续；
   - 手动触发止损单：成交入账，策略仓位归零；
   - kill switch 平仓、回撤触发后平仓；
   - 手动在交易所下单或改仓位：对账报错，开仓被禁止。
4. **账户设置：** 主网账户为单向持仓模式；杠杆和保证金模式已人工确认；API 密钥不开提现权限，并绑定 IP 白名单。
5. **开启方式：** 由人设置 `BINANCE_TESTNET=0`、主网 `RISK_SCOPE`，并加上 `--enable-execution`。资金规模从小额开始，限额只能人工调整。

## 8. 已知限制

- 研究模型是基于 Binance 原生数据的 Ridge 参考模型，不是旧的 LGBM/XGB 集成模型。旧模型需要服务器上的 `data_files/` 和 `.pkl` 文件，才能用同一流程重新检验。
- 回放敞口上限是 1 倍，实盘上限是 3 倍，两者收益规模不可直接比较；夏普在不同杠杆下基本可比。
- OHLC 数据无法证明挂单在队列中的位置，实盘挂单成交率可能高于或低于回放。
- `OrderLimiter` 的日计数保存在内存里，进程重启后会清零。回撤和敞口门禁不受影响，它们持久化或每次实时查询。
- 只支持单向持仓模式、USDT 保证金。入出金不会自动从回撤基线中扣除。
