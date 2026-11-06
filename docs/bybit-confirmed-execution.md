# Bybit 未知提交与真实成交修复

基线：master `269a8caeee8de9f5e399b25909cb4f529f4b0c83`。
验证仅使用模拟交易所、临时数据库和禁网容器；没有访问账户、部署或实际下单。
离线验证完成后单独提交此修复；发布版本以 Git 历史和对应远端 CI 结果为准。
保持原有市价单及零数量平仓意图，不修改信号、仓位计算公式、杠杆或真实交易配置。

## 修改前证据

对旧生产适配器注入模拟超时及提交回执：

```text
BEFORE ambiguous placement calls: 3
BEFORE ACK-only emitted: [('1', '0.0')]
```

旧逻辑会重试下单，底层每次生成新 orderLinkId；仅有 ACK 且无真实成交时，
仍按请求数量生成价格为零的 FillEvent。底层成交查询还没有订单过滤、完整分页和严格错误语义。

## 实现

| 文件 | 修改原因与行为 |
| --- | --- |
| `execution/adapters/bybit/adapter.py` | 接受固定客户端订单 ID；durable 订单不进入旧 TTL 撤单队列；通过 realtime/history 按身份查单；完整分页获取匹配订单的 Trade 记录；strict 账户查询失败时抛错 |
| `execution/adapters/bybit/execution_adapter.py` | 使用持久化提交、真实成交事件和可重放 outbox；不按 ACK/请求量/最近一笔无关成交补造事实；恢复前阻止新增风险；平仓使用明确的真实仓位和本策略已确认数量 |
| `shared/confirmed_execution.py` | 在现有限价默认路径外增加显式 market 模式；固定 ID、先记录后提交；超时仅查询；校验实际成交及重复身份；减仓事务核对未消费成交和 checkpoint revision |
| `shared/execution_journal.py` | 添加可空 order_type 列，持久化市场单语义，重启后不误走限价撤单流程；保留已有订单和未知状态 |
| `runner/builders/alpha_builder.py` | 为 Bybit 接入 owner、DurableFillConsumer 和真实状态回调 |
| `runner/alpha_main.py` | 将现有账户单写者 lease 扩展到 Bybit；原有启动恢复和周期对账复用通用 consumer 路径 |
| `decision/modules/alpha.py` | expired 终态清理执行 pending 状态；未修改策略数学 |
| `execution/adapters/bybit/mapper.py` | 保留 execFee 正负号与返佣事实，不改变收益计算公式 |
| `.github/workflows/ci.yml` | 将新增/更新的回归检查接入现有 lint 和 Python 测试；stdlib market 测试由 local-safety 自动发现 |

DurableFillConsumer 的实现保持不变，仅把说明改为适用于多个交易所。
TCA 日志写入、关闭失败均不得影响成交交付。
RISK_SCOPE 必须明确且没有首尾空白，避免 journal 与账户 lease 使用不同身份。

## 关键行为

1. 提交前持久化账户 scope、策略 owner、订单意图及固定客户端 ID。
2. 超时、错误回执、查不到订单均保留 reservation，既不重发，也不假定没有成交。
3. ACK 不是成交。只有订单身份、symbol、方向、trade ID、实际数量及价格匹配的事实可入账。
4. 成交记录同时是 outbox；消费回执与状态 checkpoint 在一个事务提交，崩溃后重放且去重。
5. 交易所终态只有在完整成交数量与累计成交一致时才能解除未知锁。
6. 本策略有未消费成交，或减仓读取后的 checkpoint revision 改变时，不允许用旧数量再减仓。
7. 对账失败、手工仓位、未知挂单、双向仓等不确定情况阻止新增风险，不通过改写持仓制造一致。

## 验证

原生回归命令（本轮复用已构建的 CPython 3.12/Rust 扩展镜像，将当前源码只读挂载到 /app）：

```bash
python -m pytest tests/unit/bybit/ tests/unit/execution/adapters/okx/ \
  tests/unit/execution/test_bybit_execution_adapter.py \
  tests/unit/execution/test_adapter_exceptions.py \
  tests/unit/runner/test_startup_check.py \
  -o addopts='' -q --tb=short -p no:cacheprovider
```

另运行 `tests/offline` 全部 unittest，包含 market 模式的真实子进程崩溃窗口。
运行过程中既有 Python socket 阻断，也有容器 `--network none`，无账户或网络访问。

最终结果：原生相关回归 **265 项通过**；离线 unittest **76 项通过、无跳过**；
三种错误变异均按预期使测试退出 1。正常验证总退出码为 0。

覆盖：ACK 无成交、超时返回异常和 retCode=-1 两种形态、固定 ID 恢复、部分/延迟/重复/冲突成交、
真实 V5 响应经过完整适配器与原生消费链、返佣符号、事务回滚、提交前后进程退出、
平仓真实数量、双向仓拒绝、未交付平仓保护、持仓版本竞争、expired 清理、日志失败及真实 builder 接线。

三种故意注入的错误仅存在于一次性测试进程，不改写工作树：

| 变异 | 期望结果 |
| --- | --- |
| 超时后重试三次 | 测试退出 1，观察到 3 次调用与最多 1 次的断言冲突 |
| 无成交时构造 FillEvent | 测试退出 1 |
| 移除减仓未消费成交/版本检查 | 测试退出 1，观察到不应发生的第二次平仓 |

Ruff E/W/F、工作流 YAML/run 语法、git diff --check 另行验证。
详细输出保存在 `/workspace/scratch/bybit-order-evidence/`。

## 验证边界与剩余限制

- 本报告是离线工程验证，不代表真实 Bybit testnet 或实盘协议验收；没有运行远端 Actions。
- 修改了共享执行组件，因此同时回归了现有 OKX 路径；没有修改其限价、周末数量或策略规则。
- journal 必须位于持久磁盘，恢复必须使用原 scope、owner 和数据库。当前 Compose 的运行状态持久挂载问题仍未处理，不能据本次修复宣布可部署。
- 不自动补送未知提交；历史查询不可用或超出完整保留范围时持续阻断，需独立的人工对账流程，不能删除 journal 解锁。
- 历史持仓、双向仓和非此 journal 创建的订单不自动认领。其它直接调用原始 Bybit adapter 的历史脚本不属于本次 durable Alpha 路径覆盖范围。
- journal 与 StrategyPosition 保存真实费用；全局 FillEvent/引擎账户投影尚无费用字段，本轮没有扩展它，也不宣称引擎费用与账户收益已准确对账。
- 组合风险额度预留、Binance 的订单确认问题、缺失 OI 组件及完整线上恢复仍是后续任务。
