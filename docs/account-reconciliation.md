# 成交费用与账户现金对账

账户状态分为两层，不能相互替代：

- 每个策略的成交投影记录已确认的数量、均价、已有平仓公式计算出的已实现收益，以及交易所给出的费用和币种。
- 每个账户只有一个钱包现金账本。它从不可重置的 USDT 基线开始，只累加 Bybit 账户流水的 `change`，再与币种级 `walletBalance` 比较。不能累加各 runner 的余额，也不能使用包含未实现收益的 `totalEquity`。

Bybit 官方定义 `change = cashFlow + funding - fee`。费用为正时是支出，为负时是返佣；资金费为正时是收入。成交投影中的费用不会再次写入钱包账本，行情 `FundingEvent` 也不当作真实资金费结算。

## 持久状态和执行门控

`wallet_metadata`、`wallet_transactions` 和 `wallet_reconciliation` 位于已有 `orders.sqlite`，与 runtime `state_id` 和 `scope` 绑定。基线只能显式初始化一次；删库、重新设定当前余额或推进水位都不能消除差异。分页失败、未知类型、异常币种、冲突 ID、余额不一致、过期数据和采集期间钱包变化都会阻断新增风险；减仓继续使用既有严格持仓及归属验证。

执行启动在账户连接前要求钱包表和身份已经存在。运行中按不超过 7 天的窗口读取完整分页，正常情况重复最近 7 天以接收晚到流水；不一致时从不可变基线完整回放。采集前后钱包余额必须稳定。由于交易所文档说明流水可能延迟，离线实现不能证明线上接口的实时完整性。

## 显式初始化与检查

以下入口不会由 CI 或部署自动初始化。`init` 要求执行进程停止，并由操作员提供完整账户报表对应的 USDT 余额、截止时间和非敏感证据引用：

```bash
python -m scripts.account_ledger_state init \
  --state-dir /persistent/account --scope ACCOUNT_SCOPE \
  --balance 1000.00 --start-ms 1700000000000 --source statement-reference

python -m scripts.account_ledger_state inspect \
  --state-dir /persistent/account --scope ACCOUNT_SCOPE
```

`reconcile` 使用环境中的 Bybit 只读凭证并调用账户 GET；它不连接下单入口、不设置杠杆、不发送或撤销订单：

```bash
python -m scripts.account_ledger_state reconcile \
  --state-dir /persistent/account --scope ACCOUNT_SCOPE --full
```

当前收集器只支持 Bybit UNIFIED、USDT，以及明确列出的 TRADE、SETTLEMENT、TRANSFER_IN、TRANSFER_OUT、FEE_REFUND。借贷、奖金、额外费用、非 USDT 资产和未知子类型均阻断，必须先增加经过验证的会计规则。

## 验收边界

离线回归覆盖费用和返佣符号、部分平仓收益、重复及冲突流水、晚到流水、分页失败、事务回滚、重启、异常币种和钱包差异。没有访问实际账户，因此真实交易所流水完整性、结算延迟、历史基线以及线上钱包一致性仍为未验收。
