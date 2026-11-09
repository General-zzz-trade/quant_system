# 组合额度原子预留

Alpha 执行入口改为共享 SQLite 准入；旧 `portfolio_risk.json` 只保留为历史监控接口，不再为生产 builder 放行。策略中的 BTC/ETH cap 及按买卖方向分别计量的规则保持不变。

`shared/portfolio_reservations.py` 使用 `BEGIN IMMEDIATE`，在下单日志和网络提交之前持久写入 `(scope, intent_id)` 预留。额度使用量是已核对的基线加本 epoch 全部预留；重复 intent 参数一致才幂等，参数冲突拒绝。成交、撤单、超时、重启或 TTL 均不单独释放额度。

更新基线时，持有同一个准入锁并锁住全部账户日志，严格查询全部声明账户的仓位和挂单，验证 state_id、环境、成交消费及持仓投影。未知订单、挂单、未消费成交、孤立预留和任何缺失账户都阻断更新；失败在锁内持久撤销就绪。完整验证后一次性替换基线并覆盖旧预留，避免重复计数。事务中途失败会回滚基线和覆盖操作。

## 显式初始化与使用

只在新的测试或经批准的迁移环境使用以下入口；本轮未对实际运行状态执行这些命令，也未访问账户。

注册文件为对象，`accounts` 数组中的每个条目声明 `scope`、已有 `state_dir`、`environment`（demo/testnet/live）及 `credential_prefix`（仅对账命令使用）。`state_dir` 必须是通过 `scripts.check_runtime_state` 明确初始化或迁移的状态目录，不接受临时空库替代。

```bash
python -m scripts.portfolio_risk_state init --registry accounts.json --group GROUP --path /persistent/risk.sqlite
python -m scripts.portfolio_risk_state inspect --group GROUP --path /persistent/risk.sqlite
```

初始化不联网、不建立可用基线、不覆盖已有库。运行时需要显式 `RUNTIME_STATE_DIR`、`RISK_SCOPE`、`PORTFOLIO_RISK_GROUP`、`PORTFOLIO_RISK_PATH`。全部账户必须指向同一个持久风险库；每个账户有自己的 orders/drawdown 状态和 state_id。风险库 cap 必须与已有策略 cap 一致。

单账户 Bybit 组在启动成交恢复后及运行中每约 60 秒收集对账。多账户组需要独立的全账户只读对账器：

```bash
python -m scripts.portfolio_risk_state reconcile --registry accounts.json --group GROUP --path /persistent/risk.sqlite
```

该命令读取每个 `credential_prefix` 对应的 `_API_KEY` / `_API_SECRET` 环境变量，只调用账户 GET，不下单、不设置杠杆。定期调度由部署人员配置；超期会阻断新增风险。当前只实现 Bybit linear UNIFIED 收集，其他 venue 的执行启动被拒绝，不能用不完整的旧监控快照替代。减仓仍经过原适配器的真实持仓与归属验证。

## 已验证与限制

- 20 项存储测试及 9 项入口/收集测试：独立进程抢占、崩溃恢复、过期、身份冲突、未知订单、精度、事务故障、成交后不重复计数。
- 禁网容器：293 项原生相关回归和 169 项离线测试通过。
- 故意移除额度比较：7 项回归失败，进程退出 1，包含两个进程均错误获准的场景。
- 没有实际账户或跨主机共享文件系统验收。SQLite 必须使用支持正确锁语义的本地持久存储；多主机不能各持一份数据库。
- 准入额度按原事件参考价计量；真实市场滑点和持仓市值变化不保证永远低于 cap。此变更没有改动仓位算法或订单类型。
- 预留后、下单日志前崩溃会留下孤立预留，保守阻断。需要可核实的人工恢复证据；删除库、重置基线或按时间释放都不是恢复方法。
- 共享风险库必须与全部账户状态一起协调备份和恢复；单账户备份不能证明整个组合可恢复。
