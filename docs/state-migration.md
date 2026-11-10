# 状态迁移、完整组合快照与恢复

本阶段提供离线工具，所有演练只使用临时状态。没有迁移真实账户、修改 live 配置或执行部署。

执行模式必须显式配置 `RUNTIME_STATE_DIR`。启动在账户连接之前校验 orders/drawdown 身份、钱包完整表组及规范流水；缺库、部分钱包表、不同 scope/state_id、损坏记录都拒绝。钱包字段中的 `ready` 不能掩盖实际与预期余额不一致。通过存储预检仍不代表允许下单，启动还需完成新的数据、订单、仓位、风险和钱包对账。

## 单账户迁移

```bash
python -m scripts.migrate_runtime_state migrate \
  --source-dir /persistent/old-account --target-dir /persistent/new-account \
  --scope ACCOUNT_SCOPE
python -m scripts.migrate_runtime_state inspect \
  --target-dir /persistent/new-account --scope ACCOUNT_SCOPE
```

目标必须不存在且位于源目录之外。工具持有账户执行锁，逐库用 SQLite backup API 包含已提交 WAL，再将完整临时目录原子发布。未知提交、未消费成交、持仓检查点、回撤记录及辅助状态均保留。身份已存在时保留 state_id；旧格式只在目标添加可明确识别的表结构和身份，不推断订单是否完成、持仓归属或历史余额。缺失回撤库、未知 scope、部分身份、符号链接、凭证文件及嵌入的共享风险库均拒绝。

`runtime-migration.json` 保留文件哈希、表行数和未知状态统计。`inspect` 核对刚完成且尚未运行的迁移结果；目标开始运行后文件会变化，不能再当作原始迁移快照。源数据不会被删改；获取锁和 SQLite 读取可能创建锁文件或更新临时 sidecar。

恢复或迁移目标的钱包就绪状态会撤销，基线和流水保持原值。不能同时运行源目录与恢复副本；不同目录的本地锁不能阻止两个副本重复执行。

## 完整组合快照

单账户快照不包含账户目录外的共享风险库。组合恢复必须使用完整注册表，停止所有状态写者后同时保存风险库和每个账户：

```bash
python -m scripts.risk_group_backup backup \
  --risk-path /persistent/group/risk.sqlite --registry accounts.json \
  --group GROUP --destination /backup/group-snapshot
python -m scripts.risk_group_backup restore \
  --snapshot /backup/group-snapshot --group GROUP \
  --destination /persistent/restored-group
```

注册文件 `accounts` 数组声明每个 `scope` 和 `state_dir`，不得包含凭证。工具先持有风险库写锁，再按 scope 顺序持有全部账户执行锁；账户集合和 state_id 必须与持久风险注册完全一致。快照包含风险预留、钱包流水以及所有账户的完整安全文件清单与哈希。缺少账户、仍在运行、损坏、未声明文件或身份变化均拒绝；恢复只发布到新根目录。

恢复后风险库和所有钱包强制标为需要重新对账，未知订单、孤立预留不会消失。结果中的账户槽位与 scope 映射由快照清单给出；新路径须经操作员显式绑定，工具不会修改运行配置。

## 执行配置预检

```bash
docker compose --profile live config --format json \
  | python -m scripts.check_execution_runtime --service quant-live
```

这个命令只读取配置和本地状态，不启动服务。账户和共享风险目录需要各自位于 checkout 之外的持久可写绑定；禁止嵌套挂载遮蔽状态。仓库现有 `quant-live` 没有显式执行开关及完整绑定，应当被拒绝。本轮只用原 YAML 的临时副本与空测试环境渲染，未解析真实账户配置。

限制：需要支持 SQLite 和文件锁的本地存储；所有写者必须遵守锁并停止后备份。哈希用于发现损坏，不是来源签名。快照之后的交易所事实仍须重新读取；真实迁移、磁盘故障与跨主机容灾尚未验收。
