# 运行状态持久化与离线恢复验收

基线：master `a2372f37cf1230cbf208d1499adddba636fede31`。
该 Bybit 修复已发布，远端 [CI 42259315011](https://github.com/General-zzz-trade/quant_system/actions/runs/42259315011) 全部通过。
本轮只处理 paper 状态持久化、恢复和备份，不修改策略数学、仓位规则、真实交易配置。
所有演练使用临时目录、模拟交易所及断网容器；未运行部署或访问账户。

## 修改前复现

1. Compose 只挂载数据、模型和日志，未挂载 `data/runtime`。容器 A 写入模拟未知订单后销毁；容器 B 中数据库不存在，断言退出 1。
2. 删除订单数据库后，旧 `ExecutionJournal` 自动创建空库，未知订单数由 1 变成 0。
3. 旧 `backup_remote.py` 清单没有 `orders.sqlite`；直接打包处于 WAL 模式的 `state.db` 主文件，恢复后读取已提交表失败：`OperationalError: no such table: facts`。

原始输出保留于工作区 `/workspace/scratch/runtime-state-evidence/` 的 `container-before.log`、`guard-before.log`、`backup-before.log`。其中均为合成数据。

## 修改文件与行为

| 文件 | 原因及结果 |
| --- | --- |
| `docker-compose.yml` | 仅 paper 服务增加外部持久目录绑定，固定运行状态路径；禁止 Docker 自动创建缺失宿主目录。未配置时 paper 失败，但 live/framework 仍可解析 |
| `shared/runtime_state.py`、`scripts/check_runtime_state.py` | 显式初始化及启动校验；订单库和回撤库必须同时存在并绑定同一 scope/state_id。缺失、损坏、错身份、只读或路径逃逸均拒绝，不能自动重建空库 |
| `runner/alpha_main.py` | 配置 managed state 时，在数据/账户连接前校验，并持有同目录单写者锁；目录必须对应 runner 的 `data/runtime`，覆盖原有检查点路径 |
| `scripts/check_paper_runtime.py`、`scripts/deploy.sh` | 校验实际渲染的 paper 挂载、数据库路径、账户 scope、目录可写与完整性；禁止状态放在 checkout 中，也拒绝嵌套挂载遮蔽数据库。预检失败不启动服务 |
| `shared/runtime_backup.py`、`scripts/runtime_state_backup.py` | 新的本地运行状态快照入口。持有执行锁，用 SQLite backup API 保存 WAL 中已提交数据；检查文件清单、哈希、库完整性及身份；仅原子恢复到不存在的新目录 |
| `scripts/check_runtime_recovery.py`、`tests/fixtures/runtime_recovery_worker.py` | 真实串行容器替换验收；模拟未知提交与未消费成交；检查第二写者被拒及缺失挂载失败；直接渲染已检入 Compose 验证配置路径 |
| `tests/offline/test_runtime_state.py`、`test_runtime_backup.py`、`test_ci_deploy_safety.py` | 身份、权限、损坏、锁、WAL、备份清单、部署预检回归 |
| `tests/unit/runner/test_startup_check.py` | 验证真正启动入口在状态异常时阻止账户访问，正常状态持锁后才进入线上输入检查；离线启动检查保持独立 |
| `.dockerignore`、`.github/workflows/ci.yml` | 将新入口包含在镜像中；将检查接入 lint、现有离线矩阵及容器 CI |

## 验收结果

- 原生相关回归：271 项通过，包含已有 Bybit/OKX 回归及新增入口检查。
- 全部离线 unittest：117 项通过，无跳过。宿主机缺少 hatchling 的跳过结果已被容器内完整验证覆盖。
- 容器恢复：准备、替换恢复退出 0；缺失挂载的负向启动退出 2；完整验收命令退出 0。
- 实际 Compose：使用原 YAML 的临时副本和空测试 `.env` 渲染；未读取真实账户配置，也未启动服务。验证持久目录配置有效；不配置 paper 时其他 profiles 仍可解析、paper 预检失败。
- Ruff E/W/F、工作流语法、`bash -n`、`git diff --check` 与跟踪源码安全扫描通过。

恢复结果摘录：

```json
{"phase":"recovered","new_placements":0,"unresolved":1,"applied":1,"duplicate_applied":0,"second_writer_blocked":true}
```

故障注入均在临时副本或一次性进程中完成，不改变提交源码：

| 故意破坏的行为 | 证据与结果 |
| --- | --- |
| 用普通复制替换 SQLite backup | `backup-raw-copy-mutation.log`：WAL 测试退出 1，已提交表丢失 |
| 忽略执行锁 | `ignored-lease-mutation.log`：第二写者意外退出 0，与必须拒绝的断言冲突，测试退出 1 |
| 删除已检入 Compose 的持久挂载 | `compose-mount-mutation.log`：实际渲染后的预检拒绝，退出 1 |
| 移除替换容器的状态挂载 | `container-mount-mutation.log`：正常恢复阶段被阻断，整条验收命令退出 1 |

主要复验命令：

```bash
python scripts/check_local_safety.py
python -m scripts.check_runtime_recovery --image quant-ci:startup-check
python -m pytest tests/unit/bybit/ tests/unit/execution/adapters/okx/ \
  tests/unit/execution/test_bybit_execution_adapter.py \
  tests/unit/execution/test_adapter_exceptions.py \
  tests/unit/runner/test_startup_check.py -o addopts='' -q --tb=short
```

本地使用已构建原生镜像并只读挂载当前源码；容器网络使用 `--network none`。恢复脚本可加 `--source-root .` 复用已有镜像。

## 离线操作入口

以下为命令示例，没有在真实状态目录执行。初始化仅适用于全新 paper 状态，不会认领或覆盖旧数据库。

```bash
python -m scripts.check_runtime_state init \
  --state-dir /srv/quant-paper/state --scope paper-example-account
python -m scripts.check_runtime_state check \
  --state-dir /srv/quant-paper/state --scope paper-example-account
```

paper Compose 使用 `QUANT_PAPER_STATE_DIR` 指向已初始化的绝对目录，`QUANT_PAPER_RISK_SCOPE` 必须与初始化 scope 相同。状态目录应在本地持久磁盘并位于代码 checkout 外。

停止所有状态写者后，备份到新的目录；目标父目录须已存在：

```bash
python -m scripts.runtime_state_backup backup \
  --state-dir /srv/quant-paper/state --scope paper-example-account \
  --destination /backup/quant-paper/snapshot-001
python -m scripts.runtime_state_backup restore \
  --snapshot /backup/quant-paper/snapshot-001 --scope paper-example-account \
  --destination /srv/quant-paper/restored-state-001
```

恢复保留原 scope、state_id、未知订单、成交消费记录和检查点，不覆盖原目录，不删除任何订单来解锁。

## 验证边界和剩余风险

- 新保护通过 `RUNTIME_STATE_DIR` 接入 paper；没有迁移既有真实账户状态或修改 live 服务配置，不能据此宣布实盘部署就绪。
- 锁要求同账户使用同一规范目录及支持文件锁的本地磁盘；不是跨主机分布式锁。恢复副本不能与原目录同时作为执行状态使用。
- 备份前须停止所有写者；执行锁只约束合作入口。每个 SQLite 库一致，不代表多个库和普通文件具有跨文件事务。
- 快照哈希用于发现损坏，不是可信签名。旧快照可能缺少备份之后发生的订单，恢复后仍须完成账户对账才可允许新增风险。
- 旧远程备份/恢复和 systemd 任务没有切换到新入口；它们不能作为本次运行状态恢复的验收依据。新工具只执行本地操作，不上传秘钥或访问 S3。
- 在线 OI 组件缺失仍阻止正常线上启动；Binance 订单确认、组合风险原子预留、费用投影与运行就绪探针仍属后续独立任务。
