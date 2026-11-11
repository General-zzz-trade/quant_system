# 软件工程改进执行记录

起点：master `f3664e68dfcbbb492c70d3fbe27afea58b77539c`。按数据契约、组合额度、账户账本、状态迁移、运行监控五项依次实施；每项独立提交。保持策略数学、仓位规则和真实交易配置，不执行部署或真实订单。

## 第 1 项：OI/LS 契约与 paper 主链

**工程验证通过；线上数据及完整在线 paper 验收未通过。** 本环境对 Binance 公共 OI 接口探测得到 HTTP 451，不绕过服务限制，也未访问账户。

修改前证据：缺少 `data.oi_cache`；三天旧 CSV 仍能为一小时前闭合 bar 返回数值，且没有咨询实时缓存。证据在工作区 `engineering-sequence-evidence/stage1-cache-before.log` 和 `stage1-builder-before.log`。

实现：

- `data/oi_cache.py` 保持 Binance USD-M 小时 `sumOpenInterest`（基础币数量）与 global `longShortRatio`（账户比）的历史数据语义；刷新 55 秒，接收有效期 120 秒，小时观测有效期 3900 秒。保留 8 个历史点，按 bar 开盘时间因果取样。缓存失败、过期、乱序、冲突、异常数值立即撤销就绪；仅公共 GET，无密钥或写入接口。
- `alpha_builder` 使用显式 warmup/live 阶段，验证配置要求的连续有效历史 bar；实时不能以旧 CSV 兜底。数据门控在执行入口阻断新增风险，减仓仍经过原风控。
- `runner/data_contract.py` 与 `batch_predictor.py` 验证批量预测实际读取的 bar/OI/LS CSV，避免新鲜实时缓存掩盖旧批量输入。选择最新闭合小时且禁止未来观测，保持模型计算及策略公式。
- `alpha_main` 预热不足/失败拒绝继续启动，批量输入失败阻断新增风险。新增 `--check-paper-contract` 走真实 builder、Rust 特征与 Alpha 信号链，使用明确合成模型/行情/余额，全程禁网。
- 更新当前入口文档；旧 framework 文档标为历史参考。CI加入新测试与独立paper契约检查。

正常验证：293 项原生相关回归、140 项离线 unittest 无跳过通过，详见 `stage1-final.log`；Ruff E/W/F、工作流语法和源码安全检查执行于提交前。

失败验证：旧数据持续刷新不得续命（`stage1-cache-mutation.log`，变异测试退出 1）；过期数据使paper契约检查抛错；数据超时/限流/未来值/CSV过期/预热缺口均拒绝新增订单。

复验：

```bash
python scripts/check_local_safety.py
python -m pytest tests/unit/runner/test_data_readiness.py tests/unit/runner/test_paper_check.py -o addopts='' -q
python -m runner.alpha_main --check-paper-contract
```

公共接口探测只证明本环境得到 HTTP 451；不代表真实服务数据正确，也不代表线上模型、账户或结算验收。合成模型检查的 JSON 明确将这些项设为 false。

## 第 2 项：组合额度原子预留

**工程离线验证通过；真实账户及多主机运行未验收。**

修改前：两个 scope 各申请 60、cap 100，原快照检查两边均通过，合计准入 120（`stage2-before.log`）。

修改后：共享 SQLite 原子预留接入实际执行 gate；声明账户/环境/state_id；严格全账户核对后更新冻结基线。未知订单、孤立预留、未消费成交和失败查询保留额度并阻断；不按 TTL 自动释放。生产 builder 不再读取旧监控 JSON 放行，未支持 venue 明确拒绝执行启动。实现和运行约束见 [portfolio-reservations.md](portfolio-reservations.md)。

验证：20 项存储及 9 项入口专项测试；禁网容器 293 项原生相关回归、169 项离线测试通过（`stage2-final.log`）。移除额度比较后 7 项测试失败，退出 1（`stage2-mutation.log`）。

没有访问实际账户，也未变更真实配置。市场滑点、跨主机锁语义和完整组合备份仍需专门验收。

## 第 3 项：费用与账户对账

**工程离线实现完成，真实账户结算未验收。**

修改前：Python `FillEvent` 丢失交易所费用和已实现收益，策略账户投影无法得到这些事实；多个 runner 的账户余额也不能安全相加作为交易所钱包。证据见 `stage3-before.log`。

修改后：Bybit/OKX 持久成交保留费用和币种；成交消费在同一事务中先用已有 `StrategyPosition` 公式得到本次已实现收益，再投影 Rust 账户，失败整体回滚。新增 scope 级不可变钱包基线、幂等现金流水和严格 `walletBalance` 对账；新增风险必须通过钱包就绪门控。实现、初始化和验收边界见 [account-reconciliation.md](account-reconciliation.md)。

验证：禁网容器 636 项原生相关回归、193 项离线测试通过（`stage3-final.log`）。临时变异忽略钱包余额差异后，账本回归明确失败并退出 1（`stage3-mutation.log`）。Ruff E/W/F、差异检查和源码凭证扫描在提交前执行。

真实账户、历史报表基线、线上分页完整性和结算延迟没有验证，不据此宣称实盘可用。

## 第 4 项：状态迁移与完整组合恢复

**工程离线验证通过；真实状态迁移与容灾未验收。**

修改前：缺少明确的状态迁移入口；执行 CLI 可以解析未配置持久状态的参数（主入口已有阻断，不能据此认定实际会启动）；持久钱包 `ready` 标志能够掩盖实际余额与预期余额不一致。证据见 `stage4-migration-before.log`、`stage4-before.log`、`stage4-wallet-before.log`。

修改后：增加目标目录原子发布的迁移、完整风险组备份/恢复与执行挂载预检；保留未知订单和未消费成交，恢复后强制重新对账；拒绝身份错配、部分钱包表和已改变的迁移快照。具体文件、命令及限制见 [state-migration.md](state-migration.md)。

验证：40 项专项、636 项原生相关和 235 项离线测试通过（`stage4-focused-final.log`、`stage4-final.log`）；禁网容器重建后状态保留，去掉持久挂载时退出 2（`stage4-container.log`）。实际 Compose YAML 的空环境副本能够渲染，现有 live 服务被执行预检拒绝，未启动服务（`stage4-compose-execution.log`）。

失败验证：临时移除 WAL 检查后，对同一行的已提交更新逃过检查，回归测试失败且退出 1（`stage4-wal-mutation.log`）。恢复后钱包未重新核对、缺账户快照、文件哈希变化、活动写者等均有自动化拒绝用例。独立代码复查未发现新增阻塞缺陷。

第四项提交 `54aabe4` 的 GitHub CI 曾因 Docker 白名单遗漏两个运维脚本而失败，不能把上述源码挂载验证等同于完整 CI 通过。该失败证据与构建上下文修复纳入第五项。

## 第 5 项：运行监控与发布验收

**工程实现完成；线上 paper、真实部署与实盘验收未通过。完整容器验证须以本次提交的 GitHub CI 结果为准。**

修改前：仅导入模块的健康检查在没有状态与心跳时仍返回 0（`stage5-health-before.log`）；第四项容器 CI 缺失脚本（`stage4-github-container.log`）。

修改后：主循环同步发布原子状态，运行健康与新增风险均检查新鲜心跳及对账；拒绝 PID 复用后的旧状态。发布绑定 master 精确 SHA、实际成功 CI、本地镜像 ID、完整模型/数据哈希及新进程证据，预检失败不会触发服务回滚。补齐 Docker 运维入口并测试 `--help`。实现文件、命令及已知运行限制见 [runtime-release-acceptance.md](runtime-release-acceptance.md)。

故障注入：在内存中绕过心跳门控或精确 CI 校验，回归测试均失败、退出 1（`stage5-heartbeat-mutation.log`、`stage5-release-mutation.log`）；未将变异写入源码。实际 Compose 的空环境副本验证健康入口存在；现有 paper 命令含未配置的 runner，被预检拒绝（`stage5-compose.log`）。

本地验证：81 项专项测试、Ruff E/W/F、Shell 语法和工作流 YAML 校验通过。真实 Docker 构建上下文打包成功，导出后的 13 个新增/修改 Python 与 Shell 文件和工作树逐字节相同，运维入口完整（`stage5-packaged-context-build.log`）。本地完整容器回归因 rootless Docker 存储空间不足未能启动，明确记为未通过（`stage5-container-disk-failure.log`、`stage5-packaged-container-disk-failure.log`），不能用源码专项测试替代。GitHub 的正式镜像构建、原生测试及容器测试会对最终提交独立执行。

CI 收尾修正：`4f7d256` 的真实镜像构建通过，但测试发现部署夹具在 CI 的 `PYTHONPATH` 下被真实包覆盖，以及旧启动测试替身不支持 `atexit.register` 参数。已保留失败证据，补上夹具包身份和同环境回归，并让启动测试按真实退出顺序检查 `starting → stopped` 与 lease 释放。修正后本地全套离线检查运行 302 项，301 项通过、1 项因缺少 Hatch 构建依赖跳过；完整镜像中的零跳过结果仍以最终 CI 为准。

所有操作保持 master 主干，没有访问真实账户、发通知、执行部署或提交订单。策略公式、仓位规则及 live 配置保持原值。
