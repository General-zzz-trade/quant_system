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

## 后续顺序

4. 实盘状态迁移与恢复工具：待实现；真实状态迁移不在本轮运行。
5. 运行监控与发布验收：待实现；真实部署不在本轮运行。
