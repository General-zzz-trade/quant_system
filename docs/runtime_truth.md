# 当前运行入口与验收范围

本文件以仓库代码为依据。历史版本对主机上已运行服务的描述未在当前会话验证，不能当作部署状态。

| 路径 | 当前入口 | 当前验收范围 |
| --- | --- | --- |
| Alpha 主链 | `runner.alpha_main` | 真实 builder/coordinator；OI/LS 数据契约；离线合成模型信号检查；Bybit 持久提交与成交恢复 |
| 兼容入口 | `scripts.run_bybit_alpha` | 直接调用 `runner.alpha_main.main` |
| 安装检查 | `--check-startup` | 禁网真实 Rust market→state；不验证模型或账户 |
| Paper 契约检查 | `--check-paper-contract` | 禁网真实 builder/特征/信号链，使用明确的合成行情、模型及余额；不验证线上模型、账户或交易所结算 |
| 公共行情影子 / Compose paper | `runner.shadow_main` / `quant-paper` | 单一清单、公共行情、合成起始余额及本地观察；不构造账户客户端，不提交订单；线上完整运行仍需外部数据与模型验收 |
| Compose live | `quant-live` | 存在配置不代表实盘通过；当前命令仍默认 dry-run，真实部署未执行 |
| Framework / 做市 / host systemd | 各自已有入口 | 不自动继承 Alpha 主链的已验证能力；当前主机是否启用未核实 |

标准 Alpha 链：市场事件 → FeatureComputeHook/Rust → AlphaDecisionModule → ExecutionBridge → BybitExecutionAdapter → 持久订单/真实成交 → DurableFillConsumer。

2026-11-12 入口更新：`quant-paper` 改为 [`config/shadow.json`](../config/shadow.json) 驱动的公共行情影子入口，启用 BTC/ETH 的 1h、4h，显式禁用两个 15m runner。该入口没有 ExecutionBridge，不经过上述账户及成交路径；1h 记录未提交意图，4h 保留原有 signal-only 虚拟状态。启动方法、合成余额含义及未验证范围见 [公共行情影子运行](shadow_runtime.md)。本地完整模型和数据尚缺，公共网络、长期运行及真实部署未验收。

新增风险必须同时通过数据就绪和已有执行风控。数据就绪要求显式历史预热、有效因果 OI/LS、接收及事件时间有效；小时批量预测还必须验证它真正读取的 CSV。未知提交、未消费成交和账户对账问题仍由执行恢复机制阻断。

新增执行还必须绑定持久组合风险库。Bybit 全账户严格对账后建立冻结基线，准入以原子预留计量；旧 JSON 监控文件不再为生产入口放行。多账户必须共享同一风险库并由完整账户收集器更新，见 `portfolio-reservations.md`。

账户事实由持久订单日志及交易所已确认事实驱动，不能用订单 ACK 生成成交。成交投影保留费用和本次已实现收益。各 runner 的账户投影不能相加充当交易所钱包余额；账户现金只由不可变基线加交易所流水的 `change` 核对 `walletBalance`，见 `account-reconciliation.md`。

执行模式要求完整持久状态预检。离线迁移只发布到新目录，组合快照同时保存全部注册账户和共享风险库；恢复后必须重新对账。工具与限制见 `state-migration.md`，真实迁移尚未执行。

## 更新约束

- 修改入口、状态所有权或启用范围时同步更新本文件及对应契约测试。
- 区分离线检查、公共数据探测、账户验收和部署验收，不能以其中一种代替另一种。
- 保留 `master` 为唯一长期主干，真实交易配置、策略数学及仓位规则的变更须独立说明。
- 后续阶段与剩余限制参见 `engineering-sequence.md`。
