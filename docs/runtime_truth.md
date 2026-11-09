# 当前运行入口与验收范围

本文件以仓库代码为依据。历史版本对主机上已运行服务的描述未在当前会话验证，不能当作部署状态。

| 路径 | 当前入口 | 当前验收范围 |
| --- | --- | --- |
| Alpha 主链 | `runner.alpha_main` | 真实 builder/coordinator；OI/LS 数据契约；离线合成模型信号检查；Bybit 持久提交与成交恢复 |
| 兼容入口 | `scripts.run_bybit_alpha` | 直接调用 `runner.alpha_main.main` |
| 安装检查 | `--check-startup` | 禁网真实 Rust market→state；不验证模型或账户 |
| Paper 契约检查 | `--check-paper-contract` | 禁网真实 builder/特征/信号链，使用明确的合成行情、模型及余额；不验证线上模型、账户或交易所结算 |
| Compose paper | `quant-paper` | 显式持久状态，默认 dry-run；线上完整运行仍需外部数据与模型验收 |
| Compose live | `quant-live` | 存在配置不代表实盘通过；当前命令仍默认 dry-run，真实部署未执行 |
| Framework / 做市 / host systemd | 各自已有入口 | 不自动继承 Alpha 主链的已验证能力；当前主机是否启用未核实 |

标准 Alpha 链：市场事件 → FeatureComputeHook/Rust → AlphaDecisionModule → ExecutionBridge → BybitExecutionAdapter → 持久订单/真实成交 → DurableFillConsumer。

新增风险必须同时通过数据就绪和已有执行风控。数据就绪要求显式历史预热、有效因果 OI/LS、接收及事件时间有效；小时批量预测还必须验证它真正读取的 CSV。未知提交、未消费成交和账户对账问题仍由执行恢复机制阻断。

账户事实由持久订单日志及交易所已确认事实驱动，不能用订单 ACK 生成成交。各 runner 的账户投影不能相加充当交易所钱包余额。

## 更新约束

- 修改入口、状态所有权或启用范围时同步更新本文件及对应契约测试。
- 区分离线检查、公共数据探测、账户验收和部署验收，不能以其中一种代替另一种。
- 保留 `master` 为唯一长期主干，真实交易配置、策略数学及仓位规则的变更须独立说明。
- 后续阶段与剩余限制参见 `engineering-sequence.md`。
