# Quant System

Binance USDT-M 永续合约（BTCUSDT、ETHUSDT）的目标仓位交易系统，纯 Python。

## 现状（2026-11-16）

- **没有获准实盘，也没有任何策略被证明有优势。** 文档、配置或提交说明里出现的旧夏普、旧收益数字都没有经过验证，不能当作结果。
- **前向检验**：4 个候选在前向检验中，查看日期是 2027-02-13（只报告）和 2027-05-13（判定）。
  - 4 个候选是：风险管理型多头，以及 ETH/BTC 配对、周级方向、爆仓后反转。
  - 规则见 `research/protocols/`，登记见 `research/forward_looks.jsonl`。
- **产品运行时**：`runner/binance_target.py`。
  - 默认是模拟盘，默认用测试网。
  - 执行模式要求一个由人初始化过的状态目录，并通过全部风控闸门。
  - 主网要求签名规格里的 `passed` 为真，目前是假，所以主网会被拒绝。
- **架构改造**：M1–M4 和 M5 之前的全部步骤（S1–S12）已完成，见 `docs/arch/roadmap.md` 和 `docs/arch/execution-log.md`。
  - 旧的 ML 运行时、Bybit、OKX、Polymarket、Docker 部署路径和 Rust 扩展已于 2026-11-15 删除；最后一版见 git 标签 `archive/ml-line-2026-11-15`。
  - 下一步是 M5：测试网连续运行 4 周。开始条件、每日检查和演练见 `docs/m5_checklist.md`，要先由人完成其中的准备。

## 目录

| 目录 | 内容 |
|---|---|
| `trading/` | 新架构：领域、应用、适配器、组装。严格类型检查，导入关系由 `.importlinter` 约束 |
| `runner/binance_target.py` | 运行时入口；研究撮合器在 `trading/adapters/simulated/bar_broker.py`（S10 从 `runner/offline/broker.py` 搬入） |
| `lab/` | 研究：固定日期窗口的研究、前向检验回放、试验登记 |
| `data/` | 公开数据下载（不可变的日期快照）、K 线质量检查 |
| `infra/` | 模型和规格的签名校验、服务单元、日志轮转、单元同步脚本、主机安装脚本 |
| `monitoring/` | 看门狗（只看 binance-target）、成交分析日志 |
| `research/` | 预先登记的协议、结果、试验登记表、前向查看登记 |
| `infra/systemd/` | 服务单元：`binance-target`（模拟盘）、`binance-target-testnet`、`binance-target-mainnet`、`quant-alert@`、`state-backup@`、`health-watchdog`、`log-archive` |
| `docs/arch/`、`docs/adr/` | 架构现状、迁移计划、执行记录、架构决策 |
| `tests/` | 测试；`tests/support/` 是测试用的旧执行接口 |

## 常用命令

```bash
make test                                        # 本地核心检查：资金路径测试、ruff、导入契约、mypy
make test-all                                    # 整个测试集（CI 分 4 片并行运行）
python scripts/check_local_safety.py             # 只用标准库的离线故障注入检查
python -m data.downloads.binance_public --start 2020-01-01      # 下载到新的日期快照目录
python -m trading.infrastructure.state_dir --state-dir <dir> init   # 执行模式前初始化状态目录（需要 RISK_SCOPE）
python -m trading.infrastructure.kill --state-dir <dir> activate --reason "..." [--flatten]
python -m trading.infrastructure.entry_pause --state-dir <dir> status
python -m trading.infrastructure.state_dir --state-dir <dir> restore --snapshot <文件> --reason "..."   # 只在状态库丢失或损坏时
python -m lab.paper_compare --paper-dir <模拟盘目录> --data-dir <数据快照>   # 模拟盘与研究逐笔比较
sudo bash infra/install_host.sh                  # 主机安装或更新；从不启动服务
```

完整命令和规则见 `CLAUDE.md`。

## 安全默认值

- 默认模拟盘、默认测试网。CI 从不部署，部署由人手动进行。
- 任何代码路径都不会自动提高风险限额、解除暂停、重启服务或替换模型。
- 状态读不出时阻断新敞口；权益或价格未知时下单量为零。
- 模型和策略规格只能经 `infra/model_signing.py` 加载。

## 文档

- `CLAUDE.md`：项目规则、现状、架构、注意事项（以它为准）
- `docs/binance_runbook.md`：运行手册
- `docs/m5_checklist.md`：M5 的开始条件、每日检查、演练和记录模板
- `docs/deploy_truth.md`：仓库里的服务单元和退役单元
- `docs/btc_eth_strategy.md`：第一版多周期策略的研究记录（2026-11-13）
- `docs/arch/execution-log.md`：每一步改动、审计和更正的记录
