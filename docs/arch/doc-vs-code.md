# 旧文档 vs 代码实际行为

- 基线：HEAD `e3cd2d6`。本文件在 `current-state.md` 写完之后才开始读旧文档。
- 范围：仓库内被 git 跟踪的 85 个 Markdown 文件（不含本次新产出的 `docs/arch/`、`docs/adr/`）。`docs/_archive/` 在仓库里不存在，没有读取。
- 只列不一致之处。每条给出文档位置、代码证据和建议（删除 / 更新 / 保留）。

## 0. 方法与覆盖

| 做法 | 覆盖 |
|---|---|
| 机械核对：抽取每个文档里引用的仓库路径、`python -m` 模块、systemd 单元名、`make` 目标，逐个检查是否存在 | 全部 85 个文件 |
| 全文阅读并与代码逐条比对 | `README.md`、`docs/binance_runbook.md`、`docs/btc_eth_strategy.md`、`docs/runtime_truth.md`、`docs/deploy_truth.md`、`docs/operations.md`、`docs/known_test_failures.md`、`docs/execution_recovery.md`、`docs/local_development.md` |
| 按关键词定向检索（热加载、自动重训、已删除入口、业绩数字、杠杆） | 其余 `docs/*.md`、`research.md`、`CHANGELOG.md`、`models_v8/README.md` |
| 只做机械核对，没有逐句比对行为 | `docs/superpowers/**`（10 个）、`tasks/**`（17 个）、`research/results/**`（12 个）、`research/protocols/**`（4 个），以及 2026-11 写的 Bybit / shadow 专题文档 13 个 |

机械核对的结果【实际运行结果】：85 个文件共引用 846 处仓库内对象，其中 357 处不存在。

## 1. 逐条不一致

### 1.1 `README.md`

| 位置 | 文档说法 | 代码实际行为 | 建议 |
|---|---|---|---|
| `:24-35` | 当前状态是 "Strategy H"，回测 Sharpe 2.25，$500 → $215 万，10 倍杠杆，Bybit Demo 4 个 runner | 3 倍总敞口是代码常量（`shared/exposure_gate.py:16`）。`bybit-alpha.service:13` 启动 3 个 runner 且为 dry-run。`models_v8/` 只有 json，模型二进制不在仓库里。唯一有 systemd 单元的候选运行时是 `runner.binance_target`，README 全文没有提到它 | 更新：整段删除，改写为当前状态 |
| `:70-71` | 模型热加载：`kill -HUP`，无需重启 | SIGHUP 只记一条警告，不换模型（`runner/alpha_main.py:762-769`） | 更新：删除该命令 |
| `:83-84` | `python3 -m alpha.auto_retrain` | 模块不存在 | 更新：删除 |
| `:115` | `decision/modules/alpha.py` 约 300 行 | 1,452 行，`decide()` 一个方法 671 行 | 更新 |
| `:135-137` | `execution/` 适配 Bybit、Hyperliquid、Binance；`safety/` 有电路断路器、Kill Switch、去重 | 没有 Hyperliquid 适配器。`execution/safety/` 只有 `limits.py` 和 `message_integrity.py` | 更新 |
| `:145` | `risk/` 是风控（StagedRisk、AdaptiveStop、GateChain） | Python `risk/` 包没有任何运行时使用，见 `current-state.md` 4.1 | 更新 |
| `:150`、`:154`、`:157` | `runner/strategy_config`、`portfolio/`、`regime/` | 都不存在。配置在 `strategy/config.py`，市场状态在 `strategy/regime/` | 更新 |
| `:164-165` | 102 个 `.rs` 文件，198 个 PyO3 导出，100% 被 Python 生产代码调用 | 137 个 `.rs` 文件，191 个导出，其中 46 个无处引用、12 个仅测试引用 | 更新 |
| `:6` | OKX 已加入成交驱动持仓、持久化成交消费和重启对账 | OKX 执行在 `runner/portfolio_risk.py:62-63` 被拒绝，可启动路径上不会实例化 OKX 执行适配器 | 更新 |
| `:170-180` | Alpha 研究结论：4h Sharpe 3.62–4.57 等 | 仓库里没有能复现这些数字的代码路径或数据；文件自己在 `:19` 声明"尚未完成独立复核" | 更新：删除数字，或移到带"未验证"标注的历史附录 |

处置：**更新**（实际是重写）。README 是入口文档，现在描述的是另一个系统。

### 1.2 `docs/binance_runbook.md`

| 位置 | 文档说法 | 代码实际行为 | 建议 |
|---|---|---|---|
| `:11` | 交易所杠杆设置不超过 3，由 `adapter.py` 的 `MAX_LEVERAGE` 保证 | `MAX_LEVERAGE = 10`，默认值才是 3（`execution/adapters/binance/adapter.py:44-45`）。`BINANCE_EXCHANGE_LEVERAGE` 可设到 10（`runner/binance_main.py:539-543`） | 更新。这是风控参数的错误陈述 |
| `:14` | 任何脚本都不会自动放宽风险限额 | 对 Binance 成立。旧运行时有一条自动恢复路径：`monitoring/live_ic_killswitch.py:199-203` 在 IC 回升后自动解除暂停，定时器每小时运行 | 更新：限定为 Binance 运行时，并写明旧运行时的例外 |
| `:131` | 止损默认距入场价 4% | `runner.binance_main` 是 4%（`:588`），`runner.binance_target` 是 8%（`runner/binance_target.py:374`） | 更新：分运行时写 |
| `:110-111` | 回撤默认 3 / 8，限额默认 1000 / 10000 / 50 | 对 `binance_main` 成立。`binance_target` 是 5 / 12 和 5000 / 50000 / 50（`runner/binance_target.py:260-265`） | 更新 |
| `:144` | 平仓 5 次失败后自动激活 kill switch（只拦开仓） | 代码把文件整体改写为 `active=True, flatten=False`（`runner/binance_main.py:466-469`）。如果操作员写的是 `flatten: true`，它会被改成 false | 更新：写明会覆盖操作员的平仓指令 |
| `:149-152` | `python3 -m shared.kill_file activate ...`，不带 `--path` | 默认路径是 `data/runtime/binance/`（`shared/kill_file.py:25`），只对 `binance_main` 有效。`binance_target` 读的是 `data/runtime/binance_target/` | 更新：所有命令写全 `--path` |
| `:133` | 止损触发后产生的订单会被纳入日志，成交只入账一次 | 首次查询拿不到实际订单号时，该止损被标为 `triggered` 且不再回看，成交永不入账（`current-state.md` R8，已复现） | 更新 |
| `:126-127` | 重启后从日志恢复；每 5 秒对账 | 成立。但文档没有写：状态库不存在时会被静默重建（R6） | 更新：补充 |
| 全文 | 以 `runner.binance_main`（ML 运行时）为主线写操作流程 | 没有任何启动面启动 `binance_main`；`infra/sync_systemd.sh:39-41` 把 `binance-runtime.service` 列为已退役。在用的是 `binance-target.service` | 更新：改以 `binance_target` 为主线 |

处置：**更新**。结构可保留，上表各条必须改。

### 1.3 `docs/btc_eth_strategy.md`

| 位置 | 文档说法 | 代码实际行为 | 建议 |
|---|---|---|---|
| 全文 | 描述 2026-11-13 的多周期研究（39 个配置，未通过） | 仓库里另有 `lab/overlay_study.py` 和 `research/results/overlay_2026-11-13/final_spec.json`（`screen_passed: true, passed: false`）。文档没有提到 overlay。`binance-target.service` 的 Description 指向本文档 | 更新：补充 overlay，或把 unit 的指向改到新文档 |
| `:19` | 执行模式复用 kill switch、峰值回撤平仓，重启后从日志恢复，已测试 | 正向路径有测试（`tests/unit/lab/test_target_runtime.py::test_execution_lifecycle_restart_and_flatten`）。平仓失败不重试、挂单阻塞平仓没有测试，且已复现为缺陷（R1、R2） | 更新：写明限制 |
| `:89` | kill switch 命令带 `--path data/runtime/binance_target/kill_switch.json` | 与代码一致 | 保留 |
| `:17-18` | 截尾窗口一致、纸面与研究逐笔一致，有测试 | 测试存在并通过 | 保留 |

处置：**更新**。

### 1.4 `docs/runtime_truth.md`

| 位置 | 文档说法 | 代码实际行为 | 建议 |
|---|---|---|---|
| 全表 | 列出的入口是 `runner.alpha_main`、`scripts.run_bybit_alpha`、`runner.shadow_main`、`quant-live`、Framework | 没有列 `runner.binance_target` 和 `runner.binance_main` | 更新 |
| `:12` | `quant-live`：当前命令仍默认 dry-run | 命令里的 `SUIUSDT`、`AXSUSDT` 不在 `strategy/config.py:81-109`，`runner/alpha_main.py:204-205` 启动即抛错 | 更新 |
| `:13` | Framework / 做市 / host systemd：各自已有入口 | `quant-framework` 指向不存在的 `runner.live_runner`；做市单元在 `RETIRED_UNITS` 里 | 更新 |
| `:15` | 标准 Alpha 链以 `BybitExecutionAdapter` 结尾 | 对旧运行时成立 | 保留该句，但标注为冻结线 |

处置：**更新**。文件名自称"真相"，必须覆盖在用的运行时。

### 1.5 `docs/deploy_truth.md`

2026-11-14 按 `infra/systemd/` 重写过，单元表与 `ExecStart` 一致，退役清单与 `infra/sync_systemd.sh:13-60` 一致。不一致之处：

| 位置 | 文档说法 | 代码实际行为 | 建议 |
|---|---|---|---|
| `:30` | `polymarket-maker`：frozen venue | 该单元启动即 ImportError 并以 `Restart=always` 循环（`polymarket/runner.py:23`） | 更新：标为 broken |
| `:59` | `quant-live`：profile `live`；frozen venue | 无法启动（同 1.4） | 更新：标为 broken |
| `:14` | `binance-target` 运行 `--spec models_binance/strategy/final_spec.json` | 该文件不在仓库里，`models_binance/` 为空 | 更新：写明 spec 需先发布 |
| `:32-33` | `daily-pnl-alert` 不加载 `.env` | 单元里有 `EnvironmentFile=-/quant_system/.env`（可选加载） | 更新 |

处置：**保留并更新**上述 4 处。这是旧文档里最接近代码的一份。

### 1.6 `docs/operations.md`

文件顶部自己标注了 Outdated。39 处引用里 29 处不存在（`scripts.ops.*`、`scripts.auto_retrain`、`scripts.data.*`）。另有与代码相反的操作指引：

| 位置 | 文档说法 | 代码实际行为 |
|---|---|---|
| `:61-62` | 热加载模型：`kill -HUP` | SIGHUP 被忽略（`runner/alpha_main.py:762-769`） |
| `:76-77` | `daily-retrain.timer`、`auto-retrain.timer` 自动重训并热加载 | 两者在 `RETIRED_UNITS`（`infra/sync_systemd.sh:27-30`） |
| `:144` | 检查 watchdog 是否在重启服务 | watchdog 不重启任何服务（`monitoring/watchdog.py`，只有 `systemctl is-active`） |
| `:218` | demo 使用 10 倍杠杆 | 3 倍硬上限 |

处置：**删除**。保留只会让操作员执行不存在的命令。

### 1.7 `docs/known_test_failures.md`

| 位置 | 文档说法 | 代码实际行为 | 建议 |
|---|---|---|---|
| `:1-3` | 已知失败为无；`make test` 和 `make test-all` 都应全部通过 | 实测 `2920 passed, 34 skipped`，一致 | 保留 |
| `:14` | `data/scheduler/__init__.py` 仍导入这两个模块 | `data/scheduler/` 目录已不存在 | 更新：删掉这句 |

处置：**保留**，改一处。建议并入 `Makefile` 注释或测试说明后删除单独文件。

### 1.8 `docs/execution_recovery.md`、`docs/local_development.md`

| 位置 | 文档说法 | 代码实际行为 | 建议 |
|---|---|---|---|
| `execution_recovery.md:59` | Bybit / Binance 尚未接入持久化消费与启动恢复流程 | 三个执行适配器都构造 `ConfirmedExecutor`（binance `:71-77`、bybit `:38-41`、okx `:37-40`），Binance 两个运行时都走 journal 消费 | 更新 |
| `execution_recovery.md:14`、`local_development.md:54-55` | 以 OKX 执行进程为描述对象 | OKX 执行被 `runner/portfolio_risk.py:62-63` 拒绝 | 更新 |
| `execution_recovery.md:55` | 每单最多读取 100 条成交，完整分页仍待开发 | 与代码一致（`shared/confirmed_execution.py:254`），即 R9 | 保留 |
| `local_development.md:45` | 需要新鲜的 `data/runtime/portfolio_risk.json`，缺失会阻止入场 | 读这个文件的 `runner/gates/portfolio_risk_file.py` 只有测试引用。执行准入走的是组合额度库（`runner/portfolio_risk.py`） | 更新 |
| `local_development.md:46` | 回撤默认 3 / 10 | 对 `alpha_main` 成立；`binance_main` 是 3 / 8，`binance_target` 是 5 / 12 | 更新 |

处置：**更新**，并在标题标明"仅适用于冻结的 Bybit/OKX 运行时"。

### 1.9 其余有明确矛盾的文档

| 文件 | 矛盾 | 证据 | 建议 |
|---|---|---|---|
| `research.md` | "零死代码""198 个导出 100% 被调用""所有生产文件 ≤800 行" | 109 个模块不可达；46 个导出无引用；`runner/alpha_main.py` 1,375 行、`decision/modules/alpha.py` 1,452 行 | 删除 |
| `docs/wiring_truth.md` | `:48` SIGHUP 热加载 WIRED；`:55` IC 衰减触发自动重训；`:50` 基础杠杆 10 倍；`:53` VenueRouter WIRED | SIGHUP 被忽略；重训定时器已退役；3 倍上限；`execution/adapters/venue_router.py` 不存在。34 处引用里 20 处不存在 | 删除 |
| `docs/daily-1d-rollout.md` | `:54` 预期 `ExecStart` 带 `--live` | `--live` 被代码拒绝（`runner/daily_live_runner.py:302-307`） | 删除 |
| `docs/api.md` | 描述 `runner/live_runner.py` 的 `LiveRunner` API | 文件不存在。21 处引用里 13 处不存在 | 删除 |
| `docs/execution_contracts.md` | 描述 `execution/ingress/*`、`execution/bridge/live_execution_bridge.py`、`execution/algo_adapter.py` 等契约 | 38 处引用里 29 处不存在。在用的订单路径是 `shared/confirmed_execution.py`，文档没有覆盖 | 删除，由 `docs/arch/target.md` 的端口定义取代 |
| `docs/runtime_contracts.md`、`docs/runtime_direction.md` | 引用 `runner/live_runner.py`、`scripts/ops/run_bybit_alpha.py`、`quant-runner.service` | 都不存在（6/8、6/9 处失效） | 删除 |
| `docs/production_runbook.md` | 引用 `runner/live_runner.py`、`scripts/ops/*` 和 4 个不存在的测试文件 | 13 处引用里 9 处不存在 | 删除 |
| `docs/deployment-checklist.md` | `:145-151` 指导安装 `auto-retrain.timer` | 单元文件不存在且在 `RETIRED_UNITS` 里。15 处引用里 10 处不存在 | 删除 |
| `docs/model_governance.md` | 引用 `alpha/model_loader.py`、`research/model_registry/artifact.py`、`scripts/cli.py` | 10 处引用里 8 处不存在。在用的签名机制是 `infra/model_signing.py` | 删除，签名机制写入新文档 |
| `docs/add_new_symbol.md` | `:28-29` SOL / SUI 模型已训练，Sharpe 2.09 / 2.51 | 仓库里没有这些模型；范围只有 BTC、ETH | 删除 |
| `docs/small_capital_growth_plan.md`、`docs/rust_core_rewrite.md`、`docs/rust_replacement_matrix.md`、`docs/tick_processor_unification.md` | 针对 ML 线和 Rust 重写的计划 | 引用 `risk/staged_risk.py`、`decision/backtest_module.py`、`runner/live_runner.py` 等不存在的文件 | 删除 |
| `models_v8/README.md` | `python3 -m alpha.auto_retrain` | 模块不存在 | 删除 |
| `deploy/README.md` | 引用 `deploy/docker/docker-compose.yml`、`deploy/systemd/quant-trader.service`、`bybit-mm.service` | 12 处引用里 6 处不存在 | 删除或重写 |
| `CHANGELOG.md` | 最后一次提交 2026-03-12，记录 `LiveRunner` 的修复 | `LiveRunner` 已不存在；2026-11 的全部工作没有记录 | 删除，或从 git 历史重建 |

## 2. 按文件的处置总表

| 文件或组 | 文件数 | 引用 / 失效 | 建议 | 理由 |
|---|---|---|---|---|
| `docs/superpowers/**` | 10 | 205 / 110 | 删除 | 2026-03 的实施计划与设计稿，描述的模块半数以上不存在。git 历史里可查 |
| `tasks/**` | 17 | 159 / 84 | 删除 | 2026-03 的计划、审计和清单，同上 |
| `docs/operations.md`、`docs/wiring_truth.md`、`docs/api.md`、`docs/execution_contracts.md`、`docs/runtime_contracts.md`、`docs/runtime_direction.md`、`docs/production_runbook.md`、`docs/deployment-checklist.md`、`docs/model_governance.md`、`docs/add_new_symbol.md`、`docs/daily-1d-rollout.md`、`docs/small_capital_growth_plan.md`、`docs/rust_core_rewrite.md`、`docs/rust_replacement_matrix.md`、`docs/tick_processor_unification.md` | 15 | 见 1.6、1.9 | 删除 | 含与代码相反的操作指引，或描述不存在的系统 |
| `research.md`、`models_v8/README.md`、`deploy/README.md`、`CHANGELOG.md` | 4 | 见 1.9 | 删除 | 同上 |
| `README.md` | 1 | 18 / 1 | 更新（重写） | 入口文档，描述的是已关闭的策略线 |
| `docs/binance_runbook.md`、`docs/btc_eth_strategy.md`、`docs/runtime_truth.md`、`docs/deploy_truth.md` | 4 | 0–1 失效 | 更新 | 引用基本有效，行为陈述有 1.2–1.5 所列错误 |
| `docs/known_test_failures.md` | 1 | 17 / 3 | 保留，改一处 | 与实测一致 |
| `docs/execution_recovery.md`、`docs/local_development.md` | 2 | 0 失效 | 更新 | 见 1.8 |
| `docs/account-reconciliation.md`、`docs/bybit-confirmed-execution.md`、`docs/portfolio-reservations.md`、`docs/runtime-state-recovery.md`、`docs/state-migration.md`、`docs/runtime-release-acceptance.md`、`docs/shadow_runtime.md`、`docs/offline_validation.md`、`docs/p0-ci-deploy-fix.md`、`docs/build-startup-baseline.md`、`docs/engineering-sequence.md` | 11 | 0 失效 | 保留，待核对 | 2026-11 写的 Bybit / shadow / 离线回放专题。引用全部有效，但我没有逐句核对行为。它们对应的代码属于冻结线，去留取决于冻结线的去留 |
| `research/protocols/**` | 4 | 20 / 0 | 保留 | 研究预登记记录，属于研究证据而不是架构文档 |
| `research/results/**`、`research/datasets/btcusdt_2025/README.md` | 13 | 0 失效 | 保留 | 研究结果记录 |
| `decision/README.md`、`infra/config/examples/README.md`、`infra/grafana/README.md` | 3 | 0 失效 | 随所在目录处置 | `infra/config/examples/` 的 yaml 无代码引用，见 `current-state.md` 4.5 |

合计 85 个：建议删除 46 个，更新 7 个，保留 32 个（其中 11 个待核对）。

## 3. `CLAUDE.md` 核对

`CLAUDE.md` 在本流程中只作为工作规则。它含有架构陈述，这里单独核对。它是所有旧文档里与代码最一致的一份；下面是有出入或不完整的地方。

| `CLAUDE.md` 的说法 | 代码实际行为 | 建议 |
|---|---|---|
| 模型产物只通过 `infra/model_signing.py` 加载 | `runner/binance_target.py:386` 在验签之前直接读 spec 文件取品种列表，并据此连接账户、设置杠杆（R11）。`alpha/online_ridge.py:76` 有未验签的 `pickle.load`，目前无调用方 | 更新，或按规则修代码 |
| `research/trials.jsonl` 记录每个在真实数据上评估过的配置；闸门使用整个族的 deflated Sharpe | 只有 `lab/pair_study.py:330,357` 在代码里调用登记和族统计。`lab/overlay_study.py`、`lab/breakout_study.py` 不导入 `lab.trials` | 更新 |
| `lab/forward.py`：2026-11-13 之后的数据只在登记过的 look 读取 | 守卫只约束走 `lab.forward` 的调用；`lab/data.py:load_symbol` 没有日期守卫。登记日期是自报字段 | 更新：写明是约定而非强制 |
| Kill 和回撤平仓在入场单工作时等待 | 成立，但不完整：挂着的入场单不会被撤销，会在 kill 期间继续成交（R2） | 更新 |
| Kill 文件：文件缺失即未激活 | 成立，但没有写命令行默认路径与 `binance_target` 读取路径不同（R5） | 更新 |
| `DrawdownGuard`：日 / 周闩锁，在 UTC 周期边界重置 | 成立，但没有写状态库缺失或 scope 改名时闩锁静默消失（R6） | 更新 |
| 不得有自动提高风险上限或自动重启的路径 | Binance 运行时满足。旧运行时有自动解除暂停的路径（`monitoring/live_ic_killswitch.py:199-203`）和进程内在线更新模型权重（`OnlineRidge`） | 更新：写明例外，或移除这两条路径 |
| 场所仅限 Binance；OKX、Bybit、Polymarket 代码冻结 | 代码里没有对应的硬开关。`runner.alpha_main` 拒绝 Binance 而放行 Bybit 执行（R18） | 更新，或加开关 |

其余关于退出侧缺口、`binance_main` 的两个问题、`OrderLimiter` 重启清零、`daily_reconcile_report` 循环 import 的陈述，均与代码一致。
