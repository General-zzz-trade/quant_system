# ADR-0008：遗留代码与 Rust 扩展的处置

- 状态：提议，待批准
- 日期：2026-11-14
- 子决策 2–5 涉及删除代码或下线服务，需要您确认。

## 背景

【实际运行结果】（`docs/arch/current-state.md` 第 1、4 节）

- 456 个非测试模块里，109 个（29,533 行）从启动面和测试都到不了；77 个（14,199 行）只有测试到得了。
- Rust 扩展导出 191 个名字，46 个在 Python 中无处引用。
- 候选运行时对 Rust 的依赖只有两处：事件类型（`lab/orders.py` → `event/header.py:10`）和下单限额器（`execution/safety/limits.py` → `RustOrderLimiter`）。
- 没有任何启动面使用的代码里，有与在用实现语义相反的同名类，也有默认真实下单的脚本。

## 子决策 1：新包是否依赖 Rust 扩展

- **1a 保持依赖**：继续用 Rust 的事件头和限额器。
- **1b 新包不依赖 Rust（建议）**：`trading/` 里的订单意图是纯 Python 数据类；限额规则用纯 Python 重写（约 80 行），时钟可注入，计数可持久化（ADR-0007）。

取舍：1b 让新包的安装和测试不需要编译 Rust，限额器可以用假时钟测试（现在没有任何直接测试）。代价是限额规则多一份实现，直到遗留的 Rust 限额器随冻结线一起删除。重写时以现有 Rust 实现的行为为准，先写对拍测试。

Rust 扩展本身保留，ML 线和旧运行时继续用它。

## 子决策 2：完全不可达的代码

范围：`research/` 的 48 个文件、`scripts/` 的 22 个、`monitoring/` 的 10 个、`data/downloads/` 的 13 个等，共 109 个模块；以及被在用实现取代的 `execution/bridge`、`execution/store`、`execution/models` 的 9 个文件、`execution/adapters/common`、`execution/sim`、Python `risk/` 包、三套旧回测。

- **2a 留着**。
- **2b 分批删除（建议）**：每批删除前确认没有启动面和在用测试引用，删除后全量测试必须全绿。只覆盖被删代码的测试一并删除。
- **2c 移到 `legacy/` 目录**：保留文件但移出主路径。

取舍：2c 看起来保守，但移动后 import 路径全变，等于不可运行的存档，价值不如 git 历史。2b 的风险是误删仍被主机上手工调用的脚本，这一点代码判断不了。

**纯删除的改动会超过 400 行。** 建议把"每个改动少于 400 行"理解为手写的新增与修改行数，纯删除和纯移动单独计数、单独验收。请确认是否接受这个口径；如果不接受，删除工作要拆成很多步，不在 15 步之内。

## 子决策 3：ML 线与冻结交易所（可达、有启动单元）

范围：`runner/alpha_main.py` 及其 builders、`runner/binance_main.py`、`engine`、`decision`、`features`、`alpha`、Bybit / OKX / Polymarket 适配器，以及 `bybit-alpha`、`okx-alpha`、`polymarket-maker`、`polymarket-collector`、`daily-live-runner`、`okx-ramp-up`、`okx-status-probe`、`audit-reconcile`、`cross-venue-reconcile` 这些单元。

- **3a 原地冻结，并加一个"非 Binance 交易所禁止执行"的硬开关（建议，作为过渡）**：现在代码里没有这个开关，`runner.alpha_main` 反而只放行 Bybit 执行且主网端点在白名单里（`runner/portfolio_risk.py:22-26,62-63`）。加开关属于安全修改。
- **3b 整体删除，连同单元文件**：范围最干净，但要确认这些服务在主机上已经不需要。
- **3c 拆到单独的分支或仓库**。

**需要您回答**：主机上这些单元现在是否启用、是否还需要。代码判断不了：`infra/sync_systemd.sh` 只复制单元文件，不启停。其中 `polymarket-maker` 一启动就 ImportError 并无限重启；`daily-live-runner` 读取的模型目录不在仓库里。

`runner/binance_main.py` 的去留单独说明：没有启动面启动它，但 `binance_target` 依赖它的 6 个符号，`runner/offline` 的 CI 步骤也用到 ML 回放。迁移计划第 9 步切断 `binance_target` 对它的依赖，之后它与 ML 线同进退。

## 子决策 4：`rust_bin/`（独立交易二进制）

- **4a 保留**。
- **4b 删除（建议）**：它是第二条 Binance 下单通道（经 WS-API，`execution/rust/order_submit.rs`），不经过 Python 侧的订单库和任何风控门；没有启动面引用。删除它不会减少 `Cargo.toml` 的依赖：`tokio`、`reqwest`、`clap` 在 Rust 库的其他文件里也有使用。

## 子决策 5：默认真实下单的研究脚本

范围：`research/strategies/run_hft_signal.py`、`run_liquidation_sniper.py`、`run_binary_signal.py`、`run_bybit_mm.py`。它们经适配器的私有 `_client` 直接下单和设杠杆，默认杠杆 20 至 50 倍，不传 `--dry-run` 即真实下单。对应的 systemd 单元已在 `RETIRED_UNITS` 里。

- **5a 留着**。
- **5b 删除（建议）**。它们属于子决策 2 的范围，这里单列是因为风险性质不同，建议不等其他批次，优先删除。

## 旧文档

`docs/arch/doc-vs-code.md` 建议删除 46 个、更新 7 个文档。文档处置不是代码改动，可以与任何一步并行，单独提交。

## 后果

- 选建议项时，仓库的非测试 Python 代码在第一批删除后减少约三分之一，冻结线仍在。
- `make test` 的门禁内容会变：`tests/unit/bybit`、`tests/unit/execution/adapters/okx` 现在在核心门禁里，只要冻结线还在就继续保留。
- Rust 扩展的导出清理（46 个无引用导出）跟随冻结线的去留，不单独做。

## 需要您确认

1. 子决策 1：新包不依赖 Rust。
2. 子决策 2：分批删除，以及"纯删除和纯移动不计入 400 行"的口径。
3. 子决策 3：冻结线的去留，以及主机上相关单元的实际状态。
4. 子决策 4、5：删除 `rust_bin/` 和 4 个直接下单的研究脚本。
