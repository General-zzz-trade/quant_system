# 已知测试失败：无（2026-11-13 清理后）

`make test`（核心门禁）和 `make test-all`（全量套件）都应全部通过。出现任何失败都按回归处理，不要再往本文件里登记"已知失败"。

## 2026-11-13 清理记录

此前全量套件有 35 个失败和 1 个收集错误，处理方式如下。

**删除（被测代码已不在仓库中）：**
- `tests/integration/test_convergence_parity.py` 中的 `TestEquityLeverageParity`、`TestZScaleParity`、`TestConsensusScaleParity`、`TestAdaptiveStopParity`：导入已删除的 `strategy.gates`。
- `tests/regression/test_known_bug_cases.py` 中 7 个函数：导入已删除的 `alpha.inference.bridge`、`execution.adapters.onchain_poller`、`shared.signal_postprocess`，或读取不存在的 `scripts/train_v7_alpha.py`。
- `tests/unit/scripts/test_train_v7.py` 中的 `TestSplitMetrics`、`TestV7InteractionFeatures`、`TestV7WalkForward`：导入不存在的 `scripts.train_v7_alpha`。
- `tests/unit/test_rust_parity_v2.py::TestRustFillDedupGuard::test_parity_with_python`：对照的 Python 实现 `execution.safety.duplicate_guard` 已删除。该类其余 8 个测试保留。
- `tests/unit/test_scheduler_compat.py`（整个文件）：被测的 `data.scheduler.data_scheduler`、`freshness_monitor` 不存在。注意 `data/scheduler/__init__.py` 仍导入这两个模块，整个包无法导入；仓库内没有任何代码导入它，是否删除待定。

**修正测试（测试过时，不是代码缺陷）：** 状态快照的 `.close`、`.last_price` 是 Fd8 定点整数（价格 × 1e8），浮点价格在 `close_f`、`last_price_f`。下列测试按旧接口把 `.close` 当价格用，改为读 `*_f`：
- `tests/integration/engine_state/test_event_to_state_flow.py`（2 个）
- `tests/integration/test_runtime_contracts.py::TestReplayPathContract::test_replay_preserves_event_semantics`
- `tests/replay/test_replay_vs_live_equivalence.py`
- `tests/integration/test_live_flow_e2e.py` 中的模拟决策模块（原来拿约 4×10¹² 的整数和 40100 比较，所以每根 K 线都下单）
- `tests/replay/test_replay_causal_chain.py` 中的模拟决策模块（原因同上，第 0 根就买入，之后从不卖出）

另外，`tests/integration/test_dominance_rust_parity.py` 原本比较全部特征，包括按设计就不同的 dominance 特征（无 dominance 输入时为 None）。现在只比较非 dominance 特征，这才是测试声明的意图。

**收集错误：** `tests/offline/test_execution_recovery.py` 在 pytest 的 importlib 模式下导入不到同目录的 `test_execution_safety`。现在导入前把本目录加入 `sys.path`，pytest 和 `scripts/check_local_safety.py`（unittest）下都能运行。
