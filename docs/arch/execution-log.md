# 迁移执行记录（阶段 4）

- 分支：`arch-migration`（从 `arch-diagnosis` 拉出）。每一步一个提交，未推送。
- 批准（2026-11-14）：第一批、第二批全部按建议批准；交易账户专用于本运行时；连续执行全部步骤。
- 每一步的"通用验收"是：全量 pytest、`ruff check --select E,W,F .`、`scripts/check_local_safety.py`，第 2 步之后加 `lint-imports`。下面贴的是原始输出的尾部。
- 基线（HEAD `e3cd2d6`）：`2920 passed, 34 skipped, 24 deselected`；ruff 通过；本地安全检查 `Ran 347 tests ... OK (skipped=1)`。

## 阶段 1 补查（只读）

在独立 worktree 里跑了 3 个只读审查，结果并入 `current-state.md`：

- 资金缺陷独立复核：见 `current-state.md` 5.4。12 条里 10 条成立，2 条部分成立，0 条被推翻；新增 X1、X2。

## 第 1 步：特征化测试（退出路径与订单生命周期）

**改了什么**

- 新增 `tests/characterization/`（4 个文件，372 行，全部是测试）：16 个测试。其中 12 个带 `known_defect` 标记，把 R1、R2、R3、R5、R6、R8、R9、R11、R12（两个）、R13、R20 的现状固定下来；另外 4 个固定应当保持的正常行为（kill 平仓在交易所接受时能平掉、回撤闩锁同文件同 scope 重启后仍在、kill 文件语义、带订单号的止损成交能入账）。
- `pytest.ini` 注册 `known_defect` 标记（1 行）。
- 没有改任何生产代码，没有改任何已有测试。

**通用验收（原始输出尾部）**

```
2936 passed, 34 skipped, 24 deselected, 9 warnings, 312 subtests passed in 790.26s (0:13:10)
exit=0
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
```

2936 = 基线 2920 + 新增 16。耗时比基线长是因为同时有 3 个只读审查在跑。

**没改什么 / 遗留**

- 原计划第 1 步里的"状态转移表特征化"挪到第 5 步一起做：它只服务第 5 步，放在一起更容易核对。
- 带 `known_defect` 的测试断言的是缺陷本身。它们只会在第 1b、10、11、12、13 步按已批准的修复逐条改断言，每次改动都在对应步骤里列出。

## 第 1b 步：纸面运行不因行情失败退出；纸面状态落盘（ADR-0007 子决策 3、4，已批准提前）

**改了什么**（`runner/binance_target.py`，+125 / -12 行；新增 `tests/unit/lab/test_target_runtime_resilience.py`，7 个测试）

- 行情请求（K 线、资金费）失败时不再让进程退出：本轮跳过行情步骤，写一行 `issue` 事件，按 5、10、20 … 秒退避（上限 300 秒）重试；同一周期里对账、止损维护、kill / 回撤平仓照常执行。两种模式都生效。只捕获 `BinanceRestError` 和 `OSError`，其他异常仍然退出进程。
- 纸面模式：每个周期处理完 K 线后，把两个品种的模拟账户（持仓、现金流、挂单、成交记录）原子写入 `<state_dir>/paper_state.json`（写临时文件、fsync、rename）。重启预热时恢复它，并把停机期间错过的 K 线按研究撮合逐根补算。
- 保存的状态与当前 spec、品种或撮合配置不一致，或者停机太久、行情接口已经取不到恢复点那根 K 线时，拒绝启动并提示；新增 `--reset-paper-state`，把旧记录改名归档（不删除）后重新开始。
- 两处因补算而必须的因果性修正：
  - 目标计算只使用"结算时间 ≤ 当前 K 线可见时间"的资金费（以前用字典里最新的若干笔）。正常实时运行时两者相同；只在一次追赶多根 K 线时不同。当前候选 spec 的 `funding_filter` 为 False，资金费不进入目标，对它没有影响。
  - 追赶多根 K 线时，资金费请求数量随缺口增大（以前固定 3 笔，缺口超过一天会漏掉结算）。

**新增测试的证据**

- 纸面运行 150 根 K 线后停机 7 根再重启：两个品种的全部成交与研究模拟 `lab.simulate` 逐笔相同、权益误差 < 1e-9；市价入场与挂单 4 小时两种配置都测了。
- 运行中连续 4 根 K 线行情失败：进程不退出，恢复后补处理，最终仍与研究逐笔相同；事件日志里有 4 行 `issue`，失败计数 1→4。
- 执行模式下行情失败时，kill(平仓) 仍然把仓位平掉（独立复核的 X1）。
- 退避期内不发请求；spec 不一致时拒绝启动，`--reset-paper-state` 归档旧文件；恢复点早于可取得的历史时拒绝启动。

**通用验收（原始输出尾部）**

```
2943 passed, 34 skipped, 24 deselected, 9 warnings, 312 subtests passed in 570.16s (0:09:30)
exit=0
$ ruff check --select E,W,F .
All checks passed!
$ python3 scripts/check_local_safety.py
OK (skipped=1)
```

2943 = 2936 + 新增 7。已有测试的断言一行未改；`tests/unit/lab/test_target_runtime.py` 的纸面等于研究测试照常通过。

**没改什么 / 遗留**

- 预热阶段的行情失败仍然让进程退出（启动失败由 systemd 重启）。
- 连续失败没有主动告警，只有事件日志和状态文件里的 `market_failures` 计数。告警出口在第 12 步加。
- 部署影响：主机上第一次用新代码启动时没有 `paper_state.json`，从零开始，与现在相同；之后的重启都会续上。
