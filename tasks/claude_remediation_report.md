# Remediation Report — 2026-03-14

> **Status**: COMPLETED (2026-03-24) — Historical remediation report.
> 后续生产入口已从 `runner/live_runner.py` 迁移至 `runner/alpha_main.py`.
> 当前架构请参考 [`CLAUDE.md`](/quant_system/CLAUDE.md).
>
> Completed by: Claude Code (首席治理工程师)

---

## 1. What Changed

### P1: LiveRunner Complexity Reduction
- Extracted 3 module-level helper functions (~330 LOC) from `runner/live_runner.py` to `runner/builders/`
- `_build_alert_rules()` + `_build_health_server()` → new `runner/builders/monitoring.py`
- `_build_multi_tf_ensemble()` → existing `runner/builders/inference.py`
- live_runner.py reduced from ~2342 LOC to ~2010 LOC
- No public API changes (build/from_config/start/stop behavior identical)

### P2: Config Truth Source Clarification
- Strengthened `infra/config/schema.py` deprecation docstring
- Cannot add runtime `warnings.warn()` because live_runner lazy-imports `validate_trading_config`
- Docstring now explicitly says "do not add new fields or importers"
- Points to `LiveRunnerConfig` factory methods as canonical path

### P3: Scripts Governance
- Confirmed 99/99 scripts covered in `scripts/catalog.py` (100% coverage)
- Added 14-entry quick-reference table to `scripts/README.md`
- Covers training, backtesting, data, ops, monitoring workflows

### P4: Verification
- Full unit test suite: all pass
- Integration + execution + safety + contract tests: all pass (1 xfail, 3 skips expected)
- Rust tests: 82/82 pass

---

## 2. Files Changed

| Commit | Files | Lines |
|--------|-------|-------|
| `d38152a` (P1) | `runner/live_runner.py`, `runner/builders/__init__.py`, `runner/builders/inference.py`, `runner/builders/monitoring.py` (new) | +363/-335 |
| `db6f18c` (P2+P3) | `infra/config/schema.py`, `scripts/README.md` | +32/-5 |
| `4115e43` (P0) | `tasks/claude_remediation_plan.md` (new) | +162 |

---

## 3. Tests Run

| Test Suite | Result | Count |
|-----------|--------|-------|
| tests/unit/runner/ | ✅ PASS | ~290 |
| tests/unit/engine/ | ✅ PASS | ~140 |
| tests/unit/risk/ | ✅ PASS | ~80 |
| tests/unit/execution/ | ✅ PASS | ~270 |
| tests/unit/features/ | ✅ PASS | ~250 |
| tests/unit/decision/ | ✅ PASS | ~350 |
| tests/integration/ | ✅ PASS | ~120 (1 xfail, 3 skip) |
| execution/tests/ | ✅ PASS | 67 |
| tests/execution_safety/ | ✅ PASS | ~20 |
| tests/contract/ | ✅ PASS | ~7 |
| Rust (cargo test) | ✅ PASS | 82 |

---

## 4. Remaining Risks

| # | Risk | Severity | Note |
|---|------|----------|------|
| 1 | `runner/builders/` 5 个 builder 与 live_runner `_build_*` 方法共存 | low | builders/ 用于非默认 runner，`_build_*` 用于生产 |
| 2 | `infra/config/schema.py` 仍在代码库（不能删除因 live_runner 依赖） | low | 强化文档标记为 deprecated |
| 3 | deploy.sh 引用不存在的 compose services | low | 非 CI 路径，需手动修复 |
| 4 | CI 57% 覆盖率门禁偏低 | low | 需团队讨论是否提高 |
| 5 | Slow/parity 测试不在 CI 默认路径 | low | 已文档化，可加 scheduled job |

---

## 5. Suggested Next Phase

| # | 任务 | 优先级 | 理由 |
|---|------|--------|------|
| 1 | **deploy.sh 与 compose 服务名统一** | medium | deploy.sh 引用 paper-btc/sol/eth 但 compose 只有 paper-multi |
| 2 | **Dockerfile COPY _quant_hotpath/ 修复** | medium | 路径可能不存在导致 Docker build 失败 |
| 3 | **CI 加 scheduled slow test job** | low | 覆盖 parity/NN/XGB 测试不在默认 CI |
| 4 | **LiveRunnerConfig from_config() 完整字段映射** | low | 当前仅读 3-8 个 YAML 字段 |
| 5 | **shadow_compare 结果绑定到 promote 前置检查** | low | 当前 promote 只 warn 不 block |
