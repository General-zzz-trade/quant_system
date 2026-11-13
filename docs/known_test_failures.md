# 已知的原有测试失败（2026-11-13 基线 ca44fb1 上同样失败）

`make test`（核心门禁，与 CI 相同）应全部通过。`make test-all` 运行全量套件，下列用例在改动前的基线提交上就已失败，
原因多为测试引用已重构/删除的旧代码路径（旧 AlphaRunner、旧调度器、仓库外脚本）。它们不覆盖币安运行时。
修复或删除这些用例前，不要把它们当作新回归。

- `tests/integration/engine_state/test_event_to_state_flow.py::test_market_events_advance_state`
- `tests/integration/engine_state/test_event_to_state_flow.py::test_multi_symbol_state_isolation`
- `tests/integration/test_convergence_parity.py::TestAdaptiveStopParity::test_gate_exists_and_is_callable`
- `tests/integration/test_convergence_parity.py::TestAdaptiveStopParity::test_new_position_long`
- `tests/integration/test_convergence_parity.py::TestAdaptiveStopParity::test_stop_triggers_on_large_drop`
- `tests/integration/test_convergence_parity.py::TestConsensusScaleParity::test_consensus_scale_matches_alpha_runner`
- `tests/integration/test_convergence_parity.py::TestConsensusScaleParity::test_self_key_excluded`
- `tests/integration/test_convergence_parity.py::TestConsensusScaleParity::test_short_side_contrarian`
- `tests/integration/test_convergence_parity.py::TestEquityLeverageParity::test_bracket_boundary_at_zero`
- `tests/integration/test_convergence_parity.py::TestEquityLeverageParity::test_leverage_brackets_match_alpha_runner`
- `tests/integration/test_convergence_parity.py::TestZScaleParity::test_negative_z_mirrors_positive`
- `tests/integration/test_convergence_parity.py::TestZScaleParity::test_z_scale_thresholds_match`
- `tests/integration/test_dominance_rust_parity.py::TestDominanceRustParity::test_push_dominance_independent_from_push_bar`
- `tests/integration/test_live_flow_e2e.py::test_non_triggering_market_no_orders`
- `tests/integration/test_runtime_contracts.py::TestReplayPathContract::test_replay_preserves_event_semantics`
- `tests/regression/test_known_bug_cases.py::test_bridge_enrich_deterministic`
- `tests/regression/test_known_bug_cases.py::test_bridge_warmup_optimistic`
- `tests/regression/test_known_bug_cases.py::test_checkpoint_restore_preserves_state`
- `tests/regression/test_known_bug_cases.py::test_dd_breaker_sign_convention`
- `tests/regression/test_known_bug_cases.py::test_monthly_gate_hourly_resolution`
- `tests/regression/test_known_bug_cases.py::test_poller_age_seconds`
- `tests/regression/test_known_bug_cases.py::test_pred_to_signal_canonical_import`
- `tests/replay/test_replay_causal_chain.py::test_replay_produces_orders`
- `tests/replay/test_replay_vs_live_equivalence.py::test_replay_vs_live_state_equivalence`
- `tests/unit/scripts/test_train_v7.py::TestSplitMetrics::test_basic_split_metrics`
- `tests/unit/scripts/test_train_v7.py::TestSplitMetrics::test_split_metrics_binary_mode`
- `tests/unit/scripts/test_train_v7.py::TestSplitMetrics::test_split_metrics_small_sample`
- `tests/unit/scripts/test_train_v7.py::TestV7InteractionFeatures::test_interaction_feature_list`
- `tests/unit/scripts/test_train_v7.py::TestV7InteractionFeatures::test_v6_interactions_preserved`
- `tests/unit/scripts/test_train_v7.py::TestV7WalkForward::test_expanding_folds`
- `tests/unit/scripts/test_train_v7.py::TestV7WalkForward::test_small_data_rejected`
- `tests/unit/test_rust_parity_v2.py::TestRustFillDedupGuard::test_parity_with_python`
- `tests/unit/test_scheduler_compat.py::test_data_scheduler_builds_expected_job_set`
- `tests/unit/test_scheduler_compat.py::test_data_scheduler_start_stop_is_idempotent`
- `tests/unit/test_scheduler_compat.py::test_freshness_monitor_start_stop_is_idempotent`

另有 `tests/offline/test_execution_recovery.py` 在 pytest 的 importlib 模式下无法导入同目录模块；它由 `python scripts/check_local_safety.py`（unittest）运行并通过。
