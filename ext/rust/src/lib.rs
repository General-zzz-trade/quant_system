#[cfg(feature = "python")]
use pyo3::prelude::*;

mod backtest_engine;
pub mod incremental_trackers;
mod pnl_tracker;
mod drawdown_breaker;
mod bootstrap;
mod checkpoint_store;
mod cross_sectional;
mod dedup_guard;
mod digest;
mod event_id;
mod event_store;
mod factor_math;
mod fast_1m_features;
pub mod fast_rng;
pub mod feature_engine;
mod feature_selection;
mod feature_selector;
mod fill_dedup;
mod greedy_select;
pub mod indicators;
pub mod json_parse;
mod linalg_math;
mod ml_decision;
mod monte_carlo;
mod multi_timeframe;
mod payload_dedup;
mod portfolio_math;
mod rate_limiter;
mod request_id;
mod rolling_window;
mod route_match;
mod order_state_machine;
mod sequence_buffer;
mod signer;
mod target;
mod technical;
pub mod fixed_decimal;
pub mod state_types;
pub mod state_reducers;
mod pipeline;
mod engine_guards;
pub mod risk_engine;
mod core_types;
mod event_types;
mod event_validation;
mod event_validators;
mod kernel_boundary;
pub mod decision_math;
pub mod decision_policy;
mod cross_asset;
mod decision_signals;
mod execution_store;
mod microstructure;
mod portfolio_allocator;
mod factor_signals;
mod regime_buffer;
pub mod state_store;
pub mod constraint_pipeline;
pub mod inference_bridge;
pub mod tree_predict;
pub mod unified_predictor;
pub mod tick_processor;
mod ws_client;
mod spsc_ring;
pub mod order_submit;
pub mod micro_alpha;
mod attribution;
mod ensemble_calibrate;
pub mod rust_events;
mod regime_detector;
mod adaptive_stop;
mod saga_manager;
pub mod risk_rules;
pub mod risk_aggregator;
mod correlation;
mod gate_chain;
pub mod ridge_predict;
pub mod market_maker;
#[cfg(feature = "python")]
mod py_incremental;
#[cfg(feature = "python")]
pub mod event_header;

#[cfg(feature = "python")]
#[pymodule]
fn _quant_hotpath(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 1 classes
    m.add_class::<dedup_guard::DuplicateGuard>()?;
    m.add_class::<checkpoint_store::RustCheckpointStore>()?;
    m.add_class::<rate_limiter::RustRateLimitPolicy>()?;
    m.add_class::<payload_dedup::RustPayloadDedupGuard>()?;
    m.add_class::<ml_decision::RustMLDecision>()?;
    m.add_class::<execution_store::RustAckStore>()?;
    m.add_class::<execution_store::RustDedupStore>()?;
    m.add_function(wrap_pyfunction!(json_parse::rust_parse_kline, m)?)?;
    m.add_function(wrap_pyfunction!(json_parse::rust_parse_depth, m)?)?;
    m.add_function(wrap_pyfunction!(json_parse::rust_demux_user_stream, m)?)?;
    m.add_function(wrap_pyfunction!(json_parse::rust_parse_agg_trade, m)?)?;

    // Phase 2 classes + functions
    m.add_function(wrap_pyfunction!(digest::rust_payload_digest, m)?)?;
    m.add_function(wrap_pyfunction!(digest::rust_stable_hash, m)?)?;
    m.add_class::<fill_dedup::RustFillDedupGuard>()?;
    m.add_class::<sequence_buffer::RustSequenceBuffer>()?;
    m.add_function(wrap_pyfunction!(event_id::rust_event_id, m)?)?;
    m.add_function(wrap_pyfunction!(event_id::rust_now_ns, m)?)?;
    m.add_class::<event_store::RustInMemoryEventStore>()?;
    m.add_function(wrap_pyfunction!(request_id::rust_sanitize, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_short_hash, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_make_idempotency_key, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_client_order_id, m)?)?;
    m.add_function(wrap_pyfunction!(signer::rust_hmac_sign, m)?)?;
    m.add_function(wrap_pyfunction!(signer::rust_hmac_verify, m)?)?;
    m.add_function(wrap_pyfunction!(route_match::rust_route_event_type, m)?)?;
    m.add_function(wrap_pyfunction!(route_match::rust_route_event, m)?)?;
    m.add_class::<order_state_machine::RustOrderTransition>()?;
    m.add_class::<order_state_machine::RustOrderState>()?;
    m.add_class::<order_state_machine::RustOrderStateMachine>()?;

    // C++ migration — Sprint 1: rolling windows + technical indicators
    m.add_class::<rolling_window::RollingWindow>()?;
    m.add_class::<rolling_window::VWAPWindow>()?;
    m.add_function(wrap_pyfunction!(technical::cpp_sma, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_ema, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_returns, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_macd, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_atr, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_ols, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_vwap, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_order_flow_imbalance, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_rolling_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(technical::cpp_price_impact, m)?)?;

    // C++ migration — Sprint 2: cross-sectional + portfolio + factor + linalg
    m.add_function(wrap_pyfunction!(cross_sectional::cpp_momentum_rank, m)?)?;
    m.add_function(wrap_pyfunction!(cross_sectional::cpp_rolling_beta, m)?)?;
    m.add_function(wrap_pyfunction!(cross_sectional::cpp_relative_strength, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_math::cpp_sample_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_math::cpp_ewma_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_math::cpp_rolling_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_math::cpp_portfolio_variance, m)?)?;
    m.add_function(wrap_pyfunction!(factor_math::cpp_compute_exposures, m)?)?;
    m.add_function(wrap_pyfunction!(factor_math::cpp_factor_model_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(factor_math::cpp_estimate_specific_risk, m)?)?;
    m.add_function(wrap_pyfunction!(linalg_math::cpp_black_litterman_posterior, m)?)?;

    // C++ migration — Sprint 3: feature selection + sampling
    m.add_function(wrap_pyfunction!(feature_selection::cpp_correlation_select, m)?)?;
    m.add_function(wrap_pyfunction!(feature_selection::cpp_mutual_info_select, m)?)?;
    m.add_function(wrap_pyfunction!(greedy_select::cpp_greedy_ic_select, m)?)?;
    m.add_function(wrap_pyfunction!(greedy_select::cpp_greedy_ic_select_np, m)?)?;
    m.add_function(wrap_pyfunction!(feature_selector::cpp_rolling_ic_select, m)?)?;
    m.add_function(wrap_pyfunction!(feature_selector::cpp_spearman_ic_select, m)?)?;
    m.add_function(wrap_pyfunction!(feature_selector::cpp_icir_select, m)?)?;
    m.add_function(wrap_pyfunction!(feature_selector::cpp_stable_icir_select, m)?)?;
    m.add_function(wrap_pyfunction!(feature_selector::cpp_feature_icir_report, m)?)?;
    m.add_class::<bootstrap::BootstrapResult>()?;
    m.add_function(wrap_pyfunction!(bootstrap::cpp_bootstrap_sharpe_ci, m)?)?;
    m.add_class::<monte_carlo::MCResult>()?;
    m.add_function(wrap_pyfunction!(monte_carlo::cpp_simulate_paths, m)?)?;
    m.add_function(wrap_pyfunction!(target::cpp_vol_normalized_target, m)?)?;

    // C++ migration — Sprint 4: multi-timeframe features
    m.add_function(wrap_pyfunction!(multi_timeframe::cpp_compute_4h_features, m)?)?;
    m.add_function(wrap_pyfunction!(multi_timeframe::cpp_4h_feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(fast_1m_features::cpp_compute_fast_1m_features, m)?)?;
    m.add_function(wrap_pyfunction!(fast_1m_features::cpp_fast_1m_feature_names, m)?)?;

    // C++ migration — Sprint 5: feature engine + backtest engine
    m.add_function(wrap_pyfunction!(feature_engine::cpp_compute_all_features, m)?)?;
    m.add_function(wrap_pyfunction!(feature_engine::cpp_feature_names, m)?)?;
    m.add_class::<feature_engine::RustFeatureEngine>()?;
    m.add_function(wrap_pyfunction!(backtest_engine::cpp_run_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_engine::cpp_pred_to_signal, m)?)?;

    // State types
    m.add_class::<state_types::RustMarketState>()?;
    m.add_class::<state_types::RustPositionState>()?;
    m.add_class::<state_types::RustAccountState>()?;
    m.add_class::<state_types::RustPortfolioState>()?;
    m.add_class::<state_types::RustRiskLimits>()?;
    m.add_class::<state_types::RustRiskState>()?;
    m.add_class::<state_types::RustReducerResult>()?;

    // State reducers
    m.add_class::<state_reducers::RustMarketReducer>()?;
    m.add_class::<state_reducers::RustPositionReducer>()?;
    m.add_class::<state_reducers::RustAccountReducer>()?;
    m.add_class::<state_reducers::RustPortfolioReducer>()?;
    m.add_class::<state_reducers::RustRiskReducer>()?;

    // Engine layer — Sprint B: pipeline + guards
    m.add_function(wrap_pyfunction!(pipeline::rust_detect_event_kind, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::rust_normalize_to_facts, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::rust_pipeline_apply, m)?)?;
    m.add_class::<engine_guards::RustGuardConfig>()?;
    m.add_class::<engine_guards::RustBasicGuard>()?;

    // Sprint C: risk engine
    m.add_class::<risk_engine::RustKillSwitch>()?;
    m.add_class::<risk_engine::RustCircuitBreaker>()?;
    m.add_class::<risk_engine::RustOrderLimiter>()?;
    m.add_class::<risk_engine::RustRiskGate>()?;
    m.add_class::<risk_engine::RustRiskEvaluator>()?;
    m.add_class::<risk_engine::RustRiskResult>()?;

    // Sprint C: core types
    m.add_class::<core_types::RustInterceptorChain>()?;
    m.add_class::<core_types::RustSystemClock>()?;
    m.add_class::<core_types::RustSimulatedClock>()?;
    m.add_class::<core_types::RustTradingGate>()?;

    // Sprint D: event types + validators
    m.add_function(wrap_pyfunction!(event_types::rust_event_types, m)?)?;
    m.add_function(wrap_pyfunction!(event_types::rust_sides, m)?)?;
    m.add_function(wrap_pyfunction!(event_types::rust_signal_sides, m)?)?;
    m.add_function(wrap_pyfunction!(event_types::rust_venues, m)?)?;
    m.add_function(wrap_pyfunction!(event_types::rust_order_types, m)?)?;
    m.add_function(wrap_pyfunction!(event_types::rust_time_in_force, m)?)?;
    m.add_class::<event_types::RustSignalResult>()?;
    m.add_class::<event_types::RustDecisionOutput>()?;
    m.add_class::<event_types::RustTargetPosition>()?;
    m.add_class::<event_types::RustOrderSpec>()?;
    m.add_class::<event_validation::RustEventValidator>()?;
    m.add_function(wrap_pyfunction!(event_validators::rust_validate_monotonic_time, m)?)?;
    m.add_function(wrap_pyfunction!(event_validators::rust_validate_required_fields, m)?)?;
    m.add_function(wrap_pyfunction!(event_validators::rust_validate_numeric_range, m)?)?;
    m.add_function(wrap_pyfunction!(event_validators::rust_validate_enum_value, m)?)?;
    m.add_function(wrap_pyfunction!(event_validators::rust_validate_event_batch, m)?)?;
    m.add_function(wrap_pyfunction!(event_validators::rust_validate_side, m)?)?;
    m.add_function(wrap_pyfunction!(event_validators::rust_validate_signal_side, m)?)?;
    m.add_function(wrap_pyfunction!(event_validators::rust_validate_venue, m)?)?;
    m.add_function(wrap_pyfunction!(event_validators::rust_validate_order_type, m)?)?;
    m.add_function(wrap_pyfunction!(event_validators::rust_validate_tif, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_boundary::rust_detect_kernel_event_kind, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_boundary::rust_normalize_kernel_event_to_facts, m)?)?;

    // Rust event types (Phase 2: hot-path events)
    m.add_class::<rust_events::RustMarketEvent>()?;
    m.add_class::<rust_events::RustFillEvent>()?;
    m.add_class::<rust_events::RustFundingEvent>()?;

    // Event header + framework event types (causation tracing)
    m.add_class::<event_header::RustEventHeader>()?;
    m.add_class::<event_types::RustSignalEvent>()?;
    m.add_class::<event_types::RustIntentEvent>()?;
    m.add_class::<event_types::RustOrderEvent>()?;
    m.add_class::<event_types::RustRiskEvent>()?;
    m.add_class::<event_types::RustControlEvent>()?;

    // State store (unified pipeline)
    m.add_class::<state_store::RustStateStore>()?;
    m.add_class::<state_store::RustProcessResult>()?;

    // Decision math
    m.add_function(wrap_pyfunction!(decision_math::rust_fixed_fraction_qty, m)?)?;
    m.add_function(wrap_pyfunction!(decision_math::rust_volatility_adjusted_qty, m)?)?;
    m.add_function(wrap_pyfunction!(decision_math::rust_apply_allocation_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(decision_policy::rust_build_delta_order_fields, m)?)?;
    m.add_function(wrap_pyfunction!(decision_policy::rust_limit_price, m)?)?;
    m.add_function(wrap_pyfunction!(decision_policy::rust_limit_price_f64, m)?)?;
    m.add_function(wrap_pyfunction!(decision_policy::rust_validate_order_constraints, m)?)?;

    // Microstructure
    m.add_function(wrap_pyfunction!(microstructure::rust_extract_orderbook_features, m)?)?;
    m.add_class::<microstructure::RustVPINCalculator>()?;
    m.add_class::<microstructure::RustVPINResult>()?;
    m.add_class::<microstructure::RustStreamingMicrostructure>()?;

    // Market maker engine
    m.add_class::<market_maker::RustMarketMaker>()?;
    m.add_class::<market_maker::RustMMConfig>()?;
    m.add_class::<market_maker::RustMMQuote>()?;

    // Regime buffer
    m.add_class::<regime_buffer::RustRegimeBuffer>()?;

    // Decision signals
    m.add_function(wrap_pyfunction!(decision_signals::rust_compute_rebalance_intents, m)?)?;
    m.add_function(wrap_pyfunction!(decision_signals::rust_compute_feature_signal, m)?)?;
    m.add_function(wrap_pyfunction!(decision_signals::rust_rolling_sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(decision_signals::rust_max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(decision_signals::rust_strategy_weights, m)?)?;

    // Cross-asset computer
    m.add_class::<cross_asset::RustCrossAssetComputer>()?;

    // Portfolio allocator
    m.add_function(wrap_pyfunction!(portfolio_allocator::rust_allocate_portfolio, m)?)?;
    m.add_class::<portfolio_allocator::RustPortfolioAllocator>()?;

    // Inference bridge
    m.add_class::<inference_bridge::RustInferenceBridge>()?;

    // Tree prediction (native ML inference)
    m.add_class::<tree_predict::RustTreePredictor>()?;

    // Ensemble calibration
    m.add_function(wrap_pyfunction!(ensemble_calibrate::rust_adaptive_ensemble_calibrate, m)?)?;

    // Attribution
    m.add_function(wrap_pyfunction!(attribution::rust_compute_pnl, m)?)?;
    m.add_function(wrap_pyfunction!(attribution::rust_compute_cost_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(attribution::rust_attribute_by_signal, m)?)?;
    m.add_function(wrap_pyfunction!(attribution::rust_flush_orderbook_bar, m)?)?;

    // Factor signals
    m.add_function(wrap_pyfunction!(factor_signals::rust_momentum_score, m)?)?;
    m.add_function(wrap_pyfunction!(factor_signals::rust_volatility_score, m)?)?;
    m.add_function(wrap_pyfunction!(factor_signals::rust_liquidity_score, m)?)?;
    m.add_function(wrap_pyfunction!(factor_signals::rust_volume_price_div_score, m)?)?;
    m.add_function(wrap_pyfunction!(factor_signals::rust_adx, m)?)?;
    m.add_function(wrap_pyfunction!(factor_signals::rust_carry_score, m)?)?;

    // Unified predictor (zero-copy feature→predict→signal pipeline)
    m.add_class::<unified_predictor::RustUnifiedPredictor>()?;

    // Tick processor (full hot-path: features + predict + state update + export)
    m.add_class::<tick_processor::RustTickProcessor>()?;
    m.add_class::<tick_processor::RustTickResult>()?;

    // WebSocket client (GIL-free recv)
    m.add_class::<ws_client::RustWsClient>()?;

    // SPSC ring buffer (lock-free event queue)
    m.add_class::<spsc_ring::RustSpscRing>()?;

    // WS order gateway (signed JSON-RPC messages)
    m.add_class::<order_submit::RustWsOrderGateway>()?;

    // PnL tracker
    m.add_class::<pnl_tracker::RustPnLTracker>()?;

    // Drawdown breaker
    m.add_class::<drawdown_breaker::RustDrawdownBreaker>()?;

    // Regime detector (composite vol+trend + param router)
    m.add_class::<regime_detector::RustCompositeRegimeDetector>()?;
    m.add_class::<regime_detector::RustRegimeResult>()?;
    m.add_class::<regime_detector::RustRegimeParams>()?;
    m.add_class::<regime_detector::RustRegimeParamRouter>()?;

    // Adaptive stop gate (ATR 3-phase stop-loss)
    m.add_class::<adaptive_stop::RustAdaptiveStopGate>()?;

    // Saga manager (order lifecycle state machine)
    m.add_class::<saga_manager::RustSagaManager>()?;

    // Risk aggregator (7 risk rules)
    m.add_class::<risk_aggregator::RustRiskAggregator>()?;

    // Correlation computer (portfolio correlation tracking)
    m.add_class::<correlation::RustCorrelationComputer>()?;

    // Gate chain orchestrator (configurable pure-Rust gate pipeline)
    m.add_class::<gate_chain::RustGateChain>()?;
    m.add_class::<gate_chain::RustGateResult>()?;

    // Ridge predictor (linear model for standalone binary)
    m.add_class::<ridge_predict::RustRidgePredictor>()?;

    // Incremental indicator trackers
    m.add_class::<py_incremental::PyEmaTracker>()?;
    m.add_class::<py_incremental::PyRsiTracker>()?;
    m.add_class::<py_incremental::PyAtrTracker>()?;
    m.add_class::<py_incremental::PyAdxTracker>()?;

    Ok(())
}
