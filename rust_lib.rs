// rust_lib.rs — PyO3 module registry (single #[pymodule] function).
// 528 lines — cannot split further (module registration must be one function).

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod event {
    #[cfg(feature = "python")]
    #[path = "rust/header.rs"]
    pub mod header;
    #[path = "rust/id.rs"]
    pub mod id;
    #[cfg(feature = "python")]
    #[path = "rust/decision_types.rs"]
    pub mod decision_types;
    #[cfg(feature = "python")]
    #[path = "rust/event_classes.rs"]
    pub mod event_classes;
    #[cfg(feature = "python")]
    #[path = "rust/types.rs"]
    pub mod types;
    #[path = "rust/data_events.rs"]
    pub mod data_events;
    #[path = "rust/store.rs"]
    pub mod store;
    #[path = "rust/validation.rs"]
    pub mod validation;
    #[path = "rust/validators.rs"]
    pub mod validators;
}

pub mod features {
    #[path = "rust/engine.rs"]
    pub mod engine;
    #[path = "rust/indicators.rs"]
    pub mod indicators;
    #[path = "rust/technical.rs"]
    pub mod technical;
    #[path = "rust/rolling_window.rs"]
    pub mod rolling_window;
    #[path = "rust/incremental_trackers.rs"]
    pub mod incremental_trackers;
    #[cfg(feature = "python")]
    #[path = "rust/py_incremental.rs"]
    pub mod py_incremental;
    #[path = "rust/cross_asset.rs"]
    pub mod cross_asset;
    #[path = "rust/multi_timeframe.rs"]
    pub mod multi_timeframe;
    #[path = "rust/fast_1m.rs"]
    pub mod fast_1m;
    #[path = "rust/fast_rng.rs"]
    pub mod fast_rng;
    #[path = "rust/factor_signals.rs"]
    pub mod factor_signals;
    #[path = "rust/regime_buffer.rs"]
    pub mod regime_buffer;
    #[path = "rust/correlation.rs"]
    pub mod correlation;
    #[path = "rust/microstructure.rs"]
    pub mod microstructure;
    #[path = "rust/cross_sectional.rs"]
    pub mod cross_sectional;
}

pub mod state {
    #[path = "rust/fixed_decimal.rs"]
    pub mod fixed_decimal;
    #[path = "rust/type_helpers.rs"]
    pub mod type_helpers;
    #[path = "rust/market_state.rs"]
    pub mod market_state;
    #[path = "rust/position_state.rs"]
    pub mod position_state;
    #[path = "rust/account_state.rs"]
    pub mod account_state;
    #[path = "rust/portfolio_state.rs"]
    pub mod portfolio_state;
    #[path = "rust/risk_state.rs"]
    pub mod risk_state;
    #[path = "rust/reducer_result.rs"]
    pub mod reducer_result;
    #[path = "rust/types.rs"]
    pub mod types;
    #[path = "rust/reducer_helpers.rs"]
    pub mod reducer_helpers;
    #[path = "rust/market_reducer.rs"]
    pub mod market_reducer;
    #[path = "rust/position_reducer.rs"]
    pub mod position_reducer;
    #[path = "rust/account_reducer.rs"]
    pub mod account_reducer;
    #[path = "rust/portfolio_reducer.rs"]
    pub mod portfolio_reducer;
    #[path = "rust/risk_reducer.rs"]
    pub mod risk_reducer;
    #[path = "rust/reducers.rs"]
    pub mod reducers;
    #[path = "rust/store_compute.rs"]
    pub mod store_compute;
    #[path = "rust/store.rs"]
    pub mod store;
    #[path = "rust/checkpoint.rs"]
    pub mod checkpoint;
}

pub mod decision {
    #[path = "rust/constraint_pipeline.rs"]
    pub mod constraint_pipeline;
    #[path = "rust/inference_bridge.rs"]
    pub mod inference_bridge;
    #[path = "rust/math.rs"]
    pub mod math;
    #[path = "rust/policy.rs"]
    pub mod policy;
    #[path = "rust/signals.rs"]
    pub mod signals;
    #[path = "rust/micro_alpha.rs"]
    pub mod micro_alpha;
    #[path = "rust/ensemble_calibrate.rs"]
    pub mod ensemble_calibrate;
    #[path = "rust/adaptive_sizer.rs"]
    pub mod adaptive_sizer;
    #[path = "rust/exit_manager.rs"]
    pub mod exit_manager;
    #[path = "rust/ml"]
    pub mod ml {
        #[path = "tree_predict.rs"]
        pub mod tree_predict;
        #[path = "ridge_predict.rs"]
        pub mod ridge_predict;
        #[path = "unified_predictor.rs"]
        pub mod unified_predictor;
        #[path = "ml_decision.rs"]
        pub mod ml_decision;
    }
}

pub mod execution {
    #[path = "rust/order_state_machine.rs"]
    pub mod order_state_machine;
    #[path = "rust/order_submit.rs"]
    pub mod order_submit;
    #[path = "rust/payload_dedup.rs"]
    pub mod payload_dedup;
    #[path = "rust/fill_dedup.rs"]
    pub mod fill_dedup;
    #[path = "rust/stores.rs"]
    pub mod stores;
    #[path = "rust/ws_client.rs"]
    pub mod ws_client;
}

pub mod risk {
    #[path = "rust/engine.rs"]
    pub mod engine;
    #[path = "rust/rules.rs"]
    pub mod rules;
    #[path = "rust/aggregator.rs"]
    pub mod aggregator;
    #[path = "rust/drawdown.rs"]
    pub mod drawdown;
    #[path = "rust/adaptive_stop.rs"]
    pub mod adaptive_stop;
    #[path = "rust/gate_chain.rs"]
    pub mod gate_chain;
}

pub mod regime {
    #[path = "rust/detector.rs"]
    pub mod detector;
}

pub mod engine {
    #[path = "rust/pipeline.rs"]
    pub mod pipeline;
    #[path = "rust/guards.rs"]
    pub mod guards;
    #[path = "rust/core_types.rs"]
    pub mod core_types;
    #[path = "rust/kernel.rs"]
    pub mod kernel;
    #[path = "rust/tick_result.rs"]
    pub mod tick_result;
    #[path = "rust/tick_processor.rs"]
    pub mod tick_processor;
}

pub mod research {
    #[path = "rust/backtest.rs"]
    pub mod backtest;
    #[path = "rust/pnl_tracker.rs"]
    pub mod pnl_tracker;
    #[path = "rust/attribution.rs"]
    pub mod attribution;
    #[path = "rust/feature_selection.rs"]
    pub mod feature_selection;
    #[path = "rust/feature_selector.rs"]
    pub mod feature_selector;
    #[path = "rust/greedy_select.rs"]
    pub mod greedy_select;
    #[path = "rust/bootstrap.rs"]
    pub mod bootstrap;
    #[path = "rust/monte_carlo.rs"]
    pub mod monte_carlo;
}

#[path = "common_rust"]
pub mod common {
    #[path = "request_id.rs"]
    pub mod request_id;
    #[path = "signer.rs"]
    pub mod signer;
    #[path = "digest.rs"]
    pub mod digest;
    #[path = "route_match.rs"]
    pub mod route_match;
    #[path = "saga.rs"]
    pub mod saga;
    #[path = "sequence_buffer.rs"]
    pub mod sequence_buffer;
    #[path = "spsc_ring.rs"]
    pub mod spsc_ring;
    #[path = "dedup_guard.rs"]
    pub mod dedup_guard;
    #[path = "json_parse.rs"]
    pub mod json_parse;
    #[path = "rate_limiter.rs"]
    pub mod rate_limiter;
    #[path = "target.rs"]
    pub mod target;
    #[path = "portfolio_allocator.rs"]
    pub mod portfolio_allocator;
    #[path = "portfolio_math.rs"]
    pub mod portfolio_math;
    #[path = "market_maker.rs"]
    pub mod market_maker;
    #[path = "factor_math.rs"]
    pub mod factor_math;
    #[path = "linalg_math.rs"]
    pub mod linalg_math;
}

pub mod alpha {
    #[cfg(feature = "python")]
    #[path = "rust/online_ridge.rs"]
    pub mod online_ridge;
}

#[cfg(feature = "python")]
#[pymodule]
fn _quant_hotpath(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 1 classes
    m.add_class::<common::dedup_guard::DuplicateGuard>()?;
    m.add_class::<state::checkpoint::RustCheckpointStore>()?;
    m.add_class::<common::rate_limiter::RustRateLimitPolicy>()?;
    m.add_class::<execution::payload_dedup::RustPayloadDedupGuard>()?;
    m.add_class::<decision::ml::ml_decision::RustMLDecision>()?;
    m.add_class::<execution::stores::RustAckStore>()?;
    m.add_class::<execution::stores::RustDedupStore>()?;
    m.add_function(wrap_pyfunction!(common::json_parse::rust_parse_kline, m)?)?;
    m.add_function(wrap_pyfunction!(common::json_parse::rust_parse_depth, m)?)?;
    m.add_function(wrap_pyfunction!(common::json_parse::rust_demux_user_stream, m)?)?;
    m.add_function(wrap_pyfunction!(common::json_parse::rust_parse_agg_trade, m)?)?;

    // Phase 2 classes + functions
    m.add_function(wrap_pyfunction!(common::digest::rust_payload_digest, m)?)?;
    m.add_function(wrap_pyfunction!(common::digest::rust_stable_hash, m)?)?;
    m.add_class::<execution::fill_dedup::RustFillDedupGuard>()?;
    m.add_class::<common::sequence_buffer::RustSequenceBuffer>()?;
    m.add_function(wrap_pyfunction!(event::id::rust_event_id, m)?)?;
    m.add_function(wrap_pyfunction!(event::id::rust_now_ns, m)?)?;
    m.add_class::<event::store::RustInMemoryEventStore>()?;
    m.add_function(wrap_pyfunction!(common::request_id::rust_sanitize, m)?)?;
    m.add_function(wrap_pyfunction!(common::request_id::rust_short_hash, m)?)?;
    m.add_function(wrap_pyfunction!(common::request_id::rust_make_idempotency_key, m)?)?;
    m.add_function(wrap_pyfunction!(common::request_id::rust_client_order_id, m)?)?;
    m.add_function(wrap_pyfunction!(common::signer::rust_hmac_sign, m)?)?;
    m.add_function(wrap_pyfunction!(common::signer::rust_hmac_verify, m)?)?;
    m.add_function(wrap_pyfunction!(common::route_match::rust_route_event_type, m)?)?;
    m.add_function(wrap_pyfunction!(common::route_match::rust_route_event, m)?)?;
    m.add_class::<execution::order_state_machine::RustOrderTransition>()?;
    m.add_class::<execution::order_state_machine::RustOrderState>()?;
    m.add_class::<execution::order_state_machine::RustOrderStateMachine>()?;

    // C++ migration — Sprint 1: rolling windows + technical indicators
    m.add_class::<features::rolling_window::RollingWindow>()?;
    m.add_class::<features::rolling_window::VWAPWindow>()?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_sma, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_ema, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_returns, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_macd, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_atr, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_ols, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_vwap, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_order_flow_imbalance, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_rolling_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(features::technical::cpp_price_impact, m)?)?;

    // C++ migration — Sprint 2: cross-sectional + portfolio + factor + linalg
    m.add_function(wrap_pyfunction!(features::cross_sectional::cpp_momentum_rank, m)?)?;
    m.add_function(wrap_pyfunction!(features::cross_sectional::cpp_rolling_beta, m)?)?;
    m.add_function(wrap_pyfunction!(features::cross_sectional::cpp_relative_strength, m)?)?;
    m.add_function(wrap_pyfunction!(common::portfolio_math::cpp_sample_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(common::portfolio_math::cpp_ewma_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(common::portfolio_math::cpp_rolling_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(common::portfolio_math::cpp_portfolio_variance, m)?)?;
    m.add_function(wrap_pyfunction!(common::factor_math::cpp_compute_exposures, m)?)?;
    m.add_function(wrap_pyfunction!(common::factor_math::cpp_factor_model_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(common::factor_math::cpp_estimate_specific_risk, m)?)?;
    m.add_function(wrap_pyfunction!(common::linalg_math::cpp_black_litterman_posterior, m)?)?;

    // C++ migration — Sprint 3: feature selection + sampling
    m.add_function(wrap_pyfunction!(research::feature_selection::cpp_correlation_select, m)?)?;
    m.add_function(wrap_pyfunction!(research::feature_selection::cpp_mutual_info_select, m)?)?;
    m.add_function(wrap_pyfunction!(research::greedy_select::cpp_greedy_ic_select, m)?)?;
    m.add_function(wrap_pyfunction!(research::greedy_select::cpp_greedy_ic_select_np, m)?)?;
    m.add_function(wrap_pyfunction!(research::feature_selector::cpp_rolling_ic_select, m)?)?;
    m.add_function(wrap_pyfunction!(research::feature_selector::cpp_spearman_ic_select, m)?)?;
    m.add_function(wrap_pyfunction!(research::feature_selector::cpp_icir_select, m)?)?;
    m.add_function(wrap_pyfunction!(research::feature_selector::cpp_stable_icir_select, m)?)?;
    m.add_function(wrap_pyfunction!(research::feature_selector::cpp_feature_icir_report, m)?)?;
    m.add_class::<research::bootstrap::BootstrapResult>()?;
    m.add_function(wrap_pyfunction!(research::bootstrap::cpp_bootstrap_sharpe_ci, m)?)?;
    m.add_class::<research::monte_carlo::MCResult>()?;
    m.add_function(wrap_pyfunction!(research::monte_carlo::cpp_simulate_paths, m)?)?;
    m.add_function(wrap_pyfunction!(common::target::cpp_vol_normalized_target, m)?)?;

    // C++ migration — Sprint 4: multi-timeframe features
    m.add_function(wrap_pyfunction!(features::multi_timeframe::cpp_compute_4h_features, m)?)?;
    m.add_function(wrap_pyfunction!(features::multi_timeframe::cpp_4h_feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(features::fast_1m::cpp_compute_fast_1m_features, m)?)?;
    m.add_function(wrap_pyfunction!(features::fast_1m::cpp_fast_1m_feature_names, m)?)?;

    // C++ migration — Sprint 5: feature engine + backtest engine
    m.add_function(wrap_pyfunction!(features::engine::cpp_compute_all_features, m)?)?;
    m.add_function(wrap_pyfunction!(features::engine::cpp_feature_names, m)?)?;
    m.add_class::<features::engine::RustFeatureEngine>()?;
    m.add_function(wrap_pyfunction!(research::backtest::cpp_run_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(research::backtest::cpp_pred_to_signal, m)?)?;

    // State types
    m.add_class::<state::types::RustMarketState>()?;
    m.add_class::<state::types::RustPositionState>()?;
    m.add_class::<state::types::RustAccountState>()?;
    m.add_class::<state::types::RustPortfolioState>()?;
    m.add_class::<state::types::RustRiskLimits>()?;
    m.add_class::<state::types::RustRiskState>()?;
    m.add_class::<state::types::RustReducerResult>()?;

    // State reducers
    m.add_class::<state::reducers::RustMarketReducer>()?;
    m.add_class::<state::reducers::RustPositionReducer>()?;
    m.add_class::<state::reducers::RustAccountReducer>()?;
    m.add_class::<state::reducers::RustPortfolioReducer>()?;
    m.add_class::<state::reducers::RustRiskReducer>()?;

    // Engine layer — Sprint B: pipeline + guards
    m.add_function(wrap_pyfunction!(engine::pipeline::rust_detect_event_kind, m)?)?;
    m.add_function(wrap_pyfunction!(engine::pipeline::rust_normalize_to_facts, m)?)?;
    m.add_function(wrap_pyfunction!(engine::pipeline::rust_pipeline_apply, m)?)?;
    m.add_class::<engine::guards::RustGuardConfig>()?;
    m.add_class::<engine::guards::RustBasicGuard>()?;

    // Sprint C: risk engine
    m.add_class::<risk::engine::RustKillSwitch>()?;
    m.add_class::<risk::engine::RustCircuitBreaker>()?;
    m.add_class::<risk::engine::RustOrderLimiter>()?;
    m.add_class::<risk::engine::RustRiskGate>()?;
    m.add_class::<risk::engine::RustRiskEvaluator>()?;
    m.add_class::<risk::engine::RustRiskResult>()?;

    // Sprint C: core types
    m.add_class::<engine::core_types::RustInterceptorChain>()?;
    m.add_class::<engine::core_types::RustSystemClock>()?;
    m.add_class::<engine::core_types::RustSimulatedClock>()?;
    m.add_class::<engine::core_types::RustTradingGate>()?;

    // Sprint D: event types + validators
    m.add_function(wrap_pyfunction!(event::types::rust_event_types, m)?)?;
    m.add_function(wrap_pyfunction!(event::types::rust_sides, m)?)?;
    m.add_function(wrap_pyfunction!(event::types::rust_signal_sides, m)?)?;
    m.add_function(wrap_pyfunction!(event::types::rust_venues, m)?)?;
    m.add_function(wrap_pyfunction!(event::types::rust_order_types, m)?)?;
    m.add_function(wrap_pyfunction!(event::types::rust_time_in_force, m)?)?;
    m.add_class::<event::types::RustSignalResult>()?;
    m.add_class::<event::types::RustDecisionOutput>()?;
    m.add_class::<event::types::RustTargetPosition>()?;
    m.add_class::<event::types::RustOrderSpec>()?;
    m.add_class::<event::validation::RustEventValidator>()?;
    m.add_function(wrap_pyfunction!(event::validators::rust_validate_monotonic_time, m)?)?;
    m.add_function(wrap_pyfunction!(event::validators::rust_validate_required_fields, m)?)?;
    m.add_function(wrap_pyfunction!(event::validators::rust_validate_numeric_range, m)?)?;
    m.add_function(wrap_pyfunction!(event::validators::rust_validate_enum_value, m)?)?;
    m.add_function(wrap_pyfunction!(event::validators::rust_validate_event_batch, m)?)?;
    m.add_function(wrap_pyfunction!(event::validators::rust_validate_side, m)?)?;
    m.add_function(wrap_pyfunction!(event::validators::rust_validate_signal_side, m)?)?;
    m.add_function(wrap_pyfunction!(event::validators::rust_validate_venue, m)?)?;
    m.add_function(wrap_pyfunction!(event::validators::rust_validate_order_type, m)?)?;
    m.add_function(wrap_pyfunction!(event::validators::rust_validate_tif, m)?)?;
    m.add_function(wrap_pyfunction!(engine::kernel::rust_detect_kernel_event_kind, m)?)?;
    m.add_function(wrap_pyfunction!(engine::kernel::rust_normalize_kernel_event_to_facts, m)?)?;

    // Rust event types (Phase 2: hot-path events)
    m.add_class::<event::data_events::RustMarketEvent>()?;
    m.add_class::<event::data_events::RustFillEvent>()?;
    m.add_class::<event::data_events::RustFundingEvent>()?;

    // Event header + framework event types (causation tracing)
    m.add_class::<event::header::RustEventHeader>()?;
    m.add_class::<event::types::RustSignalEvent>()?;
    m.add_class::<event::types::RustIntentEvent>()?;
    m.add_class::<event::types::RustOrderEvent>()?;
    m.add_class::<event::types::RustRiskEvent>()?;
    m.add_class::<event::types::RustControlEvent>()?;

    // State store (unified pipeline)
    m.add_class::<state::store::RustStateStore>()?;
    m.add_class::<state::store::RustProcessResult>()?;

    // Decision math
    m.add_function(wrap_pyfunction!(decision::math::rust_fixed_fraction_qty, m)?)?;
    m.add_function(wrap_pyfunction!(decision::math::rust_volatility_adjusted_qty, m)?)?;
    m.add_function(wrap_pyfunction!(decision::math::rust_apply_allocation_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(decision::adaptive_sizer::rust_adaptive_target_qty, m)?)?;

    // Exit manager
    m.add_class::<decision::exit_manager::RustExitManager>()?;
    m.add_function(wrap_pyfunction!(decision::policy::rust_build_delta_order_fields, m)?)?;
    m.add_function(wrap_pyfunction!(decision::policy::rust_limit_price, m)?)?;
    m.add_function(wrap_pyfunction!(decision::policy::rust_limit_price_f64, m)?)?;
    m.add_function(wrap_pyfunction!(decision::policy::rust_validate_order_constraints, m)?)?;

    // Microstructure
    m.add_function(wrap_pyfunction!(features::microstructure::rust_extract_orderbook_features, m)?)?;
    m.add_class::<features::microstructure::RustVPINCalculator>()?;
    m.add_class::<features::microstructure::RustVPINResult>()?;
    m.add_class::<features::microstructure::RustStreamingMicrostructure>()?;

    // Market maker engine
    m.add_class::<common::market_maker::RustMarketMaker>()?;
    m.add_class::<common::market_maker::RustMMConfig>()?;
    m.add_class::<common::market_maker::RustMMQuote>()?;

    // Regime buffer
    m.add_class::<features::regime_buffer::RustRegimeBuffer>()?;

    // Decision signals
    m.add_function(wrap_pyfunction!(decision::signals::rust_compute_rebalance_intents, m)?)?;
    m.add_function(wrap_pyfunction!(decision::signals::rust_compute_feature_signal, m)?)?;
    m.add_function(wrap_pyfunction!(decision::signals::rust_rolling_sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(decision::signals::rust_max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(decision::signals::rust_strategy_weights, m)?)?;

    // Cross-asset computer
    m.add_class::<features::cross_asset::RustCrossAssetComputer>()?;

    // Portfolio allocator
    m.add_function(wrap_pyfunction!(common::portfolio_allocator::rust_allocate_portfolio, m)?)?;
    m.add_class::<common::portfolio_allocator::RustPortfolioAllocator>()?;

    // Inference bridge
    m.add_class::<decision::inference_bridge::RustInferenceBridge>()?;

    // Tree prediction (native ML inference)
    m.add_class::<decision::ml::tree_predict::RustTreePredictor>()?;

    // Ensemble calibration
    m.add_function(wrap_pyfunction!(decision::ensemble_calibrate::rust_adaptive_ensemble_calibrate, m)?)?;

    // Attribution
    m.add_function(wrap_pyfunction!(research::attribution::rust_compute_pnl, m)?)?;
    m.add_function(wrap_pyfunction!(research::attribution::rust_compute_cost_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(research::attribution::rust_attribute_by_signal, m)?)?;
    m.add_function(wrap_pyfunction!(research::attribution::rust_flush_orderbook_bar, m)?)?;

    // Factor signals
    m.add_function(wrap_pyfunction!(features::factor_signals::rust_momentum_score, m)?)?;
    m.add_function(wrap_pyfunction!(features::factor_signals::rust_volatility_score, m)?)?;
    m.add_function(wrap_pyfunction!(features::factor_signals::rust_liquidity_score, m)?)?;
    m.add_function(wrap_pyfunction!(features::factor_signals::rust_volume_price_div_score, m)?)?;
    m.add_function(wrap_pyfunction!(features::factor_signals::rust_adx, m)?)?;
    m.add_function(wrap_pyfunction!(features::factor_signals::rust_carry_score, m)?)?;

    // Unified predictor (zero-copy feature→predict→signal pipeline)
    m.add_class::<decision::ml::unified_predictor::RustUnifiedPredictor>()?;

    // Tick processor (full hot-path: features + predict + state update + export)
    m.add_class::<engine::tick_processor::RustTickProcessor>()?;
    m.add_class::<engine::tick_processor::RustTickResult>()?;

    // WebSocket client (GIL-free recv)
    m.add_class::<execution::ws_client::RustWsClient>()?;

    // SPSC ring buffer (lock-free event queue)
    m.add_class::<common::spsc_ring::RustSpscRing>()?;

    // WS order gateway (signed JSON-RPC messages)
    m.add_class::<execution::order_submit::RustWsOrderGateway>()?;

    // PnL tracker
    m.add_class::<research::pnl_tracker::RustPnLTracker>()?;

    // Drawdown breaker
    m.add_class::<risk::drawdown::RustDrawdownBreaker>()?;

    // Regime detector (composite vol+trend + param router)
    m.add_class::<regime::detector::RustCompositeRegimeDetector>()?;
    m.add_class::<regime::detector::RustRegimeResult>()?;
    m.add_class::<regime::detector::RustRegimeParams>()?;
    m.add_class::<regime::detector::RustRegimeParamRouter>()?;

    // Adaptive stop gate (ATR 3-phase stop-loss)
    m.add_class::<risk::adaptive_stop::RustAdaptiveStopGate>()?;

    // Saga manager (order lifecycle state machine)
    m.add_class::<common::saga::RustSagaManager>()?;

    // Risk aggregator (7 risk rules)
    m.add_class::<risk::aggregator::RustRiskAggregator>()?;

    // Correlation computer (portfolio correlation tracking)
    m.add_class::<features::correlation::RustCorrelationComputer>()?;

    // Gate chain orchestrator (configurable pure-Rust gate pipeline)
    m.add_class::<risk::gate_chain::RustGateChain>()?;
    m.add_class::<risk::gate_chain::RustGateResult>()?;

    // Ridge predictor (linear model for standalone binary)
    m.add_class::<decision::ml::ridge_predict::RustRidgePredictor>()?;

    // Incremental indicator trackers
    m.add_class::<features::py_incremental::PyEmaTracker>()?;
    m.add_class::<features::py_incremental::PyRsiTracker>()?;
    m.add_class::<features::py_incremental::PyAtrTracker>()?;
    m.add_class::<features::py_incremental::PyAdxTracker>()?;

    // Online Ridge regression (RLS incremental updates)
    m.add_class::<alpha::online_ridge::RustOnlineRidge>()?;

    Ok(())
}
