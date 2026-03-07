use pyo3::prelude::*;

mod backtest_engine;
mod bootstrap;
mod cross_sectional;
mod dedup_guard;
mod digest;
mod event_id;
mod factor_math;
mod fast_1m_features;
pub mod fast_rng;
mod feature_engine;
mod feature_selection;
mod feature_selector;
mod fill_dedup;
mod greedy_select;
pub mod indicators;
mod json_parse;
mod linalg_math;
mod ml_decision;
mod monte_carlo;
mod multi_timeframe;
mod portfolio_math;
mod rate_limiter;
mod request_id;
mod rolling_window;
mod route_match;
mod sequence_buffer;
mod signer;
mod target;
mod technical;
pub mod state_types;
mod state_reducers;
mod pipeline;
mod engine_guards;
mod risk_engine;
mod core_types;
mod event_types;
mod event_validators;

#[pymodule]
fn _quant_hotpath(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 1 classes
    m.add_class::<dedup_guard::DuplicateGuard>()?;
    m.add_class::<rate_limiter::RustRateLimitPolicy>()?;
    m.add_class::<ml_decision::RustMLDecision>()?;
    m.add_function(wrap_pyfunction!(json_parse::rust_parse_kline, m)?)?;
    m.add_function(wrap_pyfunction!(json_parse::rust_parse_depth, m)?)?;
    m.add_function(wrap_pyfunction!(json_parse::rust_demux_user_stream, m)?)?;

    // Phase 2 classes + functions
    m.add_function(wrap_pyfunction!(digest::rust_payload_digest, m)?)?;
    m.add_function(wrap_pyfunction!(digest::rust_stable_hash, m)?)?;
    m.add_class::<fill_dedup::RustFillDedupGuard>()?;
    m.add_class::<sequence_buffer::RustSequenceBuffer>()?;
    m.add_function(wrap_pyfunction!(event_id::rust_event_id, m)?)?;
    m.add_function(wrap_pyfunction!(event_id::rust_now_ns, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_sanitize, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_short_hash, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_make_idempotency_key, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_client_order_id, m)?)?;
    m.add_function(wrap_pyfunction!(signer::rust_hmac_sign, m)?)?;
    m.add_function(wrap_pyfunction!(signer::rust_hmac_verify, m)?)?;
    m.add_function(wrap_pyfunction!(route_match::rust_route_event_type, m)?)?;

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
    m.add_class::<engine_guards::RustGuardConfig>()?;
    m.add_class::<engine_guards::RustBasicGuard>()?;

    // Sprint C: risk engine
    m.add_class::<risk_engine::RustKillSwitch>()?;
    m.add_class::<risk_engine::RustCircuitBreaker>()?;
    m.add_class::<risk_engine::RustOrderLimiter>()?;
    m.add_class::<risk_engine::RustRiskGate>()?;

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

    Ok(())
}
