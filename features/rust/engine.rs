use pyo3::prelude::*;

pub const N_FEATURES: usize = 141;
const PI: f64 = std::f64::consts::PI;

// Feature index constants
const F_RET_1: usize = 0;
const F_RET_3: usize = 1;
const F_RET_6: usize = 2;
const F_RET_12: usize = 3;
const F_RET_24: usize = 4;
const F_MA_CROSS_10_30: usize = 5;
const F_MA_CROSS_5_20: usize = 6;
const F_CLOSE_VS_MA20: usize = 7;
const F_CLOSE_VS_MA50: usize = 8;
const F_RSI_14: usize = 9;
const F_RSI_6: usize = 10;
const F_MACD_LINE: usize = 11;
const F_MACD_SIGNAL: usize = 12;
const F_MACD_HIST: usize = 13;
const F_BB_WIDTH_20: usize = 14;
const F_BB_PCTB_20: usize = 15;
const F_ATR_NORM_14: usize = 16;
const F_VOL_20: usize = 17;
const F_VOL_5: usize = 18;
const F_VOL_RATIO_20: usize = 19;
const F_VOL_MA_RATIO_5_20: usize = 20;
const F_BODY_RATIO: usize = 21;
const F_UPPER_SHADOW: usize = 22;
const F_LOWER_SHADOW: usize = 23;
const F_MEAN_REVERSION_20: usize = 24;
const F_PRICE_ACCELERATION: usize = 25;
const F_HOUR_SIN: usize = 26;
const F_HOUR_COS: usize = 27;
const F_DOW_SIN: usize = 28;
const F_DOW_COS: usize = 29;
const F_VOL_REGIME: usize = 30;
const F_FUNDING_RATE: usize = 31;
const F_FUNDING_MA8: usize = 32;
const F_TRADE_INTENSITY: usize = 33;
const F_TAKER_BUY_RATIO: usize = 34;
const F_TAKER_BUY_RATIO_MA10: usize = 35;
const F_TAKER_IMBALANCE: usize = 36;
const F_AVG_TRADE_SIZE: usize = 37;
const F_AVG_TRADE_SIZE_RATIO: usize = 38;
const F_VOLUME_PER_TRADE: usize = 39;
const F_TRADE_COUNT_REGIME: usize = 40;
const F_FUNDING_ZSCORE_24: usize = 41;
const F_FUNDING_MOMENTUM: usize = 42;
const F_FUNDING_EXTREME: usize = 43;
const F_FUNDING_CUMULATIVE_8: usize = 44;
const F_FUNDING_SIGN_PERSIST: usize = 45;
const F_OI_CHANGE_PCT: usize = 46;
const F_OI_CHANGE_MA8: usize = 47;
const F_OI_CLOSE_DIVERGENCE: usize = 48;
const F_LS_RATIO: usize = 49;
const F_LS_RATIO_ZSCORE_24: usize = 50;
const F_LS_EXTREME: usize = 51;
const F_CVD_10: usize = 52;
const F_CVD_20: usize = 53;
const F_CVD_PRICE_DIVERGENCE: usize = 54;
const F_AGGRESSIVE_FLOW_ZSCORE: usize = 55;
const F_VOL_OF_VOL: usize = 56;
const F_RANGE_VS_RV: usize = 57;
const F_PARKINSON_VOL: usize = 58;
const F_RV_ACCELERATION: usize = 59;
const F_OI_ACCELERATION: usize = 60;
const F_LEVERAGE_PROXY: usize = 61;
const F_OI_VOL_DIVERGENCE: usize = 62;
const F_OI_LIQUIDATION_FLAG: usize = 63;
const F_FUNDING_ANNUALIZED: usize = 64;
const F_FUNDING_VS_VOL: usize = 65;
const F_BASIS: usize = 66;
const F_BASIS_ZSCORE_24: usize = 67;
const F_BASIS_MOMENTUM: usize = 68;
const F_BASIS_EXTREME: usize = 69;
const F_FGI_NORMALIZED: usize = 70;
const F_FGI_ZSCORE_7: usize = 71;
const F_FGI_EXTREME: usize = 72;
const F_TAKER_BQ_RATIO: usize = 73;
const F_VWAP_DEV_20: usize = 74;
const F_VOLUME_MOMENTUM_10: usize = 75;
const F_MOM_VOL_DIVERGENCE: usize = 76;
const F_BASIS_CARRY_ADJ: usize = 77;
const F_VOL_REGIME_ADAPTIVE: usize = 78;
const F_LIQUIDATION_CASCADE_SCORE: usize = 79;
const F_FUNDING_TERM_SLOPE: usize = 80;
const F_CROSS_TF_REGIME_SYNC: usize = 81;
const F_IMPLIED_VOL_ZSCORE_24: usize = 82;
const F_IV_RV_SPREAD: usize = 83;
const F_PUT_CALL_RATIO: usize = 84;
const F_EXCHANGE_NETFLOW_ZSCORE: usize = 85;
const F_EXCHANGE_SUPPLY_CHANGE: usize = 86;
const F_EXCHANGE_SUPPLY_ZSCORE_30: usize = 87;
const F_ACTIVE_ADDR_ZSCORE_14: usize = 88;
const F_TX_COUNT_ZSCORE_14: usize = 89;
const F_HASHRATE_MOMENTUM: usize = 90;
const F_LIQUIDATION_VOLUME_ZSCORE_24: usize = 91;
const F_LIQUIDATION_IMBALANCE: usize = 92;
const F_LIQUIDATION_VOLUME_RATIO: usize = 93;
const F_LIQUIDATION_CLUSTER_FLAG: usize = 94;
const F_MEMPOOL_FEE_ZSCORE_24: usize = 95;
const F_MEMPOOL_SIZE_ZSCORE_24: usize = 96;
const F_FEE_URGENCY_RATIO: usize = 97;
const F_DXY_CHANGE_5D: usize = 98;
const F_SPX_BTC_CORR_30D: usize = 99;
const F_SPX_OVERNIGHT_RET: usize = 100;
const F_VIX_ZSCORE_14: usize = 101;
const F_SOCIAL_VOLUME_ZSCORE_24: usize = 102;
const F_SOCIAL_SENTIMENT_SCORE: usize = 103;
const F_SOCIAL_VOLUME_PRICE_DIV: usize = 104;

// --- Phase 1: Alias features (zero computation) ---
const F_OC_ADDR_ZSCORE_14: usize = 105;
const F_OC_TX_ZSCORE_14: usize = 106;
const F_OC_NETFLOW_ZSCORE_7: usize = 107;
const F_BTC_DOM_DEV_20: usize = 108;
const F_BTC_DOM_RET_24: usize = 109;

// --- Phase 2: Interaction features ---
const F_RSI_X_ATR: usize = 110;
const F_RSI_X_VOL: usize = 111;
const F_TREND_X_VOL: usize = 112;

// --- Phase 3: IV derived features ---
const F_IV_LEVEL: usize = 113;
const F_IV_RANK_30D: usize = 114;
const F_IV_TERM_SLOPE_DAILY: usize = 115;

// --- Phase 4: On-chain z-score features ---
const F_OC_FLOWIN_ZSCORE_7: usize = 116;
const F_OC_FLOWIN_ZSCORE_14: usize = 117;
const F_OC_FLOWOUT_ZSCORE_14: usize = 118;
const F_OC_TX_ZSCORE_7: usize = 119;

// --- Phase 5: Other features ---
const F_RET_AUTOCORR_24: usize = 120;
const F_OB_IMBALANCE_X_VOL: usize = 121;
const F_TOTAL_ZSCORE_14: usize = 122;
const F_TOTAL_ZSCORE_30: usize = 123;
const F_TOTAL_SUPPLY_CHANGE_7D: usize = 124;

// --- ETF features (computed via push_cross_market) ---
const F_SPY_RET_1D: usize = 125;
const F_SPY_RET_5D: usize = 126;
const F_SPY_RET_10D: usize = 127;
const F_SPY_VIX_CHANGE: usize = 128;
const F_TNX_CHANGE_5D: usize = 129;
const F_GOLD_RET_5D: usize = 130;
const F_IBIT_FLOW_ZSCORE: usize = 131;
const F_ETF_PREMIUM: usize = 132;

// --- Cross-asset (NaN placeholder) ---
const F_DOM_VS_SUI: usize = 133;
const F_DOM_VS_AXS: usize = 134;

// --- 4h feature (NaN placeholder) ---
const F_TF4H_BB_PCTB_20: usize = 135;

// --- USDT dominance (NaN placeholder) ---
const F_USDT_DOMINANCE: usize = 136;

// --- Independent ETF returns (computed via push_cross_market) ---
const F_TLT_RET_5D: usize = 137;
const F_USO_RET_5D: usize = 138;
const F_XLF_RET_5D: usize = 139;
const F_ETHE_RET_1D: usize = 140;

pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "ma_cross_10_30", "ma_cross_5_20", "close_vs_ma20", "close_vs_ma50",
    "rsi_14", "rsi_6",
    "macd_line", "macd_signal", "macd_hist",
    "bb_width_20", "bb_pctb_20",
    "atr_norm_14",
    "vol_20", "vol_5",
    "vol_ratio_20", "vol_ma_ratio_5_20",
    "body_ratio", "upper_shadow", "lower_shadow",
    "mean_reversion_20", "price_acceleration",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "vol_regime",
    "funding_rate", "funding_ma8",
    "trade_intensity", "taker_buy_ratio", "taker_buy_ratio_ma10",
    "taker_imbalance", "avg_trade_size", "avg_trade_size_ratio",
    "volume_per_trade", "trade_count_regime",
    "funding_zscore_24", "funding_momentum", "funding_extreme",
    "funding_cumulative_8", "funding_sign_persist",
    "oi_change_pct", "oi_change_ma8", "oi_close_divergence",
    "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
    "cvd_10", "cvd_20", "cvd_price_divergence", "aggressive_flow_zscore",
    "vol_of_vol", "range_vs_rv", "parkinson_vol", "rv_acceleration",
    "oi_acceleration", "leverage_proxy", "oi_vol_divergence", "oi_liquidation_flag",
    "funding_annualized", "funding_vs_vol",
    "basis", "basis_zscore_24", "basis_momentum", "basis_extreme",
    "fgi_normalized", "fgi_zscore_7", "fgi_extreme",
    "taker_bq_ratio", "vwap_dev_20", "volume_momentum_10",
    "mom_vol_divergence", "basis_carry_adj", "vol_regime_adaptive",
    "liquidation_cascade_score", "funding_term_slope", "cross_tf_regime_sync",
    "implied_vol_zscore_24", "iv_rv_spread", "put_call_ratio",
    "exchange_netflow_zscore", "exchange_supply_change", "exchange_supply_zscore_30",
    "active_addr_zscore_14", "tx_count_zscore_14", "hashrate_momentum",
    "liquidation_volume_zscore_24", "liquidation_imbalance",
    "liquidation_volume_ratio", "liquidation_cluster_flag",
    "mempool_fee_zscore_24", "mempool_size_zscore_24", "fee_urgency_ratio",
    "dxy_change_5d", "spx_btc_corr_30d", "spx_overnight_ret", "vix_zscore_14",
    "social_volume_zscore_24", "social_sentiment_score", "social_volume_price_div",
    // Phase 1: Aliases
    "oc_addr_zscore_14", "oc_tx_zscore_14", "oc_netflow_zscore_7",
    "btc_dom_dev_20", "btc_dom_ret_24",
    // Phase 2: Interactions
    "rsi_x_atr", "rsi_x_vol", "trend_x_vol",
    // Phase 3: IV derived
    "iv_level", "iv_rank_30d", "iv_term_slope_daily",
    // Phase 4: On-chain z-scores
    "oc_flowin_zscore_7", "oc_flowin_zscore_14", "oc_flowout_zscore_14", "oc_tx_zscore_7",
    // Phase 5: Other
    "ret_autocorr_24", "ob_imbalance_x_vol", "total_zscore_14", "total_zscore_30",
    "total_supply_change_7d",
    // ETF (NaN placeholders)
    "spy_ret_1d", "spy_ret_5d", "spy_ret_10d", "spy_vix_change",
    "tnx_change_5d", "gold_ret_5d", "ibit_flow_zscore", "etf_premium",
    // Cross-asset (NaN placeholders)
    "dom_vs_sui", "dom_vs_axs",
    // 4h feature (NaN placeholder)
    "tf4h_bb_pctb_20",
    // USDT dominance (NaN placeholder)
    "usdt_dominance",
    // Independent ETF returns
    "tlt_ret_5d", "uso_ret_5d", "xlf_ret_5d", "ethe_ret_1d",
];

// ── Primitive types: CircBuf, RollingWindow, EMAState, RSIState, ATRState, helpers ──
include!("engine_primitives.inc.rs");

// ============================================================
// RawBar — checkpoint serialization of raw bar data
// ============================================================

pub struct RawBar {
    pub close: f64, pub volume: f64, pub high: f64, pub low: f64, pub open: f64,
    pub hour: i32, pub dow: i32,
    pub funding_rate: f64, pub trades: f64,
    pub taker_buy_volume: f64, pub quote_volume: f64, pub taker_buy_quote_volume: f64,
    pub open_interest: f64, pub ls_ratio: f64, pub spot_close: f64, pub fear_greed: f64,
    pub implied_vol: f64, pub put_call_ratio: f64,
    pub oc_flow_in: f64, pub oc_flow_out: f64,
    pub oc_supply: f64, pub oc_addr: f64, pub oc_tx: f64, pub oc_hashrate: f64,
    pub liq_total_vol: f64, pub liq_buy_vol: f64, pub liq_sell_vol: f64, pub liq_count: f64,
    pub mempool_fastest_fee: f64, pub mempool_economy_fee: f64, pub mempool_size: f64,
    pub macro_dxy: f64, pub macro_spx: f64, pub macro_vix: f64, pub macro_day: i64,
    pub social_volume: f64, pub sentiment_score: f64,
}

const BAR_HISTORY_CAP: usize = 720;

// ============================================================
// BarState
// ============================================================

pub(crate) struct BarState {
    bar_history: std::collections::VecDeque<RawBar>,
    close_history: CircBuf,
    open_history: CircBuf,
    high_history: CircBuf,
    low_history: CircBuf,

    ma_5: RollingWindow,
    ma_10: RollingWindow,
    ma_20: RollingWindow,
    ma_30: RollingWindow,
    ma_50: RollingWindow,
    bb_window: RollingWindow,

    return_window_20: RollingWindow,
    return_window_5: RollingWindow,

    vol_window_20: RollingWindow,
    vol_window_5: RollingWindow,

    rsi_14: RSIState,
    rsi_6: RSIState,

    ema_12: EMAState,
    ema_26: EMAState,
    macd_signal_ema: EMAState,

    atr_14: ATRState,

    funding_ema: EMAState,
    funding_window_24: RollingWindow,
    funding_history_8: CircBuf,
    funding_sign_count: i32,
    funding_last_sign: i32,

    trades_ema_20: EMAState,
    trades_ema_5: EMAState,
    taker_buy_ratio_ema_10: EMAState,
    avg_trade_size_ema_20: EMAState,
    volume_per_trade_ema_20: EMAState,

    oi_change_ema_8: EMAState,
    last_oi: f64,
    last_oi_change_pct: f64,

    ls_ratio_window_24: RollingWindow,
    last_ls_ratio: f64,

    cvd_window_10: RollingWindow,
    cvd_window_20: RollingWindow,
    taker_ratio_window_50: RollingWindow,

    vol_5_history: CircBuf,
    hl_log_sq_window: RollingWindow,

    leverage_proxy_ema: EMAState,
    prev_oi_change_for_accel: f64,

    basis_window_24: RollingWindow,
    basis_ema_8: EMAState,
    last_basis: f64,

    fgi_window_7: RollingWindow,
    last_fgi: f64,

    vwap_cv_window: RollingWindow,
    vwap_v_window: RollingWindow,

    vol_regime_ema: EMAState,
    vol_regime_history: CircBuf,

    iv_window_24: RollingWindow,
    last_implied_vol: f64,
    last_put_call_ratio: f64,

    onchain_netflow_buf: CircBuf,
    onchain_supply_buf: CircBuf,
    onchain_addr_buf: CircBuf,
    onchain_tx_buf: CircBuf,
    onchain_hashrate_ema: EMAState,
    last_onchain_supply: f64,
    last_onchain_hashrate: f64,

    // Phase 4: separate flow-in / flow-out buffers for z-scores
    onchain_flowin_buf: CircBuf,
    onchain_flowout_buf: CircBuf,

    // Phase 3: IV rank (30-bar window)
    iv_window_30: CircBuf,

    // Phase 5: OI raw buffers for total_zscore
    oi_raw_buf_14: CircBuf,
    oi_raw_buf_30: CircBuf,

    // Phase 5: return history for autocorrelation
    return_history_buf: CircBuf,

    liq_volume_buf: CircBuf,
    liq_imbalance_buf: CircBuf,
    last_liq_volume: f64,
    last_liq_imbalance: f64,
    #[allow(dead_code)]
    last_liq_count: f64,

    mempool_fee_buf: CircBuf,
    mempool_size_buf: CircBuf,
    last_fee_urgency: f64,

    dxy_buf: CircBuf,
    spx_buf: CircBuf,
    btc_close_buf_30: CircBuf,
    last_spx_close: f64,
    prev_spx_close: f64,
    last_vix: f64,
    vix_buf: CircBuf,
    last_macro_day: i64,

    social_vol_buf: CircBuf,
    last_sentiment_score: f64,
    last_social_volume: f64,

    prev_momentum: f64,
    last_close: f64,
    last_volume: f64,
    last_hour: i32,
    last_dow: i32,
    last_funding_rate: f64,
    last_trades: f64,
    last_taker_buy_volume: f64,
    last_taker_buy_quote_volume: f64,
    last_quote_volume: f64,
    bar_count: i32,

    // Cross-market daily values (set via push_cross_market)
    cm_spy_close: f64,
    cm_tlt_close: f64,
    cm_uso_close: f64,
    cm_xlf_close: f64,
    cm_ethe_close: f64,
    cm_gbtc_vol: f64,
    cm_treasury_10y: f64,
    cm_usdt_dominance: f64,
    // Rolling buffers for computing returns
    cm_spy_buf: CircBuf,     // 10-bar SPY for ret_1d/5d/10d
    cm_tlt_buf: CircBuf,     // 6-bar TLT for ret_5d
    cm_uso_buf: CircBuf,     // 6-bar USO for ret_5d
    cm_xlf_buf: CircBuf,     // 6-bar XLF for ret_5d (gold proxy)
    cm_ethe_buf: CircBuf,    // 2-bar ETHE for ret_1d
    cm_gbtc_vol_buf: CircBuf, // 14-bar GBTC vol for zscore
    cm_treasury_buf: CircBuf, // 6-bar treasury for chg_5d
}

// ── BarState: new(), push(), get_bar_history() ──
include!("engine_push.inc.rs");

// ── BarState: get_features() ──
include!("engine_compute.inc.rs");

// ── Schedule cursors for batch feature computation ──
include!("engine_cursors.inc.rs");

// ── PyO3 exported batch functions: cpp_compute_all_features, cpp_feature_names ──
include!("engine_batch.inc.rs");

// ── RustFeatureEngine PyO3 class + methods ──
include!("engine_pyclass.inc.rs");

// ── Unit tests ──
include!("engine_tests.inc.rs");
