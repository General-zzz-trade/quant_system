use pyo3::prelude::*;

pub const N_FEATURES: usize = 105;
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
];

// ============================================================
// CircBuf — runtime-sized circular buffer
// ============================================================

pub(crate) struct CircBuf {
    pub(crate) buf: Vec<f64>,
    pub(crate) max_size: usize,
    pub(crate) count: usize,
    pub(crate) head: usize,
}

impl CircBuf {
    fn new(max_size: usize) -> Self {
        CircBuf {
            buf: vec![0.0; max_size],
            max_size,
            count: 0,
            head: 0,
        }
    }

    fn push(&mut self, x: f64) {
        if self.count < self.max_size {
            self.buf[self.count] = x;
            self.count += 1;
        } else {
            self.buf[self.head] = x;
            self.head = (self.head + 1) % self.max_size;
        }
    }

    fn size(&self) -> usize {
        self.count
    }

    fn full(&self) -> bool {
        self.count == self.max_size
    }

    fn get(&self, i: usize) -> f64 {
        self.buf[(self.head + i) % self.max_size]
    }

    fn back(&self) -> f64 {
        if self.count < self.max_size {
            self.buf[self.count - 1]
        } else {
            self.buf[(self.head + self.max_size - 1) % self.max_size]
        }
    }

    fn back_n(&self, n: usize) -> f64 {
        if self.count < self.max_size {
            self.buf[self.count - 1 - n]
        } else {
            self.buf[(self.head + self.max_size - 1 - n) % self.max_size]
        }
    }

    fn sum(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.count {
            s += self.buf[(self.head + i) % self.max_size];
        }
        s
    }
}

// ============================================================
// RollingWindow — online mean/std via sum/sumsq
// ============================================================

pub(crate) struct RollingWindow {
    pub(crate) buf: Vec<f64>,
    pub(crate) size: usize,
    pub(crate) head: usize,
    pub(crate) count: usize,
    pub(crate) sum: f64,
    pub(crate) sumsq: f64,
}

impl RollingWindow {
    fn new(size: usize) -> Self {
        RollingWindow {
            buf: vec![0.0; size],
            size,
            head: 0,
            count: 0,
            sum: 0.0,
            sumsq: 0.0,
        }
    }

    fn push(&mut self, x: f64) {
        if self.count < self.size {
            self.buf[self.count] = x;
            self.count += 1;
        } else {
            let old = self.buf[self.head];
            self.sum -= old;
            self.sumsq -= old * old;
            self.buf[self.head] = x;
            self.head = (self.head + 1) % self.size;
        }
        self.sum += x;
        self.sumsq += x * x;
    }

    fn full(&self) -> bool {
        self.count == self.size
    }

    fn n(&self) -> usize {
        self.count
    }

    fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum / self.count as f64)
        }
    }

    fn std_dev(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let mu = self.sum / self.count as f64;
        let v = self.sumsq / self.count as f64 - mu * mu;
        Some(v.max(0.0).sqrt())
    }
}

// ============================================================
// EMAState
// ============================================================

pub(crate) struct EMAState {
    pub(crate) alpha: f64,
    pub(crate) value: f64,
    pub(crate) n: i32,
}

impl EMAState {
    fn new(span: i32) -> Self {
        EMAState {
            alpha: 2.0 / (span as f64 + 1.0),
            value: 0.0,
            n: 0,
        }
    }

    fn push(&mut self, x: f64) {
        if self.n == 0 {
            self.value = x;
        } else {
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value;
        }
        self.n += 1;
    }

    fn ready(&self, span: i32) -> bool {
        self.n >= span
    }

    fn get_value(&self) -> f64 {
        if self.n > 0 {
            self.value
        } else {
            f64::NAN
        }
    }
}

// ============================================================
// RSIState
// ============================================================

pub(crate) struct RSIState {
    pub(crate) period: i32,
    pub(crate) avg_gain: f64,
    pub(crate) avg_loss: f64,
    pub(crate) n: i32,
    pub(crate) prev_close: f64,
    pub(crate) init_gains: f64,
    pub(crate) init_losses: f64,
}

impl RSIState {
    fn new(period: i32) -> Self {
        RSIState {
            period,
            avg_gain: 0.0,
            avg_loss: 0.0,
            n: 0,
            prev_close: f64::NAN,
            init_gains: 0.0,
            init_losses: 0.0,
        }
    }

    fn push(&mut self, close: f64) {
        if self.prev_close.is_nan() {
            self.prev_close = close;
            return;
        }
        let change = close - self.prev_close;
        self.prev_close = close;
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };
        self.n += 1;

        if self.n <= self.period {
            self.init_gains += gain;
            self.init_losses += loss;
            if self.n == self.period {
                self.avg_gain = self.init_gains / self.period as f64;
                self.avg_loss = self.init_losses / self.period as f64;
            }
        } else {
            let p = self.period as f64;
            self.avg_gain = (self.avg_gain * (p - 1.0) + gain) / p;
            self.avg_loss = (self.avg_loss * (p - 1.0) + loss) / p;
        }
    }

    fn get_value(&self) -> f64 {
        if self.n < self.period {
            return f64::NAN;
        }
        if self.avg_loss == 0.0 {
            return 100.0;
        }
        let rs = self.avg_gain / self.avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

// ============================================================
// ATRState
// ============================================================

pub(crate) struct ATRState {
    pub(crate) period: i32,
    pub(crate) atr: f64,
    pub(crate) n: i32,
    pub(crate) prev_close: f64,
    pub(crate) init_sum: f64,
}

impl ATRState {
    fn new(period: i32) -> Self {
        ATRState {
            period,
            atr: 0.0,
            n: 0,
            prev_close: f64::NAN,
            init_sum: 0.0,
        }
    }

    fn push(&mut self, high: f64, low: f64, close: f64) {
        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            let a = high - low;
            let b = (high - self.prev_close).abs();
            let c = (low - self.prev_close).abs();
            a.max(b).max(c)
        };
        self.prev_close = close;
        self.n += 1;

        if self.n <= self.period {
            self.init_sum += tr;
            if self.n == self.period {
                self.atr = self.init_sum / self.period as f64;
            }
        } else {
            let p = self.period as f64;
            self.atr = (self.atr * (p - 1.0) + tr) / p;
        }
    }

    fn get_value(&self) -> f64 {
        if self.n >= self.period {
            self.atr
        } else {
            f64::NAN
        }
    }
}

// ============================================================
// zscore_buf helper
// ============================================================

fn zscore_buf(data: &[f64], min_std: f64) -> f64 {
    let count = data.len();
    if count == 0 {
        return f64::NAN;
    }
    let mut sum = 0.0;
    for &v in data {
        sum += v;
    }
    let mean = sum / count as f64;
    let mut var = 0.0;
    for &v in data {
        let d = v - mean;
        var += d * d;
    }
    var /= count as f64;
    let std = var.sqrt();
    if std <= min_std {
        return 0.0;
    }
    (data[count - 1] - mean) / std
}

fn sign_f64(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

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
}

impl BarState {
    pub(crate) fn new() -> Self {
        BarState {
            bar_history: std::collections::VecDeque::with_capacity(BAR_HISTORY_CAP),
            close_history: CircBuf::new(30),
            open_history: CircBuf::new(2),
            high_history: CircBuf::new(2),
            low_history: CircBuf::new(2),

            ma_5: RollingWindow::new(5),
            ma_10: RollingWindow::new(10),
            ma_20: RollingWindow::new(20),
            ma_30: RollingWindow::new(30),
            ma_50: RollingWindow::new(50),
            bb_window: RollingWindow::new(20),

            return_window_20: RollingWindow::new(20),
            return_window_5: RollingWindow::new(5),

            vol_window_20: RollingWindow::new(20),
            vol_window_5: RollingWindow::new(5),

            rsi_14: RSIState::new(14),
            rsi_6: RSIState::new(6),

            ema_12: EMAState::new(12),
            ema_26: EMAState::new(26),
            macd_signal_ema: EMAState::new(9),

            atr_14: ATRState::new(14),

            funding_ema: EMAState::new(8),
            funding_window_24: RollingWindow::new(24),
            funding_history_8: CircBuf::new(8),
            funding_sign_count: 0,
            funding_last_sign: 0,

            trades_ema_20: EMAState::new(20),
            trades_ema_5: EMAState::new(5),
            taker_buy_ratio_ema_10: EMAState::new(10),
            avg_trade_size_ema_20: EMAState::new(20),
            volume_per_trade_ema_20: EMAState::new(20),

            oi_change_ema_8: EMAState::new(8),
            last_oi: f64::NAN,
            last_oi_change_pct: f64::NAN,

            ls_ratio_window_24: RollingWindow::new(24),
            last_ls_ratio: f64::NAN,

            cvd_window_10: RollingWindow::new(10),
            cvd_window_20: RollingWindow::new(20),
            taker_ratio_window_50: RollingWindow::new(50),

            vol_5_history: CircBuf::new(25),
            hl_log_sq_window: RollingWindow::new(20),

            leverage_proxy_ema: EMAState::new(20),
            prev_oi_change_for_accel: f64::NAN,

            basis_window_24: RollingWindow::new(24),
            basis_ema_8: EMAState::new(8),
            last_basis: f64::NAN,

            fgi_window_7: RollingWindow::new(7),
            last_fgi: f64::NAN,

            vwap_cv_window: RollingWindow::new(20),
            vwap_v_window: RollingWindow::new(20),

            vol_regime_ema: EMAState::new(5),
            vol_regime_history: CircBuf::new(30),

            iv_window_24: RollingWindow::new(24),
            last_implied_vol: f64::NAN,
            last_put_call_ratio: f64::NAN,

            onchain_netflow_buf: CircBuf::new(7),
            onchain_supply_buf: CircBuf::new(30),
            onchain_addr_buf: CircBuf::new(14),
            onchain_tx_buf: CircBuf::new(14),
            onchain_hashrate_ema: EMAState::new(14),
            last_onchain_supply: f64::NAN,
            last_onchain_hashrate: f64::NAN,

            liq_volume_buf: CircBuf::new(24),
            liq_imbalance_buf: CircBuf::new(6),
            last_liq_volume: f64::NAN,
            last_liq_imbalance: f64::NAN,
            last_liq_count: 0.0,

            mempool_fee_buf: CircBuf::new(24),
            mempool_size_buf: CircBuf::new(24),
            last_fee_urgency: f64::NAN,

            dxy_buf: CircBuf::new(10),
            spx_buf: CircBuf::new(30),
            btc_close_buf_30: CircBuf::new(30),
            last_spx_close: f64::NAN,
            prev_spx_close: f64::NAN,
            last_vix: f64::NAN,
            vix_buf: CircBuf::new(14),
            last_macro_day: -1,

            social_vol_buf: CircBuf::new(24),
            last_sentiment_score: f64::NAN,
            last_social_volume: f64::NAN,

            prev_momentum: f64::NAN,
            last_close: f64::NAN,
            last_volume: 0.0,
            last_hour: -1,
            last_dow: -1,
            last_funding_rate: f64::NAN,
            last_trades: 0.0,
            last_taker_buy_volume: 0.0,
            last_taker_buy_quote_volume: 0.0,
            last_quote_volume: 0.0,
            bar_count: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push(
        &mut self,
        close: f64, volume: f64, high: f64, low: f64, open_: f64,
        hour: i32, dow: i32,
        funding_rate: f64,
        trades: f64,
        taker_buy_volume: f64,
        quote_volume: f64,
        taker_buy_quote_volume: f64,
        open_interest: f64,
        ls_ratio: f64,
        spot_close: f64,
        fear_greed: f64,
        implied_vol: f64,
        put_call_ratio_val: f64,
        oc_flow_in: f64, oc_flow_out: f64,
        oc_supply: f64, oc_addr: f64,
        oc_tx: f64, oc_hashrate: f64,
        liq_total_vol: f64, liq_buy_vol: f64, liq_sell_vol: f64,
        liq_count: f64,
        mempool_fastest_fee: f64, mempool_economy_fee: f64,
        mempool_size: f64,
        macro_dxy: f64, macro_spx: f64, macro_vix: f64,
        macro_day: i64,
        social_volume: f64, sentiment_score: f64,
    ) {
        // Record raw bar for checkpoint persistence
        if self.bar_history.len() >= BAR_HISTORY_CAP {
            self.bar_history.pop_front();
        }
        self.bar_history.push_back(RawBar {
            close, volume, high, low, open: open_,
            hour, dow, funding_rate, trades,
            taker_buy_volume, quote_volume, taker_buy_quote_volume,
            open_interest, ls_ratio, spot_close, fear_greed,
            implied_vol, put_call_ratio: put_call_ratio_val,
            oc_flow_in, oc_flow_out, oc_supply, oc_addr, oc_tx, oc_hashrate,
            liq_total_vol, liq_buy_vol, liq_sell_vol, liq_count,
            mempool_fastest_fee, mempool_economy_fee, mempool_size,
            macro_dxy, macro_spx, macro_vix, macro_day,
            social_volume, sentiment_score,
        });

        self.last_hour = hour;
        self.last_dow = dow;

        // --- Funding ---
        if !funding_rate.is_nan() {
            self.last_funding_rate = funding_rate;
            self.funding_ema.push(funding_rate);
            self.funding_window_24.push(funding_rate);
            self.funding_history_8.push(funding_rate);
            let sign = if funding_rate > 0.0 { 1 } else if funding_rate < 0.0 { -1 } else { 0 };
            if sign != 0 {
                if sign == self.funding_last_sign {
                    self.funding_sign_count += 1;
                } else {
                    self.funding_sign_count = 1;
                    self.funding_last_sign = sign;
                }
            }
        }

        // --- OI ---
        if !open_interest.is_nan() {
            if !self.last_oi.is_nan() && self.last_oi > 0.0 {
                let change = (open_interest - self.last_oi) / self.last_oi;
                self.prev_oi_change_for_accel = self.last_oi_change_pct;
                self.last_oi_change_pct = change;
                self.oi_change_ema_8.push(change);
            }
            self.last_oi = open_interest;
            if close > 0.0 && volume > 0.0 {
                let raw_lev = open_interest / (close * volume);
                self.leverage_proxy_ema.push(raw_lev);
            }
        }

        // --- LS Ratio ---
        if !ls_ratio.is_nan() {
            self.last_ls_ratio = ls_ratio;
            self.ls_ratio_window_24.push(ls_ratio);
        }

        // --- Basis ---
        if !spot_close.is_nan() && close > 0.0 && spot_close > 0.0 {
            let basis = (close - spot_close) / spot_close;
            self.last_basis = basis;
            self.basis_window_24.push(basis);
            self.basis_ema_8.push(basis);
        }

        // --- FGI ---
        if !fear_greed.is_nan() {
            if self.last_fgi.is_nan() || (fear_greed - self.last_fgi).abs() > 0.01 {
                self.fgi_window_7.push(fear_greed);
            }
            self.last_fgi = fear_greed;
        }

        // --- Deribit IV ---
        if !implied_vol.is_nan() {
            self.last_implied_vol = implied_vol;
            self.iv_window_24.push(implied_vol);
        }
        if !put_call_ratio_val.is_nan() {
            self.last_put_call_ratio = put_call_ratio_val;
        }

        // --- On-chain ---
        if !oc_flow_in.is_nan() && !oc_flow_out.is_nan() {
            self.onchain_netflow_buf.push(oc_flow_in - oc_flow_out);
        }
        if !oc_supply.is_nan() {
            self.onchain_supply_buf.push(oc_supply);
            self.last_onchain_supply = oc_supply;
        }
        if !oc_addr.is_nan() {
            self.onchain_addr_buf.push(oc_addr);
        }
        if !oc_tx.is_nan() {
            self.onchain_tx_buf.push(oc_tx);
        }
        if !oc_hashrate.is_nan() {
            self.onchain_hashrate_ema.push(oc_hashrate);
            self.last_onchain_hashrate = oc_hashrate;
        }

        // --- V11: Liquidation ---
        if !liq_total_vol.is_nan() {
            self.liq_volume_buf.push(liq_total_vol);
            self.last_liq_volume = liq_total_vol;
            self.last_liq_count = if liq_count.is_nan() { 0.0 } else { liq_count };
            let mut imb = 0.0;
            if liq_total_vol > 0.0 && !liq_buy_vol.is_nan() && !liq_sell_vol.is_nan() {
                imb = (liq_buy_vol - liq_sell_vol) / liq_total_vol;
            }
            self.liq_imbalance_buf.push(imb);
            self.last_liq_imbalance = imb;
        }

        // --- V11: Mempool ---
        if !mempool_fastest_fee.is_nan() {
            self.mempool_fee_buf.push(mempool_fastest_fee);
        }
        if !mempool_size.is_nan() {
            self.mempool_size_buf.push(mempool_size);
        }
        if !mempool_fastest_fee.is_nan() && !mempool_economy_fee.is_nan() && mempool_economy_fee > 0.0 {
            self.last_fee_urgency = mempool_fastest_fee / mempool_economy_fee;
        }

        // --- V11: Macro (daily — only push when day changes) ---
        if macro_day >= 0 && macro_day != self.last_macro_day {
            self.last_macro_day = macro_day;
            if !macro_dxy.is_nan() {
                self.dxy_buf.push(macro_dxy);
            }
            if !macro_spx.is_nan() {
                self.prev_spx_close = self.last_spx_close;
                self.last_spx_close = macro_spx;
                self.spx_buf.push(macro_spx);
            }
            if !macro_vix.is_nan() {
                self.last_vix = macro_vix;
                self.vix_buf.push(macro_vix);
            }
        }
        if macro_day >= 0 {
            self.btc_close_buf_30.push(close);
        }

        // --- V11: Sentiment ---
        if !social_volume.is_nan() {
            self.social_vol_buf.push(social_volume);
            self.last_social_volume = social_volume;
        }
        if !sentiment_score.is_nan() {
            self.last_sentiment_score = sentiment_score;
        }

        // --- Microstructure state ---
        self.last_trades = trades;
        self.last_taker_buy_volume = taker_buy_volume;
        self.last_taker_buy_quote_volume = taker_buy_quote_volume;
        self.last_quote_volume = quote_volume;

        // --- VWAP windows ---
        if volume > 0.0 {
            self.vwap_cv_window.push(close * volume);
            self.vwap_v_window.push(volume);
        }
        if trades > 0.0 {
            self.trades_ema_20.push(trades);
            self.trades_ema_5.push(trades);
            let tbr = if volume > 0.0 { taker_buy_volume / volume } else { 0.5 };
            self.taker_buy_ratio_ema_10.push(tbr);
            let imbalance = 2.0 * tbr - 1.0;
            self.cvd_window_10.push(imbalance);
            self.cvd_window_20.push(imbalance);
            self.taker_ratio_window_50.push(tbr);
            let ats = quote_volume / trades;
            self.avg_trade_size_ema_20.push(ats);
            let vpt = volume / trades;
            self.volume_per_trade_ema_20.push(vpt);
        }

        self.bar_count += 1;
        self.close_history.push(close);
        self.open_history.push(open_);
        self.high_history.push(high);
        self.low_history.push(low);

        // --- MAs ---
        self.ma_5.push(close);
        self.ma_10.push(close);
        self.ma_20.push(close);
        self.ma_30.push(close);
        self.ma_50.push(close);
        self.bb_window.push(close);

        // --- Returns ---
        if !self.last_close.is_nan() && self.last_close != 0.0 {
            let ret = (close - self.last_close) / self.last_close;
            self.return_window_20.push(ret);
            self.return_window_5.push(ret);
        }
        self.last_close = close;

        // V5: vol_5 history
        if self.return_window_5.full() {
            if let Some(v5_std) = self.return_window_5.std_dev() {
                self.vol_5_history.push(v5_std);
            }
        }

        // V8: Adaptive vol regime
        if self.return_window_5.full() && self.return_window_20.full() {
            if let (Some(v5_std), Some(v20_std)) = (self.return_window_5.std_dev(), self.return_window_20.std_dev()) {
                if v20_std > 1e-12 {
                    let vr = v5_std / v20_std;
                    self.vol_regime_ema.push(vr);
                    self.vol_regime_history.push(vr);
                }
            }
        }

        // V5: Parkinson volatility
        if high > 0.0 && low > 0.0 && high >= low {
            let hl_ratio = high / low;
            if hl_ratio > 0.0 {
                let ln_hl = hl_ratio.ln();
                self.hl_log_sq_window.push(ln_hl * ln_hl);
            }
        }

        // Volume
        self.last_volume = volume;
        self.vol_window_20.push(volume);
        self.vol_window_5.push(volume);

        // RSI
        self.rsi_14.push(close);
        self.rsi_6.push(close);

        // MACD
        self.ema_12.push(close);
        self.ema_26.push(close);
        if self.ema_12.ready(12) && self.ema_26.ready(26) {
            let macd_val = self.ema_12.value - self.ema_26.value;
            self.macd_signal_ema.push(macd_val);
        }

        // ATR
        self.atr_14.push(high, low, close);
    }

    pub fn get_bar_history(&self) -> &std::collections::VecDeque<RawBar> {
        &self.bar_history
    }

    pub(crate) fn get_features(&self, out: &mut [f64; N_FEATURES]) {
        // Initialize all to NaN
        for v in out.iter_mut() {
            *v = f64::NAN;
        }

        let close = self.last_close;
        let n = self.close_history.size();

        // --- Multi-horizon returns ---
        let horizons: [(usize, usize); 5] = [
            (1, F_RET_1), (3, F_RET_3), (6, F_RET_6), (12, F_RET_12), (24, F_RET_24),
        ];
        for &(h, idx) in &horizons {
            if n > h {
                let past = self.close_history.back_n(h);
                if past != 0.0 {
                    out[idx] = (self.close_history.back() - past) / past;
                }
            }
        }

        // --- MA crossovers ---
        let ma10v = if self.ma_10.full() { self.ma_10.mean().unwrap() } else { f64::NAN };
        let ma30v = if self.ma_30.full() { self.ma_30.mean().unwrap() } else { f64::NAN };
        let ma5v = if self.ma_5.full() { self.ma_5.mean().unwrap() } else { f64::NAN };
        let ma20v = if self.ma_20.full() { self.ma_20.mean().unwrap() } else { f64::NAN };
        let ma50v = if self.ma_50.full() { self.ma_50.mean().unwrap() } else { f64::NAN };

        if !ma10v.is_nan() && !ma30v.is_nan() && ma30v != 0.0 {
            out[F_MA_CROSS_10_30] = ma10v / ma30v - 1.0;
        }
        if !ma5v.is_nan() && !ma20v.is_nan() && ma20v != 0.0 {
            out[F_MA_CROSS_5_20] = ma5v / ma20v - 1.0;
        }
        if !close.is_nan() && !ma20v.is_nan() && ma20v != 0.0 {
            out[F_CLOSE_VS_MA20] = close / ma20v - 1.0;
        }
        if !close.is_nan() && !ma50v.is_nan() && ma50v != 0.0 {
            out[F_CLOSE_VS_MA50] = close / ma50v - 1.0;
        }

        // --- RSI ---
        let rsi14_val = self.rsi_14.get_value();
        let rsi6_val = self.rsi_6.get_value();
        if !rsi14_val.is_nan() {
            out[F_RSI_14] = (rsi14_val - 50.0) / 50.0;
        }
        if !rsi6_val.is_nan() {
            out[F_RSI_6] = (rsi6_val - 50.0) / 50.0;
        }

        // --- MACD ---
        if self.ema_12.ready(12) && self.ema_26.ready(26) {
            let macd_line = self.ema_12.value - self.ema_26.value;
            if !close.is_nan() && close != 0.0 {
                out[F_MACD_LINE] = macd_line / close;
                if self.macd_signal_ema.ready(9) {
                    let sig = self.macd_signal_ema.value;
                    out[F_MACD_SIGNAL] = sig / close;
                    out[F_MACD_HIST] = (macd_line - sig) / close;
                }
            }
        }

        // --- Bollinger Bands ---
        if self.bb_window.full() {
            let bb_mid = self.bb_window.mean().unwrap();
            let bb_std = self.bb_window.std_dev().unwrap();
            if bb_mid != 0.0 && bb_std != 0.0 {
                let upper = bb_mid + 2.0 * bb_std;
                let lower = bb_mid - 2.0 * bb_std;
                out[F_BB_WIDTH_20] = (upper - lower) / bb_mid;
                let band_range = upper - lower;
                if band_range != 0.0 && !close.is_nan() {
                    out[F_BB_PCTB_20] = (close - lower) / band_range;
                }
            }
        }

        // --- ATR ---
        let atr_val = self.atr_14.get_value();
        if !atr_val.is_nan() && !close.is_nan() && close != 0.0 {
            out[F_ATR_NORM_14] = atr_val / close;
        }

        // --- Volatility ---
        let mut vol20_v = f64::NAN;
        let mut vol5_v = f64::NAN;
        if self.return_window_20.full() {
            vol20_v = self.return_window_20.std_dev().unwrap();
            out[F_VOL_20] = vol20_v;
        }
        if self.return_window_5.full() {
            vol5_v = self.return_window_5.std_dev().unwrap();
            out[F_VOL_5] = vol5_v;
        }

        // --- Volume features ---
        let vol_ma20 = if self.vol_window_20.full() { self.vol_window_20.mean().unwrap() } else { f64::NAN };
        let vol_ma5 = if self.vol_window_5.full() { self.vol_window_5.mean().unwrap() } else { f64::NAN };

        if !vol_ma20.is_nan() && vol_ma20 != 0.0 && self.vol_window_20.n() > 0 {
            out[F_VOL_RATIO_20] = self.last_volume / vol_ma20;
        }
        if !vol_ma5.is_nan() && !vol_ma20.is_nan() && vol_ma20 != 0.0 {
            out[F_VOL_MA_RATIO_5_20] = vol_ma5 / vol_ma20;
        }

        // --- Candle structure ---
        if n > 0 && self.open_history.size() > 0 && self.high_history.size() > 0 && self.low_history.size() > 0 {
            let o = self.open_history.back();
            let h = self.high_history.back();
            let l = self.low_history.back();
            let c = self.close_history.back();
            let hl_range = h - l;
            if hl_range > 0.0 {
                out[F_BODY_RATIO] = (c - o) / hl_range;
                out[F_UPPER_SHADOW] = (h - o.max(c)) / hl_range;
                out[F_LOWER_SHADOW] = (o.min(c) - l) / hl_range;
            }
        }

        // --- Mean reversion ---
        if self.bb_window.full() && !close.is_nan() {
            let bb_mid = self.bb_window.mean().unwrap();
            let bb_std = self.bb_window.std_dev().unwrap();
            if bb_std != 0.0 {
                out[F_MEAN_REVERSION_20] = (close - bb_mid) / bb_std;
            }
        }

        // --- Price acceleration ---
        let current_momentum = out[F_MA_CROSS_10_30];
        if !current_momentum.is_nan() && !self.prev_momentum.is_nan() {
            out[F_PRICE_ACCELERATION] = current_momentum - self.prev_momentum;
        }

        // --- Time ---
        if self.last_hour >= 0 {
            out[F_HOUR_SIN] = (2.0 * PI * self.last_hour as f64 / 24.0).sin();
            out[F_HOUR_COS] = (2.0 * PI * self.last_hour as f64 / 24.0).cos();
        }
        if self.last_dow >= 0 {
            out[F_DOW_SIN] = (2.0 * PI * self.last_dow as f64 / 7.0).sin();
            out[F_DOW_COS] = (2.0 * PI * self.last_dow as f64 / 7.0).cos();
        }

        // --- Vol regime ---
        if !vol5_v.is_nan() && !vol20_v.is_nan() && vol20_v != 0.0 {
            out[F_VOL_REGIME] = vol5_v / vol20_v;
        }

        // --- Funding rate ---
        out[F_FUNDING_RATE] = self.last_funding_rate;
        out[F_FUNDING_MA8] = if self.funding_ema.ready(8) { self.funding_ema.value } else { f64::NAN };

        // --- Kline microstructure ---
        let trades_val = self.last_trades;
        let volume_val = self.last_volume;
        if trades_val > 0.0 && self.trades_ema_20.ready(20) {
            let ema_t20 = self.trades_ema_20.value;
            if ema_t20 > 0.0 {
                out[F_TRADE_INTENSITY] = trades_val / ema_t20;
            }
        }

        let mut tbr = f64::NAN;
        if trades_val > 0.0 && volume_val > 0.0 {
            tbr = self.last_taker_buy_volume / volume_val;
            out[F_TAKER_BUY_RATIO] = tbr;
        }

        if self.taker_buy_ratio_ema_10.ready(10) {
            out[F_TAKER_BUY_RATIO_MA10] = self.taker_buy_ratio_ema_10.value;
        }

        if !tbr.is_nan() {
            out[F_TAKER_IMBALANCE] = 2.0 * tbr - 1.0;
        }

        if trades_val > 0.0 {
            let ats = self.last_quote_volume / trades_val;
            out[F_AVG_TRADE_SIZE] = ats;
            if self.avg_trade_size_ema_20.ready(20) {
                let ats_ema = self.avg_trade_size_ema_20.value;
                if ats_ema > 0.0 {
                    out[F_AVG_TRADE_SIZE_RATIO] = ats / ats_ema;
                }
            }
            let vpt = volume_val / trades_val;
            if self.volume_per_trade_ema_20.ready(20) {
                let vpt_ema = self.volume_per_trade_ema_20.value;
                if vpt_ema > 0.0 {
                    out[F_VOLUME_PER_TRADE] = vpt / vpt_ema;
                }
            }
        }

        if self.trades_ema_5.ready(5) && self.trades_ema_20.ready(20) {
            let e5 = self.trades_ema_5.value;
            let e20 = self.trades_ema_20.value;
            if e20 > 0.0 {
                out[F_TRADE_COUNT_REGIME] = e5 / e20;
            }
        }

        // --- Funding deep ---
        if self.funding_window_24.full() {
            let f_mean = self.funding_window_24.mean().unwrap();
            let f_std = self.funding_window_24.std_dev().unwrap();
            if f_std > 1e-12 && !self.last_funding_rate.is_nan() {
                let zscore = (self.last_funding_rate - f_mean) / f_std;
                out[F_FUNDING_ZSCORE_24] = zscore;
                out[F_FUNDING_EXTREME] = if zscore.abs() > 2.0 { 1.0 } else { 0.0 };
            }
        }

        let fr_ma8 = out[F_FUNDING_MA8];
        if !self.last_funding_rate.is_nan() && !fr_ma8.is_nan() {
            out[F_FUNDING_MOMENTUM] = self.last_funding_rate - fr_ma8;
        }

        if self.funding_history_8.size() == 8 {
            out[F_FUNDING_CUMULATIVE_8] = self.funding_history_8.sum();
        }

        out[F_FUNDING_SIGN_PERSIST] = if self.funding_sign_count > 0 {
            self.funding_sign_count as f64
        } else {
            f64::NAN
        };

        // --- OI features ---
        out[F_OI_CHANGE_PCT] = self.last_oi_change_pct;
        out[F_OI_CHANGE_MA8] = if self.oi_change_ema_8.ready(8) { self.oi_change_ema_8.value } else { f64::NAN };

        let ret1 = out[F_RET_1];
        if !ret1.is_nan() && !self.last_oi_change_pct.is_nan() {
            let price_sign = sign_f64(ret1);
            let oi_sign = sign_f64(self.last_oi_change_pct);
            out[F_OI_CLOSE_DIVERGENCE] = -price_sign * oi_sign;
        }

        // --- LS Ratio ---
        out[F_LS_RATIO] = self.last_ls_ratio;
        if self.ls_ratio_window_24.full() && !self.last_ls_ratio.is_nan() {
            let ls_mean = self.ls_ratio_window_24.mean().unwrap();
            let ls_std = self.ls_ratio_window_24.std_dev().unwrap();
            if ls_std > 1e-12 {
                let zscore = (self.last_ls_ratio - ls_mean) / ls_std;
                out[F_LS_RATIO_ZSCORE_24] = zscore;
                out[F_LS_EXTREME] = if zscore.abs() > 2.0 { 1.0 } else { 0.0 };
            }
        }

        // --- V5: Order Flow ---
        if self.cvd_window_10.full() {
            out[F_CVD_10] = self.cvd_window_10.mean().unwrap() * self.cvd_window_10.n() as f64;
        }

        if self.cvd_window_20.full() {
            let cvd_20_val = self.cvd_window_20.mean().unwrap() * self.cvd_window_20.n() as f64;
            out[F_CVD_20] = cvd_20_val;
            if n > 20 {
                let past20 = self.close_history.back_n(20);
                if past20 != 0.0 {
                    let ret_20 = (self.close_history.back() - past20) / past20;
                    let cvd_sign = sign_f64(cvd_20_val);
                    let ret_sign_v = sign_f64(ret_20);
                    out[F_CVD_PRICE_DIVERGENCE] = if cvd_sign != 0.0 && cvd_sign != ret_sign_v { 1.0 } else { 0.0 };
                }
            }
        }

        if self.taker_ratio_window_50.full() && !tbr.is_nan() {
            let tr_mean = self.taker_ratio_window_50.mean().unwrap();
            let tr_std = self.taker_ratio_window_50.std_dev().unwrap();
            if tr_std > 1e-12 {
                out[F_AGGRESSIVE_FLOW_ZSCORE] = (tbr - tr_mean) / tr_std;
            }
        }

        // --- V5: Volatility microstructure ---
        if self.vol_5_history.size() >= 20 {
            let cnt = 20;
            let start = self.vol_5_history.size() - cnt;
            let mut sum_v = 0.0;
            for i in start..self.vol_5_history.size() {
                sum_v += self.vol_5_history.get(i);
            }
            let mean_v = sum_v / cnt as f64;
            let mut sumsq_v = 0.0;
            for i in start..self.vol_5_history.size() {
                let d = self.vol_5_history.get(i) - mean_v;
                sumsq_v += d * d;
            }
            out[F_VOL_OF_VOL] = (sumsq_v / cnt as f64).sqrt();
        }

        if n > 0 && !close.is_nan() && close != 0.0 && !vol5_v.is_nan() && vol5_v > 1e-12 {
            let h = if self.high_history.size() > 0 { self.high_history.back() } else { close };
            let l = if self.low_history.size() > 0 { self.low_history.back() } else { close };
            out[F_RANGE_VS_RV] = ((h - l) / close) / vol5_v;
        }

        if self.hl_log_sq_window.full() {
            let mean_sq = self.hl_log_sq_window.mean().unwrap();
            if mean_sq >= 0.0 {
                out[F_PARKINSON_VOL] = (mean_sq / (4.0 * 2.0_f64.ln())).sqrt();
            }
        }

        if self.vol_5_history.size() >= 6 {
            out[F_RV_ACCELERATION] = self.vol_5_history.back_n(0) - self.vol_5_history.back_n(5);
        }

        // --- V5: Liquidation proxy ---
        if !self.last_oi_change_pct.is_nan() && !self.prev_oi_change_for_accel.is_nan() {
            out[F_OI_ACCELERATION] = self.last_oi_change_pct - self.prev_oi_change_for_accel;
        }

        if !self.last_oi.is_nan() && !close.is_nan() && close > 0.0 && self.last_volume > 0.0 {
            let raw_lev = self.last_oi / (close * self.last_volume);
            if self.leverage_proxy_ema.ready(20) {
                let lev_ema = self.leverage_proxy_ema.value;
                if lev_ema > 0.0 {
                    out[F_LEVERAGE_PROXY] = raw_lev / lev_ema;
                }
            }
        }

        // oi_vol_divergence
        let oi_chg = self.last_oi_change_pct;
        let vol_r = out[F_VOL_RATIO_20];
        if !oi_chg.is_nan() && !vol_r.is_nan() {
            out[F_OI_VOL_DIVERGENCE] = if oi_chg > 0.0 && vol_r < 1.0 { 1.0 } else { 0.0 };
        }

        // oi_liquidation_flag
        if !oi_chg.is_nan() && !vol_r.is_nan() {
            out[F_OI_LIQUIDATION_FLAG] = if oi_chg < -0.05 && vol_r > 2.0 { 1.0 } else { 0.0 };
        }

        // --- V5: Funding carry ---
        if !self.last_funding_rate.is_nan() {
            out[F_FUNDING_ANNUALIZED] = self.last_funding_rate * 3.0 * 365.0;
        }

        if !self.last_funding_rate.is_nan() && !vol20_v.is_nan() && vol20_v > 1e-12 {
            out[F_FUNDING_VS_VOL] = self.last_funding_rate / vol20_v;
        }

        // --- V7: Basis ---
        out[F_BASIS] = self.last_basis;
        if self.basis_window_24.full() && !self.last_basis.is_nan() {
            let b_mean = self.basis_window_24.mean().unwrap();
            let b_std = self.basis_window_24.std_dev().unwrap();
            if b_std > 1e-12 {
                let zscore = (self.last_basis - b_mean) / b_std;
                out[F_BASIS_ZSCORE_24] = zscore;
                out[F_BASIS_EXTREME] = if zscore > 2.0 { 1.0 } else if zscore < -2.0 { -1.0 } else { 0.0 };
            }
        }

        if !self.last_basis.is_nan() && self.basis_ema_8.ready(8) {
            out[F_BASIS_MOMENTUM] = self.last_basis - self.basis_ema_8.value;
        }

        // --- V7: FGI ---
        if !self.last_fgi.is_nan() {
            out[F_FGI_NORMALIZED] = self.last_fgi / 100.0 - 0.5;
            out[F_FGI_EXTREME] = if self.last_fgi < 25.0 { -1.0 } else if self.last_fgi > 75.0 { 1.0 } else { 0.0 };
        }

        if self.fgi_window_7.full() && !self.last_fgi.is_nan() {
            let fgi_mean = self.fgi_window_7.mean().unwrap();
            let fgi_std = self.fgi_window_7.std_dev().unwrap();
            if fgi_std > 1e-12 {
                out[F_FGI_ZSCORE_7] = (self.last_fgi - fgi_mean) / fgi_std;
            }
        }

        // --- V8: Alpha Rebuild V3 ---
        let tbqv = self.last_taker_buy_quote_volume;
        let qv = self.last_quote_volume;
        if tbqv > 0.0 && qv > 0.0 {
            out[F_TAKER_BQ_RATIO] = tbqv / qv;
        }

        if self.vwap_cv_window.full() && self.vwap_v_window.full() && !close.is_nan() && close > 0.0 {
            let sum_cv = self.vwap_cv_window.mean().unwrap() * self.vwap_cv_window.n() as f64;
            let sum_v = self.vwap_v_window.mean().unwrap() * self.vwap_v_window.n() as f64;
            if sum_v > 0.0 {
                let vwap = sum_cv / sum_v;
                out[F_VWAP_DEV_20] = (close - vwap) / close;
            }
        }

        // volume_momentum_10
        let mut ret_10 = f64::NAN;
        if n > 10 {
            let past10 = self.close_history.back_n(10);
            if past10 != 0.0 {
                ret_10 = (self.close_history.back() - past10) / past10;
            }
        }
        let vol_r_20 = out[F_VOL_RATIO_20];
        if !ret_10.is_nan() && !vol_r_20.is_nan() {
            out[F_VOLUME_MOMENTUM_10] = ret_10 * vol_r_20.min(3.0);
        }

        // mom_vol_divergence
        let ret1_2 = out[F_RET_1];
        let vol_r2 = out[F_VOL_RATIO_20];
        if !ret1_2.is_nan() && !vol_r2.is_nan() {
            let price_up = ret1_2 > 0.0;
            let vol_up = vol_r2 > 1.0;
            out[F_MOM_VOL_DIVERGENCE] = if price_up == vol_up { 1.0 } else { -1.0 };
        }

        // basis_carry_adj
        if !self.last_basis.is_nan() && !self.last_funding_rate.is_nan() {
            out[F_BASIS_CARRY_ADJ] = self.last_basis + self.last_funding_rate * 3.0;
        }

        // vol_regime_adaptive
        if self.vol_regime_ema.ready(5) && self.vol_regime_history.size() >= 30 {
            let ema_val = self.vol_regime_ema.value;
            let mut sorted_arr = [0.0_f64; 30];
            for i in 0..30 {
                sorted_arr[i] = self.vol_regime_history.get(i);
            }
            sorted_arr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median_val = sorted_arr[15];
            if ema_val > median_val * 1.05 {
                out[F_VOL_REGIME_ADAPTIVE] = 1.0;
            } else if ema_val < median_val * 0.95 {
                out[F_VOL_REGIME_ADAPTIVE] = -1.0;
            } else {
                out[F_VOL_REGIME_ADAPTIVE] = 0.0;
            }
        }

        // --- V9: Cross-factor interaction ---
        let oi_pct = out[F_OI_CHANGE_PCT];
        let vmr = out[F_VOL_MA_RATIO_5_20];
        if !oi_pct.is_nan() && !vmr.is_nan() {
            out[F_LIQUIDATION_CASCADE_SCORE] = oi_pct.abs() * vmr;
        }

        let fr = out[F_FUNDING_RATE];
        let fma8 = out[F_FUNDING_MA8];
        if !fr.is_nan() && !fma8.is_nan() {
            let denom = fma8.abs().max(1e-6);
            out[F_FUNDING_TERM_SLOPE] = (fr - fma8) / denom;
        }

        out[F_CROSS_TF_REGIME_SYNC] = f64::NAN; // requires external aggregator

        // --- V9: Deribit IV ---
        if self.iv_window_24.full() && !self.last_implied_vol.is_nan() {
            let iv_mean = self.iv_window_24.mean().unwrap();
            let iv_std = self.iv_window_24.std_dev().unwrap();
            if iv_std > 1e-8 {
                out[F_IMPLIED_VOL_ZSCORE_24] = (self.last_implied_vol - iv_mean) / iv_std;
            }
        }

        if !self.last_implied_vol.is_nan() && !vol20_v.is_nan() {
            out[F_IV_RV_SPREAD] = self.last_implied_vol - vol20_v;
        }

        out[F_PUT_CALL_RATIO] = self.last_put_call_ratio;

        // --- V10: On-chain ---
        if self.onchain_netflow_buf.size() >= 7 {
            let mut tmp = [0.0_f64; 7];
            let start = self.onchain_netflow_buf.size() - 7;
            for i in 0..7 {
                tmp[i] = self.onchain_netflow_buf.get(start + i);
            }
            out[F_EXCHANGE_NETFLOW_ZSCORE] = zscore_buf(&tmp, 1e-8);
        }

        if self.onchain_supply_buf.size() >= 2 {
            let prev = self.onchain_supply_buf.back_n(1);
            let curr = self.onchain_supply_buf.back();
            out[F_EXCHANGE_SUPPLY_CHANGE] = if prev > 1e-8 { (curr - prev) / prev } else { 0.0 };
        }

        if self.onchain_supply_buf.size() >= 30 {
            let mut tmp = [0.0_f64; 30];
            for i in 0..30 {
                tmp[i] = self.onchain_supply_buf.get(i);
            }
            out[F_EXCHANGE_SUPPLY_ZSCORE_30] = zscore_buf(&tmp, 1e-8);
        }

        if self.onchain_addr_buf.size() >= 14 {
            let mut tmp = [0.0_f64; 14];
            for i in 0..14 {
                tmp[i] = self.onchain_addr_buf.get(i);
            }
            out[F_ACTIVE_ADDR_ZSCORE_14] = zscore_buf(&tmp, 1e-8);
        }

        if self.onchain_tx_buf.size() >= 14 {
            let mut tmp = [0.0_f64; 14];
            for i in 0..14 {
                tmp[i] = self.onchain_tx_buf.get(i);
            }
            out[F_TX_COUNT_ZSCORE_14] = zscore_buf(&tmp, 1e-8);
        }

        if self.onchain_hashrate_ema.ready(14) && !self.last_onchain_hashrate.is_nan() {
            let ema_val = self.onchain_hashrate_ema.value;
            if ema_val.abs() > 1e-8 {
                out[F_HASHRATE_MOMENTUM] = (self.last_onchain_hashrate - ema_val) / ema_val;
            }
        }

        // --- V11: Liquidation features ---
        if self.liq_volume_buf.size() >= 24 {
            let mut tmp = [0.0_f64; 24];
            let start = self.liq_volume_buf.size() - 24;
            for i in 0..24 {
                tmp[i] = self.liq_volume_buf.get(start + i);
            }
            out[F_LIQUIDATION_VOLUME_ZSCORE_24] = zscore_buf(&tmp, 1e-8);
        }

        out[F_LIQUIDATION_IMBALANCE] = self.last_liq_imbalance;

        if !self.last_liq_volume.is_nan() && self.last_quote_volume > 0.0 {
            out[F_LIQUIDATION_VOLUME_RATIO] = self.last_liq_volume / self.last_quote_volume;
        }

        if self.liq_imbalance_buf.size() >= 6 && self.liq_volume_buf.size() >= 6 {
            let mut recent = [0.0_f64; 6];
            let start = self.liq_volume_buf.size() - 6;
            for i in 0..6 {
                recent[i] = self.liq_volume_buf.get(start + i);
            }
            let mut sum6 = 0.0;
            for &v in &recent {
                sum6 += v;
            }
            let mean6 = sum6 / 6.0;
            let mut var6 = 0.0;
            for &v in &recent {
                let d = v - mean6;
                var6 += d * d;
            }
            var6 /= 6.0;
            let std6 = var6.sqrt();
            out[F_LIQUIDATION_CLUSTER_FLAG] = if std6 > 1e-8 && recent[5] > mean6 + 3.0 * std6 { 1.0 } else { 0.0 };
        }

        // --- V11: Mempool features ---
        if self.mempool_fee_buf.size() >= 24 {
            let mut tmp = [0.0_f64; 24];
            let start = self.mempool_fee_buf.size() - 24;
            for i in 0..24 {
                tmp[i] = self.mempool_fee_buf.get(start + i);
            }
            out[F_MEMPOOL_FEE_ZSCORE_24] = zscore_buf(&tmp, 1e-8);
        }

        if self.mempool_size_buf.size() >= 24 {
            let mut tmp = [0.0_f64; 24];
            let start = self.mempool_size_buf.size() - 24;
            for i in 0..24 {
                tmp[i] = self.mempool_size_buf.get(start + i);
            }
            out[F_MEMPOOL_SIZE_ZSCORE_24] = zscore_buf(&tmp, 1e-8);
        }

        out[F_FEE_URGENCY_RATIO] = self.last_fee_urgency;

        // --- V11: Macro features ---
        // dxy_change_5d
        if self.dxy_buf.size() >= 6 {
            let old_dxy = self.dxy_buf.get(self.dxy_buf.size() - 6);
            let new_dxy = self.dxy_buf.back();
            out[F_DXY_CHANGE_5D] = if old_dxy > 1e-8 { (new_dxy - old_dxy) / old_dxy } else { 0.0 };
        }

        // spx_btc_corr_30d
        {
            let n_spx = self.spx_buf.size();
            let n_btc = self.btc_close_buf_30.size();
            let nc = n_spx.min(n_btc);
            if nc >= 10 {
                let m = nc - 1;
                if m >= 5 {
                    let spx_off = n_spx - nc;
                    let btc_off = n_btc - nc;
                    let mut spx_rets = vec![0.0_f64; m];
                    let mut btc_rets = vec![0.0_f64; m];
                    for i in 0..m {
                        let s0 = self.spx_buf.get(spx_off + i);
                        let s1 = self.spx_buf.get(spx_off + i + 1);
                        spx_rets[i] = if s0 > 0.0 { (s1 - s0) / s0 } else { 0.0 };
                        let b0 = self.btc_close_buf_30.get(btc_off + i);
                        let b1 = self.btc_close_buf_30.get(btc_off + i + 1);
                        btc_rets[i] = if b0 > 0.0 { (b1 - b0) / b0 } else { 0.0 };
                    }
                    let mut mean_s = 0.0;
                    let mut mean_b = 0.0;
                    for i in 0..m {
                        mean_s += spx_rets[i];
                        mean_b += btc_rets[i];
                    }
                    mean_s /= m as f64;
                    mean_b /= m as f64;
                    let mut cov = 0.0;
                    let mut var_s = 0.0;
                    let mut var_b = 0.0;
                    for i in 0..m {
                        let ds = spx_rets[i] - mean_s;
                        let db = btc_rets[i] - mean_b;
                        cov += ds * db;
                        var_s += ds * ds;
                        var_b += db * db;
                    }
                    cov /= m as f64;
                    var_s /= m as f64;
                    var_b /= m as f64;
                    let denom = (var_s * var_b).sqrt();
                    out[F_SPX_BTC_CORR_30D] = if denom > 1e-8 { cov / denom } else { 0.0 };
                }
            }
        }

        // spx_overnight_ret
        if !self.last_spx_close.is_nan() && !self.prev_spx_close.is_nan() && self.prev_spx_close > 0.0 {
            out[F_SPX_OVERNIGHT_RET] = (self.last_spx_close - self.prev_spx_close) / self.prev_spx_close;
        }

        // vix_zscore_14
        if self.vix_buf.size() >= 14 {
            let mut tmp = [0.0_f64; 14];
            for i in 0..14 {
                tmp[i] = self.vix_buf.get(i);
            }
            out[F_VIX_ZSCORE_14] = zscore_buf(&tmp, 1e-8);
        }

        // --- V11: Social sentiment features ---
        if self.social_vol_buf.size() >= 24 {
            let mut tmp = [0.0_f64; 24];
            let start = self.social_vol_buf.size() - 24;
            for i in 0..24 {
                tmp[i] = self.social_vol_buf.get(start + i);
            }
            out[F_SOCIAL_VOLUME_ZSCORE_24] = zscore_buf(&tmp, 1e-8);
        }

        out[F_SOCIAL_SENTIMENT_SCORE] = self.last_sentiment_score;

        // social_volume_price_div
        if !self.last_social_volume.is_nan() && self.social_vol_buf.size() >= 2 && self.close_history.size() >= 2 {
            let sv_change = self.social_vol_buf.back() - self.social_vol_buf.back_n(1);
            let price_change = self.close_history.back() - self.close_history.back_n(1);
            if (sv_change > 0.0 && price_change < 0.0) || (sv_change < 0.0 && price_change > 0.0) {
                out[F_SOCIAL_VOLUME_PRICE_DIV] = 1.0;
            } else {
                out[F_SOCIAL_VOLUME_PRICE_DIV] = 0.0;
            }
        }
    }
}

// ============================================================
// Schedule cursors
// ============================================================

struct StridedCursor {
    data: Vec<f64>,
    rows: usize,
    idx: usize,
    current: f64,
}

impl StridedCursor {
    fn new(sched: &[Vec<f64>]) -> Self {
        let rows = sched.len();
        let mut data = Vec::with_capacity(rows * 2);
        for row in sched {
            if row.len() >= 2 {
                data.push(row[0]);
                data.push(row[1]);
            }
        }
        let actual_rows = data.len() / 2;
        StridedCursor {
            data,
            rows: actual_rows,
            idx: 0,
            current: f64::NAN,
        }
    }

    fn advance(&mut self, bar_ts: f64) -> f64 {
        while self.idx < self.rows && self.data[self.idx * 2] <= bar_ts {
            self.current = self.data[self.idx * 2 + 1];
            self.idx += 1;
        }
        self.current
    }
}

struct OnchainCursor {
    data: Vec<f64>,
    rows: usize,
    idx: usize,
    flow_in: f64,
    flow_out: f64,
    supply: f64,
    addr: f64,
    tx: f64,
    hashrate: f64,
}

impl OnchainCursor {
    fn new(sched: &[Vec<f64>]) -> Self {
        let rows = sched.len();
        let mut data = Vec::with_capacity(rows * 7);
        let mut actual_rows = 0;
        for row in sched {
            if row.len() >= 7 {
                for j in 0..7 {
                    data.push(row[j]);
                }
                actual_rows += 1;
            }
        }
        OnchainCursor {
            data,
            rows: actual_rows,
            idx: 0,
            flow_in: f64::NAN,
            flow_out: f64::NAN,
            supply: f64::NAN,
            addr: f64::NAN,
            tx: f64::NAN,
            hashrate: f64::NAN,
        }
    }

    fn advance(&mut self, bar_ts: f64) {
        while self.idx < self.rows && self.data[self.idx * 7] <= bar_ts {
            let base = self.idx * 7;
            self.flow_in = self.data[base + 1];
            self.flow_out = self.data[base + 2];
            self.supply = self.data[base + 3];
            self.addr = self.data[base + 4];
            self.tx = self.data[base + 5];
            self.hashrate = self.data[base + 6];
            self.idx += 1;
        }
    }
}

struct LiqCursor {
    data: Vec<f64>,
    rows: usize,
    idx: usize,
    total_vol: f64,
    buy_vol: f64,
    sell_vol: f64,
}

impl LiqCursor {
    fn new(sched: &[Vec<f64>]) -> Self {
        let rows = sched.len();
        let mut data = Vec::with_capacity(rows * 4);
        let mut actual_rows = 0;
        for row in sched {
            if row.len() >= 4 {
                for j in 0..4 {
                    data.push(row[j]);
                }
                actual_rows += 1;
            }
        }
        LiqCursor {
            data,
            rows: actual_rows,
            idx: 0,
            total_vol: f64::NAN,
            buy_vol: f64::NAN,
            sell_vol: f64::NAN,
        }
    }

    fn advance(&mut self, bar_ts: f64) {
        while self.idx < self.rows && self.data[self.idx * 4] <= bar_ts {
            let base = self.idx * 4;
            self.total_vol = self.data[base + 1];
            self.buy_vol = self.data[base + 2];
            self.sell_vol = self.data[base + 3];
            self.idx += 1;
        }
    }
}

struct MempoolCursor {
    data: Vec<f64>,
    rows: usize,
    idx: usize,
    fastest_fee: f64,
    economy_fee: f64,
    mempool_size: f64,
}

impl MempoolCursor {
    fn new(sched: &[Vec<f64>]) -> Self {
        let rows = sched.len();
        let mut data = Vec::with_capacity(rows * 4);
        let mut actual_rows = 0;
        for row in sched {
            if row.len() >= 4 {
                for j in 0..4 {
                    data.push(row[j]);
                }
                actual_rows += 1;
            }
        }
        MempoolCursor {
            data,
            rows: actual_rows,
            idx: 0,
            fastest_fee: f64::NAN,
            economy_fee: f64::NAN,
            mempool_size: f64::NAN,
        }
    }

    fn advance(&mut self, bar_ts: f64) {
        while self.idx < self.rows && self.data[self.idx * 4] <= bar_ts {
            let base = self.idx * 4;
            self.fastest_fee = self.data[base + 1];
            self.economy_fee = self.data[base + 2];
            self.mempool_size = self.data[base + 3];
            self.idx += 1;
        }
    }
}

struct MacroCursor {
    data: Vec<f64>,
    rows: usize,
    idx: usize,
    dxy: f64,
    spx: f64,
    vix: f64,
    day: i64,
}

impl MacroCursor {
    fn new(sched: &[Vec<f64>]) -> Self {
        let rows = sched.len();
        let mut data = Vec::with_capacity(rows * 4);
        let mut actual_rows = 0;
        for row in sched {
            if row.len() >= 4 {
                for j in 0..4 {
                    data.push(row[j]);
                }
                actual_rows += 1;
            }
        }
        MacroCursor {
            data,
            rows: actual_rows,
            idx: 0,
            dxy: f64::NAN,
            spx: f64::NAN,
            vix: f64::NAN,
            day: -1,
        }
    }

    fn advance(&mut self, bar_ts: f64) {
        while self.idx < self.rows && self.data[self.idx * 4] <= bar_ts {
            let base = self.idx * 4;
            let ts_sec = (self.data[base] / 1000.0) as i64;
            self.day = ts_sec / 86400;
            if !self.data[base + 1].is_nan() {
                self.dxy = self.data[base + 1];
            }
            if !self.data[base + 2].is_nan() {
                self.spx = self.data[base + 2];
            }
            if !self.data[base + 3].is_nan() {
                self.vix = self.data[base + 3];
            }
            self.idx += 1;
        }
    }
}

// ============================================================
// PyO3 exported functions
// ============================================================

#[pyfunction]
#[pyo3(signature = (
    timestamps, opens, highs, lows, closes, volumes,
    trades, taker_buy_volumes, quote_volumes, taker_buy_quote_volumes,
    funding_sched, oi_sched, ls_sched, spot_sched, fgi_sched,
    iv_sched, pcr_sched, onchain_sched,
    liq_sched, mempool_sched, macro_sched
))]
pub fn cpp_compute_all_features(
    timestamps: Vec<f64>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    trades: Vec<f64>,
    taker_buy_volumes: Vec<f64>,
    quote_volumes: Vec<f64>,
    taker_buy_quote_volumes: Vec<f64>,
    funding_sched: Vec<Vec<f64>>,
    oi_sched: Vec<Vec<f64>>,
    ls_sched: Vec<Vec<f64>>,
    spot_sched: Vec<Vec<f64>>,
    fgi_sched: Vec<Vec<f64>>,
    iv_sched: Vec<Vec<f64>>,
    pcr_sched: Vec<Vec<f64>>,
    onchain_sched: Vec<Vec<f64>>,
    liq_sched: Vec<Vec<f64>>,
    mempool_sched: Vec<Vec<f64>>,
    macro_sched: Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let n_bars = timestamps.len();

    let mut funding_cur = StridedCursor::new(&funding_sched);
    let mut oi_cur = StridedCursor::new(&oi_sched);
    let mut ls_cur = StridedCursor::new(&ls_sched);
    let mut spot_cur = StridedCursor::new(&spot_sched);
    let mut fgi_cur = StridedCursor::new(&fgi_sched);
    let mut iv_cur = StridedCursor::new(&iv_sched);
    let mut pcr_cur = StridedCursor::new(&pcr_sched);
    let mut oc_cur = OnchainCursor::new(&onchain_sched);
    let mut liq_cur = LiqCursor::new(&liq_sched);
    let mut mp_cur = MempoolCursor::new(&mempool_sched);
    let mut macro_cur = MacroCursor::new(&macro_sched);

    let mut state = BarState::new();
    let mut result = Vec::with_capacity(n_bars);

    for i in 0..n_bars {
        let ts = timestamps[i];
        let mut close = closes[i];
        let volume = volumes[i];
        let mut high = highs[i];
        let mut low = lows[i];
        let mut open_ = opens[i];
        let trades_val = trades[i];
        let taker_buy_volume = taker_buy_volumes[i];
        let quote_volume = quote_volumes[i];
        let taker_buy_quote_volume = taker_buy_quote_volumes[i];

        // Default open/high/low to close if zero
        if open_ == 0.0 { open_ = close; }
        if high == 0.0 { high = close; }
        if low == 0.0 { low = close; }
        let _ = close; // suppress unused warning

        // Parse hour and dow from timestamp (ms)
        let mut hour: i32 = -1;
        let mut dow: i32 = -1;
        if ts > 0.0 {
            let ts_sec = (ts / 1000.0) as i64;
            let mut days = ts_sec / 86400;
            let mut day_sec = ts_sec % 86400;
            if day_sec < 0 {
                days -= 1;
                day_sec += 86400;
            }
            hour = (day_sec / 3600) as i32;
            dow = ((days + 3) % 7) as i32;
            if dow < 0 {
                dow += 7;
            }
        }

        // Advance schedule cursors
        let funding_rate = funding_cur.advance(ts);
        let open_interest = oi_cur.advance(ts);
        let ls_ratio = ls_cur.advance(ts);
        let spot_close = spot_cur.advance(ts);
        let fear_greed = fgi_cur.advance(ts);
        let implied_vol = iv_cur.advance(ts);
        let put_call_ratio_val = pcr_cur.advance(ts);

        oc_cur.advance(ts);
        liq_cur.advance(ts);
        mp_cur.advance(ts);
        macro_cur.advance(ts);

        close = closes[i];

        state.push(
            close, volume, high, low, open_,
            hour, dow,
            funding_rate, trades_val,
            taker_buy_volume, quote_volume, taker_buy_quote_volume,
            open_interest, ls_ratio, spot_close, fear_greed,
            implied_vol, put_call_ratio_val,
            oc_cur.flow_in, oc_cur.flow_out,
            oc_cur.supply, oc_cur.addr,
            oc_cur.tx, oc_cur.hashrate,
            // V11: Liquidation
            liq_cur.total_vol, liq_cur.buy_vol, liq_cur.sell_vol,
            if !liq_cur.total_vol.is_nan() { 1.0 } else { f64::NAN },
            // V11: Mempool
            mp_cur.fastest_fee, mp_cur.economy_fee, mp_cur.mempool_size,
            // V11: Macro
            macro_cur.dxy, macro_cur.spx, macro_cur.vix, macro_cur.day,
            // V11: Sentiment (no historical data in batch mode)
            f64::NAN, f64::NAN,
        );

        let mut row_out = [f64::NAN; N_FEATURES];
        state.get_features(&mut row_out);

        // Update prev_momentum (Python does this after get_features)
        state.prev_momentum = row_out[F_MA_CROSS_10_30];

        result.push(row_out.to_vec());
    }

    result
}

#[pyfunction]
pub fn cpp_feature_names() -> Vec<String> {
    FEATURE_NAMES.iter().map(|s| s.to_string()).collect()
}

// ============================================================
// RustFeatureEngine — Incremental feature computer for live trading
// ============================================================
// Wraps BarState as a PyO3 class. Holds all rolling state on the Rust heap.
// Call push_bar() once per bar, then get_features() to read the 105-feature vector.

#[pyclass(name = "RustFeatureEngine")]
pub struct RustFeatureEngine {
    pub(crate) state: BarState,
    pub(crate) cached_features: [f64; N_FEATURES],
    pub(crate) prev_momentum_val: f64,

    // V14 dominance state (used by push_dominance)
    dom_ratio_buf: Vec<f64>,
    dom_btc_ret_buf: Vec<f64>,
    dom_eth_ret_buf: Vec<f64>,
    dom_last_btc: f64,
    dom_last_eth: f64,
}

#[pymethods]
impl RustFeatureEngine {
    #[new]
    fn new() -> Self {
        Self {
            state: BarState::new(),
            cached_features: [f64::NAN; N_FEATURES],
            prev_momentum_val: f64::NAN,
            dom_ratio_buf: Vec::with_capacity(75),
            dom_btc_ret_buf: Vec::with_capacity(25),
            dom_eth_ret_buf: Vec::with_capacity(25),
            dom_last_btc: 0.0,
            dom_last_eth: 0.0,
        }
    }

    /// Push a new bar and update all rolling state.
    ///
    /// Args:
    ///   close, volume, high, low, open: OHLCV data
    ///   hour, dow: cyclical time features (-1 if unknown)
    ///   funding_rate: NaN if not available
    ///   trades: trade count (0 if unknown)
    ///   taker_buy_volume, quote_volume, taker_buy_quote_volume: microstructure
    ///   open_interest, ls_ratio, spot_close, fear_greed: NaN if unavailable
    ///   implied_vol, put_call_ratio: NaN if unavailable
    ///   All on-chain/liquidation/mempool/macro/social: NaN if unavailable
    #[pyo3(signature = (
        close, volume, high, low, open,
        hour=-1, dow=-1,
        funding_rate=f64::NAN,
        trades=0.0,
        taker_buy_volume=0.0,
        quote_volume=0.0,
        taker_buy_quote_volume=0.0,
        open_interest=f64::NAN,
        ls_ratio=f64::NAN,
        spot_close=f64::NAN,
        fear_greed=f64::NAN,
        implied_vol=f64::NAN,
        put_call_ratio=f64::NAN,
        oc_flow_in=f64::NAN, oc_flow_out=f64::NAN,
        oc_supply=f64::NAN, oc_addr=f64::NAN,
        oc_tx=f64::NAN, oc_hashrate=f64::NAN,
        liq_total_vol=f64::NAN, liq_buy_vol=f64::NAN, liq_sell_vol=f64::NAN,
        liq_count=f64::NAN,
        mempool_fastest_fee=f64::NAN, mempool_economy_fee=f64::NAN,
        mempool_size=f64::NAN,
        macro_dxy=f64::NAN, macro_spx=f64::NAN, macro_vix=f64::NAN,
        macro_day=-1_i64,
        social_volume=f64::NAN, sentiment_score=f64::NAN,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn push_bar(
        &mut self,
        close: f64, volume: f64, high: f64, low: f64, open: f64,
        hour: i32, dow: i32,
        funding_rate: f64,
        trades: f64,
        taker_buy_volume: f64,
        quote_volume: f64,
        taker_buy_quote_volume: f64,
        open_interest: f64,
        ls_ratio: f64,
        spot_close: f64,
        fear_greed: f64,
        implied_vol: f64,
        put_call_ratio: f64,
        oc_flow_in: f64, oc_flow_out: f64,
        oc_supply: f64, oc_addr: f64,
        oc_tx: f64, oc_hashrate: f64,
        liq_total_vol: f64, liq_buy_vol: f64, liq_sell_vol: f64,
        liq_count: f64,
        mempool_fastest_fee: f64, mempool_economy_fee: f64,
        mempool_size: f64,
        macro_dxy: f64, macro_spx: f64, macro_vix: f64,
        macro_day: i64,
        social_volume: f64, sentiment_score: f64,
    ) {
        self.state.push(
            close, volume, high, low, open,
            hour, dow,
            funding_rate, trades,
            taker_buy_volume, quote_volume, taker_buy_quote_volume,
            open_interest, ls_ratio, spot_close, fear_greed,
            implied_vol, put_call_ratio,
            oc_flow_in, oc_flow_out,
            oc_supply, oc_addr,
            oc_tx, oc_hashrate,
            liq_total_vol, liq_buy_vol, liq_sell_vol,
            liq_count,
            mempool_fastest_fee, mempool_economy_fee,
            mempool_size,
            macro_dxy, macro_spx, macro_vix,
            macro_day,
            social_volume, sentiment_score,
        );
        // Ensure BarState uses our tracked prev_momentum
        self.state.prev_momentum = self.prev_momentum_val;
        // Compute features and cache them (same sequence as batch mode)
        let mut out = [f64::NAN; N_FEATURES];
        self.state.get_features(&mut out);
        // Read momentum BEFORE caching (avoid optimizer issues)
        let new_mom = out[F_MA_CROSS_10_30];
        // Cache the feature output
        self.cached_features.copy_from_slice(&out);
        // Update prev_momentum AFTER get_features (same as batch mode)
        self.state.prev_momentum = new_mom;
        self.prev_momentum_val = new_mom;
    }

    /// Get the current 105-feature vector as a dict {name: value}.
    /// NaN values are converted to None. Returns cached features from last push_bar().
    #[pyo3(signature = ())]
    fn get_features(&self) -> std::collections::HashMap<String, Option<f64>> {
        let mut result = std::collections::HashMap::with_capacity(N_FEATURES);
        for (i, name) in FEATURE_NAMES.iter().enumerate() {
            let val = self.cached_features[i];
            result.insert(name.to_string(), if val.is_nan() { None } else { Some(val) });
        }
        result
    }

    /// Get features as a flat list of f64 (NaN for unavailable).
    /// Returns cached features from last push_bar().
    #[pyo3(signature = ())]
    fn get_features_array(&self) -> Vec<f64> {
        self.cached_features.to_vec()
    }

    /// Get feature names in order.
    #[pyo3(signature = ())]
    fn feature_names(&self) -> Vec<String> {
        FEATURE_NAMES.iter().map(|s| s.to_string()).collect()
    }

    /// Number of bars pushed so far.
    #[getter]
    fn bar_count(&self) -> i32 {
        self.state.bar_count
    }

    /// Whether warmup is complete (enough bars for all features).
    #[getter]
    fn warmed_up(&self) -> bool {
        self.state.bar_count >= 65  // _WARMUP_BARS
    }


    /// Serialize bar history to JSON for checkpoint persistence.
    ///
    /// Stores up to 720 raw bars (same format as TickProcessor.checkpoint_native).
    /// On restore, bars are replayed through push() to rebuild all rolling state.
    #[pyo3(signature = ())]
    fn checkpoint(&self) -> PyResult<String> {
        use serde_json::{json, Value};

        let bars = self.state.get_bar_history();
        let bars_json: Vec<Value> = bars.iter().map(|b| {
            json!({
                "c": b.close, "v": b.volume, "h": b.high,
                "l": b.low, "o": b.open,
                "hour": b.hour, "dow": b.dow,
                "fr": b.funding_rate, "trades": b.trades,
                "tbv": b.taker_buy_volume, "qv": b.quote_volume,
                "tbqv": b.taker_buy_quote_volume,
                "oi": b.open_interest, "ls": b.ls_ratio,
                "spot": b.spot_close, "fg": b.fear_greed,
                "iv": b.implied_vol, "pcr": b.put_call_ratio,
                "oc_fi": b.oc_flow_in, "oc_fo": b.oc_flow_out,
                "oc_s": b.oc_supply, "oc_a": b.oc_addr,
                "oc_t": b.oc_tx, "oc_h": b.oc_hashrate,
                "liq_tv": b.liq_total_vol, "liq_bv": b.liq_buy_vol,
                "liq_sv": b.liq_sell_vol, "liq_c": b.liq_count,
                "mp_ff": b.mempool_fastest_fee, "mp_ef": b.mempool_economy_fee,
                "mp_s": b.mempool_size,
                "m_dxy": b.macro_dxy, "m_spx": b.macro_spx,
                "m_vix": b.macro_vix, "m_day": b.macro_day,
                "sv": b.social_volume, "ss": b.sentiment_score,
            })
        }).collect();

        let checkpoint = json!({
            "version": 1,
            "bar_count": self.state.bar_count,
            "bars": bars_json,
        });

        serde_json::to_string(&checkpoint)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("Checkpoint serialize error: {}", e)
            ))
    }

    /// Restore from checkpoint JSON. Replays stored bars to rebuild all rolling state.
    ///
    /// Returns the number of bars replayed.
    #[pyo3(signature = (json_str))]
    fn restore_checkpoint(&mut self, json_str: &str) -> PyResult<usize> {
        let checkpoint: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("Checkpoint parse error: {}", e)
            ))?;

        let bars = checkpoint.get("bars")
            .and_then(|v| v.as_array())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No 'bars' in checkpoint"))?;

        let mut total_replayed = 0usize;

        for bar in bars {
            let close = bar.get("c").and_then(|v| v.as_f64()).unwrap_or(0.0);
            if close == 0.0 { continue; }

            self.state.push(
                close,
                bar.get("v").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("h").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("l").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("o").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("hour").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
                bar.get("dow").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
                bar.get("fr").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("trades").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("tbv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("qv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("tbqv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("oi").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("ls").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("spot").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("fg").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("iv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("pcr").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_fi").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_fo").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_s").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_a").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_t").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_h").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_tv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_bv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_sv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_c").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_ff").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_ef").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_s").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_dxy").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_spx").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_vix").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_day").and_then(|v| v.as_i64()).unwrap_or(-1),
                bar.get("sv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("ss").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
            );

            // Maintain prev_momentum tracking (same as push_bar)
            let mut out = [f64::NAN; N_FEATURES];
            self.state.prev_momentum = self.prev_momentum_val;
            self.state.get_features(&mut out);
            let new_mom = out[F_MA_CROSS_10_30];
            self.cached_features.copy_from_slice(&out);
            self.state.prev_momentum = new_mom;
            self.prev_momentum_val = new_mom;

            total_replayed += 1;
        }

        Ok(total_replayed)
    }

    fn __repr__(&self) -> String {
        format!("RustFeatureEngine(bar_count={}, warmed_up={})", self.state.bar_count, self.state.bar_count >= 65)
    }

    /// Push BTC and ETH closes to compute V14 dominance features.
    ///
    /// Returns a dict with 4 keys:
    ///   btc_dom_ratio_dev_20   — ratio deviation from 20-bar MA (None until 20 bars)
    ///   btc_dom_ratio_mom_10   — 10-bar ratio momentum          (None until 11 bars)
    ///   btc_dom_return_diff_6h  — BTC-ETH 6-bar return diff    (None until 6 return bars)
    ///   btc_dom_return_diff_24h — BTC-ETH 24-bar return diff   (None until 24 return bars)
    ///
    /// This method maintains its own independent state from push_bar().
    #[pyo3(signature = (btc_close, eth_close))]
    fn push_dominance(
        &mut self,
        btc_close: f64,
        eth_close: f64,
    ) -> std::collections::HashMap<String, Option<f64>> {
        let mut result = std::collections::HashMap::with_capacity(4);

        if eth_close <= 0.0 || btc_close <= 0.0 {
            result.insert("btc_dom_ratio_dev_20".to_string(), None);
            result.insert("btc_dom_ratio_mom_10".to_string(), None);
            result.insert("btc_dom_return_diff_6h".to_string(), None);
            result.insert("btc_dom_return_diff_24h".to_string(), None);
            return result;
        }

        let ratio = btc_close / eth_close;

        // Maintain circular buffer capped at 75
        if self.dom_ratio_buf.len() >= 75 {
            self.dom_ratio_buf.remove(0);
        }
        self.dom_ratio_buf.push(ratio);

        // Compute and store returns (only once we have a previous close)
        if self.dom_last_btc > 0.0 {
            let btc_ret = btc_close / self.dom_last_btc - 1.0;
            if self.dom_btc_ret_buf.len() >= 25 {
                self.dom_btc_ret_buf.remove(0);
            }
            self.dom_btc_ret_buf.push(btc_ret);
        }
        if self.dom_last_eth > 0.0 {
            let eth_ret = eth_close / self.dom_last_eth - 1.0;
            if self.dom_eth_ret_buf.len() >= 25 {
                self.dom_eth_ret_buf.remove(0);
            }
            self.dom_eth_ret_buf.push(eth_ret);
        }
        self.dom_last_btc = btc_close;
        self.dom_last_eth = eth_close;

        let n_ratio = self.dom_ratio_buf.len();

        // btc_dom_ratio_dev_20: ratio / MA(20) - 1
        if n_ratio >= 20 {
            let start = n_ratio - 20;
            let ma20: f64 = self.dom_ratio_buf[start..].iter().sum::<f64>() / 20.0;
            result.insert(
                "btc_dom_ratio_dev_20".to_string(),
                if ma20 > 0.0 { Some(ratio / ma20 - 1.0) } else { None },
            );
        } else {
            result.insert("btc_dom_ratio_dev_20".to_string(), None);
        }

        // btc_dom_ratio_mom_10: ratio / ratio[10 bars ago] - 1
        if n_ratio >= 11 {
            let prev = self.dom_ratio_buf[n_ratio - 11];
            result.insert(
                "btc_dom_ratio_mom_10".to_string(),
                if prev > 0.0 { Some(ratio / prev - 1.0) } else { None },
            );
        } else {
            result.insert("btc_dom_ratio_mom_10".to_string(), None);
        }

        // btc_dom_return_diff_6h: sum(btc_ret[-6:]) - sum(eth_ret[-6:])
        let n_btc = self.dom_btc_ret_buf.len();
        let n_eth = self.dom_eth_ret_buf.len();
        if n_btc >= 6 && n_eth >= 6 {
            let btc_sum: f64 = self.dom_btc_ret_buf[n_btc - 6..].iter().sum();
            let eth_sum: f64 = self.dom_eth_ret_buf[n_eth - 6..].iter().sum();
            result.insert("btc_dom_return_diff_6h".to_string(), Some(btc_sum - eth_sum));
        } else {
            result.insert("btc_dom_return_diff_6h".to_string(), None);
        }

        // btc_dom_return_diff_24h: sum(btc_ret[-24:]) - sum(eth_ret[-24:])
        if n_btc >= 24 && n_eth >= 24 {
            let btc_sum: f64 = self.dom_btc_ret_buf[n_btc - 24..].iter().sum();
            let eth_sum: f64 = self.dom_eth_ret_buf[n_eth - 24..].iter().sum();
            result.insert("btc_dom_return_diff_24h".to_string(), Some(btc_sum - eth_sum));
        } else {
            result.insert("btc_dom_return_diff_24h".to_string(), None);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine_with_bars(n: usize) -> BarState {
        let mut state = BarState::new();
        for i in 0..n {
            let price = 100.0 + (i as f64) * 0.1;
            state.push(
                price, 1000.0 + i as f64, price + 0.5, price - 0.5, price - 0.1,
                (i % 24) as i32, (i % 7) as i32,
                0.0001, 100.0, 500.0, 50000.0, 25000.0,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, -1,
                f64::NAN, f64::NAN,
            );
        }
        state
    }

    #[test]
    fn test_checkpoint_restore_features_match() {
        // Build original engine with 100 bars
        let mut original = BarState::new();
        let mut original_momentum = f64::NAN;
        for i in 0..100 {
            let price = 100.0 + (i as f64) * 0.1;
            original.push(
                price, 1000.0 + i as f64, price + 0.5, price - 0.5, price - 0.1,
                (i % 24) as i32, (i % 7) as i32,
                0.0001, 100.0, 500.0, 50000.0, 25000.0,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, -1,
                f64::NAN, f64::NAN,
            );
            original.prev_momentum = original_momentum;
            let mut out = [f64::NAN; N_FEATURES];
            original.get_features(&mut out);
            original_momentum = out[F_MA_CROSS_10_30];
            original.prev_momentum = original_momentum;
        }

        // Serialize bar history
        let bars = original.get_bar_history();
        let bars_json: Vec<serde_json::Value> = bars.iter().map(|b| {
            serde_json::json!({
                "c": b.close, "v": b.volume, "h": b.high, "l": b.low, "o": b.open,
                "hour": b.hour, "dow": b.dow,
                "fr": b.funding_rate, "trades": b.trades,
                "tbv": b.taker_buy_volume, "qv": b.quote_volume, "tbqv": b.taker_buy_quote_volume,
                "oi": b.open_interest, "ls": b.ls_ratio, "spot": b.spot_close, "fg": b.fear_greed,
                "iv": b.implied_vol, "pcr": b.put_call_ratio,
                "oc_fi": b.oc_flow_in, "oc_fo": b.oc_flow_out,
                "oc_s": b.oc_supply, "oc_a": b.oc_addr, "oc_t": b.oc_tx, "oc_h": b.oc_hashrate,
                "liq_tv": b.liq_total_vol, "liq_bv": b.liq_buy_vol, "liq_sv": b.liq_sell_vol, "liq_c": b.liq_count,
                "mp_ff": b.mempool_fastest_fee, "mp_ef": b.mempool_economy_fee, "mp_s": b.mempool_size,
                "m_dxy": b.macro_dxy, "m_spx": b.macro_spx, "m_vix": b.macro_vix, "m_day": b.macro_day,
                "sv": b.social_volume, "ss": b.sentiment_score,
            })
        }).collect();

        let checkpoint = serde_json::json!({
            "version": 1,
            "bar_count": original.bar_count,
            "bars": bars_json,
        });
        let json_str = serde_json::to_string(&checkpoint).unwrap();

        // Restore into fresh engine
        let mut restored = BarState::new();
        let mut restored_momentum = f64::NAN;
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let parsed_bars = parsed.get("bars").unwrap().as_array().unwrap();

        for bar in parsed_bars {
            restored.push(
                bar["c"].as_f64().unwrap(),
                bar["v"].as_f64().unwrap_or(0.0),
                bar["h"].as_f64().unwrap_or(0.0),
                bar["l"].as_f64().unwrap_or(0.0),
                bar["o"].as_f64().unwrap_or(0.0),
                bar["hour"].as_i64().unwrap_or(-1) as i32,
                bar["dow"].as_i64().unwrap_or(-1) as i32,
                bar.get("fr").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("trades").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("tbv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("qv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("tbqv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("oi").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("ls").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("spot").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("fg").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("iv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("pcr").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_fi").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_fo").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_s").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_a").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_t").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_h").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_tv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_bv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_sv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_c").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_ff").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_ef").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_s").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_dxy").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_spx").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_vix").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_day").and_then(|v| v.as_i64()).unwrap_or(-1),
                bar.get("sv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("ss").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
            );
            restored.prev_momentum = restored_momentum;
            let mut out = [f64::NAN; N_FEATURES];
            restored.get_features(&mut out);
            restored_momentum = out[F_MA_CROSS_10_30];
            restored.prev_momentum = restored_momentum;
        }

        assert_eq!(original.bar_count, restored.bar_count);

        // Push one more bar to both and compare features
        let next_price = 100.0 + 100.0 * 0.1;
        let push_one = |state: &mut BarState, mom: &mut f64| {
            state.push(
                next_price, 1100.0, next_price + 0.5, next_price - 0.5, next_price - 0.1,
                4, 3, 0.0001, 100.0, 500.0, 50000.0, 25000.0,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, -1,
                f64::NAN, f64::NAN,
            );
            state.prev_momentum = *mom;
            let mut out = [f64::NAN; N_FEATURES];
            state.get_features(&mut out);
            *mom = out[F_MA_CROSS_10_30];
            state.prev_momentum = *mom;
            out
        };

        let orig_features = push_one(&mut original, &mut original_momentum);
        let rest_features = push_one(&mut restored, &mut restored_momentum);

        for i in 0..N_FEATURES {
            let a = orig_features[i];
            let b = rest_features[i];
            if a.is_nan() && b.is_nan() { continue; }
            assert!(
                (a - b).abs() < 1e-10,
                "Feature {} ({}) mismatch: original={}, restored={}",
                i, FEATURE_NAMES[i], a, b,
            );
        }
    }

    #[test]
    fn test_checkpoint_bar_count() {
        let state = make_engine_with_bars(50);
        assert_eq!(state.bar_count, 50);
        assert_eq!(state.get_bar_history().len(), 50);
    }

    #[test]
    fn test_checkpoint_cap_720() {
        let state = make_engine_with_bars(800);
        assert_eq!(state.bar_count, 800);
        assert_eq!(state.get_bar_history().len(), 720);
    }
}
