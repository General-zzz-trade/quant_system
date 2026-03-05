#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "rolling_window.hpp"

namespace py = pybind11;

// ============================================================
// Feature name list — MUST match ENRICHED_FEATURE_NAMES order
// ============================================================
static const std::vector<std::string> FEATURE_NAMES = {
    // Returns
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    // MA crossovers
    "ma_cross_10_30", "ma_cross_5_20", "close_vs_ma20", "close_vs_ma50",
    // RSI
    "rsi_14", "rsi_6",
    // MACD
    "macd_line", "macd_signal", "macd_hist",
    // Bollinger Bands
    "bb_width_20", "bb_pctb_20",
    // ATR
    "atr_norm_14",
    // Volatility
    "vol_20", "vol_5",
    // Volume
    "vol_ratio_20", "vol_ma_ratio_5_20",
    // Candle structure
    "body_ratio", "upper_shadow", "lower_shadow",
    // Trend
    "mean_reversion_20", "price_acceleration",
    // Crypto-native time
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    // Volatility regime
    "vol_regime",
    // Funding rate
    "funding_rate", "funding_ma8",
    // Kline microstructure
    "trade_intensity", "taker_buy_ratio", "taker_buy_ratio_ma10",
    "taker_imbalance", "avg_trade_size", "avg_trade_size_ratio",
    "volume_per_trade", "trade_count_regime",
    // Funding deep
    "funding_zscore_24", "funding_momentum", "funding_extreme",
    "funding_cumulative_8", "funding_sign_persist",
    // OI features
    "oi_change_pct", "oi_change_ma8", "oi_close_divergence",
    // LS Ratio
    "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
    // V5: Order Flow
    "cvd_10", "cvd_20", "cvd_price_divergence", "aggressive_flow_zscore",
    // V5: Volatility microstructure
    "vol_of_vol", "range_vs_rv", "parkinson_vol", "rv_acceleration",
    // V5: Liquidation proxy
    "oi_acceleration", "leverage_proxy", "oi_vol_divergence", "oi_liquidation_flag",
    // V5: Funding carry
    "funding_annualized", "funding_vs_vol",
    // V7: Basis
    "basis", "basis_zscore_24", "basis_momentum", "basis_extreme",
    // V7: Fear & Greed
    "fgi_normalized", "fgi_zscore_7", "fgi_extreme",
    // V8: Alpha Rebuild V3
    "taker_bq_ratio", "vwap_dev_20", "volume_momentum_10",
    "mom_vol_divergence", "basis_carry_adj", "vol_regime_adaptive",
    // V9: Cross-factor interaction
    "liquidation_cascade_score", "funding_term_slope", "cross_tf_regime_sync",
    // V9: Deribit IV
    "implied_vol_zscore_24", "iv_rv_spread", "put_call_ratio",
    // V10: On-chain
    "exchange_netflow_zscore", "exchange_supply_change", "exchange_supply_zscore_30",
    "active_addr_zscore_14", "tx_count_zscore_14", "hashrate_momentum",
    // V11: Liquidation
    "liquidation_volume_zscore_24", "liquidation_imbalance",
    "liquidation_volume_ratio", "liquidation_cluster_flag",
    // V11: Mempool
    "mempool_fee_zscore_24", "mempool_size_zscore_24", "fee_urgency_ratio",
    // V11: Macro
    "dxy_change_5d", "spx_btc_corr_30d", "spx_overnight_ret", "vix_zscore_14",
    // V11: Sentiment
    "social_volume_zscore_24", "social_sentiment_score", "social_volume_price_div",
};

static constexpr int N_FEATURES = 105;  // FEATURE_NAMES.size()
static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
static constexpr double PI = 3.14159265358979323846;

// Feature index enum for readability
enum FIdx {
    F_ret_1 = 0, F_ret_3, F_ret_6, F_ret_12, F_ret_24,
    F_ma_cross_10_30, F_ma_cross_5_20, F_close_vs_ma20, F_close_vs_ma50,
    F_rsi_14, F_rsi_6,
    F_macd_line, F_macd_signal, F_macd_hist,
    F_bb_width_20, F_bb_pctb_20,
    F_atr_norm_14,
    F_vol_20, F_vol_5,
    F_vol_ratio_20, F_vol_ma_ratio_5_20,
    F_body_ratio, F_upper_shadow, F_lower_shadow,
    F_mean_reversion_20, F_price_acceleration,
    F_hour_sin, F_hour_cos, F_dow_sin, F_dow_cos,
    F_vol_regime,
    F_funding_rate, F_funding_ma8,
    F_trade_intensity, F_taker_buy_ratio, F_taker_buy_ratio_ma10,
    F_taker_imbalance, F_avg_trade_size, F_avg_trade_size_ratio,
    F_volume_per_trade, F_trade_count_regime,
    F_funding_zscore_24, F_funding_momentum, F_funding_extreme,
    F_funding_cumulative_8, F_funding_sign_persist,
    F_oi_change_pct, F_oi_change_ma8, F_oi_close_divergence,
    F_ls_ratio, F_ls_ratio_zscore_24, F_ls_extreme,
    F_cvd_10, F_cvd_20, F_cvd_price_divergence, F_aggressive_flow_zscore,
    F_vol_of_vol, F_range_vs_rv, F_parkinson_vol, F_rv_acceleration,
    F_oi_acceleration, F_leverage_proxy, F_oi_vol_divergence, F_oi_liquidation_flag,
    F_funding_annualized, F_funding_vs_vol,
    F_basis, F_basis_zscore_24, F_basis_momentum, F_basis_extreme,
    F_fgi_normalized, F_fgi_zscore_7, F_fgi_extreme,
    F_taker_bq_ratio, F_vwap_dev_20, F_volume_momentum_10,
    F_mom_vol_divergence, F_basis_carry_adj, F_vol_regime_adaptive,
    F_liquidation_cascade_score, F_funding_term_slope, F_cross_tf_regime_sync,
    F_implied_vol_zscore_24, F_iv_rv_spread, F_put_call_ratio,
    F_exchange_netflow_zscore, F_exchange_supply_change, F_exchange_supply_zscore_30,
    F_active_addr_zscore_14, F_tx_count_zscore_14, F_hashrate_momentum,
    // V11: Liquidation
    F_liquidation_volume_zscore_24, F_liquidation_imbalance,
    F_liquidation_volume_ratio, F_liquidation_cluster_flag,
    // V11: Mempool
    F_mempool_fee_zscore_24, F_mempool_size_zscore_24, F_fee_urgency_ratio,
    // V11: Macro
    F_dxy_change_5d, F_spx_btc_corr_30d, F_spx_overnight_ret, F_vix_zscore_14,
    // V11: Sentiment
    F_social_volume_zscore_24, F_social_sentiment_score, F_social_volume_price_div,
};

// ============================================================
// Incremental tracker structs — matching Python exactly
// ============================================================

struct EMAState {
    double alpha;
    double value = 0.0;
    int n = 0;

    explicit EMAState(int span) : alpha(2.0 / (span + 1.0)) {}
    EMAState() : alpha(0.0) {}

    void push(double x) noexcept {
        if (n == 0) {
            value = x;
        } else {
            value = alpha * x + (1.0 - alpha) * value;
        }
        ++n;
    }

    bool ready(int span) const noexcept { return n >= span; }
    double get_value() const noexcept { return n > 0 ? value : NaN; }
};

struct RSIState {
    int period;
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    int n = 0;
    double prev_close = NaN;
    double init_gains = 0.0;
    double init_losses = 0.0;

    explicit RSIState(int p) : period(p) {}
    RSIState() : period(14) {}

    void push(double close) noexcept {
        if (std::isnan(prev_close)) {
            prev_close = close;
            return;
        }
        double change = close - prev_close;
        prev_close = close;
        double gain = change > 0 ? change : 0.0;
        double loss = change < 0 ? -change : 0.0;
        ++n;

        if (n <= period) {
            init_gains += gain;
            init_losses += loss;
            if (n == period) {
                avg_gain = init_gains / period;
                avg_loss = init_losses / period;
            }
        } else {
            avg_gain = (avg_gain * (period - 1) + gain) / period;
            avg_loss = (avg_loss * (period - 1) + loss) / period;
        }
    }

    double get_value() const noexcept {
        if (n < period) return NaN;
        if (avg_loss == 0.0) return 100.0;
        double rs = avg_gain / avg_loss;
        return 100.0 - (100.0 / (1.0 + rs));
    }
};

struct ATRState {
    int period;
    double atr = 0.0;
    int n = 0;
    double prev_close = NaN;
    double init_sum = 0.0;

    explicit ATRState(int p) : period(p) {}
    ATRState() : period(14) {}

    void push(double high, double low, double close) noexcept {
        double tr;
        if (std::isnan(prev_close)) {
            tr = high - low;
        } else {
            tr = std::max({high - low,
                           std::abs(high - prev_close),
                           std::abs(low - prev_close)});
        }
        prev_close = close;
        ++n;

        if (n <= period) {
            init_sum += tr;
            if (n == period) {
                atr = init_sum / period;
            }
        } else {
            atr = (atr * (period - 1) + tr) / period;
        }
    }

    double get_value() const noexcept {
        return n >= period ? atr : NaN;
    }
};

// Fixed-size circular buffer (replaces Python deque(maxlen=N))
template<int MaxSize>
struct CircBuf {
    double buf[MaxSize];
    int count = 0;
    int head = 0;  // index of oldest element

    void push(double x) noexcept {
        if (count < MaxSize) {
            buf[count] = x;
            ++count;
        } else {
            buf[head] = x;
            head = (head + 1) % MaxSize;
        }
    }

    int size() const noexcept { return count; }
    bool full() const noexcept { return count == MaxSize; }

    // Access by logical index: 0 = oldest, count-1 = newest
    double operator[](int i) const noexcept {
        return buf[(head + i) % MaxSize];
    }

    // Get the last (newest) element
    double back() const noexcept {
        if (count < MaxSize) {
            return buf[count - 1];
        }
        return buf[(head + MaxSize - 1) % MaxSize];
    }

    // Get element from back: back_n(0) = newest, back_n(1) = 2nd newest
    double back_n(int n) const noexcept {
        if (count < MaxSize) {
            return buf[count - 1 - n];
        }
        return buf[(head + MaxSize - 1 - n) % MaxSize];
    }

    double sum() const noexcept {
        double s = 0.0;
        for (int i = 0; i < count; ++i) s += buf[(head + i) % MaxSize];
        return s;
    }
};

// Z-score helper for small buffers
inline double zscore_buf(const double* data, int count, double last_val, double min_std) {
    if (count == 0) return NaN;
    double sum = 0.0;
    for (int i = 0; i < count; ++i) sum += data[i];
    double mean = sum / count;
    double var = 0.0;
    for (int i = 0; i < count; ++i) {
        double d = data[i] - mean;
        var += d * d;
    }
    var /= count;
    double std = std::sqrt(var);
    if (std <= min_std) return 0.0;
    return (last_val - mean) / std;
}

// ============================================================
// BarState — per-symbol aggregated state
// ============================================================

struct BarState {
    // Close history (30 bars)
    CircBuf<30> close_history;
    CircBuf<2> open_history;
    CircBuf<2> high_history;
    CircBuf<2> low_history;

    // Moving averages
    RollingWindow ma_5{5}, ma_10{10}, ma_20{20}, ma_30{30}, ma_50{50};
    RollingWindow bb_window{20};

    // Return windows
    RollingWindow return_window_20{20}, return_window_5{5};

    // Volume windows
    RollingWindow vol_window_20{20}, vol_window_5{5};

    // RSI
    RSIState rsi_14{14}, rsi_6{6};

    // MACD EMAs
    EMAState ema_12, ema_26, macd_signal_ema;

    // ATR
    ATRState atr_14{14};

    // Funding
    EMAState funding_ema;
    RollingWindow funding_window_24{24};
    CircBuf<8> funding_history_8;
    int funding_sign_count = 0;
    int funding_last_sign = 0;

    // Microstructure
    EMAState trades_ema_20, trades_ema_5;
    EMAState taker_buy_ratio_ema_10;
    EMAState avg_trade_size_ema_20;
    EMAState volume_per_trade_ema_20;

    // OI
    EMAState oi_change_ema_8;
    double last_oi = NaN;
    double last_oi_change_pct = NaN;

    // LS Ratio
    RollingWindow ls_ratio_window_24{24};
    double last_ls_ratio = NaN;

    // V5: Order flow
    RollingWindow cvd_window_10{10}, cvd_window_20{20};
    RollingWindow taker_ratio_window_50{50};

    // V5: Volatility microstructure
    CircBuf<25> vol_5_history;
    RollingWindow hl_log_sq_window{20};

    // V5: Liquidation proxy
    EMAState leverage_proxy_ema;
    double prev_oi_change_for_accel = NaN;

    // V7: Basis
    RollingWindow basis_window_24{24};
    EMAState basis_ema_8;
    double last_basis = NaN;

    // V7: FGI
    RollingWindow fgi_window_7{7};
    double last_fgi = NaN;

    // V8: VWAP
    RollingWindow vwap_cv_window{20}, vwap_v_window{20};

    // V8: Adaptive vol regime
    EMAState vol_regime_ema;
    CircBuf<30> vol_regime_history;

    // V9: Deribit IV
    RollingWindow iv_window_24{24};
    double last_implied_vol = NaN;
    double last_put_call_ratio = NaN;

    // V10: On-chain
    CircBuf<7> onchain_netflow_buf;
    CircBuf<30> onchain_supply_buf;
    CircBuf<14> onchain_addr_buf;
    CircBuf<14> onchain_tx_buf;
    EMAState onchain_hashrate_ema;
    double last_onchain_supply = NaN;
    double last_onchain_hashrate = NaN;

    // V11: Liquidation
    CircBuf<24> liq_volume_buf;
    CircBuf<6> liq_imbalance_buf;
    double last_liq_volume = NaN;
    double last_liq_imbalance = NaN;
    double last_liq_count = 0.0;

    // V11: Mempool
    CircBuf<24> mempool_fee_buf;
    CircBuf<24> mempool_size_buf;
    double last_fee_urgency = NaN;

    // V11: Macro
    CircBuf<10> dxy_buf;
    CircBuf<30> spx_buf;
    CircBuf<30> btc_close_buf_30;
    double last_spx_close = NaN;
    double prev_spx_close = NaN;
    double last_vix = NaN;
    CircBuf<14> vix_buf;
    int64_t last_macro_day = -1;  // day index for date dedup

    // V11: Sentiment
    CircBuf<24> social_vol_buf;
    double last_sentiment_score = NaN;
    double last_social_volume = NaN;

    // Scalar state
    double prev_momentum = NaN;
    double last_close = NaN;
    double last_volume = 0.0;
    int last_hour = -1;
    int last_dow = -1;
    double last_funding_rate = NaN;
    double last_trades = 0.0;
    double last_taker_buy_volume = 0.0;
    double last_taker_buy_quote_volume = 0.0;
    double last_quote_volume = 0.0;
    int bar_count = 0;

    BarState() :
        ema_12(EMAState(12)),
        ema_26(EMAState(26)),
        macd_signal_ema(EMAState(9)),
        funding_ema(EMAState(8)),
        trades_ema_20(EMAState(20)),
        trades_ema_5(EMAState(5)),
        taker_buy_ratio_ema_10(EMAState(10)),
        avg_trade_size_ema_20(EMAState(20)),
        volume_per_trade_ema_20(EMAState(20)),
        oi_change_ema_8(EMAState(8)),
        leverage_proxy_ema(EMAState(20)),
        basis_ema_8(EMAState(8)),
        vol_regime_ema(EMAState(5)),
        onchain_hashrate_ema(EMAState(14))
    {}

    // -----------------------------------------------------------
    // push() — exactly matches Python _SymbolState.push()
    // -----------------------------------------------------------
    void push(double close, double volume, double high, double low, double open_,
              int hour, int dow,
              double funding_rate,   // NaN = not provided
              double trades,
              double taker_buy_volume,
              double quote_volume,
              double taker_buy_quote_volume,
              double open_interest,  // NaN = not provided
              double ls_ratio,       // NaN = not provided
              double spot_close,     // NaN = not provided
              double fear_greed,     // NaN = not provided
              double implied_vol,    // NaN = not provided
              double put_call_ratio_val, // NaN = not provided
              // On-chain: 6 values, all NaN if not provided
              double oc_flow_in, double oc_flow_out,
              double oc_supply, double oc_addr,
              double oc_tx, double oc_hashrate,
              // V11: Liquidation (4 values, all NaN if not provided)
              double liq_total_vol, double liq_buy_vol, double liq_sell_vol,
              double liq_count,
              // V11: Mempool (3 values, all NaN if not provided)
              double mempool_fastest_fee, double mempool_economy_fee,
              double mempool_size,
              // V11: Macro (3 values + day index, all NaN if not provided)
              double macro_dxy, double macro_spx, double macro_vix,
              int64_t macro_day,   // -1 = not provided
              // V11: Sentiment (2 values, all NaN if not provided)
              double social_volume, double sentiment_score) noexcept
    {
        last_hour = hour;
        last_dow = dow;

        // --- Funding ---
        if (!std::isnan(funding_rate)) {
            last_funding_rate = funding_rate;
            funding_ema.push(funding_rate);
            funding_window_24.push(funding_rate);
            funding_history_8.push(funding_rate);
            int sign = (funding_rate > 0) ? 1 : ((funding_rate < 0) ? -1 : 0);
            if (sign != 0) {
                if (sign == funding_last_sign) {
                    funding_sign_count += 1;
                } else {
                    funding_sign_count = 1;
                    funding_last_sign = sign;
                }
            }
        }

        // --- OI ---
        if (!std::isnan(open_interest)) {
            if (!std::isnan(last_oi) && last_oi > 0) {
                double change = (open_interest - last_oi) / last_oi;
                prev_oi_change_for_accel = last_oi_change_pct;
                last_oi_change_pct = change;
                oi_change_ema_8.push(change);
            }
            last_oi = open_interest;
            if (close > 0 && volume > 0) {
                double raw_lev = open_interest / (close * volume);
                leverage_proxy_ema.push(raw_lev);
            }
        }

        // --- LS Ratio ---
        if (!std::isnan(ls_ratio)) {
            last_ls_ratio = ls_ratio;
            ls_ratio_window_24.push(ls_ratio);
        }

        // --- Basis ---
        if (!std::isnan(spot_close) && close > 0 && spot_close > 0) {
            double basis = (close - spot_close) / spot_close;
            last_basis = basis;
            basis_window_24.push(basis);
            basis_ema_8.push(basis);
        }

        // --- FGI ---
        if (!std::isnan(fear_greed)) {
            if (std::isnan(last_fgi) || std::abs(fear_greed - last_fgi) > 0.01) {
                fgi_window_7.push(fear_greed);
            }
            last_fgi = fear_greed;
        }

        // --- Deribit IV ---
        if (!std::isnan(implied_vol)) {
            last_implied_vol = implied_vol;
            iv_window_24.push(implied_vol);
        }
        if (!std::isnan(put_call_ratio_val)) {
            last_put_call_ratio = put_call_ratio_val;
        }

        // --- On-chain ---
        if (!std::isnan(oc_flow_in) && !std::isnan(oc_flow_out)) {
            onchain_netflow_buf.push(oc_flow_in - oc_flow_out);
        }
        if (!std::isnan(oc_supply)) {
            onchain_supply_buf.push(oc_supply);
            last_onchain_supply = oc_supply;
        }
        if (!std::isnan(oc_addr)) {
            onchain_addr_buf.push(oc_addr);
        }
        if (!std::isnan(oc_tx)) {
            onchain_tx_buf.push(oc_tx);
        }
        if (!std::isnan(oc_hashrate)) {
            onchain_hashrate_ema.push(oc_hashrate);
            last_onchain_hashrate = oc_hashrate;
        }

        // --- V11: Liquidation ---
        if (!std::isnan(liq_total_vol)) {
            liq_volume_buf.push(liq_total_vol);
            last_liq_volume = liq_total_vol;
            last_liq_count = std::isnan(liq_count) ? 0.0 : liq_count;
            double imb = 0.0;
            if (liq_total_vol > 0 && !std::isnan(liq_buy_vol) && !std::isnan(liq_sell_vol)) {
                imb = (liq_buy_vol - liq_sell_vol) / liq_total_vol;
            }
            liq_imbalance_buf.push(imb);
            last_liq_imbalance = imb;
        }

        // --- V11: Mempool ---
        if (!std::isnan(mempool_fastest_fee)) {
            mempool_fee_buf.push(mempool_fastest_fee);
        }
        if (!std::isnan(mempool_size)) {
            mempool_size_buf.push(mempool_size);
        }
        if (!std::isnan(mempool_fastest_fee) && !std::isnan(mempool_economy_fee) && mempool_economy_fee > 0) {
            last_fee_urgency = mempool_fastest_fee / mempool_economy_fee;
        }

        // --- V11: Macro (daily — only push when day changes) ---
        if (macro_day >= 0 && macro_day != last_macro_day) {
            last_macro_day = macro_day;
            if (!std::isnan(macro_dxy)) {
                dxy_buf.push(macro_dxy);
            }
            if (!std::isnan(macro_spx)) {
                prev_spx_close = last_spx_close;
                last_spx_close = macro_spx;
                spx_buf.push(macro_spx);
            }
            if (!std::isnan(macro_vix)) {
                last_vix = macro_vix;
                vix_buf.push(macro_vix);
            }
        }
        // Always track BTC close for SPX-BTC correlation (when macro data exists)
        if (macro_day >= 0) {
            btc_close_buf_30.push(close);
        }

        // --- V11: Sentiment ---
        if (!std::isnan(social_volume)) {
            social_vol_buf.push(social_volume);
            last_social_volume = social_volume;
        }
        if (!std::isnan(sentiment_score)) {
            last_sentiment_score = sentiment_score;
        }

        // --- Microstructure state ---
        last_trades = trades;
        last_taker_buy_volume = taker_buy_volume;
        last_taker_buy_quote_volume = taker_buy_quote_volume;
        last_quote_volume = quote_volume;

        // --- VWAP windows ---
        if (volume > 0) {
            vwap_cv_window.push(close * volume);
            vwap_v_window.push(volume);
        }
        if (trades > 0) {
            trades_ema_20.push(trades);
            trades_ema_5.push(trades);
            double tbr = (volume > 0) ? (taker_buy_volume / volume) : 0.5;
            taker_buy_ratio_ema_10.push(tbr);
            double imbalance = 2.0 * tbr - 1.0;
            cvd_window_10.push(imbalance);
            cvd_window_20.push(imbalance);
            taker_ratio_window_50.push(tbr);
            double ats = quote_volume / trades;
            avg_trade_size_ema_20.push(ats);
            double vpt = volume / trades;
            volume_per_trade_ema_20.push(vpt);
        }

        bar_count += 1;
        close_history.push(close);
        open_history.push(open_);
        high_history.push(high);
        low_history.push(low);

        // --- MAs ---
        ma_5.push(close);
        ma_10.push(close);
        ma_20.push(close);
        ma_30.push(close);
        ma_50.push(close);
        bb_window.push(close);

        // --- Returns ---
        if (!std::isnan(last_close) && last_close != 0) {
            double ret = (close - last_close) / last_close;
            return_window_20.push(ret);
            return_window_5.push(ret);
        }
        last_close = close;

        // V5: vol_5 history
        if (return_window_5.full()) {
            auto v5_std = return_window_5.std_dev();
            if (v5_std) vol_5_history.push(*v5_std);
        }

        // V8: Adaptive vol regime
        if (return_window_5.full() && return_window_20.full()) {
            auto v5_std = return_window_5.std_dev();
            auto v20_std = return_window_20.std_dev();
            if (v5_std && v20_std && *v20_std > 1e-12) {
                double vr = *v5_std / *v20_std;
                vol_regime_ema.push(vr);
                vol_regime_history.push(vr);
            }
        }

        // V5: Parkinson volatility
        if (high > 0 && low > 0 && high >= low) {
            double hl_ratio = high / low;
            if (hl_ratio > 0) {
                double ln_hl = std::log(hl_ratio);
                hl_log_sq_window.push(ln_hl * ln_hl);
            }
        }

        // Volume
        last_volume = volume;
        vol_window_20.push(volume);
        vol_window_5.push(volume);

        // RSI
        rsi_14.push(close);
        rsi_6.push(close);

        // MACD
        ema_12.push(close);
        ema_26.push(close);
        if (ema_12.ready(12) && ema_26.ready(26)) {
            double macd_val = ema_12.value - ema_26.value;
            macd_signal_ema.push(macd_val);
        }

        // ATR
        atr_14.push(high, low, close);
    }

    // -----------------------------------------------------------
    // get_features() — exactly matches Python _SymbolState.get_features()
    // Writes N_FEATURES doubles to out[]
    // -----------------------------------------------------------
    void get_features(double* out) const noexcept {
        // Initialize all to NaN
        for (int i = 0; i < N_FEATURES; ++i) out[i] = NaN;

        double close = last_close;
        int n = close_history.size();

        // --- Multi-horizon returns ---
        const int horizons[] = {1, 3, 6, 12, 24};
        const int horizon_idx[] = {F_ret_1, F_ret_3, F_ret_6, F_ret_12, F_ret_24};
        for (int k = 0; k < 5; ++k) {
            int h = horizons[k];
            if (n > h) {
                double past = close_history.back_n(h);
                if (past != 0) {
                    out[horizon_idx[k]] = (close_history.back() - past) / past;
                }
            }
        }

        // --- MA crossovers ---
        double ma10v = ma_10.full() ? *ma_10.mean() : NaN;
        double ma30v = ma_30.full() ? *ma_30.mean() : NaN;
        double ma5v = ma_5.full() ? *ma_5.mean() : NaN;
        double ma20v = ma_20.full() ? *ma_20.mean() : NaN;
        double ma50v = ma_50.full() ? *ma_50.mean() : NaN;

        if (!std::isnan(ma10v) && !std::isnan(ma30v) && ma30v != 0)
            out[F_ma_cross_10_30] = ma10v / ma30v - 1.0;
        if (!std::isnan(ma5v) && !std::isnan(ma20v) && ma20v != 0)
            out[F_ma_cross_5_20] = ma5v / ma20v - 1.0;
        if (!std::isnan(close) && !std::isnan(ma20v) && ma20v != 0)
            out[F_close_vs_ma20] = close / ma20v - 1.0;
        if (!std::isnan(close) && !std::isnan(ma50v) && ma50v != 0)
            out[F_close_vs_ma50] = close / ma50v - 1.0;

        // --- RSI ---
        double rsi14_val = rsi_14.get_value();
        double rsi6_val = rsi_6.get_value();
        if (!std::isnan(rsi14_val)) out[F_rsi_14] = (rsi14_val - 50.0) / 50.0;
        if (!std::isnan(rsi6_val)) out[F_rsi_6] = (rsi6_val - 50.0) / 50.0;

        // --- MACD ---
        if (ema_12.ready(12) && ema_26.ready(26)) {
            double macd_line = ema_12.value - ema_26.value;
            if (!std::isnan(close) && close != 0) {
                out[F_macd_line] = macd_line / close;
                if (macd_signal_ema.ready(9)) {
                    double sig = macd_signal_ema.value;
                    out[F_macd_signal] = sig / close;
                    out[F_macd_hist] = (macd_line - sig) / close;
                }
            }
        }

        // --- Bollinger Bands ---
        if (bb_window.full()) {
            double bb_mid = *bb_window.mean();
            double bb_std = *bb_window.std_dev();
            if (bb_mid != 0 && bb_std != 0) {
                double upper = bb_mid + 2.0 * bb_std;
                double lower = bb_mid - 2.0 * bb_std;
                out[F_bb_width_20] = (upper - lower) / bb_mid;
                double band_range = upper - lower;
                if (band_range != 0 && !std::isnan(close)) {
                    out[F_bb_pctb_20] = (close - lower) / band_range;
                }
            }
        }

        // --- ATR ---
        double atr_val = atr_14.get_value();
        if (!std::isnan(atr_val) && !std::isnan(close) && close != 0)
            out[F_atr_norm_14] = atr_val / close;

        // --- Volatility ---
        double vol20_v = NaN, vol5_v = NaN;
        if (return_window_20.full()) { vol20_v = *return_window_20.std_dev(); out[F_vol_20] = vol20_v; }
        if (return_window_5.full()) { vol5_v = *return_window_5.std_dev(); out[F_vol_5] = vol5_v; }

        // --- Volume features ---
        double vol_ma20 = NaN, vol_ma5 = NaN;
        if (vol_window_20.full()) vol_ma20 = *vol_window_20.mean();
        if (vol_window_5.full()) vol_ma5 = *vol_window_5.mean();

        if (!std::isnan(vol_ma20) && vol_ma20 != 0 && vol_window_20.n() > 0)
            out[F_vol_ratio_20] = last_volume / vol_ma20;
        if (!std::isnan(vol_ma5) && !std::isnan(vol_ma20) && vol_ma20 != 0)
            out[F_vol_ma_ratio_5_20] = vol_ma5 / vol_ma20;

        // --- Candle structure ---
        if (n > 0 && open_history.size() > 0 && high_history.size() > 0 && low_history.size() > 0) {
            double o = open_history.back();
            double h = high_history.back();
            double l = low_history.back();
            double c = close_history.back();
            double hl_range = h - l;
            if (hl_range > 0) {
                out[F_body_ratio] = (c - o) / hl_range;
                out[F_upper_shadow] = (h - std::max(o, c)) / hl_range;
                out[F_lower_shadow] = (std::min(o, c) - l) / hl_range;
            }
        }

        // --- Mean reversion ---
        if (bb_window.full() && !std::isnan(close)) {
            double bb_mid = *bb_window.mean();
            double bb_std = *bb_window.std_dev();
            if (bb_std != 0) {
                out[F_mean_reversion_20] = (close - bb_mid) / bb_std;
            }
        }

        // --- Price acceleration ---
        double current_momentum = out[F_ma_cross_10_30];
        if (!std::isnan(current_momentum) && !std::isnan(prev_momentum)) {
            out[F_price_acceleration] = current_momentum - prev_momentum;
        }
        // NOTE: prev_momentum is updated after get_features() in the batch loop

        // --- Time ---
        if (last_hour >= 0) {
            out[F_hour_sin] = std::sin(2.0 * PI * last_hour / 24.0);
            out[F_hour_cos] = std::cos(2.0 * PI * last_hour / 24.0);
        }
        if (last_dow >= 0) {
            out[F_dow_sin] = std::sin(2.0 * PI * last_dow / 7.0);
            out[F_dow_cos] = std::cos(2.0 * PI * last_dow / 7.0);
        }

        // --- Vol regime ---
        if (!std::isnan(vol5_v) && !std::isnan(vol20_v) && vol20_v != 0)
            out[F_vol_regime] = vol5_v / vol20_v;

        // --- Funding rate ---
        out[F_funding_rate] = last_funding_rate;
        out[F_funding_ma8] = funding_ema.ready(8) ? funding_ema.value : NaN;

        // --- Kline microstructure ---
        double trades_val = last_trades;
        double volume_val = last_volume;
        if (trades_val > 0 && trades_ema_20.ready(20)) {
            double ema_t20 = trades_ema_20.value;
            if (ema_t20 > 0) out[F_trade_intensity] = trades_val / ema_t20;
        }

        double tbr = NaN;
        if (trades_val > 0 && volume_val > 0) {
            tbr = last_taker_buy_volume / volume_val;
            out[F_taker_buy_ratio] = tbr;
        }

        if (taker_buy_ratio_ema_10.ready(10))
            out[F_taker_buy_ratio_ma10] = taker_buy_ratio_ema_10.value;

        if (!std::isnan(tbr))
            out[F_taker_imbalance] = 2.0 * tbr - 1.0;

        if (trades_val > 0) {
            double ats = last_quote_volume / trades_val;
            out[F_avg_trade_size] = ats;
            if (avg_trade_size_ema_20.ready(20)) {
                double ats_ema = avg_trade_size_ema_20.value;
                if (ats_ema > 0) out[F_avg_trade_size_ratio] = ats / ats_ema;
            }
            double vpt = volume_val / trades_val;
            if (volume_per_trade_ema_20.ready(20)) {
                double vpt_ema = volume_per_trade_ema_20.value;
                if (vpt_ema > 0) out[F_volume_per_trade] = vpt / vpt_ema;
            }
        }

        if (trades_ema_5.ready(5) && trades_ema_20.ready(20)) {
            double e5 = trades_ema_5.value;
            double e20 = trades_ema_20.value;
            if (e20 > 0) out[F_trade_count_regime] = e5 / e20;
        }

        // --- Funding deep ---
        if (funding_window_24.full()) {
            double f_mean = *funding_window_24.mean();
            double f_std = *funding_window_24.std_dev();
            if (f_std > 1e-12 && !std::isnan(last_funding_rate)) {
                double zscore = (last_funding_rate - f_mean) / f_std;
                out[F_funding_zscore_24] = zscore;
                out[F_funding_extreme] = (std::abs(zscore) > 2.0) ? 1.0 : 0.0;
            }
        }

        double fr_ma8 = out[F_funding_ma8];
        if (!std::isnan(last_funding_rate) && !std::isnan(fr_ma8))
            out[F_funding_momentum] = last_funding_rate - fr_ma8;

        if (funding_history_8.size() == 8)
            out[F_funding_cumulative_8] = funding_history_8.sum();

        out[F_funding_sign_persist] = (funding_sign_count > 0) ? (double)funding_sign_count : NaN;

        // --- OI features ---
        out[F_oi_change_pct] = last_oi_change_pct;
        out[F_oi_change_ma8] = oi_change_ema_8.ready(8) ? oi_change_ema_8.value : NaN;

        double ret1 = out[F_ret_1];
        if (!std::isnan(ret1) && !std::isnan(last_oi_change_pct)) {
            double price_sign = (ret1 > 0) ? 1.0 : ((ret1 < 0) ? -1.0 : 0.0);
            double oi_sign = (last_oi_change_pct > 0) ? 1.0 : ((last_oi_change_pct < 0) ? -1.0 : 0.0);
            out[F_oi_close_divergence] = -price_sign * oi_sign;
        }

        // --- LS Ratio ---
        out[F_ls_ratio] = last_ls_ratio;
        if (ls_ratio_window_24.full() && !std::isnan(last_ls_ratio)) {
            double ls_mean = *ls_ratio_window_24.mean();
            double ls_std = *ls_ratio_window_24.std_dev();
            if (ls_std > 1e-12) {
                double zscore = (last_ls_ratio - ls_mean) / ls_std;
                out[F_ls_ratio_zscore_24] = zscore;
                out[F_ls_extreme] = (std::abs(zscore) > 2.0) ? 1.0 : 0.0;
            }
        }

        // --- V5: Order Flow ---
        if (cvd_window_10.full())
            out[F_cvd_10] = *cvd_window_10.mean() * cvd_window_10.n();

        if (cvd_window_20.full()) {
            double cvd_20_val = *cvd_window_20.mean() * cvd_window_20.n();
            out[F_cvd_20] = cvd_20_val;
            if (n > 20) {
                double past20 = close_history.back_n(20);
                if (past20 != 0) {
                    double ret_20 = (close_history.back() - past20) / past20;
                    double cvd_sign = (cvd_20_val > 0) ? 1.0 : ((cvd_20_val < 0) ? -1.0 : 0.0);
                    double ret_sign_v = (ret_20 > 0) ? 1.0 : ((ret_20 < 0) ? -1.0 : 0.0);
                    out[F_cvd_price_divergence] = (cvd_sign != 0 && cvd_sign != ret_sign_v) ? 1.0 : 0.0;
                }
            }
        }

        if (taker_ratio_window_50.full() && !std::isnan(tbr)) {
            double tr_mean = *taker_ratio_window_50.mean();
            double tr_std = *taker_ratio_window_50.std_dev();
            if (tr_std > 1e-12) {
                out[F_aggressive_flow_zscore] = (tbr - tr_mean) / tr_std;
            }
        }

        // --- V5: Volatility microstructure ---
        if (vol_5_history.size() >= 20) {
            // Compute mean and variance of last 20 entries
            double sum_v = 0.0, sumsq_v = 0.0;
            int cnt = 20;
            int start = vol_5_history.size() - cnt;
            for (int i = start; i < vol_5_history.size(); ++i) {
                double v = vol_5_history[i];
                sum_v += v;
            }
            double mean_v = sum_v / cnt;
            for (int i = start; i < vol_5_history.size(); ++i) {
                double d = vol_5_history[i] - mean_v;
                sumsq_v += d * d;
            }
            out[F_vol_of_vol] = std::sqrt(sumsq_v / cnt);
        }

        if (n > 0 && !std::isnan(close) && close != 0 && !std::isnan(vol5_v) && vol5_v > 1e-12) {
            double h = (high_history.size() > 0) ? high_history.back() : close;
            double l = (low_history.size() > 0) ? low_history.back() : close;
            out[F_range_vs_rv] = ((h - l) / close) / vol5_v;
        }

        if (hl_log_sq_window.full()) {
            double mean_sq = *hl_log_sq_window.mean();
            if (mean_sq >= 0) {
                out[F_parkinson_vol] = std::sqrt(mean_sq / (4.0 * std::log(2.0)));
            }
        }

        if (vol_5_history.size() >= 6) {
            out[F_rv_acceleration] = vol_5_history.back_n(0) - vol_5_history.back_n(5);
        }

        // --- V5: Liquidation proxy ---
        if (!std::isnan(last_oi_change_pct) && !std::isnan(prev_oi_change_for_accel))
            out[F_oi_acceleration] = last_oi_change_pct - prev_oi_change_for_accel;

        if (!std::isnan(last_oi) && !std::isnan(close) && close > 0 && last_volume > 0) {
            double raw_lev = last_oi / (close * last_volume);
            if (leverage_proxy_ema.ready(20)) {
                double lev_ema = leverage_proxy_ema.value;
                if (lev_ema > 0) out[F_leverage_proxy] = raw_lev / lev_ema;
            }
        }

        // oi_vol_divergence
        double oi_chg = last_oi_change_pct;
        double vol_r = out[F_vol_ratio_20];
        if (!std::isnan(oi_chg) && !std::isnan(vol_r))
            out[F_oi_vol_divergence] = (oi_chg > 0 && vol_r < 1.0) ? 1.0 : 0.0;

        // oi_liquidation_flag
        if (!std::isnan(oi_chg) && !std::isnan(vol_r))
            out[F_oi_liquidation_flag] = (oi_chg < -0.05 && vol_r > 2.0) ? 1.0 : 0.0;

        // --- V5: Funding carry ---
        if (!std::isnan(last_funding_rate))
            out[F_funding_annualized] = last_funding_rate * 3.0 * 365.0;

        if (!std::isnan(last_funding_rate) && !std::isnan(vol20_v) && vol20_v > 1e-12)
            out[F_funding_vs_vol] = last_funding_rate / vol20_v;

        // --- V7: Basis ---
        out[F_basis] = last_basis;
        if (basis_window_24.full() && !std::isnan(last_basis)) {
            double b_mean = *basis_window_24.mean();
            double b_std = *basis_window_24.std_dev();
            if (b_std > 1e-12) {
                double zscore = (last_basis - b_mean) / b_std;
                out[F_basis_zscore_24] = zscore;
                out[F_basis_extreme] = (zscore > 2.0) ? 1.0 : ((zscore < -2.0) ? -1.0 : 0.0);
            }
        }

        if (!std::isnan(last_basis) && basis_ema_8.ready(8))
            out[F_basis_momentum] = last_basis - basis_ema_8.value;

        // --- V7: FGI ---
        if (!std::isnan(last_fgi)) {
            out[F_fgi_normalized] = last_fgi / 100.0 - 0.5;
            out[F_fgi_extreme] = (last_fgi < 25) ? -1.0 : ((last_fgi > 75) ? 1.0 : 0.0);
        }

        if (fgi_window_7.full() && !std::isnan(last_fgi)) {
            double fgi_mean = *fgi_window_7.mean();
            double fgi_std = *fgi_window_7.std_dev();
            if (fgi_std > 1e-12)
                out[F_fgi_zscore_7] = (last_fgi - fgi_mean) / fgi_std;
        }

        // --- V8: Alpha Rebuild V3 ---
        double tbqv = last_taker_buy_quote_volume;
        double qv = last_quote_volume;
        if (tbqv > 0 && qv > 0)
            out[F_taker_bq_ratio] = tbqv / qv;

        if (vwap_cv_window.full() && vwap_v_window.full() && !std::isnan(close) && close > 0) {
            double sum_cv = *vwap_cv_window.mean() * vwap_cv_window.n();
            double sum_v = *vwap_v_window.mean() * vwap_v_window.n();
            if (sum_v > 0) {
                double vwap = sum_cv / sum_v;
                out[F_vwap_dev_20] = (close - vwap) / close;
            }
        }

        // volume_momentum_10
        double ret_10 = NaN;
        if (n > 10) {
            double past10 = close_history.back_n(10);
            if (past10 != 0) ret_10 = (close_history.back() - past10) / past10;
        }
        double vol_r_20 = out[F_vol_ratio_20];
        if (!std::isnan(ret_10) && !std::isnan(vol_r_20))
            out[F_volume_momentum_10] = ret_10 * std::min(vol_r_20, 3.0);

        // mom_vol_divergence
        double ret1_2 = out[F_ret_1];
        double vol_r2 = out[F_vol_ratio_20];
        if (!std::isnan(ret1_2) && !std::isnan(vol_r2)) {
            bool price_up = ret1_2 > 0;
            bool vol_up = vol_r2 > 1.0;
            out[F_mom_vol_divergence] = (price_up == vol_up) ? 1.0 : -1.0;
        }

        // basis_carry_adj
        if (!std::isnan(last_basis) && !std::isnan(last_funding_rate))
            out[F_basis_carry_adj] = last_basis + last_funding_rate * 3.0;

        // vol_regime_adaptive
        if (vol_regime_ema.ready(5) && vol_regime_history.size() >= 30) {
            double ema_val = vol_regime_ema.value;
            // Find median of vol_regime_history
            double sorted_arr[30];
            for (int i = 0; i < 30; ++i) sorted_arr[i] = vol_regime_history[i];
            std::sort(sorted_arr, sorted_arr + 30);
            double median_val = sorted_arr[15];
            if (ema_val > median_val * 1.05)
                out[F_vol_regime_adaptive] = 1.0;
            else if (ema_val < median_val * 0.95)
                out[F_vol_regime_adaptive] = -1.0;
            else
                out[F_vol_regime_adaptive] = 0.0;
        }

        // --- V9: Cross-factor interaction ---
        double oi_pct = out[F_oi_change_pct];
        double vmr = out[F_vol_ma_ratio_5_20];
        if (!std::isnan(oi_pct) && !std::isnan(vmr))
            out[F_liquidation_cascade_score] = std::abs(oi_pct) * vmr;

        double fr = out[F_funding_rate];
        double fma8 = out[F_funding_ma8];
        if (!std::isnan(fr) && !std::isnan(fma8)) {
            double denom = std::max(std::abs(fma8), 1e-6);
            out[F_funding_term_slope] = (fr - fma8) / denom;
        }

        out[F_cross_tf_regime_sync] = NaN;  // requires external aggregator

        // --- V9: Deribit IV ---
        if (iv_window_24.full() && !std::isnan(last_implied_vol)) {
            double iv_mean = *iv_window_24.mean();
            double iv_std = *iv_window_24.std_dev();
            if (iv_std > 1e-8)
                out[F_implied_vol_zscore_24] = (last_implied_vol - iv_mean) / iv_std;
        }

        if (!std::isnan(last_implied_vol) && !std::isnan(vol20_v))
            out[F_iv_rv_spread] = last_implied_vol - vol20_v;

        out[F_put_call_ratio] = last_put_call_ratio;

        // --- V10: On-chain ---
        if (onchain_netflow_buf.size() >= 7) {
            // Copy to temp array for zscore computation
            double tmp[7];
            int start = onchain_netflow_buf.size() - 7;
            for (int i = 0; i < 7; ++i) tmp[i] = onchain_netflow_buf[start + i];
            out[F_exchange_netflow_zscore] = zscore_buf(tmp, 7, tmp[6], 1e-8);
        }

        if (onchain_supply_buf.size() >= 2) {
            double prev = onchain_supply_buf.back_n(1);
            double curr = onchain_supply_buf.back();
            out[F_exchange_supply_change] = (prev > 1e-8) ? (curr - prev) / prev : 0.0;
        }

        if (onchain_supply_buf.size() >= 30) {
            double tmp[30];
            for (int i = 0; i < 30; ++i) tmp[i] = onchain_supply_buf[i];
            out[F_exchange_supply_zscore_30] = zscore_buf(tmp, 30, tmp[29], 1e-8);
        }

        if (onchain_addr_buf.size() >= 14) {
            double tmp[14];
            for (int i = 0; i < 14; ++i) tmp[i] = onchain_addr_buf[i];
            out[F_active_addr_zscore_14] = zscore_buf(tmp, 14, tmp[13], 1e-8);
        }

        if (onchain_tx_buf.size() >= 14) {
            double tmp[14];
            for (int i = 0; i < 14; ++i) tmp[i] = onchain_tx_buf[i];
            out[F_tx_count_zscore_14] = zscore_buf(tmp, 14, tmp[13], 1e-8);
        }

        if (onchain_hashrate_ema.ready(14) && !std::isnan(last_onchain_hashrate)) {
            double ema_val = onchain_hashrate_ema.value;
            if (std::abs(ema_val) > 1e-8)
                out[F_hashrate_momentum] = (last_onchain_hashrate - ema_val) / ema_val;
        }

        // --- V11: Liquidation features ---
        // liquidation_volume_zscore_24
        if (liq_volume_buf.size() >= 24) {
            double tmp[24];
            int start = liq_volume_buf.size() - 24;
            for (int i = 0; i < 24; ++i) tmp[i] = liq_volume_buf[start + i];
            out[F_liquidation_volume_zscore_24] = zscore_buf(tmp, 24, tmp[23], 1e-8);
        }

        // liquidation_imbalance
        out[F_liquidation_imbalance] = last_liq_imbalance;

        // liquidation_volume_ratio
        if (!std::isnan(last_liq_volume) && last_quote_volume > 0) {
            out[F_liquidation_volume_ratio] = last_liq_volume / last_quote_volume;
        }

        // liquidation_cluster_flag
        if (liq_imbalance_buf.size() >= 6 && liq_volume_buf.size() >= 6) {
            double recent[6];
            int start = liq_volume_buf.size() - 6;
            for (int i = 0; i < 6; ++i) recent[i] = liq_volume_buf[start + i];
            double sum6 = 0.0;
            for (int i = 0; i < 6; ++i) sum6 += recent[i];
            double mean6 = sum6 / 6.0;
            double var6 = 0.0;
            for (int i = 0; i < 6; ++i) { double d = recent[i] - mean6; var6 += d * d; }
            var6 /= 6.0;
            double std6 = std::sqrt(var6);
            out[F_liquidation_cluster_flag] = (std6 > 1e-8 && recent[5] > mean6 + 3.0 * std6) ? 1.0 : 0.0;
        }

        // --- V11: Mempool features ---
        if (mempool_fee_buf.size() >= 24) {
            double tmp[24];
            int start = mempool_fee_buf.size() - 24;
            for (int i = 0; i < 24; ++i) tmp[i] = mempool_fee_buf[start + i];
            out[F_mempool_fee_zscore_24] = zscore_buf(tmp, 24, tmp[23], 1e-8);
        }

        if (mempool_size_buf.size() >= 24) {
            double tmp[24];
            int start = mempool_size_buf.size() - 24;
            for (int i = 0; i < 24; ++i) tmp[i] = mempool_size_buf[start + i];
            out[F_mempool_size_zscore_24] = zscore_buf(tmp, 24, tmp[23], 1e-8);
        }

        out[F_fee_urgency_ratio] = last_fee_urgency;

        // --- V11: Macro features ---
        // dxy_change_5d
        if (dxy_buf.size() >= 6) {
            double old_dxy = dxy_buf[dxy_buf.size() - 6];
            double new_dxy = dxy_buf.back();
            out[F_dxy_change_5d] = (old_dxy > 1e-8) ? (new_dxy - old_dxy) / old_dxy : 0.0;
        }

        // spx_btc_corr_30d
        {
            int n_spx = spx_buf.size();
            int n_btc = btc_close_buf_30.size();
            int nc = std::min(n_spx, n_btc);
            if (nc >= 10) {
                // Compute returns from both series
                int m = nc - 1;  // number of return pairs
                if (m >= 5) {
                    double spx_rets[29], btc_rets[29];
                    int spx_off = n_spx - nc;
                    int btc_off = n_btc - nc;
                    for (int i = 0; i < m; ++i) {
                        double s0 = spx_buf[spx_off + i];
                        double s1 = spx_buf[spx_off + i + 1];
                        spx_rets[i] = (s0 > 0) ? (s1 - s0) / s0 : 0.0;
                        double b0 = btc_close_buf_30[btc_off + i];
                        double b1 = btc_close_buf_30[btc_off + i + 1];
                        btc_rets[i] = (b0 > 0) ? (b1 - b0) / b0 : 0.0;
                    }
                    // Means
                    double mean_s = 0.0, mean_b = 0.0;
                    for (int i = 0; i < m; ++i) { mean_s += spx_rets[i]; mean_b += btc_rets[i]; }
                    mean_s /= m; mean_b /= m;
                    // Covariance and variances
                    double cov = 0.0, var_s = 0.0, var_b = 0.0;
                    for (int i = 0; i < m; ++i) {
                        double ds = spx_rets[i] - mean_s;
                        double db = btc_rets[i] - mean_b;
                        cov += ds * db;
                        var_s += ds * ds;
                        var_b += db * db;
                    }
                    cov /= m; var_s /= m; var_b /= m;
                    double denom = std::sqrt(var_s * var_b);
                    out[F_spx_btc_corr_30d] = (denom > 1e-8) ? cov / denom : 0.0;
                }
            }
        }

        // spx_overnight_ret
        if (!std::isnan(last_spx_close) && !std::isnan(prev_spx_close) && prev_spx_close > 0) {
            out[F_spx_overnight_ret] = (last_spx_close - prev_spx_close) / prev_spx_close;
        }

        // vix_zscore_14
        if (vix_buf.size() >= 14) {
            double tmp[14];
            for (int i = 0; i < 14; ++i) tmp[i] = vix_buf[i];
            out[F_vix_zscore_14] = zscore_buf(tmp, 14, tmp[13], 1e-8);
        }

        // --- V11: Social sentiment features ---
        if (social_vol_buf.size() >= 24) {
            double tmp[24];
            int start = social_vol_buf.size() - 24;
            for (int i = 0; i < 24; ++i) tmp[i] = social_vol_buf[start + i];
            out[F_social_volume_zscore_24] = zscore_buf(tmp, 24, tmp[23], 1e-8);
        }

        out[F_social_sentiment_score] = last_sentiment_score;

        // social_volume_price_div
        if (!std::isnan(last_social_volume) && social_vol_buf.size() >= 2 && close_history.size() >= 2) {
            double sv_change = social_vol_buf.back() - social_vol_buf.back_n(1);
            double price_change = close_history.back() - close_history.back_n(1);
            if ((sv_change > 0 && price_change < 0) || (sv_change < 0 && price_change > 0)) {
                out[F_social_volume_price_div] = 1.0;
            } else {
                out[F_social_volume_price_div] = 0.0;
            }
        }
    }
};

// ============================================================
// Schedule cursor — forward-scanning merge for auxiliary data
// ============================================================

struct ScheduleCursor {
    const double* ts;   // timestamps (sorted ascending)
    const double* val;  // values
    int len;
    int idx = 0;
    double current = NaN;

    ScheduleCursor(const double* ts, const double* val, int len)
        : ts(ts), val(val), len(len) {}

    double advance(double bar_ts) noexcept {
        while (idx < len && ts[idx] <= bar_ts) {
            current = val[idx];
            ++idx;
        }
        return current;
    }
};

// Multi-column schedule cursor for on-chain (M, 7): [ts, FlowIn, FlowOut, Supply, Addr, Tx, HR]
struct OnchainCursor {
    const double* data;  // (M, 7) row-major
    int rows;
    int idx = 0;

    // Current values (NaN until first match)
    double flow_in = NaN, flow_out = NaN;
    double supply = NaN, addr = NaN, tx = NaN, hashrate = NaN;

    OnchainCursor(const double* data, int rows) : data(data), rows(rows) {}

    void advance(double bar_ts) noexcept {
        while (idx < rows && data[idx * 7] <= bar_ts) {
            const double* row = data + idx * 7;
            flow_in = row[1]; flow_out = row[2];
            supply = row[3]; addr = row[4];
            tx = row[5]; hashrate = row[6];
            ++idx;
        }
    }
};

// V11: Liquidation cursor (M, 4): [ts, total_vol, buy_vol, sell_vol]
struct LiqCursor {
    const double* data;
    int rows;
    int idx = 0;
    double total_vol = NaN, buy_vol = NaN, sell_vol = NaN;

    LiqCursor(const double* data, int rows) : data(data), rows(rows) {}

    void advance(double bar_ts) noexcept {
        while (idx < rows && data[idx * 4] <= bar_ts) {
            const double* row = data + idx * 4;
            total_vol = row[1]; buy_vol = row[2]; sell_vol = row[3];
            ++idx;
        }
    }
};

// V11: Mempool cursor (M, 4): [ts, fastest_fee, economy_fee, mempool_size]
struct MempoolCursor {
    const double* data;
    int rows;
    int idx = 0;
    double fastest_fee = NaN, economy_fee = NaN, mempool_size = NaN;

    MempoolCursor(const double* data, int rows) : data(data), rows(rows) {}

    void advance(double bar_ts) noexcept {
        while (idx < rows && data[idx * 4] <= bar_ts) {
            const double* row = data + idx * 4;
            fastest_fee = row[1]; economy_fee = row[2]; mempool_size = row[3];
            ++idx;
        }
    }
};

// V11: Macro cursor (M, 4): [ts, dxy, spx, vix]
struct MacroCursor {
    const double* data;
    int rows;
    int idx = 0;
    double dxy = NaN, spx = NaN, vix = NaN;
    int64_t day = -1;  // for date dedup

    MacroCursor(const double* data, int rows) : data(data), rows(rows) {}

    void advance(double bar_ts) noexcept {
        while (idx < rows && data[idx * 4] <= bar_ts) {
            const double* row = data + idx * 4;
            // Compute day index from timestamp (ms)
            int64_t ts_sec = static_cast<int64_t>(row[0] / 1000.0);
            day = ts_sec / 86400;
            if (!std::isnan(row[1])) dxy = row[1];
            if (!std::isnan(row[2])) spx = row[2];
            if (!std::isnan(row[3])) vix = row[3];
            ++idx;
        }
    }
};

// ============================================================
// cpp_compute_all_features — batch compute features for all bars
// ============================================================

inline py::array_t<double> cpp_compute_all_features(
    py::array_t<double, py::array::c_style | py::array::forcecast> timestamps,
    py::array_t<double, py::array::c_style | py::array::forcecast> opens,
    py::array_t<double, py::array::c_style | py::array::forcecast> highs,
    py::array_t<double, py::array::c_style | py::array::forcecast> lows,
    py::array_t<double, py::array::c_style | py::array::forcecast> closes,
    py::array_t<double, py::array::c_style | py::array::forcecast> volumes,
    py::array_t<double, py::array::c_style | py::array::forcecast> trades_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> tbv,
    py::array_t<double, py::array::c_style | py::array::forcecast> qv,
    py::array_t<double, py::array::c_style | py::array::forcecast> tbqv,
    py::array_t<double, py::array::c_style | py::array::forcecast> funding_sched,
    py::array_t<double, py::array::c_style | py::array::forcecast> oi_sched,
    py::array_t<double, py::array::c_style | py::array::forcecast> ls_sched,
    py::array_t<double, py::array::c_style | py::array::forcecast> spot_sched,
    py::array_t<double, py::array::c_style | py::array::forcecast> fgi_sched,
    py::array_t<double, py::array::c_style | py::array::forcecast> iv_sched,
    py::array_t<double, py::array::c_style | py::array::forcecast> pcr_sched,
    py::array_t<double, py::array::c_style | py::array::forcecast> onchain_sched,
    // V11 schedules
    py::array_t<double, py::array::c_style | py::array::forcecast> liq_sched,
    py::array_t<double, py::array::c_style | py::array::forcecast> mempool_sched,
    py::array_t<double, py::array::c_style | py::array::forcecast> macro_sched
) {
    auto ts_buf = timestamps.request();
    int n_bars = static_cast<int>(ts_buf.shape[0]);

    // Get pointers to bar data
    const double* ts_ptr = static_cast<const double*>(ts_buf.ptr);
    const double* open_ptr = static_cast<const double*>(opens.request().ptr);
    const double* high_ptr = static_cast<const double*>(highs.request().ptr);
    const double* low_ptr = static_cast<const double*>(lows.request().ptr);
    const double* close_ptr = static_cast<const double*>(closes.request().ptr);
    const double* vol_ptr = static_cast<const double*>(volumes.request().ptr);
    const double* trades_ptr = static_cast<const double*>(trades_arr.request().ptr);
    const double* tbv_ptr = static_cast<const double*>(tbv.request().ptr);
    const double* qv_ptr = static_cast<const double*>(qv.request().ptr);
    const double* tbqv_ptr = static_cast<const double*>(tbqv.request().ptr);

    // For (M, 2) schedules, stride-aware access
    struct StridedCursor {
        const double* base;
        int len;
        int idx = 0;
        double current = NaN;

        StridedCursor(const double* base, int len) : base(base), len(len) {}

        double advance(double bar_ts) noexcept {
            while (idx < len && base[idx * 2] <= bar_ts) {
                current = base[idx * 2 + 1];
                ++idx;
            }
            return current;
        }
    };

    // Setup schedule cursors from (M, 2) arrays
    auto make_cursor = [](py::array_t<double, py::array::c_style | py::array::forcecast>& arr) -> StridedCursor {
        auto buf = arr.request();
        int rows = (buf.ndim >= 1) ? static_cast<int>(buf.shape[0]) : 0;
        if (rows == 0 || buf.ndim < 2) return StridedCursor(nullptr, 0);
        return StridedCursor(static_cast<const double*>(buf.ptr), rows);
    };

    StridedCursor funding_cur = make_cursor(funding_sched);
    StridedCursor oi_cur = make_cursor(oi_sched);
    StridedCursor ls_cur = make_cursor(ls_sched);
    StridedCursor spot_cur = make_cursor(spot_sched);
    StridedCursor fgi_cur = make_cursor(fgi_sched);
    StridedCursor iv_cur = make_cursor(iv_sched);
    StridedCursor pcr_cur = make_cursor(pcr_sched);

    // On-chain cursor: (M, 7) array
    auto oc_buf = onchain_sched.request();
    int oc_rows = (oc_buf.ndim >= 1) ? static_cast<int>(oc_buf.shape[0]) : 0;
    OnchainCursor oc_cur(
        (oc_rows > 0) ? static_cast<const double*>(oc_buf.ptr) : nullptr,
        oc_rows
    );

    // V11: Liquidation cursor (M, 4)
    auto liq_buf = liq_sched.request();
    int liq_rows = (liq_buf.ndim >= 1) ? static_cast<int>(liq_buf.shape[0]) : 0;
    LiqCursor liq_cur(
        (liq_rows > 0 && liq_buf.ndim >= 2) ? static_cast<const double*>(liq_buf.ptr) : nullptr,
        liq_rows
    );

    // V11: Mempool cursor (M, 4)
    auto mp_buf = mempool_sched.request();
    int mp_rows = (mp_buf.ndim >= 1) ? static_cast<int>(mp_buf.shape[0]) : 0;
    MempoolCursor mp_cur(
        (mp_rows > 0 && mp_buf.ndim >= 2) ? static_cast<const double*>(mp_buf.ptr) : nullptr,
        mp_rows
    );

    // V11: Macro cursor (M, 4)
    auto macro_buf = macro_sched.request();
    int macro_rows = (macro_buf.ndim >= 1) ? static_cast<int>(macro_buf.shape[0]) : 0;
    MacroCursor macro_cur(
        (macro_rows > 0 && macro_buf.ndim >= 2) ? static_cast<const double*>(macro_buf.ptr) : nullptr,
        macro_rows
    );

    // Allocate output array (n_bars, N_FEATURES)
    auto result = py::array_t<double>({n_bars, N_FEATURES});
    auto result_buf = result.request();
    double* out_ptr = static_cast<double*>(result_buf.ptr);

    // Initialize state
    BarState state;

    // Process each bar
    for (int i = 0; i < n_bars; ++i) {
        double ts = ts_ptr[i];
        double close = close_ptr[i];
        double volume = vol_ptr[i];
        double high = high_ptr[i];
        double low = low_ptr[i];
        double open_ = open_ptr[i];
        double trades = trades_ptr[i];
        double taker_buy_volume = tbv_ptr[i];
        double quote_volume = qv_ptr[i];
        double taker_buy_quote_volume = tbqv_ptr[i];

        // Default open/high/low to close if zero (matching Python)
        if (open_ == 0.0) open_ = close;
        if (high == 0.0) high = close;
        if (low == 0.0) low = close;

        // Parse hour and dow from timestamp (ms)
        int hour = -1, dow = -1;
        if (ts > 0) {
            // Unix timestamp in ms → seconds
            int64_t ts_sec = static_cast<int64_t>(ts / 1000.0);
            // UTC time decomposition
            // Days since epoch (1970-01-01)
            int64_t days = ts_sec / 86400;
            int64_t day_sec = ts_sec % 86400;
            if (day_sec < 0) { days--; day_sec += 86400; }
            hour = static_cast<int>(day_sec / 3600);
            // Day of week: 1970-01-01 was Thursday → weekday()=3
            // Python weekday(): Monday=0, Sunday=6
            dow = static_cast<int>((days + 3) % 7);
            if (dow < 0) dow += 7;
        }

        // Advance schedule cursors
        double funding_rate = funding_cur.advance(ts);
        double open_interest = oi_cur.advance(ts);
        double ls_ratio = ls_cur.advance(ts);
        double spot_close = spot_cur.advance(ts);
        double fear_greed = fgi_cur.advance(ts);
        double implied_vol = iv_cur.advance(ts);
        double put_call_ratio_val = pcr_cur.advance(ts);

        // Advance on-chain cursor
        oc_cur.advance(ts);

        // Advance V11 cursors
        liq_cur.advance(ts);
        mp_cur.advance(ts);
        macro_cur.advance(ts);

        // Push bar data
        state.push(close, volume, high, low, open_,
                   hour, dow,
                   funding_rate, trades,
                   taker_buy_volume, quote_volume, taker_buy_quote_volume,
                   open_interest, ls_ratio, spot_close, fear_greed,
                   implied_vol, put_call_ratio_val,
                   oc_cur.flow_in, oc_cur.flow_out,
                   oc_cur.supply, oc_cur.addr,
                   oc_cur.tx, oc_cur.hashrate,
                   // V11: Liquidation
                   liq_cur.total_vol, liq_cur.buy_vol, liq_cur.sell_vol,
                   (!std::isnan(liq_cur.total_vol) ? 1.0 : NaN),
                   // V11: Mempool
                   mp_cur.fastest_fee, mp_cur.economy_fee, mp_cur.mempool_size,
                   // V11: Macro
                   macro_cur.dxy, macro_cur.spx, macro_cur.vix, macro_cur.day,
                   // V11: Sentiment (no historical data in batch mode)
                   NaN, NaN);

        // Get features
        double* row_out = out_ptr + i * N_FEATURES;
        state.get_features(row_out);

        // Update prev_momentum (Python does this after get_features)
        state.prev_momentum = row_out[F_ma_cross_10_30];
    }

    return result;
}

inline std::vector<std::string> cpp_feature_names() {
    return FEATURE_NAMES;
}
