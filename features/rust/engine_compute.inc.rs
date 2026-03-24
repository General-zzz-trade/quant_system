// compute.inc.rs — Included by engine.rs via include!() macro.
// Single 675-line function body (get_features) — cannot split further without restructuring.
impl BarState {
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

        // --- V7+ features (basis, FGI, alpha rebuild, cross-factor, options,
        //     on-chain, liquidation, mempool, macro, social) ---
        include!("engine_compute_v2.inc.rs");
    }
}
