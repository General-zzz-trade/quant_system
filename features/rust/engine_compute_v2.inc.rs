// engine_compute_v2.inc.rs — V7+ feature computation (basis, FGI, alpha rebuild,
// cross-factor, options, on-chain, liquidation, mempool, macro, social).
// Included inside BarState::get_features() from engine_compute.inc.rs.
// Wrapped in a block expression so include!() sees a single expression.
{
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

        // ================================================================
        // Phase 1: Alias features (zero computation)
        // ================================================================
        // oc_addr_zscore_14 = active_addr_zscore_14
        out[F_OC_ADDR_ZSCORE_14] = out[F_ACTIVE_ADDR_ZSCORE_14];
        // oc_tx_zscore_14 = tx_count_zscore_14
        out[F_OC_TX_ZSCORE_14] = out[F_TX_COUNT_ZSCORE_14];
        // oc_netflow_zscore_7 = exchange_netflow_zscore
        out[F_OC_NETFLOW_ZSCORE_7] = out[F_EXCHANGE_NETFLOW_ZSCORE];
        // btc_dom_dev_20, btc_dom_ret_24: NaN — populated externally via push_dominance()
        // out[F_BTC_DOM_DEV_20] remains NaN (set by init)
        // out[F_BTC_DOM_RET_24] remains NaN (set by init)

        // ================================================================
        // Phase 2: Interaction features (pure multiplication)
        // ================================================================
        let rsi14_out = out[F_RSI_14];
        let atr_norm_out = out[F_ATR_NORM_14];
        if !rsi14_out.is_nan() && !atr_norm_out.is_nan() {
            out[F_RSI_X_ATR] = rsi14_out * atr_norm_out;
        }
        if !rsi14_out.is_nan() && !vol20_v.is_nan() {
            out[F_RSI_X_VOL] = rsi14_out * vol20_v;
        }
        let close_vs_ma50_out = out[F_CLOSE_VS_MA50];
        if !close_vs_ma50_out.is_nan() && !vol20_v.is_nan() {
            out[F_TREND_X_VOL] = close_vs_ma50_out * vol20_v;
        }

        // ================================================================
        // Phase 3: IV derived features
        // ================================================================
        // iv_level = raw implied_vol value
        if !self.last_implied_vol.is_nan() {
            out[F_IV_LEVEL] = self.last_implied_vol;
        }
        // iv_rank_30d = percentile rank of current IV in 30-bar window
        if self.iv_window_30.size() >= 30 && !self.last_implied_vol.is_nan() {
            let current = self.last_implied_vol;
            let mut below = 0_usize;
            for i in 0..30 {
                if self.iv_window_30.get(i) <= current {
                    below += 1;
                }
            }
            out[F_IV_RANK_30D] = below as f64 / 30.0;
        }
        // iv_term_slope_daily = NaN (requires term structure data not available)
        // out[F_IV_TERM_SLOPE_DAILY] remains NaN

        // ================================================================
        // Phase 4: On-chain z-score features
        // ================================================================
        // oc_flowin_zscore_7
        if self.onchain_flowin_buf.size() >= 7 {
            let mut tmp = [0.0_f64; 7];
            let start = self.onchain_flowin_buf.size() - 7;
            for i in 0..7 {
                tmp[i] = self.onchain_flowin_buf.get(start + i);
            }
            out[F_OC_FLOWIN_ZSCORE_7] = zscore_buf(&tmp, 1e-8);
        }
        // oc_flowin_zscore_14
        if self.onchain_flowin_buf.size() >= 14 {
            let mut tmp = [0.0_f64; 14];
            for i in 0..14 {
                tmp[i] = self.onchain_flowin_buf.get(i);
            }
            out[F_OC_FLOWIN_ZSCORE_14] = zscore_buf(&tmp, 1e-8);
        }
        // oc_flowout_zscore_14
        if self.onchain_flowout_buf.size() >= 14 {
            let mut tmp = [0.0_f64; 14];
            for i in 0..14 {
                tmp[i] = self.onchain_flowout_buf.get(i);
            }
            out[F_OC_FLOWOUT_ZSCORE_14] = zscore_buf(&tmp, 1e-8);
        }
        // oc_tx_zscore_7
        if self.onchain_tx_buf.size() >= 7 {
            let mut tmp = [0.0_f64; 7];
            let start = self.onchain_tx_buf.size() - 7;
            for i in 0..7 {
                tmp[i] = self.onchain_tx_buf.get(start + i);
            }
            out[F_OC_TX_ZSCORE_7] = zscore_buf(&tmp, 1e-8);
        }

        // ================================================================
        // Phase 5: Other features
        // ================================================================
        // ret_autocorr_24: autocorrelation of returns over 24 bars
        if self.return_history_buf.size() >= 24 {
            let start = self.return_history_buf.size() - 24;
            let mut rets = [0.0_f64; 24];
            for i in 0..24 {
                rets[i] = self.return_history_buf.get(start + i);
            }
            let mut sum_r = 0.0;
            for &r in &rets { sum_r += r; }
            let mean_r = sum_r / 24.0;
            let mut var_r = 0.0;
            for &r in &rets { let d = r - mean_r; var_r += d * d; }
            if var_r > 1e-16 {
                let mut cov_r = 0.0;
                for i in 0..23 {
                    cov_r += (rets[i] - mean_r) * (rets[i + 1] - mean_r);
                }
                out[F_RET_AUTOCORR_24] = cov_r / var_r;
            }
        }

        // ob_imbalance_x_vol = taker_imbalance * vol_ratio_20
        let taker_imb_out = out[F_TAKER_IMBALANCE];
        let vol_ratio_out = out[F_VOL_RATIO_20];
        if !taker_imb_out.is_nan() && !vol_ratio_out.is_nan() {
            out[F_OB_IMBALANCE_X_VOL] = taker_imb_out * vol_ratio_out;
        }

        // total_zscore_14: z-score of raw OI over 14 bars
        if self.oi_raw_buf_14.size() >= 14 {
            let mut tmp = [0.0_f64; 14];
            for i in 0..14 {
                tmp[i] = self.oi_raw_buf_14.get(i);
            }
            out[F_TOTAL_ZSCORE_14] = zscore_buf(&tmp, 1e-8);
        }

        // total_zscore_30: z-score of raw OI over 30 bars
        if self.oi_raw_buf_30.size() >= 30 {
            let mut tmp = [0.0_f64; 30];
            for i in 0..30 {
                tmp[i] = self.oi_raw_buf_30.get(i);
            }
            out[F_TOTAL_ZSCORE_30] = zscore_buf(&tmp, 1e-8);
        }

        // total_supply_change_7d: onchain supply 7-day change rate
        if self.onchain_supply_buf.size() >= 7 {
            let old_supply = self.onchain_supply_buf.get(self.onchain_supply_buf.size() - 7);
            let new_supply = self.onchain_supply_buf.back();
            if old_supply > 1e-8 {
                out[F_TOTAL_SUPPLY_CHANGE_7D] = (new_supply - old_supply) / old_supply;
            }
        }

        // ETF features: NaN placeholders (filled by Python layer)
        // out[F_SPY_RET_1D..F_ETF_PREMIUM] remain NaN from initialization

        // Cross-asset placeholders: NaN
        // out[F_DOM_VS_SUI], out[F_DOM_VS_AXS] remain NaN

        // 4h feature placeholder: NaN
        // out[F_TF4H_BB_PCTB_20] remains NaN

        // USDT dominance placeholder: NaN
        // out[F_USDT_DOMINANCE] remains NaN
}
