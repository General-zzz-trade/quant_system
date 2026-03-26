// push.inc.rs — Included by engine.rs via include!() macro.
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

            onchain_flowin_buf: CircBuf::new(14),
            onchain_flowout_buf: CircBuf::new(14),

            iv_window_30: CircBuf::new(30),

            oi_raw_buf_14: CircBuf::new(14),
            oi_raw_buf_30: CircBuf::new(30),

            return_history_buf: CircBuf::new(48),

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

            // Cross-market
            cm_spy_close: f64::NAN,
            cm_tlt_close: f64::NAN,
            cm_uso_close: f64::NAN,
            cm_xlf_close: f64::NAN,
            cm_ethe_close: f64::NAN,
            cm_gbtc_vol: f64::NAN,
            cm_treasury_10y: f64::NAN,
            cm_usdt_dominance: f64::NAN,
            cm_spy_buf: CircBuf::new(11),
            cm_tlt_buf: CircBuf::new(6),
            cm_uso_buf: CircBuf::new(6),
            cm_xlf_buf: CircBuf::new(6),
            cm_ethe_buf: CircBuf::new(2),
            cm_gbtc_vol_buf: CircBuf::new(14),
            cm_treasury_buf: CircBuf::new(6),
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
            // Phase 5: raw OI for total_zscore
            self.oi_raw_buf_14.push(open_interest);
            self.oi_raw_buf_30.push(open_interest);
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
            self.iv_window_30.push(implied_vol);
        }
        if !put_call_ratio_val.is_nan() {
            self.last_put_call_ratio = put_call_ratio_val;
        }

        // --- On-chain ---
        if !oc_flow_in.is_nan() && !oc_flow_out.is_nan() {
            self.onchain_netflow_buf.push(oc_flow_in - oc_flow_out);
        }
        // Phase 4: separate flow-in/flow-out buffers
        if !oc_flow_in.is_nan() {
            self.onchain_flowin_buf.push(oc_flow_in);
        }
        if !oc_flow_out.is_nan() {
            self.onchain_flowout_buf.push(oc_flow_out);
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
            // Phase 5: return history for autocorrelation
            self.return_history_buf.push(ret);
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
}
