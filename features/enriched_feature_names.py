# features/enriched_feature_names.py
"""Feature name constants and dominance configuration for EnrichedFeatureComputer.

Extracted from enriched_computer.py to keep it under 800 lines.
"""

_MULTI_DOMINANCE_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "BTCUSDT": (),
    "ETHUSDT": (),
}
_MULTI_DOMINANCE_PREFIXES: tuple[str, ...] = ()
_ALL_MULTI_DOMINANCE_FEATURES: tuple[str, ...] = tuple(
    name for prefix in _MULTI_DOMINANCE_PREFIXES for name in (f"{prefix}_dev_20", f"{prefix}_ret_24")
)

# Feature names produced by this computer — used by training scripts
ENRICHED_FEATURE_NAMES: tuple[str, ...] = (
    # Returns at multiple horizons
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    # Moving average crossovers
    "ma_cross_10_30", "ma_cross_5_20", "close_vs_ma20", "close_vs_ma50",
    # RSI
    "rsi_14", "rsi_6",
    # MACD
    "macd_line", "macd_signal", "macd_hist",
    # Bollinger Bands
    "bb_width_20", "bb_pctb_20",
    # ATR
    "atr_norm_14",
    # ADX (trend strength)
    "adx_14",
    # Volatility
    "vol_20", "vol_5",
    # Volume
    "vol_ratio_20", "vol_ma_ratio_5_20",
    # Candle structure
    "body_ratio", "upper_shadow", "lower_shadow",
    # Trend
    "mean_reversion_20", "price_acceleration",
    # --- P0 Crypto-native features ---
    # Time-of-day (cyclical encoding)
    "hour_sin", "hour_cos",
    # Day-of-week (cyclical encoding)
    "dow_sin", "dow_cos",
    # Volatility regime
    "vol_regime",  # vol_5 / vol_20 ratio (>1 = expanding, <1 = contracting)
    # Funding rate features (set externally)
    "funding_rate", "funding_ma8",  # current rate + 8-period MA (= 64h)
    # --- Kline microstructure features ---
    "trade_intensity",        # trades / EMA(trades, 20) — activity anomaly
    "taker_buy_ratio",        # taker_buy_vol / volume — active buy ratio (0~1)
    "taker_buy_ratio_ma10",   # EMA(taker_buy_ratio, 10) — smoothed buy pressure
    "taker_imbalance",        # 2 * taker_buy_ratio - 1 — buy/sell imbalance (-1~1)
    "avg_trade_size",         # quote_vol / trades — average trade size
    "avg_trade_size_ratio",   # avg_trade_size / EMA(avg_trade_size, 20) — whale detection
    "volume_per_trade",       # volume / trades (normalized by EMA) — per-trade volume
    "trade_count_regime",     # EMA(trades, 5) / EMA(trades, 20) — activity expansion
    # --- Funding deep features ---
    "funding_zscore_24",      # (rate - mean_24) / std_24 — extreme funding detection
    "funding_momentum",       # rate - funding_ma8 — funding change velocity
    "funding_extreme",        # |zscore| > 2 flag — crowded trade signal
    "funding_cumulative_8",   # sum of last 8 funding settlements — holding cost
    "funding_sign_persist",   # consecutive same-sign funding count — bias persistence
    # --- Open Interest features ---
    "oi_change_pct",          # OI change rate
    "oi_change_ma8",          # OI change rate EMA(8)
    "oi_close_divergence",    # price direction vs OI direction divergence
    # --- Long/Short Ratio features ---
    "ls_ratio",               # long/short account ratio
    "ls_ratio_zscore_24",     # 24-bar z-score of LS ratio
    "ls_extreme",             # extreme long/short bias flag
    # --- V5: Order Flow features ---
    "cvd_10",                 # cumulative volume delta 10 bars
    "cvd_20",                 # cumulative volume delta 20 bars
    "cvd_price_divergence",   # CVD vs price direction divergence flag
    "aggressive_flow_zscore", # z-score of taker buy ratio over 50 bars
    # --- V5: Volatility microstructure ---
    "vol_of_vol",             # std of vol_5 over 20 bars
    "range_vs_rv",            # (H-L)/C / vol_5 — intrabar range vs realized vol
    "parkinson_vol",          # Parkinson volatility estimator (20 bars)
    "rv_acceleration",        # vol_5[t] - vol_5[t-5]
    # --- V5: Liquidation proxy ---
    "oi_acceleration",        # oi_change_pct acceleration
    "leverage_proxy",         # OI/(close*volume) normalized by EMA(20)
    "oi_vol_divergence",      # OI up + volume down divergence flag
    "oi_liquidation_flag",    # large OI drop + volume spike flag
    # --- V5: Funding carry ---
    "funding_annualized",     # funding_rate * 3 * 365
    "funding_vs_vol",         # funding_rate / vol_20
    # --- V7: Spot-futures basis ---
    "basis",                  # (futures_close - spot_close) / spot_close
    "basis_zscore_24",        # z-score of basis over 24 bars
    "basis_momentum",         # basis - EMA(basis, 8)
    "basis_extreme",          # |zscore| > 2 flag
    # --- V7: Fear & Greed Index ---
    "fgi_normalized",         # FGI / 100 - 0.5 (range [-0.5, 0.5])
    "fgi_zscore_7",           # (FGI - mean_7d) / std_7d
    "fgi_zscore_14",          # (FGI - mean_14d) / std_14d
    "fgi_extreme",            # FGI < 25 → -1, FGI > 75 → 1, else 0
    "fgi_change_7d",          # 7-day change in FGI value
    # --- V8: Alpha Rebuild V3 features ---
    "taker_bq_ratio",         # taker_buy_quote_volume / quote_volume — USD-weighted buy pressure
    "vwap_dev_20",            # (close - VWAP_20) / close — VWAP deviation (mean reversion)
    "volume_momentum_10",     # ret_10 × clip(volume/SMA_vol_20, 3.0) — volume-confirmed momentum
    "mom_vol_divergence",     # sign(ret_1)==sign(vol_ratio-1) ? +1 : -1 — exhaustion detector
    "basis_carry_adj",        # basis + funding_rate × 3 — carry-adjusted basis
    "vol_regime_adaptive",    # EMA(vol_regime,5) vs 30-bar median — regime persistence
    # --- V9: Cross-factor interaction features ---
    "liquidation_cascade_score",  # |oi_change_pct| * vol_ma_ratio_5_20 — liquidation pressure
    "funding_term_slope",         # (funding_rate - funding_ma8) / max(|funding_ma8|, 1e-6) — funding curve slope
    "cross_tf_regime_sync",       # sign(close_vs_ma20) == sign(tf4h_close_vs_ma20) — multi-TF alignment
    # --- V9: Deribit IV features (set externally) ---
    "implied_vol_zscore_24",      # IV z-score over 24 bars
    "iv_rv_spread",               # implied vol - realized vol (vol_20)
    "put_call_ratio",             # Deribit options put/call OI ratio
    # --- V10: On-chain features (Coin Metrics, daily) ---
    "exchange_netflow_zscore",    # zscore_7d(exchange inflow - outflow)
    "exchange_supply_change",     # daily pct change of exchange-held supply
    "exchange_supply_zscore_30",  # zscore_30d(SplyExNtv)
    "active_addr_zscore_14",      # zscore_14d(AdrActCnt)
    "tx_count_zscore_14",         # zscore_14d(TxTfrCnt)
    "hashrate_momentum",          # (HashRate - EMA14) / EMA14
    # --- V11: Liquidation features (Binance force orders) ---
    "liquidation_volume_zscore_24",   # z-score of hourly liquidation volume over 24 bars
    "liquidation_imbalance",          # (buy_liq - sell_liq) / total_liq — directional
    "liquidation_volume_ratio",       # liq_volume / trade_volume — leverage flush intensity
    "liquidation_cluster_flag",       # short-window cluster detection flag
    # --- V11: Mempool features (BTC-only, mempool.space) ---
    "mempool_fee_zscore_24",          # z-score of recommended fee over 24 bars
    "mempool_size_zscore_24",         # z-score of mempool size over 24 bars
    "fee_urgency_ratio",             # fastest_fee / economy_fee
    # --- V11: Macro features (DXY/SPX/VIX, daily) ---
    "dxy_change_5d",                  # 5-day DXY return
    "spx_btc_corr_30d",              # 30-day SPX-BTC correlation
    "spx_overnight_ret",              # SPX overnight return
    "vix_zscore_14",                  # 14-day VIX z-score
    # --- V21: Cross-market features (Yahoo Finance, daily) ---
    "spy_ret_1d",                     # SPY daily return (IC=-0.094 vs BTC next day)
    "qqq_ret_1d",                     # QQQ daily return (IC=-0.095, strongest cross-market signal)
    "spy_ret_5d",                     # SPY 5-day momentum
    "vix_level",                      # VIX absolute level (IC=+0.080)
    "tlt_ret_5d",                     # TLT 5-day return (IC=+0.039, liquidity proxy)
    "uso_ret_5d",                     # USO 5-day return (IC=-0.055, oil momentum)
    "coin_ret_1d",                    # Coinbase stock daily return (IC=-0.062)
    "spy_extreme",                    # SPY |ret| > 2% flag (high-conviction signal)
    # --- V11: Social sentiment ---
    "social_volume_zscore_24",        # z-score of social volume over 24 bars
    "social_sentiment_score",         # normalized sentiment score
    "social_volume_price_div",        # social volume vs price direction divergence
    # --- V12: ALT coin cross-asset features ---
    "btc_relative_strength_24",       # ALT ret_24 - BTC ret_24 — outperformance signal
    "btc_relative_strength_6",        # ALT ret_6 - BTC ret_6 — short-term alpha vs BTC
    "btc_ratio_ma20_dev",             # (ALT/BTC ratio) / MA20(ratio) - 1 — pair mean reversion
    "btc_dom_momentum",               # BTC 24h ret - ALT 24h ret — dominance shift velocity
    "btc_lead_ret_1",                 # BTC ret_1 (lagged signal for ALT)
    "btc_vol_ratio",                  # ALT vol_20 / BTC vol_20 — relative volatility regime
    # --- V14: BTC Dominance features (BTC/ETH ratio for BTC alpha) ---
    "btc_dom_dev_20",                 # BTC/ETH ratio deviation from MA(20) — capital flow signal
    "btc_dom_dev_50",                 # BTC/ETH ratio deviation from MA(50) — trend signal
    "btc_dom_ret_24",                 # BTC/ETH ratio 24-bar return — short-term dominance shift
    "btc_dom_ret_72",                 # BTC/ETH ratio 72-bar return — medium-term dominance shift
    # --- V14b: Multi-ratio dominance features (symbol vs reference pairs) ---
    "dom_vs_sui_dev_20",              # ratio vs SUI deviation from MA(20)
    "dom_vs_sui_ret_24",             # ratio vs SUI 24-bar return
    "dom_vs_axs_dev_20",             # ratio vs AXS deviation from MA(20)
    # --- V15: Interaction & statistical features (IC-screened 2026-03-18) ---
    "ret1_x_vol",                     # ret_1 × vol_20 — momentum in volatile regime (IC 0.04-0.06)
    "rsi_x_atr",                      # rsi_14 × atr_norm_14 — overbought + wide range (IC 0.04-0.05)
    "rsi_x_vol",                      # rsi_14 × vol_20 — RSI strength in volatility (IC 0.04-0.05)
    "trend_x_vol",                    # close_vs_ma50 × vol_20 — trend conviction (IC 0.03-0.05)
    "bb_x_vol",                       # bb_pctb_20 × vol_20 — band position in volatile regime (IC 0.02-0.03)
    "ret_autocorr_24",                # 24-bar return autocorrelation — momentum persistence (IC 0.03-0.04)
    "ret_skew_24",                    # 24-bar return skewness — tail risk signal (IC 0.03-0.05 on ALTs)
    # --- V16: Orderbook proxy + IV spread + liquidation features (IC-screened 2026-03-18) ---
    "ob_spread_proxy",                # (high-low)/close — intrabar range as spread (IC 0.03-0.04, 4/4 symbols)
    "ob_imbalance_proxy",             # (taker_buy - taker_sell)/total — buy/sell pressure (IC 0.03-0.04, 3/4)
    "ob_imbalance_x_vol",             # imbalance × volume_ratio — strength-weighted flow (IC 0.03-0.04, 3/4)
    "ob_imbalance_cum6",              # 6-bar cumulative imbalance — short-term order flow (IC 0.03, 3/4)
    "ob_volume_clock",                # vol_MA6/vol_MA24 - 1 — activity acceleration (IC 0.03, 3/4)
    "liq_volume_zscore_24",           # z-score of liquidation proxy volume (IC -0.18, BTC only but very strong)
    # --- V17: On-chain features (Coin Metrics, IC-screened 2026-03-18) ---
    "oc_tx_zscore_7",                 # ETH TxTfrCnt 7d z-score (IC +0.137, strongest on-chain factor)
    "oc_tx_zscore_14",                # ETH TxTfrCnt 14d z-score (IC +0.134)
    "oc_addr_zscore_7",               # Active addresses 7d z-score (IC +0.082)
    "oc_addr_zscore_14",              # Active addresses 14d z-score (IC +0.085)
    "oc_flowin_zscore_7",             # Exchange inflow 7d z-score
    "oc_flowin_zscore_14",            # Exchange inflow 14d z-score
    "oc_flowout_zscore_7",            # Exchange outflow 7d z-score
    "oc_flowout_zscore_14",           # Exchange outflow 14d z-score
    "oc_netflow_zscore_7",            # Net exchange flow 7d z-score (IC -0.074 BTC)
    "dom_vs_axs_ret_24",             # ratio vs AXS 24-bar return
    "dom_vs_eth_dev_20",             # ratio vs ETH deviation from MA(20)
    "dom_vs_eth_ret_24",             # ratio vs ETH 24-bar return
    # --- V13: Enhanced OI/LS/Taker features (IC-validated 2026-03) ---
    "oi_pct_4h",                      # 4-bar OI change rate — short-term position buildup
    "ls_deviation",                   # ls_ratio - 1.0 — directional bias magnitude
    "taker_buy_sell_ratio",           # taker_buy_vol / taker_sell_vol — raw buy/sell pressure
    "top_retail_divergence",          # top_trader_ls - global_ls — smart vs dumb money divergence
    "oi_price_divergence_12",         # 12-bar OI change - 12-bar price change — position/price mismatch
    # --- V18: OI change rate + funding cumulative (IC-screened 2026-03-21) ---
    "oi_change_24",                   # 24-bar OI change rate — medium-term position buildup (IC -0.291 at h96)
    "oi_change_96",                   # 96-bar OI change rate — long-term OI shift (strongest unused feature)
    "funding_cum_3",                  # 3-period cumulative funding rate — short-term carry cost (IC +0.041)
    # --- V19: Implied Volatility features (DVOL from Deribit) ---
    "dvol_chg_72",                    # 72-bar DVOL change rate (IC +0.132, strongest IV feature)
    "iv_term_struct",                 # short/long IV ratio: MA(24)/MA(168) - 1 (IC +0.104)
    "dvol_z",                         # DVOL z-score over 168 bars (IC +0.101)
    "dvol_chg_24",                    # 24-bar DVOL change rate (IC +0.107)
    "dvol_mean_rev",                  # DVOL / MA(720) - 1 — mean reversion signal (IC +0.094)
)

_WARMUP_BARS = 65  # bars needed before all features are valid (funding_ma8 needs 64h)
