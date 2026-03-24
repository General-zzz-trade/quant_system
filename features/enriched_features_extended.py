# features/enriched_features_extended.py
"""Extended feature computation (V5-V19) for EnrichedFeatureComputer.

Extracted from enriched_computer.py to reduce file size.
Called by _SymbolState.get_features() to populate V5+ features.
"""
from __future__ import annotations

import math
from math import sqrt
from typing import TYPE_CHECKING, Deque, Dict, Optional

if TYPE_CHECKING:
    from features.enriched_computer import _SymbolState


def _window_zscore(values: Deque[float], window: int) -> Optional[float]:
    if len(values) < window:
        return None
    window_vals = list(values)[-window:]
    mean = sum(window_vals) / len(window_vals)
    var = sum((value - mean) ** 2 for value in window_vals) / len(window_vals)
    std = sqrt(var) if var > 0 else 0.0
    return (window_vals[-1] - mean) / std if std > 1e-8 else 0.0


def compute_extended_features(
    state: "_SymbolState",
    feats: Dict[str, Optional[float]],
    close: float,
    n: int,
    hist: "Deque[float]",
    tbr: Optional[float],
    btc_close: Optional[float],
) -> None:
    """Compute V5-V19 features and add them to feats dict.

    Args:
        state: The _SymbolState instance with all buffers/trackers.
        feats: Feature dict to populate (mutated in place).
        close: Current bar close price.
        n: Length of close history.
        hist: Close price history deque.
        tbr: Taker buy ratio (None if unavailable).
        btc_close: BTC close price for cross-asset features.
    """
    # --- V5: Order Flow features ---
    if state.cvd_window_10.full:
        feats["cvd_10"] = state.cvd_window_10.mean * state.cvd_window_10.n
    else:
        feats["cvd_10"] = None

    if state.cvd_window_20.full:
        cvd_20_val = state.cvd_window_20.mean * state.cvd_window_20.n
        feats["cvd_20"] = cvd_20_val
        # CVD-price divergence: sign(cvd_20) != sign(ret_20)
        if n > 20 and hist[-21] != 0:
            ret_20 = (hist[-1] - hist[-21]) / hist[-21]
            cvd_sign = 1.0 if cvd_20_val > 0 else (-1.0 if cvd_20_val < 0 else 0.0)
            ret_sign = 1.0 if ret_20 > 0 else (-1.0 if ret_20 < 0 else 0.0)
            feats["cvd_price_divergence"] = 1.0 if cvd_sign != 0 and cvd_sign != ret_sign else 0.0
        else:
            feats["cvd_price_divergence"] = None
    else:
        feats["cvd_20"] = None
        feats["cvd_price_divergence"] = None

    if state.taker_ratio_window_50.full:
        tr_mean = state.taker_ratio_window_50.mean
        tr_std = state.taker_ratio_window_50.std
        if tr_std is not None and tr_std > 1e-12 and tbr is not None:
            feats["aggressive_flow_zscore"] = (tbr - tr_mean) / tr_std
        else:
            feats["aggressive_flow_zscore"] = None
    else:
        feats["aggressive_flow_zscore"] = None

    # --- V5: Volatility microstructure ---
    vol5_val = feats.get("vol_5")
    vol20_val = feats.get("vol_20")

    if len(state.vol_5_history) >= 20:
        recent = list(state.vol_5_history)[-20:]
        mean_v = sum(recent) / len(recent)
        var_v = sum((x - mean_v) ** 2 for x in recent) / len(recent)
        feats["vol_of_vol"] = sqrt(var_v)
    else:
        feats["vol_of_vol"] = None

    if n > 0 and close and close != 0 and vol5_val is not None and vol5_val > 1e-12:
        h = state.high_history[-1] if len(state.high_history) > 0 else close
        l = state.low_history[-1] if len(state.low_history) > 0 else close  # noqa: E741
        feats["range_vs_rv"] = ((h - l) / close) / vol5_val
    else:
        feats["range_vs_rv"] = None

    if state.hl_log_sq_window.full:
        mean_sq = state.hl_log_sq_window.mean
        if mean_sq is not None and mean_sq >= 0:
            feats["parkinson_vol"] = sqrt(mean_sq / (4.0 * math.log(2)))
        else:
            feats["parkinson_vol"] = None
    else:
        feats["parkinson_vol"] = None

    if len(state.vol_5_history) >= 6:
        feats["rv_acceleration"] = state.vol_5_history[-1] - state.vol_5_history[-6]
    else:
        feats["rv_acceleration"] = None

    # --- V5: Liquidation proxy ---
    if state._last_oi_change_pct is not None and state._prev_oi_change_for_accel is not None:
        feats["oi_acceleration"] = state._last_oi_change_pct - state._prev_oi_change_for_accel
    else:
        feats["oi_acceleration"] = None

    if state._last_oi is not None and close and close > 0 and state._last_volume > 0:
        raw_lev = state._last_oi / (close * state._last_volume)
        lev_ema = state.leverage_proxy_ema.value
        if state.leverage_proxy_ema.ready and lev_ema and lev_ema > 0:
            feats["leverage_proxy"] = raw_lev / lev_ema
        else:
            feats["leverage_proxy"] = None
    else:
        feats["leverage_proxy"] = None

    # OI up + volume down divergence
    oi_chg = state._last_oi_change_pct
    vol_r = feats.get("vol_ratio_20")
    if oi_chg is not None and vol_r is not None:
        feats["oi_vol_divergence"] = 1.0 if oi_chg > 0 and vol_r < 1.0 else 0.0
    else:
        feats["oi_vol_divergence"] = None

    # Large OI drop + volume spike → liquidation
    if oi_chg is not None and vol_r is not None:
        feats["oi_liquidation_flag"] = 1.0 if oi_chg < -0.05 and vol_r > 2.0 else 0.0
    else:
        feats["oi_liquidation_flag"] = None

    # --- V5: Funding carry ---
    if state._last_funding_rate is not None:
        feats["funding_annualized"] = state._last_funding_rate * 3.0 * 365.0
    else:
        feats["funding_annualized"] = None

    if state._last_funding_rate is not None and vol20_val is not None and vol20_val > 1e-12:
        feats["funding_vs_vol"] = state._last_funding_rate / vol20_val
    else:
        feats["funding_vs_vol"] = None

    # --- V7: Spot-futures basis ---
    feats["basis"] = state._last_basis
    if state.basis_window_24.full and state._last_basis is not None:
        b_mean = state.basis_window_24.mean
        b_std = state.basis_window_24.std
        if b_std is not None and b_std > 1e-12:
            zscore = (state._last_basis - b_mean) / b_std
            feats["basis_zscore_24"] = zscore
            feats["basis_extreme"] = 1.0 if zscore > 2.0 else (-1.0 if zscore < -2.0 else 0.0)
        else:
            feats["basis_zscore_24"] = None
            feats["basis_extreme"] = None
    else:
        feats["basis_zscore_24"] = None
        feats["basis_extreme"] = None

    basis_ema_val = state.basis_ema_8.value if state.basis_ema_8.ready else None
    if state._last_basis is not None and basis_ema_val is not None:
        feats["basis_momentum"] = state._last_basis - basis_ema_val
    else:
        feats["basis_momentum"] = None

    # --- V7: Fear & Greed Index ---
    if state._last_fgi is not None:
        feats["fgi_normalized"] = state._last_fgi / 100.0 - 0.5
        feats["fgi_extreme"] = (
            -1.0 if state._last_fgi < 25 else (1.0 if state._last_fgi > 75 else 0.0)
        )
    else:
        feats["fgi_normalized"] = None
        feats["fgi_extreme"] = None

    if state.fgi_window_7.full and state._last_fgi is not None:
        fgi_mean = state.fgi_window_7.mean
        fgi_std = state.fgi_window_7.std
        if fgi_std is not None and fgi_std > 1e-12:
            feats["fgi_zscore_7"] = (state._last_fgi - fgi_mean) / fgi_std
        else:
            feats["fgi_zscore_7"] = None
    else:
        feats["fgi_zscore_7"] = None

    if state.fgi_window_14.full and state._last_fgi is not None:
        fgi_mean_14 = state.fgi_window_14.mean
        fgi_std_14 = state.fgi_window_14.std
        if fgi_std_14 is not None and fgi_std_14 > 1e-12:
            feats["fgi_zscore_14"] = (state._last_fgi - fgi_mean_14) / fgi_std_14
        else:
            feats["fgi_zscore_14"] = None
    else:
        feats["fgi_zscore_14"] = None

    if len(state.fgi_history_7d) >= 8 and state._last_fgi is not None:
        feats["fgi_change_7d"] = state._last_fgi - state.fgi_history_7d[0]
    else:
        feats["fgi_change_7d"] = None

    # --- V8: Alpha Rebuild V3 features ---

    # taker_bq_ratio: USD-weighted buy pressure
    tbqv = state._last_taker_buy_quote_volume
    qv = state._last_quote_volume
    if tbqv > 0 and qv > 0:
        feats["taker_bq_ratio"] = tbqv / qv
    else:
        feats["taker_bq_ratio"] = None

    # vwap_dev_20: VWAP deviation (mean reversion signal)
    if state.vwap_cv_window.full and state.vwap_v_window.full and close and close > 0:
        sum_cv = state.vwap_cv_window.mean * state.vwap_cv_window.n
        sum_v = state.vwap_v_window.mean * state.vwap_v_window.n
        if sum_v > 0:
            vwap = sum_cv / sum_v
            feats["vwap_dev_20"] = (close - vwap) / close
        else:
            feats["vwap_dev_20"] = None
    else:
        feats["vwap_dev_20"] = None

    # volume_momentum_10: ret_10 × clip(volume/SMA_vol_20, 3.0)
    ret_10 = feats.get("ret_12")  # closest available; use close_history directly
    if n > 10 and hist[-11] != 0:
        ret_10 = (hist[-1] - hist[-11]) / hist[-11]
    else:
        ret_10 = None
    vol_r_20 = feats.get("vol_ratio_20")
    if ret_10 is not None and vol_r_20 is not None:
        feats["volume_momentum_10"] = ret_10 * min(vol_r_20, 3.0)
    else:
        feats["volume_momentum_10"] = None

    # mom_vol_divergence: price direction vs volume direction agreement
    ret1 = feats.get("ret_1")
    vol_r = feats.get("vol_ratio_20")
    if ret1 is not None and vol_r is not None:
        price_up = ret1 > 0
        vol_up = vol_r > 1.0
        feats["mom_vol_divergence"] = 1.0 if price_up == vol_up else -1.0
    else:
        feats["mom_vol_divergence"] = None

    # basis_carry_adj: basis + funding_rate × 3
    if state._last_basis is not None and state._last_funding_rate is not None:
        feats["basis_carry_adj"] = state._last_basis + state._last_funding_rate * 3.0
    else:
        feats["basis_carry_adj"] = None

    # vol_regime_adaptive: EMA(vol_regime,5) vs 30-bar median
    if state.vol_regime_ema.ready and len(state.vol_regime_history) >= 30:
        ema_val = state.vol_regime_ema.value
        if ema_val is None:
            feats["vol_regime_adaptive"] = None
        else:
            sorted_hist = sorted(state.vol_regime_history)
            median_val = sorted_hist[len(sorted_hist) // 2]
            if ema_val > median_val * 1.05:
                feats["vol_regime_adaptive"] = 1.0
            elif ema_val < median_val * 0.95:
                feats["vol_regime_adaptive"] = -1.0
            else:
                feats["vol_regime_adaptive"] = 0.0
    else:
        feats["vol_regime_adaptive"] = None

    # --- V9: Cross-factor interaction features ---
    # liquidation_cascade_score: |oi_change_pct| * vol_ma_ratio_5_20
    oi_pct = feats.get("oi_change_pct")
    vmr = feats.get("vol_ma_ratio_5_20")
    if oi_pct is not None and vmr is not None:
        feats["liquidation_cascade_score"] = abs(oi_pct) * vmr
    else:
        feats["liquidation_cascade_score"] = None

    # funding_term_slope: normalized (funding_rate - funding_ma8)
    fr = feats.get("funding_rate")
    fma8 = feats.get("funding_ma8")
    if fr is not None and fma8 is not None:
        denom = max(abs(fma8), 1e-6)
        feats["funding_term_slope"] = (fr - fma8) / denom
    else:
        feats["funding_term_slope"] = None

    # cross_tf_regime_sync: sign(close_vs_ma20) == sign(tf4h_close_vs_ma20)
    # tf4h_close_vs_ma20 is computed externally (multi-timeframe aggregator), not available here
    feats["cross_tf_regime_sync"] = None

    # --- V9: Deribit IV features ---
    # implied_vol_zscore_24
    if state.iv_window_24.full and state._last_implied_vol is not None:
        iv_mean = state.iv_window_24.mean
        iv_std = state.iv_window_24.std
        if iv_std is not None and iv_std > 1e-8:
            feats["implied_vol_zscore_24"] = (state._last_implied_vol - iv_mean) / iv_std
        else:
            feats["implied_vol_zscore_24"] = None
    else:
        feats["implied_vol_zscore_24"] = None

    # iv_rv_spread: implied vol - realized vol (vol_20)
    vol20 = feats.get("vol_20")
    if state._last_implied_vol is not None and vol20 is not None:
        feats["iv_rv_spread"] = state._last_implied_vol - vol20
    else:
        feats["iv_rv_spread"] = None

    # put_call_ratio (passthrough from external feed)
    feats["put_call_ratio"] = state._last_put_call_ratio

    # --- V10: On-chain features ---
    # exchange_netflow_zscore: zscore_7d(inflow - outflow)
    feats["exchange_netflow_zscore"] = _window_zscore(state._onchain_netflow_buf, 7)

    # exchange_supply_change: (today - yesterday) / yesterday
    if len(state._onchain_supply_buf) >= 2:
        prev = state._onchain_supply_buf[-2]
        curr = state._onchain_supply_buf[-1]
        feats["exchange_supply_change"] = (curr - prev) / prev if prev > 1e-8 else 0.0
    else:
        feats["exchange_supply_change"] = None

    # exchange_supply_zscore_30: zscore_30d(SplyExNtv)
    if len(state._onchain_supply_buf) >= 30:
        vals = list(state._onchain_supply_buf)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = sqrt(var) if var > 0 else 0.0
        feats["exchange_supply_zscore_30"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
    else:
        feats["exchange_supply_zscore_30"] = None

    # active_addr_zscore_14: zscore_14d(AdrActCnt)
    feats["active_addr_zscore_14"] = _window_zscore(state._onchain_addr_buf, 14)

    # tx_count_zscore_14: zscore_14d(TxTfrCnt)
    feats["tx_count_zscore_14"] = _window_zscore(state._onchain_tx_buf, 14)

    # hashrate_momentum: (HashRate - EMA14) / EMA14
    if state._onchain_hashrate_ema.ready and state._last_onchain_hashrate is not None:
        ema_val = state._onchain_hashrate_ema.value
        if ema_val is not None and abs(ema_val) > 1e-8:
            feats["hashrate_momentum"] = (state._last_onchain_hashrate - ema_val) / ema_val
        else:
            feats["hashrate_momentum"] = None
    else:
        feats["hashrate_momentum"] = None

    # --- V17: On-chain IC-screened aliases ---
    feats["oc_tx_zscore_7"] = _window_zscore(state._onchain_tx_buf, 7)
    feats["oc_tx_zscore_14"] = _window_zscore(state._onchain_tx_buf, 14)
    feats["oc_addr_zscore_7"] = _window_zscore(state._onchain_addr_buf, 7)
    feats["oc_addr_zscore_14"] = feats["active_addr_zscore_14"]
    feats["oc_flowin_zscore_7"] = _window_zscore(state._onchain_flowin_buf, 7)
    feats["oc_flowin_zscore_14"] = _window_zscore(state._onchain_flowin_buf, 14)
    feats["oc_flowout_zscore_7"] = _window_zscore(state._onchain_flowout_buf, 7)
    feats["oc_flowout_zscore_14"] = _window_zscore(state._onchain_flowout_buf, 14)
    feats["oc_netflow_zscore_7"] = feats["exchange_netflow_zscore"]

    # --- V11: Liquidation features ---
    # liquidation_volume_zscore_24
    if len(state._liq_volume_buf) >= 24:
        vals = list(state._liq_volume_buf)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = sqrt(var) if var > 0 else 0.0
        feats["liquidation_volume_zscore_24"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
    else:
        feats["liquidation_volume_zscore_24"] = None

    # liquidation_imbalance
    feats["liquidation_imbalance"] = state._last_liq_imbalance

    # liquidation_volume_ratio: liq_vol / quote_vol
    if state._last_liq_volume is not None and state._last_quote_volume > 0:
        feats["liquidation_volume_ratio"] = state._last_liq_volume / state._last_quote_volume
    else:
        feats["liquidation_volume_ratio"] = None

    # liquidation_cluster_flag: >3 std liq volume in short window (6 bars)
    if len(state._liq_imbalance_buf) >= 6 and len(state._liq_volume_buf) >= 6:
        recent = list(state._liq_volume_buf)[-6:]
        mean = sum(recent) / len(recent)
        var = sum((v - mean) ** 2 for v in recent) / len(recent)
        std = sqrt(var) if var > 0 else 0.0
        if std > 1e-8 and recent[-1] > mean + 3.0 * std:
            feats["liquidation_cluster_flag"] = 1.0
        else:
            feats["liquidation_cluster_flag"] = 0.0
    else:
        feats["liquidation_cluster_flag"] = None

    # --- V11: Mempool features ---
    if len(state._mempool_fee_buf) >= 24:
        vals = list(state._mempool_fee_buf)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = sqrt(var) if var > 0 else 0.0
        feats["mempool_fee_zscore_24"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
    else:
        feats["mempool_fee_zscore_24"] = None

    if len(state._mempool_size_buf) >= 24:
        vals = list(state._mempool_size_buf)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = sqrt(var) if var > 0 else 0.0
        feats["mempool_size_zscore_24"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
    else:
        feats["mempool_size_zscore_24"] = None

    feats["fee_urgency_ratio"] = state._last_fee_urgency

    # --- V11: Macro features ---
    # dxy_change_5d
    if len(state._dxy_buf) >= 6:
        feats["dxy_change_5d"] = (
            (state._dxy_buf[-1] - state._dxy_buf[-6]) / state._dxy_buf[-6]
            if state._dxy_buf[-6] > 1e-8 else 0.0
        )
    else:
        feats["dxy_change_5d"] = None

    # spx_btc_corr_30d
    if len(state._spx_buf) >= 10 and len(state._btc_close_buf_30) >= 10:
        n = min(len(state._spx_buf), len(state._btc_close_buf_30))
        spx_vals = list(state._spx_buf)[-n:]
        btc_vals = list(state._btc_close_buf_30)[-n:]
        # Compute returns
        if n >= 2:
            spx_rets = [(spx_vals[i] - spx_vals[i-1]) / spx_vals[i-1] if spx_vals[i-1] > 0 else 0.0 for i in
                range(1, n)]
            btc_rets = [(btc_vals[i] - btc_vals[i-1]) / btc_vals[i-1] if btc_vals[i-1] > 0 else 0.0 for i in
                range(1, n)]
            m = len(spx_rets)
            if m >= 5:
                mean_s = sum(spx_rets) / m
                mean_b = sum(btc_rets) / m
                cov = sum((spx_rets[i] - mean_s) * (btc_rets[i] - mean_b) for i in range(m)) / m
                var_s = sum((r - mean_s) ** 2 for r in spx_rets) / m
                var_b = sum((r - mean_b) ** 2 for r in btc_rets) / m
                denom = sqrt(var_s * var_b)
                feats["spx_btc_corr_30d"] = cov / denom if denom > 1e-8 else 0.0
            else:
                feats["spx_btc_corr_30d"] = None
        else:
            feats["spx_btc_corr_30d"] = None
    else:
        feats["spx_btc_corr_30d"] = None

    # spx_overnight_ret
    if state._last_spx_close is not None and state._prev_spx_close is not None and state._prev_spx_close > 0:
        feats["spx_overnight_ret"] = (state._last_spx_close - state._prev_spx_close) / state._prev_spx_close
    else:
        feats["spx_overnight_ret"] = None

    # vix_zscore_14
    if len(state._vix_buf) >= 14:
        vals = list(state._vix_buf)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = sqrt(var) if var > 0 else 0.0
        feats["vix_zscore_14"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
    else:
        feats["vix_zscore_14"] = None

    # --- V11: Social sentiment features ---
    if len(state._social_vol_buf) >= 24:
        vals = list(state._social_vol_buf)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = sqrt(var) if var > 0 else 0.0
        feats["social_volume_zscore_24"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
    else:
        feats["social_volume_zscore_24"] = None

    feats["social_sentiment_score"] = state._last_sentiment_score

    # social_volume_price_div: social volume up + price down (or vice versa) → divergence
    if state._last_social_volume is not None and len(state._social_vol_buf) >= 2 and len(state.close_history) >= 2:
        sv_change = state._social_vol_buf[-1] - state._social_vol_buf[-2]
        price_change = state.close_history[-1] - state.close_history[-2]
        if (sv_change > 0 and price_change < 0) or (sv_change < 0 and price_change > 0):
            feats["social_volume_price_div"] = 1.0
        else:
            feats["social_volume_price_div"] = 0.0
    else:
        feats["social_volume_price_div"] = None

    # ── V12: ALT cross-asset features (requires btc_close parameter) ──
    if btc_close is not None and btc_close > 0:
        state._btc_ref_buf.append(btc_close)

        # BTC ret_1 (lagged BTC return as lead signal for ALT)
        if len(state._btc_ref_buf) >= 2:
            feats["btc_lead_ret_1"] = state._btc_ref_buf[-1] / state._btc_ref_buf[-2] - 1
        else:
            feats["btc_lead_ret_1"] = None

        # BTC vol_20 for relative vol
        if len(state._btc_ref_buf) >= 3:
            btc_rets = []
            for j in range(1, min(21, len(state._btc_ref_buf))):
                btc_rets.append(state._btc_ref_buf[-j] / state._btc_ref_buf[-j - 1] - 1)
            if len(btc_rets) >= 5:
                _m = sum(btc_rets) / len(btc_rets)
                btc_vol = sqrt(sum((r - _m) ** 2 for r in btc_rets) / len(btc_rets))
            else:
                btc_vol = None
            state._btc_ref_vol_buf.append(btc_vol if btc_vol else 0)
        else:
            btc_vol = None

        # ALT/BTC ratio tracking
        if close is not None and close > 0:
            ratio = close / btc_close
            state._alt_btc_ratio_buf.append(ratio)

        # btc_relative_strength_24: ALT ret_24 - BTC ret_24
        if close is not None and len(state.close_history) >= 24 and len(state._btc_ref_buf) >= 24:
            alt_ret24 = close / state.close_history[-24] - 1
            btc_ret24 = state._btc_ref_buf[-1] / state._btc_ref_buf[-24] - 1
            feats["btc_relative_strength_24"] = alt_ret24 - btc_ret24
        else:
            feats["btc_relative_strength_24"] = None

        # btc_relative_strength_6
        if close is not None and len(state.close_history) >= 6 and len(state._btc_ref_buf) >= 6:
            alt_ret6 = close / state.close_history[-6] - 1
            btc_ret6 = state._btc_ref_buf[-1] / state._btc_ref_buf[-6] - 1
            feats["btc_relative_strength_6"] = alt_ret6 - btc_ret6
        else:
            feats["btc_relative_strength_6"] = None

        # btc_ratio_ma20_dev: (ratio / MA20_ratio) - 1
        if len(state._alt_btc_ratio_buf) >= 20:
            ratio_ma = sum(list(state._alt_btc_ratio_buf)[-20:]) / 20
            if ratio_ma > 0:
                feats["btc_ratio_ma20_dev"] = state._alt_btc_ratio_buf[-1] / ratio_ma - 1
            else:
                feats["btc_ratio_ma20_dev"] = None
        else:
            feats["btc_ratio_ma20_dev"] = None

        # btc_dom_momentum: BTC ret_24 - ALT ret_24 (opposite of relative strength)
        _rs24 = feats.get("btc_relative_strength_24")
        feats["btc_dom_momentum"] = -float(_rs24) if _rs24 is not None else None

        # btc_vol_ratio: ALT vol_20 / BTC vol_20
        vol20 = feats.get("vol_20")
        if vol20 is not None and btc_vol is not None and btc_vol > 1e-8:
            feats["btc_vol_ratio"] = vol20 / btc_vol
        else:
            feats["btc_vol_ratio"] = None
    else:
        feats["btc_relative_strength_24"] = None
        feats["btc_relative_strength_6"] = None
        feats["btc_ratio_ma20_dev"] = None
        feats["btc_dom_momentum"] = None
        feats["btc_lead_ret_1"] = None
        feats["btc_vol_ratio"] = None

    # ── V16: Orderbook proxy + IV spread + liquidation features ──
    # OB spread proxy: (high - low) / close — intrabar range
    if len(state.high_history) >= 1 and len(state.low_history) >= 1 and close and close > 0:
        feats["ob_spread_proxy"] = (state.high_history[-1] - state.low_history[-1]) / close
    else:
        feats["ob_spread_proxy"] = None

    # OB imbalance proxy: (taker_buy - taker_sell) / total
    if len(state._taker_imb_buf_6) >= 1:
        feats["ob_imbalance_proxy"] = state._taker_imb_buf_6[-1]
    else:
        feats["ob_imbalance_proxy"] = None

    # OB imbalance × volume: strength-weighted
    if feats["ob_imbalance_proxy"] is not None and len(state._vol_buf_24) >= 20:
        vol_mean = sum(list(state._vol_buf_24)[-20:]) / 20
        cur_vol = state._vol_buf_6[-1] if state._vol_buf_6 else 0
        vol_ratio = cur_vol / vol_mean if vol_mean > 0 else 1.0
        feats["ob_imbalance_x_vol"] = feats["ob_imbalance_proxy"] * vol_ratio
    else:
        feats["ob_imbalance_x_vol"] = None

    # OB imbalance cumulative 6-bar
    if len(state._taker_imb_buf_6) >= 6:
        feats["ob_imbalance_cum6"] = sum(state._taker_imb_buf_6)
    else:
        feats["ob_imbalance_cum6"] = None

    # OB volume clock: MA6/MA24 - 1
    if len(state._vol_buf_6) >= 6 and len(state._vol_buf_24) >= 24:
        ma6 = sum(state._vol_buf_6) / 6
        ma24 = sum(state._vol_buf_24) / 24
        feats["ob_volume_clock"] = ma6 / ma24 - 1 if ma24 > 0 else 0.0
    else:
        feats["ob_volume_clock"] = None

    # Liquidation volume z-score (24-bar)
    if len(state._liq_vol_buf_24) >= 12:
        liq_arr = list(state._liq_vol_buf_24)
        mu = sum(liq_arr) / len(liq_arr)
        var = sum((x - mu) ** 2 for x in liq_arr) / len(liq_arr)
        std = var ** 0.5
        feats["liq_volume_zscore_24"] = (liq_arr[-1] - mu) / std if std > 1e-10 else 0.0
    else:
        feats["liq_volume_zscore_24"] = None

    # ── V15: Interaction & statistical features ──
    # IC-screened 2026-03-18: all significant (p<0.001) across 4 symbols.
    _ret1 = feats.get("ret_1")
    _vol20 = feats.get("vol_20")
    _rsi14 = feats.get("rsi_14")
    _atr14 = feats.get("atr_norm_14")
    _bbpctb = feats.get("bb_pctb_20")
    _trend50 = feats.get("close_vs_ma50")

    # Interaction: ret_1 × vol_20
    feats["ret1_x_vol"] = _ret1 * _vol20 if _ret1 is not None and _vol20 is not None else None
    # Interaction: rsi_14 × atr_norm_14
    feats["rsi_x_atr"] = _rsi14 * _atr14 if _rsi14 is not None and _atr14 is not None else None
    # Interaction: rsi_14 × vol_20
    feats["rsi_x_vol"] = _rsi14 * _vol20 if _rsi14 is not None and _vol20 is not None else None
    # Interaction: close_vs_ma50 × vol_20
    feats["trend_x_vol"] = _trend50 * _vol20 if _trend50 is not None and _vol20 is not None else None
    # Interaction: bb_pctb_20 × vol_20
    feats["bb_x_vol"] = _bbpctb * _vol20 if _bbpctb is not None and _vol20 is not None else None

    # Return autocorrelation (24-bar)
    rbuf = state._ret_buf_24
    if len(rbuf) >= 24:
        r = list(rbuf)
        r1, r2 = r[:-1], r[1:]
        mr1 = sum(r1) / len(r1)
        mr2 = sum(r2) / len(r2)
        cov = sum((a - mr1) * (b - mr2) for a, b in zip(r1, r2)) / len(r1)
        v1 = sum((a - mr1) ** 2 for a in r1) / len(r1)
        v2 = sum((b - mr2) ** 2 for b in r2) / len(r2)
        denom = (v1 * v2) ** 0.5
        feats["ret_autocorr_24"] = cov / denom if denom > 1e-12 else 0.0
    else:
        feats["ret_autocorr_24"] = None

    # Return skewness (24-bar)
    if len(rbuf) >= 24:
        r = list(rbuf)
        mu = sum(r) / len(r)
        std = (sum((x - mu) ** 2 for x in r) / len(r)) ** 0.5
        if std > 1e-12:
            feats["ret_skew_24"] = sum(((x - mu) / std) ** 3 for x in r) / len(r)
        else:
            feats["ret_skew_24"] = 0.0
    else:
        feats["ret_skew_24"] = None

    # ── V19: Implied Volatility features (DVOL from Deribit) ──
    dvol_buf = state._dvol_buf
    dvol_n = len(dvol_buf)

    # dvol_chg_72: 72-bar DVOL change rate
    if dvol_n >= 73 and dvol_buf[-73] > 0:
        feats["dvol_chg_72"] = (dvol_buf[-1] - dvol_buf[-73]) / dvol_buf[-73]
    else:
        feats["dvol_chg_72"] = None

    # iv_term_struct: MA(24) / MA(168) - 1
    if dvol_n >= 168:
        short_ma = sum(list(dvol_buf)[-24:]) / 24
        long_ma = sum(list(dvol_buf)[-168:]) / 168
        feats["iv_term_struct"] = short_ma / long_ma - 1 if long_ma > 0 else 0.0
    else:
        feats["iv_term_struct"] = None

    # dvol_z: z-score over 168 bars
    if dvol_n >= 168:
        window = list(dvol_buf)[-168:]
        mu_dv = sum(window) / 168
        var_dv = sum((x - mu_dv) ** 2 for x in window) / 168
        std_dv = var_dv ** 0.5
        feats["dvol_z"] = (dvol_buf[-1] - mu_dv) / max(std_dv, 0.1)
    else:
        feats["dvol_z"] = None

    # dvol_chg_24: 24-bar DVOL change rate
    if dvol_n >= 25 and dvol_buf[-25] > 0:
        feats["dvol_chg_24"] = (dvol_buf[-1] - dvol_buf[-25]) / dvol_buf[-25]
    else:
        feats["dvol_chg_24"] = None

    # dvol_mean_rev: DVOL / MA(720) - 1
    if dvol_n >= 720:
        ma720 = sum(list(dvol_buf)[-720:]) / 720
        feats["dvol_mean_rev"] = dvol_buf[-1] / ma720 - 1 if ma720 > 0 else 0.0
    else:
        feats["dvol_mean_rev"] = None

