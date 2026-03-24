# features/enriched_features_extended_v2.py
"""Extended feature computation part 2 (V11-V19) for EnrichedFeatureComputer.

Extracted from enriched_features_extended.py to reduce file size.
Called by compute_extended_features() to populate V11+ features.
"""
from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Deque, Dict, Optional

if TYPE_CHECKING:
    from features.enriched_computer import _SymbolState


def compute_extended_features_v2(
    state: "_SymbolState",
    feats: Dict[str, Optional[float]],
    close: float,
    n: int,
    hist: "Deque[float]",
    tbr: Optional[float],
    btc_close: Optional[float],
) -> None:
    """Compute V11-V19 features and add them to feats dict.

    This is the exact code that was in enriched_features_extended.py lines 369-706,
    extracted verbatim to reduce file size.
    """
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


