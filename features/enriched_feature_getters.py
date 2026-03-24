# features/enriched_feature_getters.py
"""Feature extraction logic for _SymbolState.get_features().

Extracted from enriched_computer.py to reduce file size.
Called by _SymbolState.get_features() to compute base features (V1-V4, V13-V14, V18).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from features.enriched_computer import _SymbolState

from features.enriched_computer import _ALL_MULTI_DOMINANCE_FEATURES, _MULTI_DOMINANCE_PREFIXES


def compute_base_features(
    state: "_SymbolState",
    btc_close: float | None = None,
) -> Dict[str, Optional[float]]:
    """Compute base features (V1-V4, funding, OI, LS, dominance, V13, V18) from state.

    Returns the feature dict which will be extended by V5-V19 features
    from enriched_features_extended.py.
    """
    feats: Dict[str, Optional[float]] = {}
    close = state._last_close

    # --- Multi-horizon returns ---
    hist = state.close_history
    n = len(hist)
    for horizon, name in [(1, "ret_1"), (3, "ret_3"), (6, "ret_6"),
                          (12, "ret_12"), (24, "ret_24")]:
        if n > horizon and hist[-1 - horizon] != 0:
            feats[name] = (hist[-1] - hist[-1 - horizon]) / hist[-1 - horizon]
        else:
            feats[name] = None

    # --- MA crossovers ---
    ma10 = state.ma_10.mean if state.ma_10.full else None
    ma30 = state.ma_30.mean if state.ma_30.full else None
    ma5 = state.ma_5.mean if state.ma_5.full else None
    ma20 = state.ma_20.mean if state.ma_20.full else None
    ma50 = state.ma_50.mean if state.ma_50.full else None

    if ma10 is not None and ma30 is not None and ma30 != 0:
        feats["ma_cross_10_30"] = ma10 / ma30 - 1.0
    else:
        feats["ma_cross_10_30"] = None

    if ma5 is not None and ma20 is not None and ma20 != 0:
        feats["ma_cross_5_20"] = ma5 / ma20 - 1.0
    else:
        feats["ma_cross_5_20"] = None

    if close is not None and ma20 is not None and ma20 != 0:
        feats["close_vs_ma20"] = close / ma20 - 1.0
    else:
        feats["close_vs_ma20"] = None

    if close is not None and ma50 is not None and ma50 != 0:
        feats["close_vs_ma50"] = close / ma50 - 1.0
    else:
        feats["close_vs_ma50"] = None

    # --- RSI ---
    feats["rsi_14"] = state.rsi_14.value
    feats["rsi_6"] = state.rsi_6.value
    # Normalize RSI to [-1, 1] range for ML
    if feats["rsi_14"] is not None:
        feats["rsi_14"] = (feats["rsi_14"] - 50.0) / 50.0
    if feats["rsi_6"] is not None:
        feats["rsi_6"] = (feats["rsi_6"] - 50.0) / 50.0

    # --- MACD ---
    if state.ema_12.ready and state.ema_26.ready:
        macd_line = state.ema_12.value - state.ema_26.value  # type: ignore[operator]
        # Normalize by close
        if close and close != 0:
            feats["macd_line"] = macd_line / close
        else:
            feats["macd_line"] = None

        if state.macd_signal_ema.ready:
            sig = state.macd_signal_ema.value
            if close and close != 0:
                feats["macd_signal"] = sig / close  # type: ignore[operator]
                feats["macd_hist"] = (macd_line - sig) / close  # type: ignore[operator]
            else:
                feats["macd_signal"] = None
                feats["macd_hist"] = None
        else:
            feats["macd_signal"] = None
            feats["macd_hist"] = None
    else:
        feats["macd_line"] = None
        feats["macd_signal"] = None
        feats["macd_hist"] = None

    # --- Bollinger Bands ---
    if state.bb_window.full:
        bb_mid = state.bb_window.mean
        bb_std = state.bb_window.std
        if bb_mid and bb_std and bb_mid != 0:
            upper = bb_mid + 2.0 * bb_std
            lower = bb_mid - 2.0 * bb_std
            feats["bb_width_20"] = (upper - lower) / bb_mid
            band_range = upper - lower
            if band_range != 0 and close is not None:
                feats["bb_pctb_20"] = (close - lower) / band_range
            else:
                feats["bb_pctb_20"] = None
        else:
            feats["bb_width_20"] = None
            feats["bb_pctb_20"] = None
    else:
        feats["bb_width_20"] = None
        feats["bb_pctb_20"] = None

    # --- ATR (normalized by close) ---
    atr_val = state.atr_14.value
    if atr_val is not None and close and close != 0:
        feats["atr_norm_14"] = atr_val / close
    else:
        feats["atr_norm_14"] = None

    # --- ADX (trend strength, 0-100) ---
    feats["adx_14"] = state.adx_14.value

    # --- Volatility ---
    feats["vol_20"] = state.return_window_20.std if state.return_window_20.full else None
    feats["vol_5"] = state.return_window_5.std if state.return_window_5.full else None

    # --- Volume features ---
    vol_ma20 = state.vol_window_20.mean if state.vol_window_20.full else None
    vol_ma5 = state.vol_window_5.mean if state.vol_window_5.full else None
    if vol_ma20 and vol_ma20 != 0 and state.vol_window_20.n > 0:
        feats["vol_ratio_20"] = state._last_volume / vol_ma20
    else:
        feats["vol_ratio_20"] = None

    if vol_ma5 is not None and vol_ma20 is not None and vol_ma20 != 0:
        feats["vol_ma_ratio_5_20"] = vol_ma5 / vol_ma20
    else:
        feats["vol_ma_ratio_5_20"] = None

    # --- Candle structure ---
    if n > 0 and len(state.open_history) > 0 and len(state.high_history) > 0 and len(state.low_history) > 0:
        o = state.open_history[-1]
        h = state.high_history[-1]
        l = state.low_history[-1]  # noqa: E741
        c = hist[-1]
        hl_range = h - l
        if hl_range > 0:
            feats["body_ratio"] = (c - o) / hl_range
            feats["upper_shadow"] = (h - max(o, c)) / hl_range
            feats["lower_shadow"] = (min(o, c) - l) / hl_range
        else:
            feats["body_ratio"] = None
            feats["upper_shadow"] = None
            feats["lower_shadow"] = None
    else:
        feats["body_ratio"] = None
        feats["upper_shadow"] = None
        feats["lower_shadow"] = None

    # --- Mean reversion (z-score) ---
    if state.bb_window.full and close is not None:
        bb_mid = state.bb_window.mean
        bb_std = state.bb_window.std
        if bb_mid is not None and bb_std is not None and bb_std != 0:
            feats["mean_reversion_20"] = (close - bb_mid) / bb_std
        else:
            feats["mean_reversion_20"] = None
    else:
        feats["mean_reversion_20"] = None

    # --- Price acceleration (change in momentum) ---
    current_momentum = feats.get("ma_cross_10_30")
    if current_momentum is not None and state._prev_momentum is not None:
        feats["price_acceleration"] = current_momentum - state._prev_momentum
    else:
        feats["price_acceleration"] = None
    state._prev_momentum = current_momentum

    # --- Time-of-day (cyclical encoding) ---
    if state._last_hour >= 0:
        feats["hour_sin"] = math.sin(2 * math.pi * state._last_hour / 24.0)
        feats["hour_cos"] = math.cos(2 * math.pi * state._last_hour / 24.0)
    else:
        feats["hour_sin"] = None
        feats["hour_cos"] = None

    # --- Day-of-week (cyclical encoding) ---
    if state._last_dow >= 0:
        feats["dow_sin"] = math.sin(2 * math.pi * state._last_dow / 7.0)
        feats["dow_cos"] = math.cos(2 * math.pi * state._last_dow / 7.0)
    else:
        feats["dow_sin"] = None
        feats["dow_cos"] = None

    # --- Volatility regime ---
    vol5 = feats.get("vol_5")
    vol20 = feats.get("vol_20")
    if vol5 is not None and vol20 is not None and vol20 != 0:
        feats["vol_regime"] = vol5 / vol20
    else:
        feats["vol_regime"] = None

    # --- Funding rate features ---
    feats["funding_rate"] = state._last_funding_rate
    feats["funding_ma8"] = state.funding_ema.value if state.funding_ema.ready else None

    # --- Kline microstructure features ---
    trades = state._last_trades
    if trades > 0 and state.trades_ema_20.ready:
        ema_trades_20 = state.trades_ema_20.value
        feats["trade_intensity"] = trades / ema_trades_20 if ema_trades_20 and ema_trades_20 > 0 else None
    else:
        feats["trade_intensity"] = None

    volume = state._last_volume
    tbr: Optional[float] = None
    if trades > 0 and volume > 0:
        tbr = state._last_taker_buy_volume / volume
        feats["taker_buy_ratio"] = tbr
    else:
        feats["taker_buy_ratio"] = None

    feats["taker_buy_ratio_ma10"] = state.taker_buy_ratio_ema_10.value if state.taker_buy_ratio_ema_10.ready else None

    if tbr is not None:
        feats["taker_imbalance"] = 2.0 * tbr - 1.0
    else:
        feats["taker_imbalance"] = None

    if trades > 0:
        ats = state._last_quote_volume / trades
        feats["avg_trade_size"] = ats
        ats_ema = state.avg_trade_size_ema_20.value
        if state.avg_trade_size_ema_20.ready and ats_ema and ats_ema > 0:
            feats["avg_trade_size_ratio"] = ats / ats_ema
        else:
            feats["avg_trade_size_ratio"] = None

        vpt = volume / trades
        vpt_ema = state.volume_per_trade_ema_20.value
        if state.volume_per_trade_ema_20.ready and vpt_ema and vpt_ema > 0:
            feats["volume_per_trade"] = vpt / vpt_ema
        else:
            feats["volume_per_trade"] = None
    else:
        feats["avg_trade_size"] = None
        feats["avg_trade_size_ratio"] = None
        feats["volume_per_trade"] = None

    if state.trades_ema_5.ready and state.trades_ema_20.ready:
        ema5 = state.trades_ema_5.value
        ema20 = state.trades_ema_20.value
        if ema5 is not None and ema20 is not None and ema20 > 0:
            feats["trade_count_regime"] = ema5 / ema20
        else:
            feats["trade_count_regime"] = None
    else:
        feats["trade_count_regime"] = None

    # --- Funding deep features ---
    if state.funding_window_24.full:
        f_mean = state.funding_window_24.mean
        f_std = state.funding_window_24.std
        if f_std is not None and f_std > 1e-12 and state._last_funding_rate is not None:
            zscore = (state._last_funding_rate - f_mean) / f_std
            feats["funding_zscore_24"] = zscore
            feats["funding_extreme"] = 1.0 if abs(zscore) > 2.0 else 0.0
        else:
            feats["funding_zscore_24"] = None
            feats["funding_extreme"] = None
    else:
        feats["funding_zscore_24"] = None
        feats["funding_extreme"] = None

    fr_ma8 = feats.get("funding_ma8")
    if state._last_funding_rate is not None and fr_ma8 is not None:
        feats["funding_momentum"] = state._last_funding_rate - fr_ma8
    else:
        feats["funding_momentum"] = None

    if len(state.funding_history_8) == 8:
        feats["funding_cumulative_8"] = sum(state.funding_history_8)
    else:
        feats["funding_cumulative_8"] = None

    feats["funding_sign_persist"] = float(state._funding_sign_count) if state._funding_sign_count > 0 else None

    # --- OI features ---
    feats["oi_change_pct"] = state._last_oi_change_pct
    feats["oi_change_ma8"] = state.oi_change_ema_8.value if state.oi_change_ema_8.ready else None

    # OI-price divergence: price up but OI down (or vice versa)
    ret1 = feats.get("ret_1")
    if ret1 is not None and state._last_oi_change_pct is not None:
        # Divergence: opposite signs -> positive value; same signs -> negative
        price_sign = 1.0 if ret1 > 0 else (-1.0 if ret1 < 0 else 0.0)
        oi_sign = 1.0 if state._last_oi_change_pct > 0 else (-1.0 if state._last_oi_change_pct < 0 else 0.0)
        feats["oi_close_divergence"] = -price_sign * oi_sign
    else:
        feats["oi_close_divergence"] = None

    # --- LS Ratio features ---
    feats["ls_ratio"] = state._last_ls_ratio
    if state.ls_ratio_window_24.full and state._last_ls_ratio is not None:
        ls_mean = state.ls_ratio_window_24.mean
        ls_std = state.ls_ratio_window_24.std
        if ls_std is not None and ls_std > 1e-12:
            zscore = (state._last_ls_ratio - ls_mean) / ls_std
            feats["ls_ratio_zscore_24"] = zscore
            feats["ls_extreme"] = 1.0 if abs(zscore) > 2.0 else 0.0
        else:
            feats["ls_ratio_zscore_24"] = None
            feats["ls_extreme"] = None
    else:
        feats["ls_ratio_zscore_24"] = None
        feats["ls_extreme"] = None

    # --- V14: BTC Dominance features ---
    buf = state._dom_ratio_buf
    if len(buf) >= 21:
        cur = buf[-1]
        ma20 = sum(list(buf)[-20:]) / 20
        feats["btc_dom_dev_20"] = cur / ma20 - 1 if ma20 > 0 else None
    else:
        feats["btc_dom_dev_20"] = None

    if len(buf) >= 51:
        cur = buf[-1]
        ma50 = sum(list(buf)[-50:]) / 50
        feats["btc_dom_dev_50"] = cur / ma50 - 1 if ma50 > 0 else None
    else:
        feats["btc_dom_dev_50"] = None

    if len(buf) >= 25:
        feats["btc_dom_ret_24"] = buf[-1] / buf[-25] - 1 if buf[-25] > 0 else None
    else:
        feats["btc_dom_ret_24"] = None

    if len(buf) >= 73:
        feats["btc_dom_ret_72"] = buf[-1] / buf[-73] - 1 if buf[-73] > 0 else None
    else:
        feats["btc_dom_ret_72"] = None

    # --- V14b: Multi-ratio dominance features ---
    for name in _ALL_MULTI_DOMINANCE_FEATURES:
        feats[name] = None
    for prefix in _MULTI_DOMINANCE_PREFIXES:
        ratio_buf = state._multi_dom_ratio_bufs.get(prefix)
        if not ratio_buf:
            continue
        if len(ratio_buf) >= 21:
            cur = ratio_buf[-1]
            ma20 = sum(list(ratio_buf)[-20:]) / 20
            feats[f"{prefix}_dev_20"] = cur / ma20 - 1 if ma20 > 0 else None
        if len(ratio_buf) >= 25:
            prev = ratio_buf[-25]
            feats[f"{prefix}_ret_24"] = ratio_buf[-1] / prev - 1 if prev > 0 else None

    # --- V13: Enhanced OI/LS/Taker features ---
    # oi_pct_4h: 4-bar OI change rate
    if len(state._oi_buf_12) >= 5 and state._oi_buf_12[-5] > 0:
        feats["oi_pct_4h"] = (state._oi_buf_12[-1] - state._oi_buf_12[-5]) / state._oi_buf_12[-5]
    else:
        feats["oi_pct_4h"] = None

    # ls_deviation: ls_ratio - 1.0
    if state._last_ls_ratio is not None:
        feats["ls_deviation"] = state._last_ls_ratio - 1.0
    else:
        feats["ls_deviation"] = None

    # taker_buy_sell_ratio: buy_vol / sell_vol
    if state._last_taker_buy_volume > 0 and state._last_taker_sell_volume > 0:
        feats["taker_buy_sell_ratio"] = state._last_taker_buy_volume / state._last_taker_sell_volume
    else:
        feats["taker_buy_sell_ratio"] = None

    # top_retail_divergence: top_trader_ls - global_ls
    if state._last_top_trader_ls is not None and state._last_ls_ratio is not None:
        feats["top_retail_divergence"] = state._last_top_trader_ls - state._last_ls_ratio
    else:
        feats["top_retail_divergence"] = None

    # oi_price_divergence_12: 12-bar OI change - 12-bar price change
    if len(state._oi_buf_12) >= 12 and state._oi_buf_12[0] > 0:
        oi_change_12 = (state._oi_buf_12[-1] - state._oi_buf_12[0]) / state._oi_buf_12[0]
        ret_12 = feats.get("ret_12")
        if ret_12 is not None:
            feats["oi_price_divergence_12"] = oi_change_12 - ret_12
        else:
            feats["oi_price_divergence_12"] = None
    else:
        feats["oi_price_divergence_12"] = None

    # --- V18: OI change rate (24-bar and 96-bar) ---
    buf97 = state._oi_buf_97
    if len(buf97) >= 25 and buf97[-25] > 0:
        feats["oi_change_24"] = (buf97[-1] - buf97[-25]) / buf97[-25]
    else:
        feats["oi_change_24"] = None

    if len(buf97) >= 97 and buf97[0] > 0:
        feats["oi_change_96"] = (buf97[-1] - buf97[0]) / buf97[0]
    else:
        feats["oi_change_96"] = None

    # --- V18: 3-period cumulative funding ---
    if len(state.funding_history_8) >= 3:
        feats["funding_cum_3"] = sum(list(state.funding_history_8)[-3:])
    else:
        feats["funding_cum_3"] = None

    # --- V5-V19: Extended features (extracted to enriched_features_extended.py) ---
    from features.enriched_features_extended import compute_extended_features
    compute_extended_features(state, feats, close, n, hist, tbr, btc_close)

    return feats
