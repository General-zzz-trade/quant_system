"""push_bar() — the bar update logic for _SymbolState.

Extracted from enriched_computer.py to keep individual files under 500 lines.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional


def push_bar(state, close: float, volume: float, high: float, low: float, open_: float,
             *, hour: int = -1, dow: int = -1, funding_rate: Optional[float] = None,
             trades: float = 0.0, taker_buy_volume: float = 0.0,
             quote_volume: float = 0.0,
             taker_buy_quote_volume: float = 0.0,
             open_interest: Optional[float] = None,
             ls_ratio: Optional[float] = None,
             top_trader_ls_ratio: Optional[float] = None,
             eth_close: Optional[float] = None,
             spot_close: Optional[float] = None,
             fear_greed: Optional[float] = None,
             implied_vol: Optional[float] = None,
             put_call_ratio: Optional[float] = None,
             onchain_metrics: Optional[Dict[str, float]] = None,
             liquidation_metrics: Optional[Dict[str, float]] = None,
             mempool_metrics: Optional[Dict[str, float]] = None,
             macro_metrics: Optional[Dict[str, float]] = None,
             sentiment_metrics: Optional[Dict[str, float]] = None,
             multi_dom_ratios: Optional[Dict[str, float]] = None,
             dvol: Optional[float] = None) -> None:
    """Push a new bar of data and update all trackers on state."""
    state._last_hour = hour
    state._last_dow = dow
    if funding_rate is not None:
        state._last_funding_rate = funding_rate
        state.funding_ema.push(funding_rate)
        state.funding_window_24.push(funding_rate)
        state.funding_history_8.append(funding_rate)
        sign = 1 if funding_rate > 0 else (-1 if funding_rate < 0 else 0)
        if sign != 0:
            if sign == state._funding_last_sign:
                state._funding_sign_count += 1
            else:
                state._funding_sign_count = 1
                state._funding_last_sign = sign

    if open_interest is not None:
        if state._last_oi is not None and state._last_oi > 0:
            change = (open_interest - state._last_oi) / state._last_oi
            state._prev_oi_change_for_accel = state._last_oi_change_pct
            state._last_oi_change_pct = change
            state.oi_change_ema_8.push(change)
        state._last_oi = open_interest
        state._oi_buf_12.append(open_interest)
        state._oi_buf_97.append(open_interest)
        if close > 0 and volume > 0:
            raw_lev = open_interest / (close * volume)
            state.leverage_proxy_ema.push(raw_lev)

    if top_trader_ls_ratio is not None:
        state._last_top_trader_ls = top_trader_ls_ratio

    taker_sell = volume - taker_buy_volume if volume > 0 and taker_buy_volume > 0 else 0.0
    state._last_taker_sell_volume = taker_sell

    if eth_close is not None and eth_close > 0:
        state._last_eth_close = eth_close
        dom_ratio = close / eth_close
        state._dom_ratio_buf.append(dom_ratio)
    if multi_dom_ratios:
        for prefix, ratio in multi_dom_ratios.items():
            if ratio <= 0:
                continue
            state._multi_dom_ratio_bufs.setdefault(prefix, deque(maxlen=25)).append(ratio)

    if dvol is not None and not math.isnan(dvol):
        state._dvol_buf.append(dvol)

    if ls_ratio is not None:
        state._last_ls_ratio = ls_ratio
        state.ls_ratio_window_24.push(ls_ratio)

    if spot_close is not None and close > 0 and spot_close > 0:
        basis = (close - spot_close) / spot_close
        state._last_basis = basis
        state.basis_window_24.push(basis)
        state.basis_ema_8.push(basis)

    if fear_greed is not None:
        if state._last_fgi is None or abs(fear_greed - state._last_fgi) > 0.01:
            state.fgi_window_7.push(fear_greed)
            state.fgi_window_14.push(fear_greed)
            state.fgi_history_7d.append(fear_greed)
        state._last_fgi = fear_greed

    if implied_vol is not None:
        state._last_implied_vol = implied_vol
        state.iv_window_24.push(implied_vol)
    if put_call_ratio is not None:
        state._last_put_call_ratio = put_call_ratio

    if onchain_metrics is not None:
        flow_in = onchain_metrics.get("FlowInExUSD")
        flow_out = onchain_metrics.get("FlowOutExUSD")
        if flow_in is not None:
            state._onchain_flowin_buf.append(flow_in)
        if flow_out is not None:
            state._onchain_flowout_buf.append(flow_out)
        if flow_in is not None and flow_out is not None:
            state._onchain_netflow_buf.append(flow_in - flow_out)
        supply = onchain_metrics.get("SplyExNtv")
        if supply is not None:
            state._onchain_supply_buf.append(supply)
            state._last_onchain_supply = supply
        addr = onchain_metrics.get("AdrActCnt")
        if addr is not None:
            state._onchain_addr_buf.append(addr)
        tx = onchain_metrics.get("TxTfrCnt")
        if tx is not None:
            state._onchain_tx_buf.append(tx)
        hr = onchain_metrics.get("HashRate")
        if hr is not None:
            state._onchain_hashrate_ema.push(hr)
            state._last_onchain_hashrate = hr

    if liquidation_metrics is not None:
        total = liquidation_metrics.get("liq_total_volume", 0.0)
        buy = liquidation_metrics.get("liq_buy_volume", 0.0)
        sell = liquidation_metrics.get("liq_sell_volume", 0.0)
        state._liq_volume_buf.append(total)
        state._last_liq_volume = total
        state._last_liq_count = liquidation_metrics.get("liq_count", 0.0)
        imb = (buy - sell) / total if total > 0 else 0.0
        state._liq_imbalance_buf.append(imb)
        state._last_liq_imbalance = imb

    if mempool_metrics is not None:
        fee = mempool_metrics.get("fastest_fee")
        if fee is not None:
            state._mempool_fee_buf.append(fee)
        size = mempool_metrics.get("mempool_size")
        if size is not None:
            state._mempool_size_buf.append(size)
        eco = mempool_metrics.get("economy_fee")
        if fee is not None and eco is not None and eco > 0:
            state._last_fee_urgency = fee / eco

    if macro_metrics is not None:
        date_str_raw = macro_metrics.get("date")
        date_str: Optional[str] = str(date_str_raw) if date_str_raw is not None else None
        if date_str is None or date_str != state._last_macro_date:
            state._last_macro_date = date_str
            dxy = macro_metrics.get("dxy")
            if dxy is not None:
                state._dxy_buf.append(dxy)
            spx = macro_metrics.get("spx")
            if spx is not None:
                state._prev_spx_close = state._last_spx_close
                state._last_spx_close = spx
                state._spx_buf.append(spx)
            vix = macro_metrics.get("vix")
            if vix is not None:
                state._last_vix = vix
                state._vix_buf.append(vix)
        state._btc_close_buf_30.append(close)

    if sentiment_metrics is not None:
        sv = sentiment_metrics.get("social_volume")
        if sv is not None:
            state._social_vol_buf.append(sv)
            state._last_social_volume = sv
        ss = sentiment_metrics.get("sentiment_score")
        if ss is not None:
            state._last_sentiment_score = ss

    state._last_trades = trades
    state._last_taker_buy_volume = taker_buy_volume
    state._last_taker_buy_quote_volume = taker_buy_quote_volume
    state._last_quote_volume = quote_volume

    if volume > 0:
        state.vwap_cv_window.push(close * volume)
        state.vwap_v_window.push(volume)
    if trades > 0:
        state.trades_ema_20.push(trades)
        state.trades_ema_5.push(trades)
        tbr = taker_buy_volume / volume if volume > 0 else 0.5
        state.taker_buy_ratio_ema_10.push(tbr)
        imbalance = 2.0 * tbr - 1.0
        state.cvd_window_10.push(imbalance)
        state.cvd_window_20.push(imbalance)
        state.taker_ratio_window_50.push(tbr)
        ats = quote_volume / trades
        state.avg_trade_size_ema_20.push(ats)
        vpt = volume / trades
        state.volume_per_trade_ema_20.push(vpt)

    state._bar_count += 1
    state.close_history.append(close)
    state.open_history.append(open_)
    state.high_history.append(high)
    state.low_history.append(low)

    state.ma_5.push(close)
    state.ma_10.push(close)
    state.ma_20.push(close)
    state.ma_30.push(close)
    state.ma_50.push(close)
    state.bb_window.push(close)

    if state._last_close is not None and state._last_close != 0:
        ret = (close - state._last_close) / state._last_close
        state.return_window_20.push(ret)
        state.return_window_5.push(ret)
        state._ret_buf_24.append(ret)
    state._last_close = close

    state._vol_buf_6.append(volume)
    state._vol_buf_24.append(volume)
    if volume > 0:
        _tbv = taker_buy_volume or 0.0
        _tsv = volume - _tbv
        _total = _tbv + _tsv
        _imb = (_tbv - _tsv) / _total if _total > 0 else 0
        state._taker_imb_buf_6.append(_imb)
    if (open_interest is not None and state._last_oi is not None
            and state._last_oi > 0 and volume > 0):
        oi_change = abs(open_interest - state._last_oi) / state._last_oi
        vol_ma = sum(state._vol_buf_24) / max(len(state._vol_buf_24), 1)
        vol_spike = volume / vol_ma if vol_ma > 0 else 1.0
        liq_proxy = oi_change * vol_spike * volume
        state._liq_vol_buf_24.append(liq_proxy)
    elif len(state._liq_vol_buf_24) > 0:
        state._liq_vol_buf_24.append(0.0)

    if state.return_window_5.full:
        state.vol_5_history.append(state.return_window_5.std)

    if state.return_window_5.full and state.return_window_20.full:
        v5 = state.return_window_5.std
        v20 = state.return_window_20.std
        if v20 is not None and v20 > 1e-12:
            vr = v5 / v20
            state.vol_regime_ema.push(vr)
            state.vol_regime_history.append(vr)

    if high > 0 and low > 0 and high >= low:
        hl_ratio = high / low
        if hl_ratio > 0:
            ln_hl = math.log(hl_ratio)
            state.hl_log_sq_window.push(ln_hl * ln_hl)

    state._last_volume = volume
    state.vol_window_20.push(volume)
    state.vol_window_5.push(volume)

    state.rsi_14.push(close)
    state.rsi_6.push(close)

    state.ema_12.push(close)
    state.ema_26.push(close)
    if state.ema_12.ready and state.ema_26.ready:
        macd_val = state.ema_12.value - state.ema_26.value
        state.macd_signal_ema.push(macd_val)

    state.atr_14.push(high, low, close)
    state.adx_14.push(high, low, close)
