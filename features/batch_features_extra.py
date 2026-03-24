# features/batch_features_extra.py
"""Extra batch feature functions (V15-V24) for batch_feature_engine.

Extracted from batch_feature_engine.py to reduce file size.
Contains _add_v15_features through _add_etf_volume_features plus
supporting data loaders and utilities.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

def _add_v15_features(feat_df: pd.DataFrame) -> None:
    """Add V15 interaction and statistical features to batch DataFrame.

    These are derived from existing features — no new data sources needed.
    IC-screened 2026-03-18: all significant (p<0.001) across BTC/ETH/SUI/AXS.
    """
    # Interaction terms: multiply existing features for regime-conditional alpha
    # NaN propagates naturally through multiplication
    if "ret_1" in feat_df.columns and "vol_20" in feat_df.columns:
        feat_df["ret1_x_vol"] = feat_df["ret_1"] * feat_df["vol_20"]
    if "rsi_14" in feat_df.columns and "atr_norm_14" in feat_df.columns:
        feat_df["rsi_x_atr"] = feat_df["rsi_14"] * feat_df["atr_norm_14"]
    if "rsi_14" in feat_df.columns and "vol_20" in feat_df.columns:
        feat_df["rsi_x_vol"] = feat_df["rsi_14"] * feat_df["vol_20"]
    if "close_vs_ma50" in feat_df.columns and "vol_20" in feat_df.columns:
        feat_df["trend_x_vol"] = feat_df["close_vs_ma50"] * feat_df["vol_20"]
    if "bb_pctb_20" in feat_df.columns and "vol_20" in feat_df.columns:
        feat_df["bb_x_vol"] = feat_df["bb_pctb_20"] * feat_df["vol_20"]

    # Return autocorrelation (24-bar rolling)
    if "ret_1" in feat_df.columns:
        ret = feat_df["ret_1"].values
        ac = np.full(len(ret), np.nan)
        for i in range(24, len(ret)):
            chunk = ret[i - 24:i]
            if np.any(np.isnan(chunk)):
                continue
            r1, r2 = chunk[:-1], chunk[1:]
            std1, std2 = np.std(r1), np.std(r2)
            if std1 > 1e-10 and std2 > 1e-10:
                ac[i] = np.corrcoef(r1, r2)[0, 1]
        feat_df["ret_autocorr_24"] = ac

    # Return skewness (24-bar rolling)
    if "ret_1" in feat_df.columns:
        ret = feat_df["ret_1"].values
        skew = np.full(len(ret), np.nan)
        for i in range(24, len(ret)):
            chunk = ret[i - 24:i]
            if np.any(np.isnan(chunk)):
                continue
            mu = np.mean(chunk)
            std = np.std(chunk)
            if std > 1e-10:
                skew[i] = np.mean(((chunk - mu) / std) ** 3)
        feat_df["ret_skew_24"] = skew


def _add_v16_features(
    symbol: str, feat_df: pd.DataFrame,
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volumes: np.ndarray, taker_buy_vol: np.ndarray,
) -> None:
    """Add V16 orderbook proxy, IV spread, and liquidation features."""
    # OB spread proxy: (high - low) / close
    feat_df["ob_spread_proxy"] = (highs - lows) / np.maximum(closes, 1e-10)

    # OB imbalance proxy: (taker_buy - taker_sell) / total
    tsv = volumes - taker_buy_vol
    total = taker_buy_vol + tsv
    feat_df["ob_imbalance_proxy"] = np.where(total > 0, (taker_buy_vol - tsv) / total, 0.0)

    # OB imbalance × volume ratio
    vol_ma20 = pd.Series(volumes).rolling(20).mean().values
    vol_ratio = np.where(vol_ma20 > 0, volumes / vol_ma20, 1.0)
    feat_df["ob_imbalance_x_vol"] = feat_df["ob_imbalance_proxy"].values * vol_ratio

    # OB imbalance cumulative 6-bar
    feat_df["ob_imbalance_cum6"] = pd.Series(feat_df["ob_imbalance_proxy"]).rolling(6).sum().values

    # OB volume clock: MA6/MA24 - 1
    vol_ma6 = pd.Series(volumes).rolling(6).mean().values
    vol_ma24 = pd.Series(volumes).rolling(24).mean().values
    feat_df["ob_volume_clock"] = np.where(vol_ma24 > 0, vol_ma6 / vol_ma24 - 1, 0.0)

    # IV-RV spread (load from Deribit IV file if available)
    from pathlib import Path
    iv_path = Path(f"data_files/{symbol}_deribit_iv.csv")
    if iv_path.exists():
        try:
            iv_df = pd.read_csv(iv_path)
            iv_df["ts_ms"] = pd.to_datetime(iv_df["timestamp"]).astype(np.int64) // 10**6
            iv_s = iv_df.sort_values("ts_ms")
            # TODO: interpolate IV onto kline timestamps for iv_rv_spread
            _ = iv_s  # placeholder for future IV interpolation
        except Exception:
            pass
    # If IV not available from file, try from existing features
    if "iv_rv_spread" not in feat_df.columns:
        if "implied_vol_zscore_24" in feat_df.columns and "vol_20" in feat_df.columns:
            # Can't compute IV-RV without raw IV, leave as NaN
            feat_df["iv_rv_spread"] = np.nan
        else:
            feat_df["iv_rv_spread"] = np.nan

    # Liquidation volume z-score (from proxy file)
    liq_path = Path(f"data_files/{symbol}_liquidation_proxy.csv")
    if liq_path.exists():
        try:
            liq_df = pd.read_csv(liq_path)
            if "liq_proxy_volume" in liq_df.columns:
                liq_vol = liq_df["liq_proxy_volume"].values
                n = min(len(liq_vol), len(feat_df))
                padded = np.concatenate([np.full(len(feat_df) - n, np.nan), liq_vol[-n:]])
                feat_df["liq_volume_zscore_24"] = _rolling_zscore_arr(padded, 24)
        except Exception:
            feat_df["liq_volume_zscore_24"] = np.nan
    else:
        feat_df["liq_volume_zscore_24"] = np.nan


def _add_v17_onchain_features(symbol: str, feat_df: pd.DataFrame, timestamps: np.ndarray) -> None:
    """Add V17 on-chain features from Coin Metrics data.

    IC-screened 2026-03-18:
    - ETH TxTfrCnt_zscore_7: IC=+0.137 (strongest single on-chain factor)
    - ETH AdrActCnt_zscore_14: IC=+0.085
    - BTC oc_netflow: IC=-0.074
    """
    # Map symbol to asset
    asset_map = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
    asset = asset_map.get(symbol)
    if not asset:
        return

    oc_path = Path(f"data/onchain/{asset}_onchain_combined.csv")
    if not oc_path.exists():
        return

    try:
        oc = pd.read_csv(oc_path)
        oc_ts = pd.to_numeric(oc["timestamp"], errors="coerce").values
        oc = oc.sort_values("timestamp")

        # Interpolate daily on-chain → hourly kline
        for col, prefix in [
            ("TxTfrCnt", "oc_tx"),
            ("AdrActCnt", "oc_addr"),
            ("FlowInExUSD", "oc_flowin"),
            ("FlowOutExUSD", "oc_flowout"),
        ]:
            if col not in oc.columns:
                continue
            vals = np.interp(timestamps, oc_ts,
                            pd.to_numeric(oc[col], errors="coerce").fillna(0).values)
            # Z-scores at 7d and 14d
            for win_days, win_label in [(7, "7"), (14, "14")]:
                win = win_days * 24
                feat_df[f"{prefix}_zscore_{win_label}"] = _rolling_zscore_arr(vals, win)

        # Net flow = inflow - outflow
        if "FlowInExUSD" in oc.columns and "FlowOutExUSD" in oc.columns:
            flow_in = np.interp(timestamps, oc_ts,
                               pd.to_numeric(oc["FlowInExUSD"], errors="coerce").fillna(0).values)
            flow_out = np.interp(timestamps, oc_ts,
                                pd.to_numeric(oc["FlowOutExUSD"], errors="coerce").fillna(0).values)
            net = flow_in - flow_out
            feat_df["oc_netflow_zscore_7"] = _rolling_zscore_arr(net, 7 * 24)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("V17 on-chain features failed for %s: %s", symbol, e)


def _rolling_zscore_arr(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr)
    mu = s.rolling(window).mean()
    std = s.rolling(window).std()
    return ((s - mu) / std.replace(0, np.nan)).values


def _compute_ratio_features(
    closes: np.ndarray, ref_closes: np.ndarray, prefix: str,
    feat_df: pd.DataFrame,
) -> None:
    """Compute dev_20 and ret_24 ratio features for closes vs ref_closes.

    Writes '{prefix}_dev_20' and '{prefix}_ret_24' into feat_df.
    """
    import pandas as _pd

    n = len(closes)
    n_ref = len(ref_closes)
    min_n = min(n, n_ref)

    ratio = np.full(n, np.nan)
    ratio[-min_n:] = closes[-min_n:] / np.where(ref_closes[-min_n:] > 0, ref_closes[-min_n:], 1)

    ratio_s = _pd.Series(ratio)
    ma20 = ratio_s.rolling(20).mean().values

    feat_df[f"{prefix}_dev_20"] = ratio / np.where(ma20 > 0, ma20, np.nan) - 1
    feat_df[f"{prefix}_ret_24"] = ratio_s.pct_change(24).values


def _load_ref_closes(symbol: str) -> "np.ndarray[tuple[Any, ...], np.dtype[Any]] | None":
    """Load 1h closes for a reference symbol. Returns None if file missing."""
    path = Path(f"data_files/{symbol}_1h.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    result: np.ndarray[tuple[Any, ...], np.dtype[Any]] = df["close"].values.astype(np.float64)
    return result


# Multi-ratio dominance config: symbol -> list of (ref_symbol, feature_prefix)
_DOMINANCE_PAIRS: dict[str, list[tuple[str, str]]] = {
    "BTCUSDT": [("SUIUSDT", "dom_vs_sui")],
    "ETHUSDT": [("SUIUSDT", "dom_vs_sui"), ("AXSUSDT", "dom_vs_axs")],
    "SUIUSDT": [("AXSUSDT", "dom_vs_axs")],
    "AXSUSDT": [("ETHUSDT", "dom_vs_eth")],
}

# All possible multi-ratio feature names (for NaN fill when pair not applicable)
_ALL_MULTI_RATIO_NAMES = [
    "dom_vs_sui_dev_20", "dom_vs_sui_ret_24",
    "dom_vs_axs_dev_20", "dom_vs_axs_ret_24",
    "dom_vs_eth_dev_20", "dom_vs_eth_ret_24",
]


def _add_cross_market_features(feat_df: pd.DataFrame, timestamps: np.ndarray) -> None:
    """Add V21 cross-market features from Yahoo Finance daily data.

    Loads data_files/cross_market_daily.csv and forward-fills daily values
    to hourly bars via timestamp alignment.

    Features: spy_ret_1d, qqq_ret_1d, spy_ret_5d, vix_level,
              tlt_ret_5d, uso_ret_5d, coin_ret_1d, spy_extreme
    """
    cm_path = Path("data_files/cross_market_daily.csv")
    if not cm_path.exists():
        logger.debug("No cross_market_daily.csv — skipping V21 features")
        return

    try:
        cm = pd.read_csv(cm_path, index_col="date", parse_dates=True)
    except Exception:
        logger.debug("Failed to load cross_market_daily.csv", exc_info=True)
        return

    # Convert bar timestamps to dates for alignment
    bar_dates = pd.to_datetime(timestamps, unit="ms", utc=True).date

    # For each cross-market column, forward-fill daily value to each bar
    for col in ["spy_ret_1d", "qqq_ret_1d", "spy_ret_5d", "vix_level",
                "tlt_ret_5d", "uso_ret_5d", "coin_ret_1d", "spy_extreme",
                "treasury_10y_chg_5d", "eem_ret_5d", "gld_ret_5d",
                "ethe_ret_1d", "gbtc_ret_1d", "ibit_ret_1d", "bito_ret_1d",
                "gbtc_premium_dev",
                "etha_ret_1d", "bitx_ret_1d", "biti_ret_1d",
                "mara_ret_1d", "riot_ret_1d"]:
        if col not in cm.columns:
            continue

        # Build date → value lookup
        cm_vals = cm[col].dropna()
        cm_dates = cm_vals.index.date
        cm_dict: dict[object, float] = dict(zip(cm_dates, cm_vals.values))

        # Forward-fill with T-1 shift: for a bar on date D, use the most
        # recent cross-market value from date < D (strictly before) to avoid
        # look-ahead bias (US markets close ~21:00 UTC).
        result = np.full(len(timestamps), np.nan, dtype=np.float64)
        sorted_cm_dates = sorted(cm_dict.keys())
        idx = 0
        last_val = np.nan
        for i, bd in enumerate(bar_dates):
            while idx < len(sorted_cm_dates) and sorted_cm_dates[idx] < bd:
                last_val = cm_dict[sorted_cm_dates[idx]]
                idx += 1
            result[i] = last_val

        feat_df[col] = result

    n_filled = sum(1 for col in feat_df.columns if col.startswith("spy_") or col.startswith("qqq_")
                   or col.startswith("vix_") or col.startswith("tlt_") or col.startswith("uso_")
                   or col.startswith("coin_ret"))
    logger.debug("V21 cross-market: %d features added", n_filled)


def _add_dominance_features(symbol: str, feat_df: pd.DataFrame, closes: np.ndarray) -> None:
    """Add dominance ratio features: original BTC/ETH + multi-ratio pairs."""
    import pandas as _pd
    from pathlib import Path as _P

    # --- Original V14: BTC/ETH ratio (4 features, kept for backward compat) ---
    dom_names = ["btc_dom_dev_20", "btc_dom_dev_50", "btc_dom_ret_24", "btc_dom_ret_72"]

    eth_path = _P("data_files/ETHUSDT_1h.csv")
    if not eth_path.exists() or symbol == "ETHUSDT":
        for name in dom_names:
            feat_df[name] = np.nan
    else:
        eth_df = pd.read_csv(eth_path)
        eth_closes = eth_df["close"].values.astype(np.float64)

        n = len(closes)
        n_eth = len(eth_closes)
        min_n = min(n, n_eth)

        ratio = np.full(n, np.nan)
        ratio[-min_n:] = closes[-min_n:] / np.where(eth_closes[-min_n:] > 0, eth_closes[-min_n:], 1)

        ratio_s = _pd.Series(ratio)
        ma20 = ratio_s.rolling(20).mean().values
        ma50 = ratio_s.rolling(50).mean().values

        feat_df["btc_dom_dev_20"] = ratio / np.where(ma20 > 0, ma20, np.nan) - 1
        feat_df["btc_dom_dev_50"] = ratio / np.where(ma50 > 0, ma50, np.nan) - 1
        feat_df["btc_dom_ret_24"] = ratio_s.pct_change(24).values
        feat_df["btc_dom_ret_72"] = ratio_s.pct_change(72).values

    # --- V14b: Multi-ratio dominance features ---
    # Initialize all multi-ratio columns to NaN
    for name in _ALL_MULTI_RATIO_NAMES:
        feat_df[name] = np.nan

    # Compute pairs applicable to this symbol
    pairs = _DOMINANCE_PAIRS.get(symbol, [])
    # Cache loaded reference data to avoid re-reading
    _ref_cache: dict[str, np.ndarray | None] = {}
    for ref_sym, prefix in pairs:
        if ref_sym == symbol:
            continue
        if ref_sym not in _ref_cache:
            _ref_cache[ref_sym] = _load_ref_closes(ref_sym)
        ref_closes = _ref_cache[ref_sym]
        if ref_closes is None:
            continue
        _compute_ratio_features(closes, ref_closes, prefix, feat_df)


def _load_liq_schedule(symbol: str) -> np.ndarray:
    """Load liquidation proxy as (M, 4): [ts, total_vol, buy_vol, sell_vol]."""
    path = Path(f"data_files/{symbol}_liquidation_proxy.csv")
    if not path.exists():
        return np.empty((0, 4), dtype=np.float64)
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row.get("ts", 0))
            if ts == 0:
                continue
            total = float(row.get("liq_proxy_volume", 0))
            buy = float(row.get("liq_proxy_buy", 0))
            sell = float(row.get("liq_proxy_sell", 0))
            rows.append([ts, total, buy, sell])
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    arr = np.array(rows, dtype=np.float64)
    arr = arr[arr[:, 0].argsort()]
    return arr


def _load_mempool_schedule() -> np.ndarray:
    """Load mempool fees as (M, 4): [ts, fastest_fee, economy_fee, mempool_size]."""
    path = Path("data_files/btc_mempool_fees.csv")
    if not path.exists():
        return np.empty((0, 4), dtype=np.float64)
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row.get("timestamp", 0))
            if ts == 0:
                continue
            fastest = float(row.get("max_fee", row.get("avg_fee", 0)))
            economy = float(row.get("min_fee", 1))
            size = float(row.get("avg_fee", 0)) * 1000
            rows.append([ts, fastest, economy, size])
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    arr = np.array(rows, dtype=np.float64)
    arr = arr[arr[:, 0].argsort()]
    return arr


def _load_macro_schedule_arr() -> np.ndarray:
    """Load macro daily as (M, 4): [ts_ms, dxy, spx, vix]."""
    path = Path("data_files/macro_daily.csv")
    if not path.exists():
        return np.empty((0, 4), dtype=np.float64)
    # Merge all values per date
    date_data: Dict[str, Dict[str, float]] = {}
    date_ts: Dict[str, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get("date", "")
            ts_ms = float(row.get("timestamp_ms", 0))
            if not date or ts_ms == 0:
                continue
            if date not in date_data:
                date_data[date] = {}
                date_ts[date] = ts_ms
            else:
                date_ts[date] = min(date_ts[date], ts_ms)
            for key in ("dxy", "spx", "vix"):
                val = row.get(key, "")
                if val != "":
                    date_data[date][key] = float(val)
    rows = []
    for date in sorted(date_data.keys()):
        d = date_data[date]
        if not d:
            continue
        ts = date_ts[date]
        rows.append([ts, d.get("dxy", float("nan")),
                     d.get("spx", float("nan")),
                     d.get("vix", float("nan"))])
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    arr = np.array(rows, dtype=np.float64)
    arr = arr[arr[:, 0].argsort()]
    return arr


def _load_onchain_schedule(symbol: str) -> np.ndarray:
    """Load on-chain metrics as (M, 7) array: [ts, FlowIn, FlowOut, Supply, Addr, Tx, HR]."""
    path = Path(f"data_files/{symbol}_onchain.csv")
    if not path.exists():
        return np.empty((0, 7), dtype=np.float64)

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row.get("timestamp", 0))
            flow_in = float(row.get("FlowInExUSD", "nan") or "nan")
            flow_out = float(row.get("FlowOutExUSD", "nan") or "nan")
            supply = float(row.get("SplyExNtv", "nan") or "nan")
            addr = float(row.get("AdrActCnt", "nan") or "nan")
            tx = float(row.get("TxTfrCnt", "nan") or "nan")
            hr = float(row.get("HashRate", "nan") or "nan")
            rows.append([ts, flow_in, flow_out, supply, addr, tx, hr])

    if not rows:
        return np.empty((0, 7), dtype=np.float64)
    arr = np.array(rows, dtype=np.float64)
    # Sort by timestamp
    arr = arr[arr[:, 0].argsort()]
    return arr


def _add_iv_features(
    feat_df: pd.DataFrame, symbol: str, timestamps: np.ndarray
) -> None:
    """Add V22 Deribit IV features from daily DVOL data.

    Loads {currency}_iv_daily.csv (or falls back to {SYMBOL}_dvol_1h.csv),
    computes 5 IV features, and merges into feat_df with T-1 date shift
    to prevent look-ahead bias (daily DVOL close is end-of-day).

    Features added:
      - iv_level: DVOL / 100 (normalized)
      - iv_rank_30d: percentile rank of IV over 30 days
      - iv_change_1d: 1-day change in DVOL (percentage points)
      - iv_term_slope_daily: intraday DVOL range / level
      - rv_iv_spread: realized vol - implied vol (annualized %)
    """
    from features.options_flow import (
        load_dvol_daily,
        compute_iv_features_from_dvol,
        IV_FEATURE_NAMES,
    )

    # Determine currency from symbol
    currency = "BTC" if "BTC" in symbol.upper() else "ETH"

    try:
        dvol_df = load_dvol_daily(currency)
    except Exception as e:
        logger.debug("IV data unavailable for %s: %s", currency, e)
        for name in IV_FEATURE_NAMES:
            feat_df[name] = np.nan
        return

    if dvol_df.empty:
        for name in IV_FEATURE_NAMES:
            feat_df[name] = np.nan
        return

    # Compute realized vol from bar closes for rv_iv_spread
    # Use 20-day annualized vol (sqrt(365) * std of daily log returns)
    rv_series = None
    if "vol_20" in feat_df.columns:
        # vol_20 is already computed in C++ engine (20-bar rolling std of returns)
        # Convert hourly bars to daily by taking last bar per date
        bar_dates = pd.to_datetime(timestamps, unit="ms", utc=True).date
        bar_date_series = pd.Series(bar_dates, index=feat_df.index)
        vol20 = feat_df["vol_20"].copy()
        # Annualize: vol_20 is hourly std * sqrt(20); we need annualized
        # Approximate: daily vol ~ hourly vol * sqrt(24), annualized ~ daily * sqrt(365)
        vol20_ann = vol20 * np.sqrt(24 * 365) * 100  # to percentage to match DVOL units

        # Group by date, take last value
        daily_rv = pd.DataFrame({"date": bar_date_series, "rv": vol20_ann})
        daily_rv_last = daily_rv.groupby("date")["rv"].last()

        # Align to dvol_df dates
        dvol_dates = dvol_df["date"].dt.date
        rv_aligned = dvol_dates.map(daily_rv_last).values
        rv_series = pd.Series(rv_aligned, index=dvol_df.index)

    # Compute IV features on daily DVOL data
    iv_feats = compute_iv_features_from_dvol(dvol_df, rv_series)

    # Build date -> feature value lookup (using T-1 shift for look-ahead prevention)
    # For a bar on date D, use IV features from date D-1
    dvol_dates_sorted = dvol_df["date"].dt.date.values
    iv_values: Dict[str, np.ndarray] = {}
    for col in IV_FEATURE_NAMES:
        iv_values[col] = iv_feats[col].values if col in iv_feats.columns else np.full(len(dvol_df), np.nan)

    # Map bar timestamps to dates
    bar_dates = pd.to_datetime(timestamps, unit="ms", utc=True).date

    # Forward-fill with T-1 shift: for bar on date D, find most recent DVOL from date < D
    for col in IV_FEATURE_NAMES:
        result = np.full(len(timestamps), np.nan, dtype=np.float64)
        col_vals = iv_values[col]
        idx = 0
        last_val = np.nan
        for i, bd in enumerate(bar_dates):
            while idx < len(dvol_dates_sorted) and dvol_dates_sorted[idx] < bd:
                last_val = col_vals[idx]
                idx += 1
            result[i] = last_val
        feat_df[col] = result

    n_valid = sum(1 for col in IV_FEATURE_NAMES if not np.all(np.isnan(feat_df[col].values)))
    logger.debug("V22 IV features: %d/%d populated for %s", n_valid, len(IV_FEATURE_NAMES), symbol)


STABLECOIN_FEATURE_NAMES = [
    "total_supply_change_1d",
    "total_supply_change_7d",
    "total_zscore_14",
    "total_zscore_30",
    "usdt_dominance",
    "supply_acceleration",
]


def _add_stablecoin_features(feat_df: pd.DataFrame, timestamps: np.ndarray) -> None:
    """Add V23 stablecoin supply features from DeFiLlama daily data.

    Loads data_files/stablecoin_daily.csv and forward-fills daily values
    to hourly bars with T-1 shift (bar on date D uses data from D-1).

    Features:
      - total_supply_change_1d: 1-day % change in total stablecoin supply
      - total_supply_change_7d: 7-day % change
      - total_zscore_14: z-score of total supply over 14-day window
      - total_zscore_30: z-score over 30-day window
      - usdt_dominance: USDT share of total supply
      - supply_acceleration: change_1d - change_1d.shift(7) (inflow momentum)
    """
    sc_path = Path("data_files/stablecoin_daily.csv")
    if not sc_path.exists():
        logger.debug("No stablecoin_daily.csv — skipping V23 features")
        for name in STABLECOIN_FEATURE_NAMES:
            feat_df[name] = np.nan
        return

    try:
        sc = pd.read_csv(sc_path)
    except Exception:
        logger.debug("Failed to load stablecoin_daily.csv", exc_info=True)
        for name in STABLECOIN_FEATURE_NAMES:
            feat_df[name] = np.nan
        return

    # Parse and sort
    sc["date_parsed"] = pd.to_datetime(sc["date"])
    sc = sc.sort_values("date_parsed").reset_index(drop=True)

    # Compute daily features on the daily data
    total = pd.to_numeric(sc["total_supply"], errors="coerce")
    usdt = pd.to_numeric(sc["usdt_supply"], errors="coerce")

    change_1d = total.pct_change(1)
    change_7d = total.pct_change(7)

    mu_14 = total.rolling(14).mean()
    std_14 = total.rolling(14).std()
    zscore_14 = (total - mu_14) / std_14.replace(0, np.nan)

    mu_30 = total.rolling(30).mean()
    std_30 = total.rolling(30).std()
    zscore_30 = (total - mu_30) / std_30.replace(0, np.nan)

    dominance = usdt / total.replace(0, np.nan)

    acceleration = change_1d - change_1d.shift(7)

    # Pack into dict for T-1 forward-fill
    sc_dates = sc["date_parsed"].dt.date.values
    daily_features = {
        "total_supply_change_1d": change_1d.values,
        "total_supply_change_7d": change_7d.values,
        "total_zscore_14": zscore_14.values,
        "total_zscore_30": zscore_30.values,
        "usdt_dominance": dominance.values,
        "supply_acceleration": acceleration.values,
    }

    # Convert bar timestamps to dates for alignment
    bar_dates = pd.to_datetime(timestamps, unit="ms", utc=True).date

    # Forward-fill with T-1 shift: for bar on date D, use most recent value from date < D
    for col_name, col_vals in daily_features.items():
        result = np.full(len(timestamps), np.nan, dtype=np.float64)
        idx = 0
        last_val = np.nan
        for i, bd in enumerate(bar_dates):
            while idx < len(sc_dates) and sc_dates[idx] < bd:
                v = col_vals[idx]
                if not np.isnan(v):
                    last_val = v
                idx += 1
            result[i] = last_val
        feat_df[col_name] = result

    n_filled = sum(1 for name in STABLECOIN_FEATURE_NAMES
                   if not np.all(np.isnan(feat_df[name].values)))
    logger.debug("V23 stablecoin: %d/%d features populated", n_filled, len(STABLECOIN_FEATURE_NAMES))


# ── V24: ETF Volume/Flow Features ──────────────────────────────────────

ETF_VOLUME_FEATURE_NAMES = (
    "etf_vol_change_5d",      # 5-day change in aggregate BTC ETF dollar volume
    "etf_vol_zscore_7",       # 7-day z-score of aggregate ETF volume
    "etf_vol_zscore_14",      # 14-day z-score
    "gbtc_vol_zscore_14",     # GBTC-specific volume z-score (outflow proxy)
    "etha_vol_zscore_14",     # ETHA volume z-score
)


def _add_etf_volume_features(feat_df: pd.DataFrame, timestamps: np.ndarray) -> None:
    """Add V24 ETF volume features from etf_volume_daily.csv.

    ETF dollar volume is a contrarian signal: volume spikes predict negative
    next-day returns (IC=-0.09 to -0.11, T-1 shifted). This captures
    retail inflow at tops and GBTC redemption pressure.

    Data source: Yahoo Finance via scripts/data/download_cross_market.py.
    """
    etf_path = Path("data_files/etf_volume_daily.csv")
    if not etf_path.exists():
        for name in ETF_VOLUME_FEATURE_NAMES:
            feat_df[name] = np.nan
        return

    try:
        etf_df = pd.read_csv(etf_path, parse_dates=["date"])
    except Exception:
        for name in ETF_VOLUME_FEATURE_NAMES:
            feat_df[name] = np.nan
        return

    etf_df["date"] = etf_df["date"].dt.date
    etf_df = etf_df.sort_values("date").reset_index(drop=True)

    # Compute daily features
    agg_vol = etf_df["btc_etf_dollar_vol"]

    daily_feats = pd.DataFrame(index=etf_df.index)
    daily_feats["date"] = etf_df["date"]
    daily_feats["etf_vol_change_5d"] = agg_vol.pct_change(5)
    daily_feats["etf_vol_zscore_7"] = (
        (agg_vol - agg_vol.rolling(7).mean()) / agg_vol.rolling(7).std()
    )
    daily_feats["etf_vol_zscore_14"] = (
        (agg_vol - agg_vol.rolling(14).mean()) / agg_vol.rolling(14).std()
    )

    # Per-ETF features
    if "gbtc_dollar_vol" in etf_df.columns:
        gbtc = etf_df["gbtc_dollar_vol"]
        daily_feats["gbtc_vol_zscore_14"] = (
            (gbtc - gbtc.rolling(14).mean()) / gbtc.rolling(14).std()
        )
    else:
        daily_feats["gbtc_vol_zscore_14"] = np.nan

    if "etha_dollar_vol" in etf_df.columns:
        etha = etf_df["etha_dollar_vol"]
        daily_feats["etha_vol_zscore_14"] = (
            (etha - etha.rolling(14).mean()) / etha.rolling(14).std()
        )
    else:
        daily_feats["etha_vol_zscore_14"] = np.nan

    # Build date → feature lookup with T-1 shift
    bar_dates = pd.to_datetime(timestamps, unit="ms").date
    sorted_dates = sorted(daily_feats["date"].dropna().unique())

    for col_name in ETF_VOLUME_FEATURE_NAMES:
        if col_name not in daily_feats.columns:
            feat_df[col_name] = np.nan
            continue

        col_vals = daily_feats.set_index("date")[col_name]
        result = np.full(len(feat_df), np.nan)
        last_val = np.nan

        for i, bd in enumerate(bar_dates):
            # T-1 shift: find most recent date strictly before bar date
            idx = np.searchsorted(sorted_dates, bd, side="left") - 1
            if idx >= 0:
                lookup_date = sorted_dates[idx]
                v = col_vals.get(lookup_date, np.nan)
                if not (isinstance(v, float) and np.isnan(v)):
                    last_val = float(v)
            result[i] = last_val
        feat_df[col_name] = result

    n_filled = sum(1 for name in ETF_VOLUME_FEATURE_NAMES
                   if not np.all(np.isnan(feat_df[name].values)))
    logger.debug("V24 ETF volume: %d/%d features populated", n_filled, len(ETF_VOLUME_FEATURE_NAMES))
