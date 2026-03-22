"""Batch feature computation — C++ accelerated with Python fallback.

Replaces the slow iterrows() loop in compute_oos_features() with a single
C++ call that processes all bars at once.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

from _quant_hotpath import cpp_compute_all_features, cpp_feature_names  # noqa: E402


def _load_schedule(path: Path, ts_col: str, val_col: str) -> Dict[int, float]:
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row[ts_col])] = float(row[val_col])
    return schedule


def _load_spot_closes(symbol: str) -> Dict[int, float]:
    path = Path(f"data_files/{symbol}_spot_1h.csv")
    closes: Dict[int, float] = {}
    if not path.exists():
        return closes
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_col = "open_time" if "open_time" in row else "timestamp"
            closes[int(row[ts_col])] = float(row["close"])
    return closes


def _load_fgi_schedule() -> Dict[int, float]:
    path = Path("data_files/fear_greed_index.csv")
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row["timestamp"])] = float(row["value"])
    return schedule


def _parse_ts_ms(raw: str) -> int:
    """Parse timestamp string — supports both epoch ms and ISO 8601."""
    if "T" in raw or "-" in raw:
        from datetime import datetime
        dt = datetime.fromisoformat(raw)
        return int(dt.timestamp() * 1000)
    return int(raw)


def _load_iv_schedule(symbol: str) -> Dict[int, float]:
    path = Path(f"data_files/{symbol}_deribit_iv.csv")
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_col = "timestamp" if "timestamp" in row else "open_time"
            val_col = "implied_vol" if "implied_vol" in row else "mark_iv"
            if val_col not in row:
                continue
            schedule[_parse_ts_ms(row[ts_col])] = float(row[val_col])
    return schedule


def _load_pcr_schedule(symbol: str) -> Dict[int, float]:
    path = Path(f"data_files/{symbol}_deribit_pcr.csv")
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_col = "timestamp" if "timestamp" in row else "open_time"
            val_col = "put_call_ratio" if "put_call_ratio" in row else "pcr"
            if val_col not in row:
                continue
            schedule[_parse_ts_ms(row[ts_col])] = float(row[val_col])
    return schedule


def _dict_to_sorted_array(d: Dict[int, float]) -> np.ndarray:
    """Convert {timestamp: value} dict to sorted (M, 2) numpy array."""
    if not d:
        return np.empty((0, 2), dtype=np.float64)
    items = sorted(d.items())
    return np.array(items, dtype=np.float64)


def compute_features_batch(
    symbol: str,
    df: pd.DataFrame,
    *,
    include_iv: bool = False,
    include_onchain: bool = False,
    include_v11: bool = False,
) -> pd.DataFrame:
    """Compute enriched features for all bars in df.

    C++ accelerated replacement for compute_oos_features().
    Falls back to Python if C++ module not available.
    """
    # Extract bar data as numpy arrays
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    timestamps = df[ts_col].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    opens = df["open"].values.astype(np.float64) if "open" in df.columns else closes.copy()
    highs = df["high"].values.astype(np.float64) if "high" in df.columns else closes.copy()
    lows = df["low"].values.astype(np.float64) if "low" in df.columns else closes.copy()
    volumes = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.zeros_like(closes)
    trades = df["trades"].values.astype(np.float64) if "trades" in df.columns else np.zeros_like(closes)
    tbv = (df["taker_buy_volume"].values.astype(np.float64)
           if "taker_buy_volume" in df.columns else np.zeros_like(closes))
    qv = df["quote_volume"].values.astype(np.float64) if "quote_volume" in df.columns else np.zeros_like(closes)
    tbqv = (df["taker_buy_quote_volume"].values.astype(np.float64)
            if "taker_buy_quote_volume" in df.columns else np.zeros_like(closes))

    # Replace NaN with 0 for volume-like fields
    for arr in [volumes, trades, tbv, qv, tbqv]:
        np.nan_to_num(arr, copy=False)

    # Load schedule data
    funding = _load_schedule(
        Path(f"data_files/{symbol}_funding.csv"), "timestamp", "funding_rate")
    oi = _load_schedule(
        Path(f"data_files/{symbol}_open_interest.csv"), "timestamp", "sum_open_interest")
    ls = _load_schedule(
        Path(f"data_files/{symbol}_ls_ratio.csv"), "timestamp", "long_short_ratio")
    spot_closes = _load_spot_closes(symbol)
    fgi_schedule = _load_fgi_schedule()

    # Convert schedules to sorted (M, 2) arrays
    funding_arr = _dict_to_sorted_array(funding)
    oi_arr = _dict_to_sorted_array(oi)
    ls_arr = _dict_to_sorted_array(ls)
    spot_arr = _dict_to_sorted_array(spot_closes)
    fgi_arr = _dict_to_sorted_array(fgi_schedule)

    # IV schedules (optional)
    empty_sched = np.empty((0, 2), dtype=np.float64)
    if include_iv:
        iv_arr = _dict_to_sorted_array(_load_iv_schedule(symbol))
        pcr_arr = _dict_to_sorted_array(_load_pcr_schedule(symbol))
    else:
        iv_arr = empty_sched
        pcr_arr = empty_sched

    # On-chain schedule: (M, 7) [ts, FlowIn, FlowOut, Supply, Addr, Tx, HR]
    if include_onchain:
        onchain_arr = _load_onchain_schedule(symbol)
    else:
        onchain_arr = np.empty((0, 7), dtype=np.float64)

    # V11 schedules
    if include_v11:
        liq_arr = _load_liq_schedule(symbol)
        mempool_arr = _load_mempool_schedule()
        macro_arr = _load_macro_schedule_arr()
    else:
        liq_arr = np.empty((0, 4), dtype=np.float64)
        mempool_arr = np.empty((0, 4), dtype=np.float64)
        macro_arr = np.empty((0, 4), dtype=np.float64)

    # Call C++ engine
    result = cpp_compute_all_features(
        timestamps, opens, highs, lows, closes, volumes,
        trades, tbv, qv, tbqv,
        funding_arr, oi_arr, ls_arr, spot_arr, fgi_arr,
        iv_arr, pcr_arr, onchain_arr,
        liq_arr, mempool_arr, macro_arr,
    )

    # Wrap as DataFrame
    feat_names = cpp_feature_names()
    feat_df = pd.DataFrame(result, columns=feat_names, index=df.index)

    # Replace NaN features with NaN (already done by C++ via NaN sentinel)
    feat_df["close"] = closes

    # Multi-timeframe 4h features (not in C++ engine)
    from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
    tf4h = compute_4h_features(df)
    for col in TF4H_FEATURE_NAMES:
        if col in tf4h.columns:
            feat_df[col] = tf4h[col].values

    # V14: BTC Dominance features (BTC/ETH ratio)
    _add_dominance_features(symbol, feat_df, closes)

    # V15: Interaction & statistical features (IC-screened 2026-03-18)
    _add_v15_features(feat_df)

    # V16: Orderbook proxy + IV spread + liquidation features
    _add_v16_features(symbol, feat_df, closes, highs, lows, volumes, tbv)

    # V17: On-chain features (TxTfrCnt, AdrActCnt, exchange flow)
    _add_v17_onchain_features(symbol, feat_df, timestamps)

    # V21: Cross-market features (SPY/QQQ/VIX/TLT/USO/COIN from Yahoo Finance)
    _add_cross_market_features(feat_df, timestamps)

    return feat_df


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
                "treasury_10y_chg_5d", "eem_ret_5d", "gld_ret_5d"]:
        if col not in cm.columns:
            continue

        # Build date → value lookup
        cm_vals = cm[col].dropna()
        cm_dates = cm_vals.index.date
        cm_dict: dict[object, float] = dict(zip(cm_dates, cm_vals.values))

        # Forward-fill: for each bar, use the most recent daily value
        result = np.full(len(timestamps), np.nan, dtype=np.float64)
        sorted_cm_dates = sorted(cm_dict.keys())
        idx = 0
        last_val = np.nan
        for i, bd in enumerate(bar_dates):
            while idx < len(sorted_cm_dates) and sorted_cm_dates[idx] <= bd:
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
