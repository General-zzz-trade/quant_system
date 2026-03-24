# features/batch_features_data_loaders.py
"""Data loading helpers and IV/stablecoin/ETF feature functions for batch features.

Extracted from batch_features_extra.py to reduce file size.
Contains _load_* functions and _add_iv_features, _add_stablecoin_features,
_add_etf_volume_features.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def _load_liq_schedule(symbol: str) -> np.ndarray:
    """Load liquidation schedule timestamps for a symbol.

    Returns unix timestamps (seconds) of known liquidation events.
    """
    path = Path(f"data_files/{symbol}_liq_schedule.csv")
    if not path.exists():
        return np.array([], dtype=np.float64)
    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            return df["timestamp"].values.astype(np.float64)
        if "ts" in df.columns:
            return df["ts"].values.astype(np.float64)
        return np.array([], dtype=np.float64)
    except Exception as e:
        logger.warning("Failed to load liq schedule for %s: %s", symbol, e)
        return np.array([], dtype=np.float64)


def _load_mempool_schedule() -> np.ndarray:
    """Load mempool congestion timestamps.

    Returns unix timestamps (seconds) of high-mempool events.
    """
    path = Path("data_files/mempool_schedule.csv")
    if not path.exists():
        return np.array([], dtype=np.float64)
    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            return df["timestamp"].values.astype(np.float64)
        return np.array([], dtype=np.float64)
    except Exception as e:
        logger.warning("Failed to load mempool schedule: %s", e)
        return np.array([], dtype=np.float64)


def _load_macro_schedule_arr() -> np.ndarray:
    """Load macro event schedule (FOMC, CPI, etc.).

    Returns array of (timestamp, event_type, impact_score) tuples
    stored as structured array.
    """
    path = Path("data_files/macro_schedule.csv")
    if not path.exists():
        # Fallback: try FOMC-only schedule
        fomc_path = Path("data_files/fomc_dates.csv")
        if not fomc_path.exists():
            return np.array([], dtype=np.float64)
        try:
            df = pd.read_csv(fomc_path)
            if "timestamp" in df.columns:
                return df["timestamp"].values.astype(np.float64)
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
                return (dates.astype(np.int64) // 10**9).values.astype(np.float64)
            return np.array([], dtype=np.float64)
        except Exception as e:
            logger.warning("Failed to load FOMC dates: %s", e)
            return np.array([], dtype=np.float64)

    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            return df["timestamp"].values.astype(np.float64)
        return np.array([], dtype=np.float64)
    except Exception as e:
        logger.warning("Failed to load macro schedule: %s", e)
        return np.array([], dtype=np.float64)


def _load_onchain_schedule(symbol: str) -> np.ndarray:
    """Load on-chain event schedule for a symbol.

    Returns unix timestamps of significant on-chain events
    (large transfers, whale movements, etc.).
    """
    path = Path(f"data_files/{symbol}_onchain_events.csv")
    if not path.exists():
        # Try generic BTC on-chain events
        btc_path = Path("data_files/BTCUSDT_onchain_events.csv")
        if symbol.upper().startswith("BTC") and btc_path.exists():
            path = btc_path
        else:
            return np.array([], dtype=np.float64)

    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            return df["timestamp"].values.astype(np.float64)
        return np.array([], dtype=np.float64)
    except Exception as e:
        logger.warning("Failed to load onchain schedule for %s: %s", symbol, e)
        return np.array([], dtype=np.float64)


def _add_iv_features(
    symbol: str,
    feat_df: pd.DataFrame,
    timestamps: np.ndarray,
    closes: np.ndarray,
) -> None:
    """Add implied volatility features from Deribit DVOL data (V19/V20).

    Features added:
    - dvol_zscore: z-score of DVOL over 720 bars
    - dvol_chg_24: 24-bar DVOL change rate
    - dvol_chg_72: 72-bar DVOL change rate
    - iv_term_struct: MA(24) / MA(168) - 1
    - dvol_mean_rev: DVOL / MA(720) - 1
    - iv_rv_spread: IV - RV (realized volatility)
    """
    base_sym = symbol.replace("USDT", "").upper()
    dvol_path = Path(f"data_files/{base_sym}USDT_dvol_1h.csv")

    features_to_add = [
        "dvol_zscore", "dvol_chg_24", "dvol_chg_72",
        "iv_term_struct", "dvol_mean_rev", "iv_rv_spread",
    ]

    if not dvol_path.exists():
        for feat in features_to_add:
            feat_df[feat] = np.nan
        return

    try:
        dvol_df = pd.read_csv(dvol_path)
        if "open_time" not in dvol_df.columns or "dvol" not in dvol_df.columns:
            for feat in features_to_add:
                feat_df[feat] = np.nan
            return

        dvol_ts = dvol_df["open_time"].values.astype(np.float64)
        dvol_vals = dvol_df["dvol"].values.astype(np.float64)

        # Align DVOL to feature timestamps via nearest-match
        dvol_aligned = np.full(len(timestamps), np.nan)
        j = 0
        for i, ts in enumerate(timestamps):
            while j < len(dvol_ts) - 1 and dvol_ts[j + 1] <= ts:
                j += 1
            if j < len(dvol_ts) and abs(dvol_ts[j] - ts) < 7200_000:  # within 2h
                dvol_aligned[i] = dvol_vals[j]

        # dvol_zscore: z-score over 168 bars
        window = 168
        dvol_series = pd.Series(dvol_aligned)
        rolling_mean = dvol_series.rolling(window, min_periods=window).mean()
        rolling_std = dvol_series.rolling(window, min_periods=window).std()
        feat_df["dvol_zscore"] = np.where(
            rolling_std > 0.1,
            (dvol_aligned - rolling_mean) / rolling_std,
            np.nan,
        )

        # dvol_chg_24
        feat_df["dvol_chg_24"] = dvol_series.pct_change(24)

        # dvol_chg_72
        feat_df["dvol_chg_72"] = dvol_series.pct_change(72)

        # iv_term_struct: MA(24) / MA(168) - 1
        ma24 = dvol_series.rolling(24, min_periods=24).mean()
        ma168 = dvol_series.rolling(168, min_periods=168).mean()
        feat_df["iv_term_struct"] = np.where(ma168 > 0, ma24 / ma168 - 1, np.nan)

        # dvol_mean_rev: DVOL / MA(720) - 1
        ma720 = dvol_series.rolling(720, min_periods=720).mean()
        feat_df["dvol_mean_rev"] = np.where(ma720 > 0, dvol_aligned / ma720 - 1, np.nan)

        # iv_rv_spread: IV - RV
        if len(closes) > 20:
            log_ret = np.log(closes[1:] / closes[:-1])
            rv_series = pd.Series(log_ret).rolling(20, min_periods=20).std() * np.sqrt(8760)
            rv_aligned = np.full(len(timestamps), np.nan)
            rv_aligned[1:] = rv_series.values
            # DVOL is annualized percentage, RV is annualized decimal
            feat_df["iv_rv_spread"] = dvol_aligned / 100.0 - rv_aligned
        else:
            feat_df["iv_rv_spread"] = np.nan

    except Exception as e:
        logger.warning("IV feature computation failed for %s: %s", symbol, e)
        for feat in features_to_add:
            feat_df[feat] = np.nan


def _add_stablecoin_features(feat_df: pd.DataFrame, timestamps: np.ndarray) -> None:
    """Add stablecoin supply features (V22).

    Features: stablecoin_supply_chg_7d, stablecoin_supply_zscore
    """
    path = Path("data_files/stablecoin_daily.csv")
    features = ["stablecoin_supply_chg_7d", "stablecoin_supply_zscore"]

    if not path.exists():
        for f in features:
            feat_df[f] = np.nan
        return

    try:
        sc_df = pd.read_csv(path)
        if "date" not in sc_df.columns or "total_supply" not in sc_df.columns:
            for f in features:
                feat_df[f] = np.nan
            return

        sc_df["date"] = pd.to_datetime(sc_df["date"])
        sc_df = sc_df.sort_values("date")

        supply = sc_df["total_supply"].values.astype(np.float64)
        sc_dates = sc_df["date"].values.astype("datetime64[ms]").astype(np.int64)

        # 7-day supply change
        chg_7d = np.full(len(supply), np.nan)
        for i in range(7, len(supply)):
            if supply[i - 7] > 0:
                chg_7d[i] = (supply[i] - supply[i - 7]) / supply[i - 7]

        # Z-score over 30 days
        supply_series = pd.Series(supply)
        rolling_mean = supply_series.rolling(30, min_periods=30).mean()
        rolling_std = supply_series.rolling(30, min_periods=30).std()
        zscore = np.where(rolling_std > 0, (supply - rolling_mean) / rolling_std, np.nan)

        # Align to bar timestamps (daily -> hourly via forward fill)
        chg_aligned = np.full(len(timestamps), np.nan)
        zscore_aligned = np.full(len(timestamps), np.nan)
        j = 0
        for i, ts in enumerate(timestamps):
            while j < len(sc_dates) - 1 and sc_dates[j + 1] <= ts:
                j += 1
            if j < len(sc_dates):
                chg_aligned[i] = chg_7d[j]
                zscore_aligned[i] = zscore[j]

        feat_df["stablecoin_supply_chg_7d"] = chg_aligned
        feat_df["stablecoin_supply_zscore"] = zscore_aligned

    except Exception as e:
        logger.warning("Stablecoin feature computation failed: %s", e)
        for f in features:
            feat_df[f] = np.nan


def _add_etf_volume_features(feat_df: pd.DataFrame, timestamps: np.ndarray) -> None:
    """Add ETF volume features (V23).

    Features: etf_volume_zscore, etf_volume_chg_5d
    """
    path = Path("data_files/etf_volume_daily.csv")
    features = ["etf_volume_zscore", "etf_volume_chg_5d"]

    if not path.exists():
        for f in features:
            feat_df[f] = np.nan
        return

    try:
        etf_df = pd.read_csv(path)
        if "date" not in etf_df.columns or "total_volume" not in etf_df.columns:
            for f in features:
                feat_df[f] = np.nan
            return

        etf_df["date"] = pd.to_datetime(etf_df["date"])
        etf_df = etf_df.sort_values("date")

        volume = etf_df["total_volume"].values.astype(np.float64)
        etf_dates = etf_df["date"].values.astype("datetime64[ms]").astype(np.int64)

        # Z-score over 20 days
        vol_series = pd.Series(volume)
        rolling_mean = vol_series.rolling(20, min_periods=20).mean()
        rolling_std = vol_series.rolling(20, min_periods=20).std()
        zscore = np.where(rolling_std > 0, (volume - rolling_mean) / rolling_std, np.nan)

        # 5-day volume change
        chg_5d = np.full(len(volume), np.nan)
        for i in range(5, len(volume)):
            if volume[i - 5] > 0:
                chg_5d[i] = (volume[i] - volume[i - 5]) / volume[i - 5]

        # Align to bar timestamps
        zscore_aligned = np.full(len(timestamps), np.nan)
        chg_aligned = np.full(len(timestamps), np.nan)
        j = 0
        for i, ts in enumerate(timestamps):
            while j < len(etf_dates) - 1 and etf_dates[j + 1] <= ts:
                j += 1
            if j < len(etf_dates):
                zscore_aligned[i] = zscore[j]
                chg_aligned[i] = chg_5d[j]

        feat_df["etf_volume_zscore"] = zscore_aligned
        feat_df["etf_volume_chg_5d"] = chg_aligned

    except Exception as e:
        logger.warning("ETF volume feature computation failed: %s", e)
        for f in features:
            feat_df[f] = np.nan
