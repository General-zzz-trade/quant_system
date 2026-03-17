"""Precomputed feature hook for backtesting.

Pre-computes all features from historical OHLCV data using the same batch
feature engine as model training. Serves features bar-by-bar as the engine
replays market events. This ensures feature consistency between training
and backtesting without needing RustFeatureEngine's live data sources.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class PrecomputedFeatureHook:
    """Serves pre-computed features to the engine pipeline.

    Usage:
        hook = PrecomputedFeatureHook.from_csv(symbol, csv_path)
        # Pass as feature_computer to run_backtest()
    """

    def __init__(self, features_by_ts: Dict[int, Dict[str, float]]) -> None:
        """
        Parameters
        ----------
        features_by_ts : dict
            Maps timestamp (ms int64) → feature dict.
        """
        self._features_by_ts = features_by_ts
        self._last_features: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_dataframe(
        cls,
        symbol: str,
        df: pd.DataFrame,
        include_4h: bool = True,
        include_interactions: bool = True,
    ) -> "PrecomputedFeatureHook":
        """Build from raw OHLCV DataFrame using batch feature engine."""
        from features.batch_feature_engine import compute_features_batch

        has_macro = False
        try:
            from pathlib import Path
            has_macro = Path("data_files/macro_daily.csv").exists()
        except Exception as e:
            logger.debug("Failed to check macro data file: %s", e)

        feat_df = compute_features_batch(symbol, df, include_v11=has_macro)

        if include_4h:
            try:
                from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
                tf4h = compute_4h_features(df)
                for col in TF4H_FEATURE_NAMES:
                    if col in tf4h.columns:
                        feat_df[col] = tf4h[col].values
            except Exception as e:
                logger.warning("Failed to compute 4h features: %s", e)

        if include_interactions:
            try:
                from scripts.train_v7_alpha import INTERACTION_FEATURES  # type: ignore[import-not-found]
                for int_name, fa, fb in INTERACTION_FEATURES:
                    if fa in feat_df.columns and fb in feat_df.columns:
                        feat_df[int_name] = (
                            feat_df[fa].astype(float) * feat_df[fb].astype(float)
                        )
            except Exception as e:
                logger.warning("Failed to compute interaction features: %s", e)

        # Build timestamp → features mapping
        ts_col = "open_time" if "open_time" in df.columns else "timestamp"
        timestamps = df[ts_col].values.astype(np.int64)

        feature_cols = [
            c for c in feat_df.columns
            if c not in ("close", "open_time", "timestamp", "open", "high", "low", "volume")
        ]

        features_by_ts: Dict[int, Dict[str, float]] = {}
        feat_vals = feat_df[feature_cols].values

        for i, ts in enumerate(timestamps):
            if i < len(feat_vals):
                row = feat_vals[i]
                fdict = {}
                for j, col in enumerate(feature_cols):
                    v = float(row[j])
                    if not math.isnan(v):
                        fdict[col] = v
                features_by_ts[int(ts)] = fdict

        return cls(features_by_ts)

    @classmethod
    def from_csv(
        cls,
        symbol: str,
        csv_path: str,
        include_4h: bool = True,
    ) -> "PrecomputedFeatureHook":
        """Build from CSV file path."""
        df = pd.read_csv(csv_path)
        return cls.from_dataframe(symbol, df, include_4h=include_4h)

    def on_event(self, event: Any) -> Optional[Mapping[str, Any]]:
        """Feature hook protocol: event → features dict or None."""
        from event.types import EventType as _EventType

        kind = getattr(event, "event_type", None)
        if kind is not None and kind is not _EventType.MARKET:
            kind_val = getattr(kind, "value", kind)
            if not isinstance(kind_val, str) or "market" not in kind_val.lower():
                sym = getattr(event, "symbol", None)
                if sym and sym in self._last_features:
                    return dict(self._last_features[sym])
                return None

        symbol = getattr(event, "symbol", None)
        ts = getattr(event, "ts", None)
        if symbol is None or ts is None:
            return None

        # Convert datetime to ms timestamp for lookup
        ts_ms = self._resolve_ts(ts)
        if ts_ms is None:
            return None

        features = self._features_by_ts.get(ts_ms)
        if features is not None:
            self._last_features[symbol] = features
            return features

        # Try nearby timestamps (±1ms tolerance for float rounding)
        for offset in (-1, 1, -2, 2):
            features = self._features_by_ts.get(ts_ms + offset)
            if features is not None:
                self._last_features[symbol] = features
                return features

        # Return last known features for this symbol
        return self._last_features.get(symbol)

    @staticmethod
    def _resolve_ts(ts: Any) -> Optional[int]:
        """Convert various timestamp formats to milliseconds int."""
        if isinstance(ts, (int, np.integer)):
            n = int(ts)
            return n if n > 1_000_000_000_000 else n * 1000

        from datetime import datetime
        if isinstance(ts, datetime):
            return int(ts.timestamp() * 1000)

        if isinstance(ts, str):
            try:
                from datetime import datetime as dt, timezone
                s = ts.strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                parsed = dt.fromisoformat(s)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return int(parsed.timestamp() * 1000)
            except ValueError:
                pass

        return None
