# features/options_flow.py
"""Options flow feature computer — gamma imbalance, max pain, vega exposure.

Computes features from Deribit options snapshots stored in SQLite:
  - gamma_imbalance: net gamma by strike relative to spot (dealer hedging pressure)
  - max_pain: strike price where total option losses are maximized
  - vega_exposure: net vega (IV directional bet size)
  - iv_term_slope: ATM IV near vs far (term structure slope)
  - pcr_zscore: put/call ratio z-score (extreme positioning)
  - vol_skew_25d: 25-delta risk reversal (crash fear premium)

All features are computed incrementally and cached.
Data source: scripts/data/download_deribit_options.py → SQLite DB.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

_log = logging.getLogger(__name__)


@dataclass
class OptionsFlowConfig:
    """Configuration for options flow features."""
    btc_db_path: str = "data/options/btc_options.db"
    eth_db_path: str = "data/options/eth_options.db"
    zscore_window: int = 168     # 7 days of hourly data
    max_pain_n_strikes: int = 50  # top N strikes for max pain calc
    refresh_interval_s: float = 300.0  # 5 min refresh


class OptionsFlowComputer:
    """Compute options flow features from Deribit snapshot database.

    Features produced:
      - gamma_imbalance_zscore: z-score of net gamma exposure
      - max_pain_distance: (spot - max_pain) / spot, signed
      - vega_net_zscore: z-score of net vega exposure
      - iv_term_slope: near ATM IV - far ATM IV (contango/backwardation)
      - pcr_zscore: put/call ratio z-score over zscore_window
      - iv_rv_premium: ATM IV - realized vol (overpriced = high premium)
      - dvol_zscore: DVOL z-score over zscore_window
    """

    FEATURE_NAMES = (
        "gamma_imbalance_zscore",
        "max_pain_distance",
        "vega_net_zscore",
        "iv_term_slope",
        "pcr_zscore",
        "iv_rv_premium",
        "dvol_zscore",
    )

    def __init__(self, cfg: OptionsFlowConfig | None = None) -> None:
        self._cfg = cfg or OptionsFlowConfig()
        # Rolling windows for z-score computation
        self._pcr_history: Deque[float] = deque(maxlen=self._cfg.zscore_window)
        self._dvol_history: Deque[float] = deque(maxlen=self._cfg.zscore_window)
        self._gamma_history: Deque[float] = deque(maxlen=self._cfg.zscore_window)
        self._vega_history: Deque[float] = deque(maxlen=self._cfg.zscore_window)
        self._last_features: Dict[str, float] = {}
        self._last_refresh_ts: float = 0.0

    def compute(
        self,
        symbol: str,
        spot_price: float,
        realized_vol: float = 0.0,
        ts_ms: int = 0,
    ) -> Dict[str, float]:
        """Compute options flow features for the given symbol.

        Args:
            symbol: Trading symbol (e.g., BTCUSDT, ETHUSDT)
            spot_price: Current spot/index price
            realized_vol: Current realized vol (e.g., vol_20 from enriched computer)
            ts_ms: Current timestamp in milliseconds

        Returns:
            Dict of feature name → value. Missing features are NaN.
        """
        currency = "BTC" if "BTC" in symbol.upper() else "ETH"
        db_path = self._cfg.btc_db_path if currency == "BTC" else self._cfg.eth_db_path

        if not Path(db_path).exists():
            return {name: float("nan") for name in self.FEATURE_NAMES}

        try:
            snap = self._load_latest_snapshot(db_path, ts_ms)
        except Exception as e:
            _log.debug("Options flow data unavailable: %s", e)
            return {name: float("nan") for name in self.FEATURE_NAMES}

        if snap is None:
            return {name: float("nan") for name in self.FEATURE_NAMES}

        features: Dict[str, float] = {}

        # 1. PCR z-score
        pcr = snap.get("pcr", 0.0) or 0.0
        self._pcr_history.append(pcr)
        features["pcr_zscore"] = _rolling_zscore(self._pcr_history, pcr)

        # 2. DVOL z-score
        dvol = snap.get("dvol", 0.0) or 0.0
        self._dvol_history.append(dvol)
        features["dvol_zscore"] = _rolling_zscore(self._dvol_history, dvol)

        # 3. IV term slope (near - far: positive = inverted term structure = fear)
        atm_near = snap.get("atm_iv_near")
        atm_far = snap.get("atm_iv_far")
        if atm_near is not None and atm_far is not None:
            features["iv_term_slope"] = atm_near - atm_far
        else:
            features["iv_term_slope"] = float("nan")

        # 4. IV-RV premium
        if atm_near is not None and realized_vol > 0:
            features["iv_rv_premium"] = atm_near - realized_vol * 100  # IV in %, RV as decimal
        else:
            features["iv_rv_premium"] = float("nan")

        # 5. Gamma imbalance from OI distribution
        call_oi = snap.get("call_oi", 0.0) or 0.0
        put_oi = snap.get("put_oi", 0.0) or 0.0
        total_oi = call_oi + put_oi
        if total_oi > 0:
            # Gamma imbalance: (call_oi - put_oi) / total_oi
            # Positive = dealer short gamma on calls = upside hedging pressure
            gamma_raw = (call_oi - put_oi) / total_oi
            self._gamma_history.append(gamma_raw)
            features["gamma_imbalance_zscore"] = _rolling_zscore(self._gamma_history, gamma_raw)
        else:
            features["gamma_imbalance_zscore"] = float("nan")

        # 6. Vega exposure (proxy: volume-weighted PCR direction)
        call_vol = snap.get("call_vol_24h", 0.0) or 0.0
        put_vol = snap.get("put_vol_24h", 0.0) or 0.0
        total_vol = call_vol + put_vol
        if total_vol > 0:
            vega_raw = (put_vol - call_vol) / total_vol  # positive = put buying
            self._vega_history.append(vega_raw)
            features["vega_net_zscore"] = _rolling_zscore(self._vega_history, vega_raw)
        else:
            features["vega_net_zscore"] = float("nan")

        # 7. Max pain distance
        max_pain = self._estimate_max_pain(snap, spot_price)
        if max_pain is not None and spot_price > 0:
            features["max_pain_distance"] = (spot_price - max_pain) / spot_price
        else:
            features["max_pain_distance"] = float("nan")

        self._last_features = features
        return features

    def _load_latest_snapshot(
        self, db_path: str, ts_ms: int = 0
    ) -> Optional[Dict[str, float]]:
        """Load the most recent snapshot from SQLite."""
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            if ts_ms > 0:
                row = conn.execute(
                    "SELECT * FROM snapshots WHERE ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1",
                    (ts_ms,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM snapshots ORDER BY ts_ms DESC LIMIT 1"
                ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def _estimate_max_pain(
        self, snap: Dict, spot_price: float
    ) -> Optional[float]:
        """Estimate max pain from term structure data.

        Max pain = strike where total option holder losses are maximized.
        With limited data (no per-strike OI), approximate using PCR and
        ATM strikes from term structure.
        """
        import json as _json

        ts_str = snap.get("term_structure", "[]")
        try:
            ts = _json.loads(ts_str) if isinstance(ts_str, str) else ts_str
        except Exception:
            return None

        if not ts or not isinstance(ts, list):
            return None

        # Use near-term ATM strike as max pain proxy
        # (real max pain would need per-strike OI which we may not have)
        strikes = [entry.get("strike") for entry in ts if entry.get("strike")]
        if not strikes:
            return None

        # Max pain tends to be near the ATM strike of the nearest expiry
        return float(strikes[0])

    @property
    def last_features(self) -> Dict[str, float]:
        return dict(self._last_features)


def _rolling_zscore(window: Deque[float], value: float) -> float:
    """Compute z-score of value against rolling window."""
    if len(window) < 10:
        return 0.0
    vals = list(window)
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var) if var > 0 else 1e-8
    return (value - mean) / std
