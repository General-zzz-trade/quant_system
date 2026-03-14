# tests/unit/features/test_batch_cross_asset.py
"""Parity tests: batch_cross_asset (vectorized) vs CrossAssetComputer (Rust incremental)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.cross_asset_computer import CrossAssetComputer, CROSS_ASSET_FEATURE_NAMES
from features.batch_cross_asset import (
    compute_cross_features_for_symbol,
    _rolling_ret,
    _rsi_wilder,
    _macd_line,
    _sma,
    _rolling_std,
    _forward_fill_schedule,
)


def _make_df(n: int, base: float, step: float, ts_start: int = 1000000) -> pd.DataFrame:
    """Create OHLCV DataFrame with slight uptrend."""
    close = base + np.arange(n) * step
    return pd.DataFrame({
        "timestamp": np.arange(ts_start, ts_start + n * 3600000, 3600000, dtype=np.int64),
        "open": close - 1,
        "high": close + 5,
        "low": close - 5,
        "close": close,
        "volume": np.full(n, 100.0),
    })


def _make_funding(timestamps: np.ndarray, rate: float) -> dict[int, float]:
    """Create funding schedule at every 8h (every 8th bar)."""
    schedule = {}
    for i, ts in enumerate(timestamps):
        if i % 8 == 0:
            schedule[int(ts)] = rate
    return schedule


class TestBatchCrossAssetParity:
    """Verify vectorized batch matches Rust incremental computation."""

    def _run_parity(self, n: int = 100, btc_step: float = 10.0, eth_step: float = 8.0,
                    btc_fr: float = 0.0001, eth_fr: float = 0.00015):
        """Run both paths and return (batch_df, incremental_records)."""
        ts_start = 1704067200000  # 2024-01-01
        btc_df = _make_df(n, 40000.0, btc_step, ts_start)
        eth_df = _make_df(n, 3000.0, eth_step, ts_start)

        btc_timestamps = btc_df["timestamp"].values
        eth_timestamps = eth_df["timestamp"].values

        btc_funding = _make_funding(btc_timestamps, btc_fr)
        eth_funding = _make_funding(eth_timestamps, eth_fr)

        # Path 1: Vectorized batch
        batch_df = compute_cross_features_for_symbol(
            "ETHUSDT", eth_df, btc_df, btc_funding, eth_funding)

        # Path 2: Rust incremental (bar-by-bar)
        comp = CrossAssetComputer()
        records = []

        btc_f_times = sorted(btc_funding.keys())
        eth_f_times = sorted(eth_funding.keys())
        btc_fi, eth_fi = 0, 0

        for i in range(n):
            ts = int(eth_timestamps[i])

            # Forward-fill BTC funding
            cur_btc_fr = None
            while btc_fi < len(btc_f_times) and btc_f_times[btc_fi] <= ts:
                cur_btc_fr = btc_funding[btc_f_times[btc_fi]]
                btc_fi += 1
            if cur_btc_fr is None and btc_fi > 0:
                cur_btc_fr = btc_funding[btc_f_times[btc_fi - 1]]

            # Forward-fill ETH funding
            cur_eth_fr = None
            while eth_fi < len(eth_f_times) and eth_f_times[eth_fi] <= ts:
                cur_eth_fr = eth_funding[eth_f_times[eth_fi]]
                eth_fi += 1
            if cur_eth_fr is None and eth_fi > 0:
                cur_eth_fr = eth_funding[eth_f_times[eth_fi - 1]]

            # Push BTC bar first (same timestamp = exists)
            btc_close = float(btc_df.iloc[i]["close"])
            btc_high = float(btc_df.iloc[i]["high"])
            btc_low = float(btc_df.iloc[i]["low"])
            comp.on_bar("BTCUSDT", close=btc_close, funding_rate=cur_btc_fr,
                        high=btc_high, low=btc_low)

            # Push ETH bar
            eth_close = float(eth_df.iloc[i]["close"])
            comp.on_bar("ETHUSDT", close=eth_close, funding_rate=cur_eth_fr)

            feats = comp.get_features("ETHUSDT")
            records.append(feats)

        return batch_df, records

    def test_btc_ret_features_match(self):
        batch_df, records = self._run_parity(80)
        for name in ("btc_ret_1", "btc_ret_3", "btc_ret_6", "btc_ret_12", "btc_ret_24"):
            for i in range(len(records)):
                rust_val = records[i][name]
                batch_val = batch_df.iloc[i][name]
                if rust_val is None:
                    assert np.isnan(batch_val), f"{name}[{i}]: Rust=None, batch={batch_val}"
                else:
                    assert batch_val == pytest.approx(rust_val, abs=1e-8), \
                        f"{name}[{i}]: Rust={rust_val}, batch={batch_val}"

    def test_btc_rsi_matches(self):
        batch_df, records = self._run_parity(80)
        name = "btc_rsi_14"
        matched = 0
        for i in range(len(records)):
            rust_val = records[i][name]
            batch_val = batch_df.iloc[i][name]
            if rust_val is None:
                continue
            if np.isnan(batch_val):
                continue
            # RSI uses EMA which can have small initialization differences
            assert batch_val == pytest.approx(rust_val, rel=0.05), \
                f"{name}[{i}]: Rust={rust_val}, batch={batch_val}"
            matched += 1
        assert matched > 40, f"Too few RSI matches: {matched}"

    def test_btc_macd_matches(self):
        batch_df, records = self._run_parity(80)
        name = "btc_macd_line"
        matched = 0
        for i in range(len(records)):
            rust_val = records[i][name]
            batch_val = batch_df.iloc[i][name]
            if rust_val is None:
                continue
            if np.isnan(batch_val):
                continue
            assert batch_val == pytest.approx(rust_val, rel=0.05), \
                f"{name}[{i}]: Rust={rust_val}, batch={batch_val}"
            matched += 1
        assert matched > 40, f"Too few MACD matches: {matched}"

    def test_btc_technical_features_match(self):
        batch_df, records = self._run_parity(80)
        for name in ("btc_mean_reversion_20", "btc_bb_width_20"):
            matched = 0
            for i in range(len(records)):
                rust_val = records[i][name]
                batch_val = batch_df.iloc[i][name]
                if rust_val is None:
                    continue
                if np.isnan(batch_val):
                    continue
                assert batch_val == pytest.approx(rust_val, rel=0.05), \
                    f"{name}[{i}]: Rust={rust_val}, batch={batch_val}"
                matched += 1
            assert matched > 40, f"Too few {name} matches: {matched}"

    def test_btc_atr_norm_matches(self):
        batch_df, records = self._run_parity(80)
        name = "btc_atr_norm_14"
        matched = 0
        for i in range(len(records)):
            rust_val = records[i][name]
            batch_val = batch_df.iloc[i][name]
            if rust_val is None:
                continue
            if np.isnan(batch_val):
                continue
            assert batch_val == pytest.approx(rust_val, rel=0.1), \
                f"{name}[{i}]: Rust={rust_val}, batch={batch_val}"
            matched += 1
        assert matched > 40, f"Too few ATR matches: {matched}"

    def test_rolling_beta_matches(self):
        batch_df, records = self._run_parity(80)
        for name in ("rolling_beta_30", "rolling_beta_60"):
            matched = 0
            for i in range(len(records)):
                rust_val = records[i][name]
                batch_val = batch_df.iloc[i][name]
                if rust_val is None:
                    continue
                if np.isnan(batch_val):
                    continue
                assert batch_val == pytest.approx(rust_val, rel=0.1), \
                    f"{name}[{i}]: Rust={rust_val}, batch={batch_val}"
                matched += 1
            if name == "rolling_beta_30":
                assert matched > 30, f"Too few beta_30 matches: {matched}"

    def test_rolling_corr_matches(self):
        batch_df, records = self._run_parity(80)
        name = "rolling_corr_30"
        matched = 0
        for i in range(len(records)):
            rust_val = records[i][name]
            batch_val = batch_df.iloc[i][name]
            if rust_val is None:
                continue
            if np.isnan(batch_val):
                continue
            assert batch_val == pytest.approx(rust_val, rel=0.1), \
                f"{name}[{i}]: Rust={rust_val}, batch={batch_val}"
            matched += 1
        assert matched > 30, f"Too few correlation matches: {matched}"

    def test_relative_strength_matches(self):
        batch_df, records = self._run_parity(80)
        name = "relative_strength_20"
        matched = 0
        for i in range(len(records)):
            rust_val = records[i][name]
            batch_val = batch_df.iloc[i][name]
            if rust_val is None:
                continue
            if np.isnan(batch_val):
                continue
            assert batch_val == pytest.approx(rust_val, rel=0.1), \
                f"{name}[{i}]: Rust={rust_val}, batch={batch_val}"
            matched += 1
        assert matched > 30, f"Too few rel_str matches: {matched}"

    def test_funding_features_match(self):
        batch_df, records = self._run_parity(80)
        for name in ("funding_diff", "funding_diff_ma8"):
            matched = 0
            for i in range(len(records)):
                rust_val = records[i][name]
                batch_val = batch_df.iloc[i][name]
                if rust_val is None:
                    continue
                if np.isnan(batch_val):
                    continue
                assert batch_val == pytest.approx(rust_val, abs=1e-8), \
                    f"{name}[{i}]: Rust={rust_val}, batch={batch_val}"
                matched += 1
            assert matched > 40, f"Too few {name} matches: {matched}"

    def test_all_17_features_present(self):
        batch_df, _ = self._run_parity(80)
        for name in CROSS_ASSET_FEATURE_NAMES:
            assert name in batch_df.columns, f"Missing feature: {name}"

    def test_output_shape(self):
        n = 100
        batch_df, records = self._run_parity(n)
        assert len(batch_df) == n
        assert len(records) == n
        assert len(batch_df.columns) == 17


class TestVectorizedHelpers:
    """Unit tests for vectorized helper functions."""

    def test_rolling_ret(self):
        close = np.array([100, 110, 121, 133.1])
        ret = _rolling_ret(close, 1)
        assert np.isnan(ret[0])
        assert ret[1] == pytest.approx(0.10)
        assert ret[2] == pytest.approx(0.10)

    def test_sma(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        sma = _sma(x, 3)
        assert np.isnan(sma[0])
        assert np.isnan(sma[1])
        assert sma[2] == pytest.approx(2.0)
        assert sma[3] == pytest.approx(3.0)
        assert sma[4] == pytest.approx(4.0)

    def test_rolling_std(self):
        x = np.array([1, 1, 1, 1], dtype=np.float64)
        std = _rolling_std(x, 3)
        assert std[2] == pytest.approx(0.0, abs=1e-10)

    def test_forward_fill_schedule(self):
        timestamps = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        schedule = {100: 0.5, 300: 1.0}
        filled = _forward_fill_schedule(timestamps, schedule)
        assert filled[0] == 0.5
        assert filled[1] == 0.5  # forward-fill
        assert filled[2] == 1.0
        assert filled[3] == 1.0  # forward-fill
        assert filled[4] == 1.0

    def test_forward_fill_empty_schedule(self):
        timestamps = np.array([100, 200], dtype=np.int64)
        filled = _forward_fill_schedule(timestamps, {})
        assert np.isnan(filled[0])
        assert np.isnan(filled[1])

    def test_rsi_range(self):
        # Monotonic uptrend → RSI near 100
        close = 100.0 + np.arange(50) * 1.0
        rsi = _rsi_wilder(close, 14)
        # After warmup, RSI should be high for pure uptrend
        valid = ~np.isnan(rsi)
        assert np.all(rsi[valid] > 50)

    def test_macd_uptrend_positive(self):
        close = 100.0 + np.arange(50) * 1.0
        macd = _macd_line(close)
        # In uptrend, EMA(12) > EMA(26) → positive MACD
        valid = ~np.isnan(macd)
        assert np.all(macd[valid] > 0)
