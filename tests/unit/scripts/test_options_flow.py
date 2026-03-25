"""Tests for OptionsFlowComputer."""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent.parent))

import math
import sqlite3
import json

from features.options_flow import OptionsFlowComputer, OptionsFlowConfig


def _create_test_db(path: str, n_snapshots: int = 20) -> None:
    """Create a test options database with fake data."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            index_price REAL,
            call_oi REAL, put_oi REAL, pcr REAL,
            call_vol_24h REAL, put_vol_24h REAL, vol_pcr REAL,
            dvol REAL, hist_iv REAL,
            atm_iv_near REAL, atm_iv_far REAL, term_spread REAL,
            n_expiries INTEGER,
            term_structure TEXT
        )
    """)
    for i in range(n_snapshots):
        ts_ms = 1700000000000 + i * 3600000  # hourly
        pcr = 0.8 + (i % 5) * 0.1  # varies 0.8-1.2
        conn.execute("""
            INSERT INTO snapshots (ts_ms, timestamp, index_price,
                call_oi, put_oi, pcr, call_vol_24h, put_vol_24h, vol_pcr,
                dvol, hist_iv, atm_iv_near, atm_iv_far, term_spread,
                n_expiries, term_structure)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ts_ms, f"2023-11-{14+i//24}T{i%24:02d}:00:00Z", 37000 + i * 10,
            100000 + i * 100, 80000 + i * 80, pcr,
            5000 + i * 50, 4000 + i * 40, (4000 + i*40) / (5000 + i*50),
            55.0 + i * 0.1, 45.0, 56.0, 50.0, 6.0,
            3, json.dumps([{"expiry": "24NOV23", "atm_iv": 56.0, "strike": 37000}]),
        ))
    conn.commit()
    conn.close()


class TestOptionsFlowComputer:
    def test_missing_db_returns_nan(self):
        cfg = OptionsFlowConfig(btc_db_path="/nonexistent.db")
        comp = OptionsFlowComputer(cfg)
        feats = comp.compute("BTCUSDT", 37000.0)
        assert all(math.isnan(v) for v in feats.values())

    def test_compute_with_data(self, tmp_path):
        db_path = str(tmp_path / "btc_options.db")
        _create_test_db(db_path, n_snapshots=30)

        cfg = OptionsFlowConfig(btc_db_path=db_path)
        comp = OptionsFlowComputer(cfg)

        # Compute features multiple times to build up z-score window
        for i in range(15):
            feats = comp.compute("BTCUSDT", 37000.0 + i * 10, realized_vol=0.45)

        # After enough samples, z-scores should be computable
        assert "pcr_zscore" in feats
        assert "dvol_zscore" in feats
        assert "iv_term_slope" in feats
        assert "iv_rv_premium" in feats
        assert "gamma_imbalance_zscore" in feats

    def test_feature_names(self):
        comp = OptionsFlowComputer()
        assert len(comp.FEATURE_NAMES) == 7
        assert "gamma_imbalance_zscore" in comp.FEATURE_NAMES
        assert "max_pain_distance" in comp.FEATURE_NAMES
        assert "pcr_zscore" in comp.FEATURE_NAMES

    def test_iv_term_slope(self, tmp_path):
        db_path = str(tmp_path / "btc_options.db")
        _create_test_db(db_path, n_snapshots=5)

        cfg = OptionsFlowConfig(btc_db_path=db_path)
        comp = OptionsFlowComputer(cfg)
        feats = comp.compute("BTCUSDT", 37000.0)

        # atm_iv_near=56, atm_iv_far=50, slope should be 6.0
        assert feats["iv_term_slope"] == 6.0

    def test_iv_rv_premium(self, tmp_path):
        db_path = str(tmp_path / "btc_options.db")
        _create_test_db(db_path, n_snapshots=5)

        cfg = OptionsFlowConfig(btc_db_path=db_path)
        comp = OptionsFlowComputer(cfg)
        feats = comp.compute("BTCUSDT", 37000.0, realized_vol=0.45)

        # atm_iv_near=56%, realized_vol=0.45 → 45%, premium = 11
        assert abs(feats["iv_rv_premium"] - 11.0) < 0.5

    def test_eth_uses_eth_db(self, tmp_path):
        btc_path = str(tmp_path / "btc_options.db")
        eth_path = str(tmp_path / "eth_options.db")
        _create_test_db(eth_path, n_snapshots=5)

        cfg = OptionsFlowConfig(btc_db_path=btc_path, eth_db_path=eth_path)
        comp = OptionsFlowComputer(cfg)

        # BTC path doesn't exist → NaN
        btc_feats = comp.compute("BTCUSDT", 37000.0)
        assert all(math.isnan(v) for v in btc_feats.values())

        # ETH path exists → has values
        eth_feats = comp.compute("ETHUSDT", 2000.0)
        assert not math.isnan(eth_feats["iv_term_slope"])

    def test_last_features_property(self, tmp_path):
        db_path = str(tmp_path / "btc_options.db")
        _create_test_db(db_path, n_snapshots=5)

        cfg = OptionsFlowConfig(btc_db_path=db_path)
        comp = OptionsFlowComputer(cfg)
        comp.compute("BTCUSDT", 37000.0)

        last = comp.last_features
        assert len(last) == 7
