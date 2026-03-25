"""Configuration constants for auto_retrain.

Extracted from auto_retrain.py to keep it under 500 lines.
"""
from pathlib import Path


SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_HORIZONS = [12, 24]          # h48 dropped by default (negative IC)
MODEL_DIR_TEMPLATE = "models_v8/{symbol}_gate_v2"
MODEL_DIR_OVERRIDES: dict[str, str] = {}
DATA_DIR_TEMPLATE = "data_files/{symbol}_1h.csv"
RETRAIN_LOG = Path("logs/retrain_history.jsonl")

# ── 15m Configuration ──
SYMBOLS_15M = ["BTCUSDT", "ETHUSDT"]  # SOLUSDT 15m FAIL (1/4 PASS), removed
DEFAULT_HORIZONS_15M = {
    "BTCUSDT": [4, 8],       # 1h, 2h -- high frequency
    "ETHUSDT": [4, 8],       # 1h, 2h -- high frequency
    "SOLUSDT": [4, 8, 16],   # 1h, 2h, 4h
}
MODEL_DIR_15M_TEMPLATE = "models_v8/{symbol}_15m"

# ── 4h Configuration ──
SYMBOLS_4H = ["BTCUSDT", "ETHUSDT"]
MODEL_DIR_4H_TEMPLATE = "models_v8/{symbol}_4h"

# Validation thresholds
MIN_IC = 0.02                         # minimum IC for new model to deploy
MIN_SHARPE = 1.0                      # minimum Sharpe for new model
DECAY_TOLERANCE = 0.7                 # new Sharpe >= old * this (30% decay OK)
MIN_TRADES = 15                       # minimum OOS trades
BOOTSTRAP_P5_MIN = 0.0               # bootstrap p5 must be positive
MIN_FINAL_SHARPE = 0.5               # final fold Sharpe must be > this
MIN_FINAL_AVG_NET_BPS = 2.0          # final fold avg net bps must be > this

# ── Daily retrain configuration ──
DAILY_MAX_AGE_HOURS = 24              # only retrain if model older than this
DAILY_IC_TOLERANCE = 0.95             # new IC >= old IC * this (5% tolerance)
DAILY_VALIDATION_MONTHS = 3           # shorter validation window for speed
