"""Configuration constants for auto_retrain.

Extracted from auto_retrain.py to keep it under 500 lines.
"""
from pathlib import Path


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_HORIZONS = [24]               # h12 dropped: IC collapsed in live (BTC -341%, ETH -91%)

# Per-symbol forced features: always included in IC selection regardless of rank.
# These are hand-picked for IC stability (consistent sign across 30/60/90/180d windows)
# and live availability (computed in feature_hook, not batch-only).
FORCED_FEATURES: dict[str, list[str]] = {
    "ETHUSDT": [
        # 9 features — audit 2026-04-11: dropped macd_hist (SIGN_FLIP 180d),
        # dropped vix_chg_1d (WEAK: 30d -0.008), added stablecoin_supply_chg_7d
        # (STRONG across all windows). Rationale: ETH 1h h12 60d IC ~0 indicated
        # overfitting; tightened forced list to STRONG-only features.
        "ls_ratio",                 # +0.151/+0.166/+0.166/+0.166 (STRONG)
        "oc_netflow_zscore_7",      # +0.236/+0.279/+0.087/+0.025 (STRONG, on-chain)
        "vix_level",                # +0.086/+0.131/+0.138/+0.039 (STRONG, macro)
        "fgi_extreme",              # -0.160/-0.102/-0.063/-0.080 (STRONG, contrarian)
        "funding_sign_persist",     # +0.131/+0.101/+0.098/+0.033 (STRONG, sentiment)
        "spy_ret_1d",               # -0.016/-0.131/-0.058/-0.050 (STRONG, risk-off)
        "qqq_ret_1d",               # -0.015/-0.144/-0.046/-0.058 (STRONG, tech)
        "iwm_ret_1d",               # -0.024/-0.110/-0.058/-0.052 (STRONG, small-cap)
        "stablecoin_supply_chg_7d", # +0.110/+0.027/+0.063/+0.040 (STRONG, liquidity)
    ],
    "BTCUSDT": [
        # Soft forcing: only 5 core macro features (2026-04-10 audit)
        # Previous attempt with 10 forced hit comparison_gate — too rigid.
        # These 5 are the highest-conviction stable signals across 30/60/90/180d;
        # let greedy IC selection fill remaining slots for diversification.
        "vix_chg_1d",    # +0.09/+0.17/+0.09/+0.04 (unique contrarian alpha)
        "gld_ret_5d",    # +0.17/+0.04/+0.06/+0.01 (safe-haven divergence)
        "spy_ret_1d",    # -0.07/-0.16/-0.07/-0.06 (equity risk-off)
        "qqq_ret_1d",    # -0.07/-0.17/-0.07/-0.08 (tech risk-off)
        "iwm_ret_1d",    # -0.08/-0.13/-0.07/-0.05 (small-cap risk-off)
    ],
}
MODEL_DIR_TEMPLATE = "models_v8/{symbol}_gate_v2"
MODEL_DIR_OVERRIDES: dict[str, str] = {}

# Per-symbol training window cutoff (regime-focused training).
# Older crypto data (pre-2023) is often too different from current regime.
# 0 = use all data (default). N = last N years from train_end.
# 2026-04-11: ETH set to 3.0 after discovering h12 60d IC ~0 (overfitting on
# long 2019-2022 tail that has different correlations vs 2023-2026 regime).
MAX_TRAIN_YEARS: dict[str, float] = {
    "ETHUSDT": 3.0,
    "SOLUSDT": 3.0,  # post-FTX (2022-11) regime is meaningfully different
}
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
DEFAULT_HORIZONS_4H = {
    "BTCUSDT": [6, 12, 24],  # ensemble (Sharpe 6.08) > any single horizon; keep all
    "ETHUSDT": [6],           # h12 dropped: train IC=0.007 (near zero); h6 IC=0.110 (good)
}

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
