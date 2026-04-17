"""Configuration constants for auto_retrain.

Extracted from auto_retrain.py to keep it under 500 lines.
"""
from pathlib import Path


# 2026-04-12: SOLUSDT dropped from scheduled retrain per user decision
# to focus on BTC+ETH only. Model artefacts remain on disk under
# models_v8/SOLUSDT_gate_v2/ but no longer participate in the weekly /
# daily retrain cycle.  Keep DEFAULT_HORIZONS_15M entry for SOL so that
# a manual ``--only-15m --symbol SOLUSDT`` one-off still works if the
# user changes their mind.
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_HORIZONS = [24]               # h12 dropped: IC collapsed in live (BTC -341%, ETH -91%)

# Per-symbol forced features: always included in IC selection regardless of rank.
# These are hand-picked for IC stability (consistent sign across 30/60/90/180d windows)
# and live availability (computed in feature_hook, not batch-only).
FORCED_FEATURES: dict[str, list[str]] = {
    "ETHUSDT": [
        # 2026-04-13 v3: 19 forced — bear market IC audit + trend features:
        #   ADDED vix_chg_1d (IC=-0.14, replaces weak vix_level IC=0.01)
        #   ADDED etha_ret_1d (residual IC=-0.078, ETH ETF flow signal)
        #   REPLACED gld_ret_1d → gold_btc_return_spread_5d (gld_ret sign-flip)
        "ls_ratio",                     # on-chain long/short
        "oc_netflow_zscore_7",          # on-chain netflow
        "fgi_extreme",                  # contrarian sentiment
        "funding_sign_persist",         # funding direction persistence
        "spy_ret_1d",                   # risk-off
        "qqq_ret_1d",                   # tech risk-off
        "iwm_ret_1d",                   # small-cap
        "stablecoin_supply_chg_7d",     # liquidity
        "coin_ret_1d",                  # Coinbase sentiment
        "funding_cumulative_8",         # funding pressure
        "fxi_ret_1d",                   # China equity proxy
        "hyg_ret_1d",                   # HY credit
        "m2_yoy_change",               # M2 money supply
        "vix_chg_1d",                   # NEW: VIX change (IC=-0.14, replaces vix_level)
        "etha_ret_1d",                  # NEW: ETH ETF flow (residual IC=-0.078)
        "gold_btc_return_spread_5d",    # NEW: gold-BTC divergence (replaces gld_ret_1d)
    ],
    "BTCUSDT": [
        # 2026-04-13: 11 forced — DO NOT ADD MORE (overfit proven twice).
        # gbtc_premium_dev has residual alpha (IC=-0.056) but adding it
        # degraded OOS backtest (9.62→5.45). The greedy selector picks
        # it naturally when it helps; forcing it hurts model stability.
        "vix_chg_1d",               # 0.185 avg |IC| (contrarian)
        "gld_ret_5d",               # safe-haven
        "spy_ret_1d",               # 0.221 avg |IC| (risk-off)
        "qqq_ret_1d",               # 0.216 avg |IC| (tech risk-off)
        "iwm_ret_1d",               # 0.218 avg |IC| (small-cap)
        "trend_x_vol",              # momentum × vol interaction
        "coin_ret_1d",              # 0.301 avg |IC| (Coinbase sentiment)
        "rsi_x_atr",                # overbought × range
        "fxi_ret_1d",               # 0.109 avg |IC| (China equity proxy)
        "hyg_ret_1d",               # 0.164 avg |IC| (HY credit)
        "gld_ret_1d",               # 1d gold return
    ],
}

# Per-symbol feature BLACKLIST — greedy IC selection + forced features
# both filter these out when the list is non-empty.
#
# 2026-04-12 learning: populating this from scripts/feature_ablation.py
# single-feature Δ IC measurements DID NOT improve models — the marginal
# (one-feature-removed) ablation effect doesn't compose when the greedy
# IC selector reshuffles the feature pool.  Tested on both BTC and ETH:
#
#   BTC: train_v12 Sharpe  2.68 -> 2.01  (dropped -25%)
#   ETH: train_v12 Sharpe  2.49 -> 1.67  (dropped -33%)
#   ETH: live-eq Sharpe    5.05 -> 2.26  (dropped -55%)
#
# Both were reverted.  The BLACKLIST infrastructure is KEPT (wired
# through train_v12.py and pipeline.py) in case a genuinely broken
# feature surfaces (eg. data pipeline bug producing garbage values),
# but do not populate it from ablation results without multi-feature
# joint ablation validation.
#
# Future improvement: implement proper LOO (leave-one-out) ablation
# that RETRAINS for each candidate removal and measures final OOS
# Sharpe rather than IC delta.  Single-pass Ridge-only ablation is
# too crude for the ensemble.
BLACKLIST_FEATURES: dict[str, list[str]] = {
    # Empty — see comment above.  Populate manually if a feature is
    # KNOWN to be broken / contaminated.
}
MODEL_DIR_TEMPLATE = "models_v8/{symbol}_gate_v2"
MODEL_DIR_OVERRIDES: dict[str, str] = {}

# Per-symbol training window cutoff (regime-focused training).
# Older crypto data (pre-2023) is often too different from current regime.
# 0 = use all data (default). N = last N years from train_end.
# 2026-04-11: ETH set to 3.0 after discovering h12 60d IC ~0 (overfitting on
# long 2019-2022 tail that has different correlations vs 2023-2026 regime).
MAX_TRAIN_YEARS: dict[str, float] = {
    "ETHUSDT": 1.5,
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
