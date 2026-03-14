"""Per-symbol strategy configuration — single source of truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SymbolStrategyConfig:
    fixed_features: List[str]
    candidate_pool: List[str]
    n_flexible: int = 4
    deadzone: float = 0.5
    min_hold: int = 24
    monthly_gate_window: int = 480
    long_only: bool = True
    ensemble: bool = True
    vol_target: Optional[float] = None
    vol_feature: str = "atr_norm_14"
    dd_limit: Optional[float] = None
    dd_cooldown: int = 48
    model_dir: str = ""


SYMBOL_CONFIG = {
    "BTCUSDT": SymbolStrategyConfig(
        fixed_features=[
            "basis", "ret_24", "fgi_normalized", "fgi_extreme", "parkinson_vol",
            "atr_norm_14", "rsi_14", "tf4h_atr_norm_14", "basis_zscore_24", "cvd_20",
        ],
        candidate_pool=[
            "funding_zscore_24", "basis_momentum", "vol_ma_ratio_5_20",
            "mean_reversion_20", "funding_sign_persist", "hour_sin",
            "liquidation_cascade_score",
            "implied_vol_zscore_24", "iv_rv_spread",
            "exchange_supply_zscore_30",
            "mempool_size_zscore_24",
            "spx_overnight_ret",
            "mempool_fee_zscore_24",
            "liquidation_volume_ratio",
        ],
        n_flexible=4,
        deadzone=0.5,
        min_hold=24,
        monthly_gate_window=480,
        model_dir="models_v8/BTCUSDT_gate_v2",
    ),
    "SOLUSDT": SymbolStrategyConfig(
        fixed_features=[
            "basis", "ret_24", "fgi_normalized", "fgi_extreme", "parkinson_vol",
            "atr_norm_14", "rsi_14", "tf4h_atr_norm_14", "basis_zscore_24", "cvd_20",
            "btc_ret_24", "btc_rsi_14", "btc_mean_reversion_20",
        ],
        candidate_pool=[
            "funding_zscore_24", "basis_momentum", "vol_ma_ratio_5_20",
            "mean_reversion_20", "funding_sign_persist", "hour_sin",
            "btc_ret_12", "btc_macd_line", "btc_atr_norm_14", "btc_bb_width_20",
            "mempool_fee_zscore_24", "mempool_size_zscore_24",
        ],
        n_flexible=5,
        deadzone=1.0,
        min_hold=48,
        monthly_gate_window=480,
        model_dir="models_v8/SOLUSDT_gate_v3",
    ),
    "ETHUSDT": SymbolStrategyConfig(
        fixed_features=[
            "ret_24", "atr_norm_14", "parkinson_vol", "bb_width_20", "vol_20",
            "rsi_14", "tf4h_atr_norm_14", "tf4h_ret_6", "basis_zscore_24",
            "basis_momentum", "btc_ret_24", "btc_rsi_14", "btc_mean_reversion_20",
        ],
        candidate_pool=[
            "mean_reversion_20", "rsi_6", "ma_cross_10_30", "close_vs_ma50",
            "vol_of_vol", "fgi_normalized", "funding_ma8", "vwap_dev_20",
            "btc_atr_norm_14", "btc_bb_width_20",
            "mempool_fee_zscore_24", "mempool_size_zscore_24",
            "active_addr_zscore_14", "tx_count_zscore_14",
            "hashrate_momentum",
        ],
        n_flexible=4,
        deadzone=0.5,
        min_hold=24,
        monthly_gate_window=480,
        model_dir="models_v8/ETHUSDT_gate_v2",
    ),
}


def get_config(symbol: str) -> SymbolStrategyConfig:
    """Get strategy config for a symbol. Raises KeyError if not found."""
    return SYMBOL_CONFIG[symbol]
