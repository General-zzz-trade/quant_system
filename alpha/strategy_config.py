"""Per-symbol strategy configuration — single source of truth."""
from __future__ import annotations

from dataclasses import dataclass, field
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
        ],
        candidate_pool=[
            "funding_zscore_24", "basis_momentum", "vol_ma_ratio_5_20",
            "mean_reversion_20", "funding_sign_persist", "hour_sin",
        ],
        n_flexible=4,
        deadzone=0.5,
        min_hold=24,
        monthly_gate_window=480,
        model_dir="models_v8/SOLUSDT_gate_v2",
    ),
}


def get_config(symbol: str) -> SymbolStrategyConfig:
    """Get strategy config for a symbol. Raises KeyError if not found."""
    return SYMBOL_CONFIG[symbol]
