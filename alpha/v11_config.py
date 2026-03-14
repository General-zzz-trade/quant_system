"""V11 Alpha Configuration — single source of truth for all alpha parameters.

Loads from model config.json with backward-compatible defaults matching v10 behavior.
All modules (backtest, exit, regime, ensemble) read from this config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExitConfig:
    """Exit strategy parameters."""
    trailing_stop_pct: float = 0.0       # 0 = disabled
    zscore_cap: float = 0.0              # 0 = disabled (entry gate: skip if |z| > cap)
    reversal_threshold: float = -0.3     # position * z < this → exit
    deadzone_fade: float = 0.2           # |z| < this → exit (signal too weak)
    time_filter: TimeFilterConfig = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.time_filter is None:
            self.time_filter = TimeFilterConfig()


@dataclass
class RegimeGateConfig:
    """Regime-based position scaling."""
    enabled: bool = False
    ranging_high_vol_action: str = "reduce"  # "skip" | "reduce"
    reduce_factor: float = 0.3
    bb_width_high_pct: float = 75.0      # percentile threshold for "high" bb_width
    vol_of_vol_high_pct: float = 75.0    # percentile threshold for "high" vol_of_vol
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0

    def __post_init__(self):
        if self.ranging_high_vol_action not in ("skip", "reduce"):
            raise ValueError(
                f"ranging_high_vol_action must be 'skip' or 'reduce', "
                f"got '{self.ranging_high_vol_action}'"
            )
        if not 0.0 <= self.reduce_factor <= 1.0:
            raise ValueError(
                f"reduce_factor must be in [0, 1], got {self.reduce_factor}"
            )


@dataclass
class TimeFilterConfig:
    """Time-based entry filter."""
    enabled: bool = False
    skip_hours_utc: List[int] = field(default_factory=list)

    def __post_init__(self):
        for h in self.skip_hours_utc:
            if not 0 <= h <= 23:
                raise ValueError(f"skip_hours_utc values must be 0-23, got {h}")


@dataclass
class V11Config:
    """Unified alpha configuration.

    Backward-compatible: v10 config.json loads with all defaults matching
    current v10 behavior (mean_zscore, no exit/regime/time features).
    """
    # Core
    horizons: List[int] = field(default_factory=lambda: [12, 24, 48])
    ensemble_method: str = "mean_zscore"    # "mean_zscore" | "ic_weighted"
    lgbm_xgb_weight: float = 0.5           # weight for lgbm (xgb gets 1 - this)
    zscore_window: int = 720
    zscore_warmup: int = 180                # SINGLE source of truth

    # Entry
    deadzone: float = 0.3
    min_hold: int = 12
    max_hold: int = 96
    long_only: bool = False

    # New modules
    exit: ExitConfig = field(default_factory=ExitConfig)
    regime_gate: RegimeGateConfig = field(default_factory=RegimeGateConfig)
    time_filter: TimeFilterConfig = field(default_factory=TimeFilterConfig)

    # IC-weighted ensemble params
    ic_ema_span: int = 720                  # EMA span for rolling IC
    ic_min_threshold: float = -0.01         # horizons below this get weight=0

    def __post_init__(self):
        if self.ensemble_method not in ("mean_zscore", "ic_weighted"):
            raise ValueError(
                f"ensemble_method must be 'mean_zscore' or 'ic_weighted', "
                f"got '{self.ensemble_method}'"
            )
        if not 0.0 <= self.lgbm_xgb_weight <= 1.0:
            raise ValueError(
                f"lgbm_xgb_weight must be in [0, 1], got {self.lgbm_xgb_weight}"
            )
        if self.zscore_warmup < 1:
            raise ValueError(f"zscore_warmup must be >= 1, got {self.zscore_warmup}")
        if self.zscore_window < self.zscore_warmup:
            raise ValueError(
                f"zscore_window ({self.zscore_window}) must be >= "
                f"zscore_warmup ({self.zscore_warmup})"
            )
        if not self.horizons:
            raise ValueError("horizons must not be empty")
        # Sync exit.time_filter with top-level time_filter
        self.exit.time_filter = self.time_filter

    @classmethod
    def from_config_json(cls, cfg: Dict[str, Any]) -> V11Config:
        """Load from model config.json, filling v10 defaults for missing fields.

        Accepts both v10 and v11 format config dicts.
        """
        # Parse nested configs
        exit_raw = cfg.get("exit", {})
        time_filter_raw = exit_raw.pop("time_filter", None) or cfg.get("time_filter", {})
        regime_raw = cfg.get("regime_gate", {})

        time_filter = TimeFilterConfig(
            enabled=time_filter_raw.get("enabled", False),
            skip_hours_utc=time_filter_raw.get("skip_hours_utc", []),
        )

        exit_cfg = ExitConfig(
            trailing_stop_pct=exit_raw.get("trailing_stop_pct", 0.0),
            zscore_cap=exit_raw.get("zscore_cap", 0.0),
            reversal_threshold=exit_raw.get("reversal_threshold", -0.3),
            deadzone_fade=exit_raw.get("deadzone_fade", 0.2),
            time_filter=time_filter,
        )

        regime_cfg = RegimeGateConfig(
            enabled=regime_raw.get("enabled", False),
            ranging_high_vol_action=regime_raw.get("ranging_high_vol_action", "reduce"),
            reduce_factor=regime_raw.get("reduce_factor", 0.3),
            bb_width_high_pct=regime_raw.get("bb_width_high_pct", 75.0),
            vol_of_vol_high_pct=regime_raw.get("vol_of_vol_high_pct", 75.0),
            adx_trend_threshold=regime_raw.get("adx_trend_threshold", 25.0),
            adx_range_threshold=regime_raw.get("adx_range_threshold", 20.0),
        )

        return cls(
            horizons=cfg.get("horizons", [12, 24, 48]),
            ensemble_method=cfg.get("ensemble_method", "mean_zscore"),
            lgbm_xgb_weight=cfg.get("lgbm_xgb_weight", 0.5),
            zscore_window=cfg.get("zscore_window", 720),
            zscore_warmup=cfg.get("zscore_warmup", 180),
            deadzone=cfg.get("deadzone", 0.3),
            min_hold=cfg.get("min_hold", 12),
            max_hold=cfg.get("max_hold", 96),
            long_only=cfg.get("long_only", False),
            exit=exit_cfg,
            regime_gate=regime_cfg,
            time_filter=time_filter,
            ic_ema_span=cfg.get("ic_ema_span", 720),
            ic_min_threshold=cfg.get("ic_min_threshold", -0.01),
        )
