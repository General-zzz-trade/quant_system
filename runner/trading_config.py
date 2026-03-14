"""Simplified trading config — ~50 fields (down from 102 in LiveRunnerConfig)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass(frozen=True, slots=True)
class TradingConfig:
    """Lean config for the decomposed runner. No experimental flags."""

    # Identity
    symbols: tuple[str, ...]
    venue: str = "binance"
    testnet: bool = True
    shadow_mode: bool = True
    currency: str = "USDT"

    # Market data
    ws_base_url: str = "wss://fstream.binance.com/stream"
    kline_interval: str = "1m"

    # Feature / ML
    model_dir: str = "models_v8"

    # Signal / strategy
    deadzone: Union[float, Dict[str, float]] = 0.5
    min_hold_bars: Optional[Dict[str, int]] = None
    long_only_symbols: Optional[set] = None
    trend_follow: bool = False
    trend_indicator: str = "tf4h_close_vs_ma20"
    trend_threshold: float = 0.0
    max_hold: int = 120
    monthly_gate: bool = False
    monthly_gate_window: Union[int, Dict[str, int]] = 480
    vol_target: Union[None, float, Dict[str, Optional[float]]] = None
    vol_feature: Union[str, Dict[str, str]] = "atr_norm_14"

    # Risk / control
    max_gross_leverage: float = 3.0
    max_net_leverage: float = 1.0
    max_concentration: float = 0.4
    dd_warning_pct: float = 10.0
    dd_reduce_pct: float = 15.0
    dd_kill_pct: float = 20.0
    pending_order_timeout_sec: float = 30.0
    margin_check_interval_sec: float = 30.0
    margin_warning_ratio: float = 0.15
    margin_critical_ratio: float = 0.08

    # Execution
    use_ws_orders: bool = False
    enable_preflight: bool = True
    preflight_min_balance: float = 0.0

    # Recovery / persistence
    data_dir: str = "data/live"
    enable_persistent_stores: bool = True
    enable_reconcile: bool = True
    reconcile_interval_sec: float = 60.0
    reconcile_on_startup: bool = True
    checkpoint_interval_sec: float = 300.0

    # Monitoring
    enable_monitoring: bool = True
    health_port: Optional[int] = None
    health_host: str = "127.0.0.1"
    enable_alpha_health: bool = True
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Model registry (optional)
    model_registry_db: Optional[str] = None
    model_names: tuple[str, ...] = ()

    @classmethod
    def paper(cls, *, symbols: list[str], **kw) -> "TradingConfig":
        return cls(symbols=tuple(symbols), testnet=True, shadow_mode=True,
                   enable_reconcile=False, **kw)

    @classmethod
    def testnet_full(cls, *, symbols: list[str], **kw) -> "TradingConfig":
        return cls(symbols=tuple(symbols), testnet=True, shadow_mode=False,
                   enable_reconcile=True, use_ws_orders=True, **kw)

    @classmethod
    def prod(cls, *, symbols: list[str], **kw) -> "TradingConfig":
        return cls(symbols=tuple(symbols), testnet=False, shadow_mode=False,
                   enable_reconcile=True, use_ws_orders=True, **kw)
