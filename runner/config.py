# runner/config.py
"""LiveRunner configuration and operator status types."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True, slots=True)
class LiveRunnerConfig:
    symbols: tuple[str, ...] = ("BTCUSDT",)
    currency: str = "USDT"
    ws_base_url: str = "wss://fstream.binance.com/stream"
    kline_interval: str = "1m"
    enable_regime_gate: bool = True
    enable_monitoring: bool = True
    enable_reconcile: bool = True
    reconcile_interval_sec: float = 60.0
    health_stale_data_sec: float = 120.0
    venue: str = "binance"
    # Margin monitoring
    margin_check_interval_sec: float = 30.0
    margin_warning_ratio: float = 0.15
    margin_critical_ratio: float = 0.08
    # Shutdown
    pending_order_timeout_sec: float = 30.0
    # Production infrastructure
    data_dir: str = "data/live"
    enable_persistent_stores: bool = True
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    # Pre-flight checks
    enable_preflight: bool = True
    preflight_min_balance: float = 0.0
    # Startup reconciliation
    reconcile_on_startup: bool = True
    # Shadow mode — simulate orders without executing
    shadow_mode: bool = False
    # Latency SLA
    latency_p99_threshold_ms: float = 5000.0
    # Correlation risk
    max_avg_correlation: float = 0.7
    # Health HTTP endpoint
    health_port: Optional[int] = None
    health_host: str = "127.0.0.1"
    health_auth_token_env: Optional[str] = None
    # Testnet mode
    testnet: bool = False
    # ModelRegistry auto-loading
    model_registry_db: Optional[str] = None
    artifact_store_root: Optional[str] = None
    model_names: tuple[str, ...] = ()
    # Portfolio risk aggregator
    enable_portfolio_risk: bool = True
    max_gross_leverage: float = 3.0
    max_net_leverage: float = 1.0
    max_concentration: float = 0.4
    # Data scheduler + freshness monitor
    enable_data_scheduler: bool = False
    data_files_dir: str = "data_files"
    # Signal constraints (must match backtest)
    min_hold_bars: Optional[Dict[str, int]] = None
    long_only_symbols: Optional[set] = None
    deadzone: Union[float, Dict[str, float]] = 0.5
    # Trend hold
    trend_follow: bool = False
    trend_indicator: str = "tf4h_close_vs_ma20"
    trend_threshold: float = 0.0
    max_hold: int = 120
    # Monthly gate
    monthly_gate: bool = False
    monthly_gate_window: Union[int, Dict[str, int]] = 480
    # Bear regime thresholds for Strategy F
    bear_thresholds: Optional[list] = None
    # Vol-adaptive sizing (scalar or per-symbol dict)
    vol_target: Union[None, float, Dict[str, Optional[float]]] = None
    vol_feature: Union[str, Dict[str, str]] = "atr_norm_14"
    # Ensemble weights for model averaging (matches config.json ensemble_weights)
    ensemble_weights: Optional[List[float]] = None
    # Alpha health monitoring (IC tracking + position scaling)
    enable_alpha_health: bool = True
    alpha_health_horizons: tuple[int, ...] = (12, 24)
    # WS-API order gateway (4ms vs 30ms REST)
    use_ws_orders: bool = False
    # Adaptive config for BTC (select_robust() every 24h)
    adaptive_btc_enabled: bool = False
    adaptive_btc_interval_hours: int = 24
    # Multi-timeframe ensemble (Direction 13)
    enable_multi_tf_ensemble: bool = False
    multi_tf_models: Optional[Dict[str, list]] = None  # {"BTCUSDT": ["gate_v2", "15m"]}
    ensemble_conflict_policy: str = "flat"  # "flat" = go flat on conflict, "average" = ignore
    # Drawdown circuit breaker (continuous equity monitoring)
    dd_warning_pct: float = 10.0
    dd_reduce_pct: float = 15.0
    dd_kill_pct: float = 20.0
    # Regime-aware position sizing (Direction 17)
    enable_regime_sizing: bool = False
    regime_low_vol_scale: float = 1.0
    regime_mid_vol_scale: float = 0.6
    regime_high_vol_scale: float = 0.25
    # Burn-in gate (Direction 14)
    enable_burnin_gate: bool = False
    burnin_report_path: str = "data/live/burnin_report.json"


@dataclass(frozen=True, slots=True)
class OperatorControlRecord:
    command: str
    reason: str
    source: str
    ts: datetime
    result: str


@dataclass(frozen=True, slots=True)
class OperatorKillSwitchStatus:
    scope: str
    key: str
    mode: str
    reason: str
    source: str


@dataclass(frozen=True, slots=True)
class OperatorReconcileStatus:
    ok: bool
    should_halt: bool
    drift_count: int


@dataclass(frozen=True, slots=True)
class OperatorStatusSnapshot:
    running: bool
    stopped: bool
    stream_status: str
    incident_state: str
    last_incident_category: Optional[str]
    last_incident_ts: Optional[datetime]
    recommended_action: str
    kill_switch: Optional[OperatorKillSwitchStatus]
    last_reconcile: Optional[OperatorReconcileStatus]
    last_control: Optional[OperatorControlRecord]
