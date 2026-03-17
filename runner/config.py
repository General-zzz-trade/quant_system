# runner/config.py
"""LiveRunner configuration and operator status types."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union


@dataclass(frozen=True, slots=True)
class LiveRunnerConfig:
    # --- Production Core ---
    symbols: tuple[str, ...] = ("BTCUSDT",)
    currency: str = "USDT"
    ws_base_url: str = "wss://fstream.binance.com/stream"
    kline_interval: str = "1m"
    venue: str = "binance"
    testnet: bool = False
    shadow_mode: bool = False

    # --- Risk / Control ---
    enable_regime_gate: bool = True
    enable_portfolio_risk: bool = True
    max_gross_leverage: float = 3.0
    max_net_leverage: float = 1.0
    max_concentration: float = 0.4
    max_avg_correlation: float = 0.7
    dd_warning_pct: float = 10.0
    dd_reduce_pct: float = 15.0
    dd_kill_pct: float = 20.0
    latency_p99_threshold_ms: float = 5000.0
    margin_check_interval_sec: float = 30.0
    margin_warning_ratio: float = 0.15
    margin_critical_ratio: float = 0.08
    pending_order_timeout_sec: float = 30.0
    enable_preflight: bool = True
    preflight_min_balance: float = 0.0
    enable_reconcile: bool = True
    reconcile_interval_sec: float = 60.0
    reconcile_on_startup: bool = True

    # --- Signal / Strategy Constraints ---
    deadzone: Union[float, Dict[str, float]] = 0.5
    min_hold_bars: Optional[Dict[str, int]] = None
    long_only_symbols: Optional[set] = None
    trend_follow: bool = False
    trend_indicator: str = "tf4h_close_vs_ma20"
    trend_threshold: float = 0.0
    max_hold: int = 120
    monthly_gate: bool = False
    monthly_gate_window: Union[int, Dict[str, int]] = 480
    bear_thresholds: Optional[list] = None
    vol_target: Union[None, float, Dict[str, Optional[float]]] = None
    vol_feature: Union[str, Dict[str, str]] = "atr_norm_14"
    ensemble_weights: Optional[List[float]] = None

    # --- Monitoring / Health ---
    enable_monitoring: bool = True
    health_port: Optional[int] = None
    health_host: str = "127.0.0.1"
    health_auth_token_env: Optional[str] = None
    health_stale_data_sec: float = 120.0
    enable_alpha_health: bool = True
    alpha_health_horizons: tuple[int, ...] = (12, 24)

    # --- Data / Storage ---
    data_dir: str = "data/live"
    enable_persistent_stores: bool = True
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    data_files_dir: str = "data_files"
    enable_data_scheduler: bool = False
    model_registry_db: Optional[str] = None
    artifact_store_root: Optional[str] = None
    model_names: tuple[str, ...] = ()
    enable_decision_recording: bool = False
    decision_recording_path: str = "data/live/decisions.jsonl"

    # --- Execution ---
    use_ws_orders: bool = False

    # --- Rust hot-path ---
    enable_rust_components: bool = False

    # --- Experimental ---
    adaptive_btc_enabled: bool = False
    adaptive_btc_interval_hours: int = 24
    enable_multi_tf_ensemble: bool = False
    multi_tf_models: Optional[Dict[str, list]] = None  # {"BTCUSDT": ["gate_v2", "15m"]}
    ensemble_conflict_policy: str = "flat"  # "flat" = go flat on conflict, "average" = ignore
    enable_regime_sizing: bool = False
    regime_low_vol_scale: float = 1.0
    regime_mid_vol_scale: float = 0.6
    regime_high_vol_scale: float = 0.25
    initial_equity: float = 500.0
    enable_burnin_gate: bool = False
    burnin_report_path: str = "data/live/burnin_report.json"

    # -- Factory classmethods --------------------------------------------------

    @classmethod
    def lite(cls, *, symbols: list[str], **overrides) -> "LiveRunnerConfig":
        """Minimal config for local testing. Disables all optional subsystems."""
        defaults = dict(
            symbols=tuple(symbols),
            testnet=True,
            enable_monitoring=False,
            enable_persistent_stores=False,
            enable_reconcile=False,
            enable_alpha_health=False,
            enable_regime_sizing=False,
            adaptive_btc_enabled=False,
            use_ws_orders=False,
            enable_multi_tf_ensemble=False,
            enable_burnin_gate=False,
            enable_decision_recording=False,
            enable_structured_logging=False,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def paper(cls, *, symbols: list[str], **overrides) -> "LiveRunnerConfig":
        """Paper trading config. Enables monitoring but no real execution."""
        defaults = dict(
            symbols=tuple(symbols),
            testnet=True,
            enable_monitoring=True,
            enable_persistent_stores=True,
            enable_reconcile=False,
            enable_alpha_health=True,
            shadow_mode=True,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def testnet_full(cls, *, symbols: list[str], **overrides) -> "LiveRunnerConfig":
        """Testnet config. Full stack on testnet exchange."""
        defaults = dict(
            symbols=tuple(symbols),
            testnet=True,
            enable_monitoring=True,
            enable_persistent_stores=True,
            enable_reconcile=True,
            enable_alpha_health=True,
            use_ws_orders=True,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def prod(cls, *, symbols: list[str], **overrides) -> "LiveRunnerConfig":
        """Production config. All safety systems enabled."""
        defaults = dict(
            symbols=tuple(symbols),
            testnet=False,
            enable_monitoring=True,
            enable_persistent_stores=True,
            enable_reconcile=True,
            enable_alpha_health=True,
            enable_structured_logging=True,
            use_ws_orders=True,
        )
        defaults.update(overrides)
        return cls(**defaults)


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
