"""PolymarketConfig — configuration for prediction market trading."""
from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True, slots=True)
class PolymarketConfig:
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://clob.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    scan_interval_sec: int = 3600
    min_liquidity_usd: float = 10_000
    min_hours_to_expiry: float = 24.0
    crypto_keywords: tuple[str, ...] = ("BTC", "Bitcoin", "ETH", "Ethereum", "crypto", "Solana", "SOL")
    max_position_pct: float = 0.10
    max_total_pct: float = 0.30
    stop_loss_pct: float = 0.15
    signal_threshold: float = 0.15
    min_probability: float = 0.15
    max_probability: float = 0.85
    kelly_fraction: float = 0.5
    dd_warning_pct: float = 10.0
    dd_reduce_pct: float = 15.0
    dd_kill_pct: float = 20.0
    data_dir: str = "data/polymarket"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str) -> "PolymarketConfig":
        """Load config from YAML file, falling back to defaults for missing keys."""
        import yaml
        raw: Dict[str, Any] = {}
        p = Path(path)
        if p.exists():
            with open(p) as f:
                raw = yaml.safe_load(f) or {}
        valid_keys = {fld.name for fld in fields(cls)}
        filtered = {k: tuple(v) if isinstance(v, list) else v
                    for k, v in raw.items() if k in valid_keys}
        return cls(**filtered)
