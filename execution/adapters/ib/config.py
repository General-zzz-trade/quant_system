# execution/adapters/ib/config.py
"""IB adapter configuration."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True, slots=True)
class IBConfig:
    """Interactive Brokers connection and trading configuration.

    Attributes:
        host: IB Gateway/TWS host (localhost or remote).
        port: API port — 4001=live GW, 4002=paper GW, 7496=live TWS, 7497=paper TWS.
        client_id: Unique client ID for this connection (0-31).
        account: IB account ID (auto-detected if empty).
        timeout: Connection timeout in seconds.
        readonly: If True, no orders can be placed (data only).
        symbols: Default symbols to subscribe.
        max_positions: Maximum simultaneous positions.
        max_order_value: Maximum single order value (USD).
        max_daily_loss: Daily loss kill switch (USD).
    """
    host: str = "127.0.0.1"
    port: int = 4002  # paper trading gateway by default
    client_id: int = 1
    account: str = ""
    timeout: int = 30
    readonly: bool = False
    symbols: tuple[str, ...] = ()
    max_positions: int = 20
    max_order_value: float = 50_000.0
    max_daily_loss: float = 2_000.0

    @property
    def is_paper(self) -> bool:
        """Check if connected to paper trading."""
        return self.port in (4002, 7497)

    @classmethod
    def paper(cls, **kwargs: Any) -> "IBConfig":
        """Factory for paper trading (port 4002)."""
        return cls(port=4002, **kwargs)

    @classmethod
    def live(cls, **kwargs: Any) -> "IBConfig":
        """Factory for live trading (port 4001)."""
        return cls(port=4001, readonly=False, **kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "IBConfig":
        """Load config from YAML file."""
        import yaml

        raw: Dict[str, Any] = {}
        p = Path(path)
        if p.exists():
            with open(p) as f:
                raw = yaml.safe_load(f) or {}
        valid_keys = {fld.name for fld in fields(cls)}
        filtered = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in raw.items()
            if k in valid_keys
        }
        return cls(**filtered)
