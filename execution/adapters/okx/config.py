"""OKX adapter configuration (from environment variables)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from execution.adapters.okx.urls import OKX_REST_BASE


@dataclass(frozen=True, slots=True)
class OkxConfig:
    api_key: str
    api_secret: str
    passphrase: str
    base_url: str = OKX_REST_BASE
    simulated: bool = False  # x-simulated-trading flag
    recv_window_ms: int = 5000
    timeout_s: float = 10.0

    # Safety caps specific to OKX first-24h rollout
    # MUST be overridden by the caller with realistic values once validated.
    # Default is intentionally tiny to prevent accidental large orders.
    max_order_notional_usd: float = 50.0      # per-order cap
    max_daily_notional_usd: float = 500.0     # daily cumulative cap
    max_rate_per_sec: float = 2.0             # order rate limit

    @classmethod
    def from_env(cls) -> "OkxConfig":
        key = os.environ.get("OKX_API_KEY", "").strip()
        secret = os.environ.get("OKX_API_SECRET", "").strip()
        passphrase = os.environ.get("OKX_API_PASSPHRASE", "").strip()
        if not (key and secret and passphrase):
            raise RuntimeError(
                "OKX_API_KEY, OKX_API_SECRET and OKX_API_PASSPHRASE "
                "environment variables are required."
            )
        base_url = os.environ.get("OKX_BASE_URL", OKX_REST_BASE).strip()
        simulated = os.environ.get("OKX_SIMULATED", "0").strip() in ("1", "true", "True")

        def _fenv(name: str, default: float) -> float:
            v = os.environ.get(name, "").strip()
            try:
                return float(v) if v else default
            except ValueError:
                return default

        return cls(
            api_key=key,
            api_secret=secret,
            passphrase=passphrase,
            base_url=base_url,
            simulated=simulated,
            max_order_notional_usd=_fenv("OKX_MAX_ORDER_NOTIONAL", 50.0),
            max_daily_notional_usd=_fenv("OKX_MAX_DAILY_NOTIONAL", 500.0),
            max_rate_per_sec=_fenv("OKX_MAX_RATE_PER_SEC", 2.0),
        )

    def __repr__(self) -> str:
        return (
            f"OkxConfig(base_url={self.base_url!r}, api_key='***', "
            f"api_secret='***', passphrase='***', simulated={self.simulated}, "
            f"max_order=${self.max_order_notional_usd}, "
            f"max_daily=${self.max_daily_notional_usd})"
        )
