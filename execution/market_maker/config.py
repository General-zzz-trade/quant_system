"""Market maker configuration."""

from __future__ import annotations

import dataclasses as dc


@dc.dataclass
class MarketMakerConfig:
    """All tunables for the perpetual market maker."""

    # ── Symbol ───────────────────────────────────────────────
    symbol: str = "ETHUSDT"
    tick_size: float = 0.01
    qty_step: float = 0.001
    min_notional: float = 20.0

    # ── Sizing ───────────────────────────────────────────────
    order_size_eth: float = 0.01          # ~$20 per side
    max_inventory_notional: float = 50.0  # $50 max net exposure
    capital: float = 100.0                # starting capital

    # ── A-S model ────────────────────────────────────────────
    gamma: float = 0.3                    # risk aversion
    kappa: float = 1.5                    # order arrival intensity
    time_horizon_s: float = 300.0         # 5 min rolling reset
    funding_bias_mult: float = 1.0        # funding rate bias multiplier

    # ── Spread bounds ────────────────────────────────────────
    min_spread_bps: float = 2.0           # minimum spread in bps
    max_spread_bps: float = 30.0          # maximum spread in bps

    # ── VPIN defense ─────────────────────────────────────────
    vpin_threshold: float = 0.7
    vpin_spread_mult: float = 1.5

    # ── Vol estimator ────────────────────────────────────────
    vol_trade_window: int = 200           # EMA window for vol
    vol_ema_alpha: float = 0.01           # 2/(200+1) ≈ 0.01

    # ── Order management ─────────────────────────────────────
    stale_order_s: float = 2.0            # cancel orders older than this
    stale_tick_distance: int = 2          # cancel if > N ticks from BBO
    quote_update_interval_s: float = 0.1  # 100ms quote refresh

    # ── Risk ─────────────────────────────────────────────────
    daily_loss_limit: float = 10.0        # $10 daily loss → kill
    circuit_breaker_losses: int = 3       # consecutive losses → pause
    circuit_breaker_pause_s: float = 120.0

    # ── Infrastructure ───────────────────────────────────────
    testnet: bool = False
    dry_run: bool = False                 # log quotes but don't submit
    log_file: str = "logs/market_maker.log"
    market_data_stale_s: float = 15.0    # fail fast if WS/depth stops moving

    # ── Microstructure ───────────────────────────────────────
    vpin_bucket_volume: float = 100.0
    vpin_n_buckets: int = 50
    trade_buffer_size: int = 200

    def max_inventory_qty(self, ref_price: float) -> float:
        """Max inventory in base asset units at current price."""
        if ref_price <= 0:
            return 0.0
        return self.max_inventory_notional / ref_price

    def validate(self) -> None:
        """Raise ValueError on invalid config."""
        if self.order_size_eth <= 0:
            raise ValueError("order_size_eth must be positive")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.daily_loss_limit <= 0:
            raise ValueError("daily_loss_limit must be positive")
        if self.min_spread_bps >= self.max_spread_bps:
            raise ValueError("min_spread_bps must be < max_spread_bps")
        if self.market_data_stale_s <= 0:
            raise ValueError("market_data_stale_s must be positive")
