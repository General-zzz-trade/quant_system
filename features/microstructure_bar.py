# features/microstructure_bar.py
"""Bar-level microstructure features from tick data (Tier 1a).

Aggregates aggTrade + L2 depth ticks into bar-level features for
injection into the alpha model. Captures information NOT available
in standard OHLCV klines:
  - VPIN (volume-synchronised probability of informed trading)
  - Order book imbalance and depth ratio
  - Trade flow toxicity
  - Aggressive trade clustering
  - Spread dynamics

These features are computed incrementally (O(1) per tick) and
snapshotted at bar boundaries for model input.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class MicrostructureBarFeatures:
    """Tick-derived features aggregated to bar level."""
    # VPIN (0-1, higher = more informed flow)
    vpin: float = 0.0
    # Order book imbalance (-1 to +1, positive = bid-heavy)
    ob_imbalance: float = 0.0
    # Depth ratio (bid_depth / ask_depth, >1 = bid-heavy)
    depth_ratio: float = 1.0
    # Spread in bps
    spread_bps: float = 0.0
    # Trade flow metrics
    buy_volume_pct: float = 0.5      # fraction of volume from buys
    trade_intensity: float = 0.0      # trades per second in bar
    avg_trade_size: float = 0.0       # mean trade size (base asset)
    large_trade_pct: float = 0.0      # fraction of volume from large trades
    # Clustering
    buy_run_max: int = 0              # longest consecutive buy streak
    sell_run_max: int = 0             # longest consecutive sell streak
    # Volatility microstructure
    tick_volatility: float = 0.0      # vol from trade-to-trade returns
    # Weighted mid movement
    weighted_mid_ret: float = 0.0     # weighted mid return over bar


class MicrostructureBarComputer:
    """Incrementally compute microstructure features from ticks.

    Usage:
        computer = MicrostructureBarComputer()

        # Feed ticks as they arrive
        computer.on_trade(price=2320.0, qty=0.5, side="buy")
        computer.on_depth(best_bid=2319.99, best_ask=2320.01,
                          bid_depth_5=300000, ask_depth_5=250000)

        # At bar boundary, snapshot and reset
        features = computer.snapshot_and_reset()
        # features.vpin, features.ob_imbalance, etc.
    """

    def __init__(
        self,
        large_trade_threshold: float = 1.0,  # trades > 1.0 ETH = "large"
        vpin_bucket_volume: float = 50.0,
        vpin_n_buckets: int = 50,
    ) -> None:
        self._large_threshold = large_trade_threshold
        self._vpin_bucket_vol = vpin_bucket_volume
        self._vpin_n_buckets = vpin_n_buckets
        self._reset_state()

    def _reset_state(self) -> None:
        # Trade accumulators
        self._buy_volume = 0.0
        self._sell_volume = 0.0
        self._total_volume = 0.0
        self._trade_count = 0
        self._large_volume = 0.0
        self._trade_sizes: list[float] = []
        self._trade_prices: list[float] = []
        self._trade_sides: list[str] = []
        self._bar_start_ts = 0.0

        # Run tracking
        self._current_run_side = ""
        self._current_run_len = 0
        self._buy_run_max = 0
        self._sell_run_max = 0

        # Depth accumulators (use latest snapshot)
        self._last_ob_imbalance = 0.0
        self._last_depth_ratio = 1.0
        self._last_spread_bps = 0.0
        self._ob_samples = 0
        self._ob_imbalance_sum = 0.0
        self._depth_ratio_sum = 0.0
        self._spread_bps_sum = 0.0
        self._first_weighted_mid = 0.0
        self._last_weighted_mid = 0.0

        # VPIN
        self._vpin_buy_bucket = 0.0
        self._vpin_sell_bucket = 0.0
        self._vpin_buckets: deque[tuple[float, float]] = deque(maxlen=self._vpin_n_buckets)

    def on_trade(self, price: float, qty: float, side: str, ts: float = 0.0) -> None:
        """Process a single aggTrade."""
        if price <= 0 or qty <= 0:
            return

        if self._trade_count == 0 and ts > 0:
            self._bar_start_ts = ts

        self._trade_count += 1
        self._total_volume += qty
        self._trade_sizes.append(qty)
        self._trade_prices.append(price)
        self._trade_sides.append(side)

        if side == "buy":
            self._buy_volume += qty
        else:
            self._sell_volume += qty

        if qty >= self._large_threshold:
            self._large_volume += qty

        # Run tracking
        if side == self._current_run_side:
            self._current_run_len += 1
        else:
            self._current_run_side = side
            self._current_run_len = 1
        if side == "buy":
            self._buy_run_max = max(self._buy_run_max, self._current_run_len)
        else:
            self._sell_run_max = max(self._sell_run_max, self._current_run_len)

        # VPIN bucket accumulation
        if side == "buy":
            self._vpin_buy_bucket += qty
        else:
            self._vpin_sell_bucket += qty

        bucket_total = self._vpin_buy_bucket + self._vpin_sell_bucket
        if bucket_total >= self._vpin_bucket_vol:
            self._vpin_buckets.append((self._vpin_buy_bucket, self._vpin_sell_bucket))
            self._vpin_buy_bucket = 0.0
            self._vpin_sell_bucket = 0.0

    def on_depth(
        self,
        best_bid: float,
        best_ask: float,
        bid_depth_5: float = 0.0,
        ask_depth_5: float = 0.0,
        weighted_mid: float = 0.0,
    ) -> None:
        """Process an orderbook depth snapshot."""
        if best_bid <= 0 or best_ask <= 0:
            return

        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 10000

        # Imbalance
        total_depth = bid_depth_5 + ask_depth_5
        if total_depth > 0:
            imbalance = (bid_depth_5 - ask_depth_5) / total_depth
        else:
            imbalance = 0.0

        depth_ratio = bid_depth_5 / ask_depth_5 if ask_depth_5 > 0 else 1.0

        self._ob_samples += 1
        self._ob_imbalance_sum += imbalance
        self._depth_ratio_sum += depth_ratio
        self._spread_bps_sum += spread_bps

        self._last_ob_imbalance = imbalance
        self._last_depth_ratio = depth_ratio
        self._last_spread_bps = spread_bps

        wm = weighted_mid if weighted_mid > 0 else mid
        if self._first_weighted_mid == 0.0:
            self._first_weighted_mid = wm
        self._last_weighted_mid = wm

    def snapshot_and_reset(self) -> MicrostructureBarFeatures:
        """Snapshot current bar features and reset for next bar."""
        feat = MicrostructureBarFeatures()

        # VPIN
        if len(self._vpin_buckets) >= 5:
            total_imbalance = sum(abs(b - s) for b, s in self._vpin_buckets)
            total_vol = sum(b + s for b, s in self._vpin_buckets)
            feat.vpin = total_imbalance / total_vol if total_vol > 0 else 0.0
        elif self._total_volume > 0:
            feat.vpin = abs(self._buy_volume - self._sell_volume) / self._total_volume

        # OB averages over bar
        if self._ob_samples > 0:
            feat.ob_imbalance = self._ob_imbalance_sum / self._ob_samples
            feat.depth_ratio = self._depth_ratio_sum / self._ob_samples
            feat.spread_bps = self._spread_bps_sum / self._ob_samples

        # Trade flow
        if self._total_volume > 0:
            feat.buy_volume_pct = self._buy_volume / self._total_volume
            feat.large_trade_pct = self._large_volume / self._total_volume

        if self._trade_count > 0:
            feat.avg_trade_size = self._total_volume / self._trade_count

        feat.trade_intensity = self._trade_count  # per bar (normalize externally)
        feat.buy_run_max = self._buy_run_max
        feat.sell_run_max = self._sell_run_max

        # Tick volatility
        if len(self._trade_prices) >= 2:
            log_rets = []
            for i in range(1, len(self._trade_prices)):
                if self._trade_prices[i - 1] > 0:
                    lr = math.log(self._trade_prices[i] / self._trade_prices[i - 1])
                    log_rets.append(lr)
            if log_rets:
                feat.tick_volatility = float(_std(log_rets))

        # Weighted mid return
        if self._first_weighted_mid > 0 and self._last_weighted_mid > 0:
            feat.weighted_mid_ret = (
                self._last_weighted_mid / self._first_weighted_mid - 1.0
            )

        self._reset_state()
        return feat

    def get_feature_dict(self, features: MicrostructureBarFeatures) -> dict[str, float]:
        """Convert to flat dict for model input."""
        return {
            "micro_vpin": features.vpin,
            "micro_ob_imbalance": features.ob_imbalance,
            "micro_depth_ratio": features.depth_ratio,
            "micro_spread_bps": features.spread_bps,
            "micro_buy_pct": features.buy_volume_pct,
            "micro_trade_intensity": features.trade_intensity,
            "micro_avg_trade_size": features.avg_trade_size,
            "micro_large_trade_pct": features.large_trade_pct,
            "micro_buy_run_max": float(features.buy_run_max),
            "micro_sell_run_max": float(features.sell_run_max),
            "micro_tick_vol": features.tick_volatility,
            "micro_wmid_ret": features.weighted_mid_ret,
        }


# Number of microstructure features
MICROSTRUCTURE_FEATURE_COUNT = 12

MICROSTRUCTURE_FEATURE_NAMES = (
    "micro_vpin",
    "micro_ob_imbalance",
    "micro_depth_ratio",
    "micro_spread_bps",
    "micro_buy_pct",
    "micro_trade_intensity",
    "micro_avg_trade_size",
    "micro_large_trade_pct",
    "micro_buy_run_max",
    "micro_sell_run_max",
    "micro_tick_vol",
    "micro_wmid_ret",
)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(max(var, 0.0))
