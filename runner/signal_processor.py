"""SignalProcessor — z-score computation, discretization, and signal logic.

Extracted from AlphaRunner to reduce god-class size.
Handles: z-score normalization, z-clamp, apply_constraints, secondary horizon,
direction alignment, and force-exit checks.
"""
from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class SignalProcessor:
    """Processes raw predictions into discrete trading signals.

    Owns: RustInferenceBridge, z-score state, min/max hold, deadzone.
    Does NOT own: regime filter, feature engine, model prediction.
    """

    def __init__(
        self,
        inference: Any,
        symbol: str,
        deadzone_base: float,
        min_hold_base: int,
        max_hold_base: int,
        zscore_window: int,
        zscore_warmup: int,
        long_only: bool = False,
    ):
        self._inference = inference
        self._symbol = symbol
        self._deadzone_base = deadzone_base
        self._deadzone = deadzone_base
        self._min_hold_base = min_hold_base
        self._max_hold_base = max_hold_base
        self._min_hold = min_hold_base
        self._max_hold = max_hold_base
        self._zscore_window = zscore_window
        self._zscore_warmup = zscore_warmup
        self._long_only = long_only

        self._last_z_score: float = 0.0
        self._z_scale: float = 1.0

    # ── Public properties ────────────────────────────────────────────

    @property
    def deadzone(self) -> float:
        return self._deadzone

    @deadzone.setter
    def deadzone(self, value: float) -> None:
        self._deadzone = value

    @property
    def deadzone_base(self) -> float:
        return self._deadzone_base

    @property
    def min_hold(self) -> int:
        return self._min_hold

    @min_hold.setter
    def min_hold(self, value: int) -> None:
        self._min_hold = value

    @property
    def max_hold(self) -> int:
        return self._max_hold

    @max_hold.setter
    def max_hold(self, value: int) -> None:
        self._max_hold = value

    @property
    def last_z_score(self) -> float:
        return self._last_z_score

    @property
    def z_scale(self) -> float:
        return self._z_scale

    @property
    def inference(self) -> Any:
        return self._inference

    # ── Core signal processing ───────────────────────────────────────

    @staticmethod
    def compute_z_scale(z: float) -> float:
        """Non-linear position sizing based on z-score magnitude.

        Shared live/backtest contract:
        - |z| > 2.0: scale=1.5
        - |z| > 1.0: scale=1.0
        - |z| > 0.5: scale=0.7
        - else:       scale=0.5
        """
        abs_z = abs(z)
        if abs_z > 2.0:
            return 1.5
        elif abs_z > 1.0:
            return 1.0
        elif abs_z > 0.5:
            return 0.7
        else:
            return 0.5

    def normalize_z(self, pred: float, hour_key: int) -> float | None:
        """Z-score normalize a prediction. Returns None during warmup."""
        z_val = self._inference.zscore_normalize(self._symbol, pred, hour_key)
        if z_val is None:
            return None
        # Clip extreme z-scores (service restart → limited history → z=13+)
        z = max(-5.0, min(5.0, z_val))
        self._z_scale = self.compute_z_scale(z)
        self._last_z_score = z
        return z

    def apply_z_clamp(self, z: float, current_signal: int) -> float:
        """Clamp extreme z-scores when not in position.

        |z| > 3.5 with no position → cap ±3.0. Prevents false signals
        from low-variance prediction buffers.
        """
        if abs(z) > 3.5 and current_signal == 0:
            old_z = z
            z = 3.0 if z > 0 else -3.0
            self._z_scale = self.compute_z_scale(z)
            self._last_z_score = z
            logger.info(
                "%s Z_CLAMP: |z|=%.1f capped to %.1f (extreme z with no position → likely unreliable)",
                self._symbol, abs(old_z), z,
            )
        return z

    def discretize(
        self,
        pred: float,
        hour_key: int,
        regime_ok: bool,
    ) -> int:
        """Apply constraints: z-score → deadzone → min-hold → max-hold → signal.

        Returns discrete signal: +1 (long), -1 (short), 0 (flat).
        """
        effective_dz = 999.0 if not regime_ok else self._deadzone
        return int(self._inference.apply_constraints(
            self._symbol, pred, hour_key,
            deadzone=effective_dz,
            min_hold=self._min_hold,
            max_hold=self._max_hold,
            long_only=self._long_only,
        ))

    def secondary_horizon_fill(
        self,
        new_signal: int,
        regime_ok: bool,
        feat_dict: dict,
        hour_key: int,
        horizon_models: list,
        predict_fn,
    ) -> int:
        """When primary signal is flat, check secondary (shorter) horizon.

        Fills ~21% more bars. Walk-forward validated: Sharpe 1.68→2.22.
        """
        if new_signal != 0 or not regime_ok:
            return new_signal

        pred_h2 = predict_fn(feat_dict)
        if pred_h2 is None:
            return new_signal

        h2_key = f"{self._symbol}_h2"
        h2_signal = int(self._inference.apply_constraints(
            h2_key, pred_h2, hour_key,
            deadzone=self._deadzone,
            min_hold=self._min_hold,
            max_hold=self._max_hold,
        ))
        if h2_signal != 0:
            self._inference.set_position(self._symbol, h2_signal, 1)
            return h2_signal
        return new_signal

    def apply_direction_alignment(
        self,
        new_signal: int,
        prev_signal: int,
        consensus_signals: dict,
    ) -> int:
        """Block ETH new entries opposing BTC consensus direction.

        BTC is crypto market leader (70% dominance). Only applies to
        new entries (prev=0 → new≠0) on ETH runners.
        """
        if new_signal == 0 or prev_signal != 0 or "ETH" not in self._symbol:
            return new_signal

        btc_sigs = [v for k, v in consensus_signals.items()
                    if "BTC" in k and v != 0]
        if btc_sigs:
            btc_dir = 1 if sum(btc_sigs) > 0 else -1
            if new_signal != btc_dir:
                logger.info(
                    "%s DIRECTION_ALIGN: ETH wants %s but BTC consensus is %s → blocked",
                    self._symbol,
                    "LONG" if new_signal > 0 else "SHORT",
                    "LONG" if btc_dir > 0 else "SHORT",
                )
                return 0
        return new_signal

    def check_force_exits(
        self,
        prev_signal: int,
        z: float,
        bar: dict,
        entry_price: float,
        consensus_signals: dict,
        runner_key: str,
        config: dict,
        compute_stop_fn,
    ) -> tuple[bool, str]:
        """Check all force-exit conditions. Returns (should_exit, reason).

        Priority order:
        1. ATR adaptive stop-loss
        2. Quick loss exit (-1% → -5% at 5x leverage)
        3. 4h z-score reversal
        4. Z-score reversal
        """
        if prev_signal == 0 or entry_price <= 0:
            return False, ""

        # 1. Adaptive stop loss: ATR-based with trailing
        stop = compute_stop_fn(bar["close"])
        if prev_signal > 0 and bar["low"] <= stop:
            return True, "atr_stop"
        elif prev_signal < 0 and bar["high"] >= stop:
            return True, "atr_stop"

        # 2. Quick loss exit (leverage protection)
        if prev_signal > 0:
            unrealized_pct = (bar["close"] - entry_price) / entry_price
        else:
            unrealized_pct = (entry_price - bar["close"]) / entry_price
        if unrealized_pct < -0.01:
            return True, f"quick_loss_{unrealized_pct:.2%}"

        # 3. 4h z-score reversal exit
        base_sym = self._symbol.replace("USDT", "") + "USDT"
        key_4h = f"{base_sym}_4h"
        sig_4h = consensus_signals.get(key_4h, 0)
        is_4h_runner = "4h" in config.get("version", "")
        if not is_4h_runner and sig_4h != 0 and sig_4h == -prev_signal:
            return True, f"4h_reversal_sig={sig_4h}"

        # 4. Z-score reversal exit
        z_reversal_threshold = -0.3
        if prev_signal > 0 and z < z_reversal_threshold:
            return True, "z_reversal"
        elif prev_signal < 0 and z > -z_reversal_threshold:
            return True, "z_reversal"

        return False, ""

    def sync_flat(self) -> None:
        """Sync inference bridge to flat state."""
        self._inference.set_position(self._symbol, 0, 1)
