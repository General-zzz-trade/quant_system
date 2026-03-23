"""PositionSizer — equity-adaptive position sizing with risk controls.

Extracted from AlphaRunner to reduce god-class size.
Handles: leverage ladder, equity-tier weights, IC scaling, consensus scaling,
correlation sizing, portfolio exposure cap, dynamic Sharpe degradation.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from execution.balance_utils import get_total_and_free_balance
from core.exceptions import VenueError

logger = logging.getLogger(__name__)

# Equity-adaptive base weights per runner.
# Small account (<$500): concentrate on fewer positions, higher cap.
# Medium ($500-$10K): standard allocation.
# Large (>$10K): diversify, lower per-symbol cap.
_BASE_WEIGHTS_TIERS = {
    "small": {  # equity < 500
        "BTCUSDT": 0.25, "ETHUSDT": 0.25, "ETHUSDT_15m": 0.0,
        "BTCUSDT_15m": 0.0, "BTCUSDT_4h": 0.35, "ETHUSDT_4h": 0.30,
    },
    "medium": {  # 500 <= equity < 10000
        "BTCUSDT": 0.18, "ETHUSDT": 0.18, "ETHUSDT_15m": 0.05,
        "BTCUSDT_15m": 0.05, "BTCUSDT_4h": 0.25, "ETHUSDT_4h": 0.20,
    },
    "large": {  # equity >= 10000
        "BTCUSDT": 0.12, "ETHUSDT": 0.12, "ETHUSDT_15m": 0.05,
        "BTCUSDT_15m": 0.05, "BTCUSDT_4h": 0.18, "ETHUSDT_4h": 0.15,
    },
}

# Map runner_key to model name for IC health lookup.
_RUNNER_TO_MODEL = {
    "BTCUSDT": "BTCUSDT_gate_v2", "ETHUSDT": "ETHUSDT_gate_v2",
    "BTCUSDT_4h": "BTCUSDT_4h", "ETHUSDT_4h": "ETHUSDT_4h",
}


class PositionSizer:
    """Computes position sizes based on equity, risk, and market context.

    Owns: leverage ladder, equity-tier weights, IC health cache,
    dynamic Sharpe scale, consensus scaling.
    """

    # Leverage ladder: 10x across all tiers for demo signal validation.
    LEVERAGE_LADDER = [
        (0, 10.0),
    ]

    def __init__(
        self,
        adapter: Any,
        symbol: str,
        runner_key: str = "",
        step_size: float = 0.01,
        min_size: float = 0.01,
        max_qty: float = 0,
        base_position_size: float = 0.001,
        adaptive_sizing: bool = True,
    ):
        self._adapter = adapter
        self._symbol = symbol
        self._runner_key = runner_key
        self._step_size = step_size
        self._min_size = min_size
        self._max_qty = max_qty
        self._base_position_size = base_position_size
        self._adaptive_sizing = adaptive_sizing
        self._position_size = base_position_size

        # Dynamic scale from rolling trade Sharpe
        self._dynamic_scale: float = 1.0
        self._recent_trade_pnls: list[float] = []

        # IC health cache
        self._ic_scale_cache: dict[str, float] = {}
        self._ic_scale_cache_ts: float = 0.0

    @property
    def position_size(self) -> float:
        return self._position_size

    @position_size.setter
    def position_size(self, value: float) -> None:
        self._position_size = value

    @property
    def dynamic_scale(self) -> float:
        return self._dynamic_scale

    @property
    def recent_trade_pnls(self) -> list[float]:
        return self._recent_trade_pnls

    def round_to_step(self, size: float) -> float:
        """Round qty to exchange step size (floor to never exceed)."""
        if self._step_size <= 0:
            return size
        steps = int(size / self._step_size)
        size = steps * self._step_size
        if self._step_size >= 1:
            size = int(size)
        else:
            step_decimals = max(0, -int(np.floor(np.log10(self._step_size))))
            size = round(size, step_decimals)
        return size

    def get_leverage_for_equity(self, equity: float, rets: list[float],
                                vol_history: list[float],
                                vol_median: float) -> float:
        """Look up base leverage from ladder, then scale by volatility.

        Low vol → higher leverage (same risk budget), high vol → lower.
        """
        lev = 1.0
        for threshold, lev_val in self.LEVERAGE_LADDER:
            if equity >= threshold:
                lev = lev_val

        if len(rets) >= 20:
            rv_20 = float(np.std(rets[-20:]))
            _vol_median = float(np.median(vol_history)) if len(vol_history) >= 50 else vol_median
            if _vol_median > 0:
                vol_ratio = rv_20 / _vol_median
                vol_ratio = max(0.5, min(3.0, vol_ratio))
                vol_scale = min(1.0, 1.0 / (vol_ratio ** 0.3))
                lev *= vol_scale

        return lev

    def get_consensus_scale(self, current_signal: int,
                            consensus_signals: dict) -> float:
        """Position scale based on cross-symbol signal consensus.

        Contrarian research: all agree bearish → market goes UP (+28bp).
        Returns scale factor in [0.5, 1.3].
        """
        if not consensus_signals or not self._runner_key:
            return 1.0
        if current_signal == 0:
            return 1.0

        n_bull = n_bear = n_total = 0
        for rkey, sig in consensus_signals.items():
            if rkey == self._runner_key:
                continue
            if sig > 0:
                n_bull += 1
            elif sig < 0:
                n_bear += 1
            n_total += 1

        if n_total == 0:
            return 1.0

        same_dir = n_bull if current_signal > 0 else n_bear
        opposite_dir = n_bear if current_signal > 0 else n_bull

        if opposite_dir == n_total and n_total >= 2:
            return 1.3  # contrarian boost

        agree_frac = same_dir / n_total if n_total > 0 else 0
        if agree_frac >= 0.5:
            return 1.0
        elif opposite_dir > same_dir:
            return 0.85
        else:
            return 0.90

    def update_dynamic_scale(self) -> None:
        """Update dynamic position scale based on rolling trade Sharpe.

        Negative Sharpe → reduce. < -0.5 → pause trading.
        """
        pnls = self._recent_trade_pnls
        if len(pnls) < 5:
            self._dynamic_scale = 1.0
            return
        arr = np.array(pnls)
        mu = np.mean(arr)
        std = np.std(arr)
        sharpe = mu / std if std > 1e-8 else 0.0
        if sharpe > 0:
            self._dynamic_scale = 1.0
        elif sharpe > -0.5:
            self._dynamic_scale = 0.5
        else:
            self._dynamic_scale = 0.0
            logger.warning(
                "%s DYNAMIC SCALE → 0: rolling Sharpe=%.2f (%d trades), pausing",
                self._symbol, sharpe, len(pnls),
            )

    def record_trade_pnl(self, pnl_pct: float) -> None:
        """Record a trade PnL for dynamic scale tracking."""
        self._recent_trade_pnls.append(pnl_pct)
        if len(self._recent_trade_pnls) > 30:
            self._recent_trade_pnls = self._recent_trade_pnls[-30:]

    def _refresh_ic_cache(self) -> None:
        """Refresh IC health cache every 10 minutes."""
        if time.time() - self._ic_scale_cache_ts <= 600:
            return
        try:
            ic_path = Path("data/runtime/ic_health.json")
            if not ic_path.exists():
                return
            ic_data = json.loads(ic_path.read_text())
            for m in ic_data.get("models", []):
                model_name = m.get("model", "")
                status = m.get("overall_status", "GREEN")
                scale = {"GREEN": 1.2, "YELLOW": 0.8, "RED": 0.4}.get(status, 1.0)
                self._ic_scale_cache[model_name] = scale
            self._ic_scale_cache_ts = time.time()
        except Exception as exc:
            logger.debug("%s IC scale cache update failed: %s", self._symbol, exc)

    def compute(
        self,
        price: float,
        *,
        pnl_tracker: Any,
        entry_scaler_module: Any,
        deadzone: float,
        deadzone_base: float,
        regime_active: bool,
        z_scale: float,
        last_z_score: float,
        current_signal: int,
        consensus_signals: dict,
        rets: list[float],
        vol_history: list[float],
        vol_median: float,
        state_store: Any = None,
    ) -> float:
        """Compute position size with full adaptive pipeline.

        Returns qty (already rounded to step size).
        """
        if not self._adaptive_sizing:
            size = self.round_to_step(self._base_position_size)
            self._position_size = size
            return size

        try:
            equity, _free = get_total_and_free_balance(self._adapter.get_balances())
            if equity is None:
                logger.warning("%s sizing fallback: USDT total unavailable", self._symbol)
                equity = 0.0
        except VenueError as exc:
            logger.error("VENUE_ERROR symbol=%s context=sizing type=%s retry=true",
                         self._symbol, type(exc).__name__)
            size = self.round_to_step(self._base_position_size)
            self._position_size = size
            return size
        except Exception as exc:
            logger.warning("%s sizing fallback: failed to fetch balances: %s", self._symbol, exc)
            size = self.round_to_step(self._base_position_size)
            self._position_size = size
            return size

        if equity <= 0 or price <= 0:
            size = self.round_to_step(self._base_position_size)
            self._position_size = size
            return size

        # Leverage from ladder + vol-adaptive
        target_lev = self.get_leverage_for_equity(equity, rets, vol_history, vol_median)

        # Drawdown leverage reduction
        dd_pct = pnl_tracker.drawdown_pct if pnl_tracker.peak_equity > 0 else 0
        vol_ratio = (deadzone / deadzone_base) if deadzone_base > 0 else 1.0
        dd_scale = entry_scaler_module.leverage_scale(dd_pct, vol_ratio=vol_ratio)
        target_lev *= dd_scale
        if dd_scale < 1.0:
            logger.info("%s DD_SCALE: dd=%.1f%% → lev_scale=%.2f → effective_lev=%.1fx",
                        self._symbol, dd_pct, dd_scale, target_lev)

        # Equity-tier base weight
        if equity < 500:
            weights = _BASE_WEIGHTS_TIERS["small"]
        elif equity < 10000:
            weights = _BASE_WEIGHTS_TIERS["medium"]
        else:
            weights = _BASE_WEIGHTS_TIERS["large"]

        base_cap = weights.get(self._runner_key or self._symbol, 0.10)

        # Regime scaling
        regime_cap_scale = 1.0 if regime_active else 0.6
        base_cap *= regime_cap_scale

        # IC health scaling
        self._refresh_ic_cache()
        model_name = _RUNNER_TO_MODEL.get(self._runner_key or self._symbol, "")
        ic_scale = self._ic_scale_cache.get(model_name, 1.0)
        per_sym_cap = base_cap * ic_scale

        # Dynamic Sharpe degradation
        per_sym_cap *= self._dynamic_scale

        # Correlation-aware sizing
        try:
            from runner.main import _correlation_computer
            if _correlation_computer is not None:
                active = [s for s, sig in consensus_signals.items()
                          if sig != 0 and s != self._symbol and s != self._runner_key]
                if active:
                    avg_corr = _correlation_computer.position_correlation(self._symbol, active)
                    if avg_corr is not None and avg_corr > 0.6:
                        corr_scale = max(0.3, 1.0 - (avg_corr - 0.6) / 0.4)
                        per_sym_cap *= corr_scale
        except Exception:
            logger.debug("%s correlation sizing unavailable", self._symbol, exc_info=True)

        # Directional hedge boost
        my_key = self._runner_key or self._symbol
        peer_key = "ETHUSDT" if "BTC" in my_key else "BTCUSDT"
        peer_sigs = [v for k, v in consensus_signals.items() if peer_key in k and v != 0]
        if peer_sigs and current_signal != 0:
            peer_dir = 1 if sum(peer_sigs) > 0 else -1
            if peer_dir != current_signal:
                per_sym_cap *= 1.15
                logger.debug("%s HEDGE_BOOST: opposite to %s → cap*1.15", self._symbol, peer_key)

        max_notional = equity * per_sym_cap * target_lev
        size = max_notional / price

        # Apply z-scale + consensus + confidence
        consensus_scale = self.get_consensus_scale(current_signal, consensus_signals)
        confidence_scale = entry_scaler_module.confidence_cap_scale(
            z_score=last_z_score, base_dz=deadzone_base,
        )
        combined_scale = z_scale * consensus_scale * confidence_scale
        pre_scale_size = size
        size *= combined_scale

        logger.info(
            "%s SCALE: cap=%.3f lev=%.1f base_size=%.4f × z=%.2f×cons=%.2f×conf=%.2f=%.2f → size=%.4f",
            self._symbol, per_sym_cap, target_lev, pre_scale_size,
            z_scale, consensus_scale, confidence_scale, combined_scale, size,
        )

        # Clamp to exchange limits
        size = max(self._min_size, size)
        if self._max_qty > 0:
            size = min(size, self._max_qty)
        size = self.round_to_step(size)

        if size != self._position_size:
            logger.info(
                "%s SIZING: equity=$%.0f lev=%.0fx → %.2f %s ($%.0f notional)",
                self._symbol, equity, target_lev, size,
                self._symbol.replace("USDT", ""), size * price,
            )

        # Set exchange leverage
        lev_int = max(2, int(round(target_lev)))
        if not hasattr(self, "_current_exchange_lev") or self._current_exchange_lev != lev_int:
            try:
                result = self._adapter._client.post("/v5/position/set-leverage", {
                    "category": "linear", "symbol": self._symbol,
                    "buyLeverage": str(lev_int),
                    "sellLeverage": str(lev_int),
                })
                if isinstance(result, dict):
                    ret_code = result.get("retCode", -1)
                    if ret_code != 0:
                        logger.warning(
                            "%s set_leverage API failed: retCode=%s retMsg=%s",
                            self._symbol, ret_code, result.get("retMsg"),
                        )
                self._current_exchange_lev = lev_int
                logger.info("%s exchange leverage set to %dx", self._symbol, lev_int)
            except Exception as e:
                logger.warning("%s set_leverage failed (non-fatal): %s", self._symbol, e)

        # Portfolio exposure cap
        try:
            _SCALE = 100_000_000
            total_exposure = sum(
                abs(state_store.get_position(s).qty) / _SCALE * price
                for s in ["BTCUSDT", "ETHUSDT"]
                if state_store is not None
            ) if state_store else 0
            if equity < 500:
                max_total = equity * 8.0
            elif equity < 10000:
                max_total = equity * 6.0
            else:
                max_total = equity * 5.0
            if total_exposure + (size * price) > max_total and size > self._min_size:
                allowed = max(0, max_total - total_exposure) / price
                if allowed < self._min_size:
                    size = 0
                else:
                    size = self.round_to_step(allowed)
                logger.info("%s PORTFOLIO_CAP: total_exposure=$%.0f + new=$%.0f > max=$%.0f → size=%.4f",
                            self._symbol, total_exposure, size * price, max_total, size)
        except Exception as exc:
            logger.debug("%s portfolio exposure check unavailable: %s", self._symbol, exc)

        # Final safety
        if np.isnan(size) or np.isinf(size) or size < 0:
            logger.error("%s SIZING BLOCKED: invalid size=%.6f — falling back to min_size",
                         self._symbol, size)
            size = self._min_size

        self._position_size = size
        return size
