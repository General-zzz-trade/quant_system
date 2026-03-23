"""Gate evaluator for AlphaRunner — evaluate alpha expansion gates.

Extracted from AlphaRunner._evaluate_gates() to separate gate logic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Features consumed by gates
GATE_FEATURES = [
    "liquidation_volume_zscore_24", "oi_acceleration",
    "liquidation_cascade_score", "liquidation_imbalance",
    "tf4h_close_vs_ma20", "tf4h_rsi_14", "tf4h_macd_hist",
    "funding_rate", "basis",
    "vpin", "ob_imbalance", "spread_bps",
]


class GateEvaluator:
    """Evaluates alpha expansion gates and returns cumulative scale factor.

    Gates:
      1. LiquidationCascade: block/scale during liquidation events
      2. MultiTFConfluence: scale by 1h vs 4h alignment
      3. CarryCost: adjust by funding+basis carry cost
      4. VPIN: delays entry when microstructure unfavorable
    """

    def __init__(
        self,
        liq_gate: Any,
        mtf_gate: Any,
        carry_gate: Any,
        vpin_gate: Any,
    ):
        self._liq_gate = liq_gate
        self._mtf_gate = mtf_gate
        self._carry_gate = carry_gate
        self._vpin_gate = vpin_gate
        self.last_scale: float = 1.0

    def evaluate(
        self,
        signal: int,
        feat_dict: Dict[str, Any],
        consensus_signals: Dict[str, int],
        runner_key: str,
        symbol: str,
    ) -> float:
        """Evaluate all gates and return cumulative scale factor.

        Args:
            signal: Discrete signal (+1, -1, 0).
            feat_dict: Feature dictionary from RustFeatureEngine.
            consensus_signals: Shared signal state across runners.
            runner_key: Runner identifier (e.g. "BTCUSDT_4h").
            symbol: Trading symbol (e.g. "BTCUSDT").

        Returns:
            Scale factor in [0.0, ~1.5]. 0.0 means blocked.
        """
        if signal == 0:
            self.last_scale = 1.0
            return 1.0

        ev = type("_GateEv", (), {"metadata": {"signal": signal}})()
        ctx: Dict[str, Any] = {"signal": signal}

        # Populate context from features
        for key in GATE_FEATURES:
            val = feat_dict.get(key)
            if val is not None:
                fval = float(val)
                if fval == fval:  # NaN check
                    ctx[key] = fval

        # Inject 4h model signal from consensus state
        base_sym = symbol.replace("USDT", "") + "USDT"
        key_4h = f"{base_sym}_4h"
        sig_4h = consensus_signals.get(key_4h, 0)
        if sig_4h != 0:
            ctx["tf4h_model_signal"] = sig_4h

        scale = 1.0

        # Gate 1: Liquidation cascade
        r = self._liq_gate.check(ev, ctx)
        if not r.allowed:
            logger.info("%s LiquidationCascade BLOCKED: %s", symbol, r.reason)
            self.last_scale = 0.0
            return 0.0
        scale *= r.scale

        # Gate 2: Multi-timeframe confluence
        r = self._mtf_gate.check(ev, ctx)
        scale *= r.scale

        # Gate 3: Carry cost
        r = self._carry_gate.check(ev, ctx)
        scale *= r.scale

        # Gate 4: VPIN entry timing
        ctx["symbol"] = symbol
        r = self._vpin_gate.check(ev, ctx)
        if not r.allowed:
            logger.info("%s VPIN BLOCKED: %s", symbol, r.reason)
            self.last_scale = 0.0
            return 0.0
        scale *= r.scale

        self.last_scale = scale
        return scale
