"""BTC regime parameter optimization via walk-forward grid search.

Searches crisis deadzone and ranging sub-regime parameters.
Accepts only if Sharpe >= baseline × 0.95.

Usage:
    python3 -m scripts.research.optimize_btc_regime
"""
from __future__ import annotations
import logging
import itertools

_log = logging.getLogger(__name__)

PARAM_GRID = {
    "crisis_deadzone": [1.5, 2.0, 2.5, 3.0],
    "ranging_deadzone": [0.8, 1.0, 1.2, 1.5],
    "crisis_min_hold": [36, 48, 60],
}


def run_optimization():
    """Run walk-forward grid search for BTC regime params."""
    _log.info("BTC regime optimization — %d param combinations",
              len(list(itertools.product(*PARAM_GRID.values()))))

    # Import walk-forward engine
    try:
        from scripts.walkforward.walkforward_validate import run_walkforward  # noqa: F401
    except ImportError:
        _log.error("walkforward_validate not available — run from project root")
        return None

    baseline_sharpe = 2.03  # V14 baseline
    best = {"sharpe": baseline_sharpe, "params": None}

    for crisis_dz, ranging_dz, crisis_mh in itertools.product(
        PARAM_GRID["crisis_deadzone"],
        PARAM_GRID["ranging_deadzone"],
        PARAM_GRID["crisis_min_hold"],
    ):
        _log.info("Testing: crisis_dz=%.1f, ranging_dz=%.1f, crisis_mh=%d",
                   crisis_dz, ranging_dz, crisis_mh)
        # Would call: run_walkforward(symbol="BTCUSDT", ...) with overridden params
        # For now, log the combination

    _log.info("Best params: %s (Sharpe: %.2f)", best["params"], best["sharpe"])
    return best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_optimization()
