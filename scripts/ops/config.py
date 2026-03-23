"""Compatibility re-export — moved to runner.strategy_config."""
from runner.strategy_config import *  # noqa: F401,F403
from runner.strategy_config import SYMBOL_CONFIG, MODEL_BASE, INTERVAL, WARMUP_BARS, POLL_INTERVAL  # noqa: F401
from runner.strategy_config import get_max_order_notional, MAX_ORDER_NOTIONAL, _consensus_signals  # noqa: F401
from runner.strategy_config import MAX_ORDER_NOTIONAL_PCT, MAX_ORDER_NOTIONAL_FLOOR, MAX_ORDER_NOTIONAL_CEILING  # noqa: F401
