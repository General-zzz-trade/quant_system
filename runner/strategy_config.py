"""Backward-compat — strategy config moved to strategy/config.py."""
from strategy.config import *  # noqa: F401, F403
from strategy.config import _IS_LIVE, _consensus_signals, get_max_order_notional  # noqa: F401
