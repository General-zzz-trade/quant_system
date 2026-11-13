"""Multi-timeframe (1h / 4h / 1d) strategy lab for BTC and ETH.

One strategy function (``lab.strategy.compute_targets``) maps closed bars up to
time t to a target position weight. The vectorized research engine and the
live runtime (``runner.binance_target``) call the same function, and both
execute through the research broker model (``runner.offline.broker``).
"""
