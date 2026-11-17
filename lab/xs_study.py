"""Cross-sectional long/short ideas of the 2026-11-17e study (owner 2026-11-17: "开广度和成本", then "1包含，
2做多空行情趋势震荡": BTC and ETH stay in the list; long and short; momentum while the market trends, reversal while
it ranges). Research functions of (hourly frame, context) as the lab's judge takes them through
``lab.um_lab.legacy(fn, cross=True, reads=("BTCUSDT",))``; the runtime cannot run them until they are registered in
trading/domain/strategy.

Each idea decides once a day from completed days only (``_hours``): long ``K`` and short ``K`` symbols of the list it
was given, at ``ON`` each; the rest 0. A symbol without the score's history (listed later, delisted) is not ranked.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from trading.domain.strategy.ideas import ON, _daily, _hours, _rank_pick, _xs_reversal_ls

K = 3
REGIME_DAYS = 30                 # BTC's efficiency ratio over the last 30 daily closes
REGIME_MOMENTUM_DAYS = 14        # the regime idea's momentum leg (fixed before the study)
REGIME_REVERSAL_DAYS = 3         # its reversal leg: B2_xs_reversal_ls with days=3


def _panel(ctx: dict, column: str) -> pd.DataFrame:
    """The daily ``column`` of every symbol the strategy was given, in name order (ties go by name)."""
    return pd.DataFrame({s: _daily(ctx["frames"][s])[column] for s in sorted(ctx["universe"])})


def _long_short(df: pd.DataFrame, ctx: dict, score: pd.DataFrame, *, long_high: bool) -> pd.Series:
    symbol = ctx["symbol"]
    high, low = _rank_pick(score, symbol, K, top=True), _rank_pick(score, symbol, K, top=False)
    long, short = (high, low) if long_high else (low, high)
    return _hours(df, (long.astype(float) - short.astype(float)) * ON)


def momentum_ls(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """Long the ``K`` largest ``n``-day returns, short the ``K`` smallest."""
    close = _panel(ctx, "close")
    return _long_short(df, ctx, close / close.shift(ctx["params"]["n"]) - 1, long_high=True)


def carry_ls(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """Short the ``K`` highest funding sums over ``window`` days (they pay the longs' funding to the shorts), long the
    ``K`` lowest."""
    score = _panel(ctx, "funding").rolling(ctx["params"]["window"]).sum()
    return _long_short(df, ctx, score, long_high=False)


def lowvol_ls(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """Long the ``K`` lowest standard deviations of the last ``n`` daily returns, short the ``K`` highest."""
    returns = _panel(ctx, "close").pct_change()
    return _long_short(df, ctx, returns.rolling(ctx["params"]["n"]).std(), long_high=False)


def efficiency(close: pd.Series, n: int) -> pd.Series:
    """Kaufman's efficiency ratio: the net move over ``n`` closes over the sum of the ``n`` moves between them; 1 on a
    straight line, near 0 in a range. Unknown before ``n`` moves."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return (net / path).where(path > 0)


def regime_ls(df: pd.DataFrame, ctx: dict) -> pd.Series:
    """Momentum (``REGIME_MOMENTUM_DAYS``) on days BTC's ``REGIME_DAYS``-day efficiency ratio is at least ``thr``,
    reversal (``REGIME_REVERSAL_DAYS``) on days it is below; nothing before it is known."""
    er = efficiency(_daily(ctx["btc"])["close"], REGIME_DAYS)
    trend = _hours(df, (er >= ctx["params"]["thr"]).astype(float))
    ranging = _hours(df, (er < ctx["params"]["thr"]).astype(float))
    momentum = momentum_ls(df, {**ctx, "params": {"n": REGIME_MOMENTUM_DAYS}})
    reversal = _xs_reversal_ls(df, {**ctx, "params": {"days": REGIME_REVERSAL_DAYS}})
    return momentum * trend + reversal * ranging


IDEAS: dict[str, tuple[Any, list[dict]]] = {
    "X1_xs_momentum_ls": (momentum_ls, [{"n": 7}, {"n": 14}, {"n": 30}]),
    "X2_xs_carry_ls": (carry_ls, [{"window": 3}, {"window": 7}, {"window": 14}]),
    "X3_xs_lowvol_ls": (lowvol_ls, [{"n": 14}, {"n": 30}]),
    "X4_xs_regime_ls": (regime_ls, [{"thr": 0.25}, {"thr": 0.35}]),
}
