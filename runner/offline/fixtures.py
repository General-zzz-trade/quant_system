"""Synthetic engineering fixtures. These contain no evidence of market alpha."""
import math
from decimal import Decimal as D

from data.quality.causal_bars import Bar, HOUR


def synthetic_bars(count=1200):
    def price(i):
        return D(str(round(40000 + 2000 * math.sin(i / 50) + 500 * math.sin(i / 9), 2)))
    return [Bar(1704067200000 + i * HOUR, price(i), max(price(i), price(i + 1)) * D("1.005"),
                min(price(i), price(i + 1)) * D(".995"), price(i + 1), D(100)) for i in range(count)]


def synthetic_funding(bars):
    return {b.open_time: (D(".0001"), b.open) for b in bars if b.open_time % (8 * HOUR) == 0}
