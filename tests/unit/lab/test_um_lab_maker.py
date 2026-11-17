"""The lab's replay with maker entries (the cost lever, owner 2026-11-17): ``execution`` and ``ttl_bars`` reach the
shared broker; the default stays market. Synthetic snapshot only."""
from __future__ import annotations

from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from lab import um_lab as lab
from tests.unit.lab.test_um_lab import snapshot  # noqa: F401  (the synthetic snapshot fixture)

H = lab.H


def _flip(frame, ctx=None):
    return pd.Series(np.where((frame.index // H) % 48 < 24, 0.5, -0.5), index=frame.index)


def _fee_rates(result):
    return {round(float(D(f["fee"]) / (D(f["qty"]) * D(f["price"])) * 10000), 6) for f in result["fills"]}


def test_the_default_replay_takes_market_orders_at_the_taker_fee(snapshot):  # noqa: F811
    df = lab.frame("BTCUSDT", snapshot)
    default = lab.replay("BTCUSDT", _flip(df), snapshot=snapshot, stop_pct="0.08")
    market = lab.replay("BTCUSDT", _flip(df), snapshot=snapshot, stop_pct="0.08", execution="market")
    assert _fee_rates(default) == {5.0}
    assert default["fills"] == market["fills"] and default["curve"].equals(market["curve"])


def test_a_maker_replay_rests_entries_at_the_maker_fee(snapshot):  # noqa: F811
    df = lab.frame("BTCUSDT", snapshot)
    once = lab.replay("BTCUSDT", _flip(df), snapshot=snapshot, stop_pct="0.08", execution="maker")
    twice = lab.replay("BTCUSDT", _flip(df), snapshot=snapshot, stop_pct="0.08", execution="maker",
                       cost_multiplier=2)
    assert 2.0 in _fee_rates(once) and _fee_rates(once) <= {2.0, 5.0}   # entries maker, reductions and stops taker
    assert 4.0 in _fee_rates(twice) and _fee_rates(twice) <= {4.0, 10.0}
    market = lab.replay("BTCUSDT", _flip(df), snapshot=snapshot, stop_pct="0.08")
    assert once["fills"] != market["fills"]


def test_a_maker_replay_at_leverage_runs_on_isolated_margin(snapshot):  # noqa: F811
    df = lab.frame("BTCUSDT", snapshot)
    out = lab.replay("BTCUSDT", _flip(df), snapshot=snapshot, stop_pct="0.045", leverage="2", execution="maker")
    assert 2.0 in _fee_rates(out)


def test_an_unknown_execution_or_lifetime_is_refused(snapshot):  # noqa: F811
    df = lab.frame("BTCUSDT", snapshot)
    with pytest.raises(ValueError):
        lab.replay("BTCUSDT", _flip(df), snapshot=snapshot, execution="limit")
    with pytest.raises(ValueError):
        lab.replay("BTCUSDT", _flip(df), snapshot=snapshot, execution="maker", ttl_bars=0)


def test_run_idea_gives_every_replay_the_execution(snapshot, monkeypatch):  # noqa: F811
    seen = []
    real = lab.replay

    def spy(symbol, weights, **kwargs):
        seen.append((symbol, kwargs.get("execution"), kwargs.get("ttl_bars")))
        return real(symbol, weights, **kwargs)

    monkeypatch.setattr(lab, "replay", spy)
    terms = lab.LiveTerms(capital="300", stop_pct="0.08", pause_hours=4)
    lab.run_idea(_flip, {}, ["BTCUSDT", "ALTUSDT"], processes=1, snapshot=snapshot, live=terms,
                 execution="maker", ttl_bars=2)
    assert sorted(seen) == [("ALTUSDT", "maker", 2), ("BTCUSDT", "maker", 2)]
    seen.clear()
    lab.run_idea(_flip, {}, ["BTCUSDT"], processes=1, snapshot=snapshot, live=terms)
    assert seen == [("BTCUSDT", "market", 1)]
