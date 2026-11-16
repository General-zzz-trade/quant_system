"""The research lab's harness (lab/um_lab.py) on a tiny synthetic snapshot: no network, no real data."""
from __future__ import annotations

import csv
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from lab import um_lab as lab
from lab.causality import violations

H, DAY = lab.H, lab.DAY


def _write(snapshot, symbol, start, hours, *, drift=0.0, volume=1e9, seed=0):
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, hours)))
    quote = volume / 24 * rng.uniform(0.2, 1.8, hours)
    rows = [(start + k * H, p, p * 1.01, p * 0.99, p, 1000, q, 10, 500, q / 2)
            for k, (p, q) in enumerate(zip(price, quote))]
    with gzip.open(snapshot / f"{symbol}_1h.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades",
                         "taker_buy_volume", "taker_buy_quote_volume"))
        writer.writerows(rows)
    with gzip.open(snapshot / f"{symbol}_funding.csv.gz", "wt", newline="") as handle:
        handle.write("funding_time,rate\n" + "".join(f"{start + k * 8 * H},0.0001\n" for k in range(hours // 8)))


@pytest.fixture
def snapshot(tmp_path):
    start = lab.START - 40 * DAY
    _write(tmp_path, "BTCUSDT", start, 24 * 120, drift=0.0005, seed=1)
    _write(tmp_path, "ALTUSDT", start + 30 * DAY, 24 * 90, seed=2)
    (tmp_path / "manifest.json").write_text(json.dumps({"contracts": [
        {"symbol": "BTCUSDT", "type": "perpetual", "margin": "USDT", "files": {"x": {}}},
        {"symbol": "ALTUSDT", "type": "perpetual", "margin": "USDT", "files": {"x": {}}},
        {"symbol": "GONEUSDT", "type": "perpetual", "margin": "USDT", "files": {}}]}))
    (tmp_path / "exchange_info.json").write_text(json.dumps({"BTCUSDT": {"step": "0.001", "min_notional": "5"}}))
    lab.frame.cache_clear()
    lab.manifest.cache_clear()
    lab.exchange_info.cache_clear()
    yield tmp_path
    lab.frame.cache_clear()


def test_contracts_with_data_and_their_rules(snapshot):
    assert lab.contracts(snapshot) == ["ALTUSDT", "BTCUSDT"]
    assert str(lab.rules("BTCUSDT", snapshot).min_notional) == "5"
    assert lab.rules("ALTUSDT", snapshot) == lab.DEFAULT_RULES            # no published filters (delisted)


def test_funding_lands_on_its_hour(snapshot):
    df = lab.frame("BTCUSDT", snapshot)
    assert df["funding"].iloc[0] == pytest.approx(0.0001) and df["funding"].iloc[1] == 0.0


def test_a_replay_goes_through_the_shared_broker_and_costs_money(snapshot):
    df = lab.frame("BTCUSDT", snapshot)
    flip = pd.Series(np.where((df.index // H) % 48 < 24, 0.5, -0.5), index=df.index)
    once = lab.replay("BTCUSDT", flip, snapshot=snapshot)
    twice = lab.replay("BTCUSDT", flip, cost_multiplier=2, snapshot=snapshot)
    assert len(once["curve"]) > 0 and len(once["fills"]) > 10
    assert twice["curve"].iloc[-1] < once["curve"].iloc[-1]               # double fees cost more


def test_the_portfolio_counts_a_late_contract_from_its_first_day_only():
    first = pd.Series([100.0, 110.0, 121.0], index=[0, DAY, 2 * DAY])
    late = pd.Series([100.0, 90.0], index=[DAY, 2 * DAY])
    daily = lab.portfolio_daily([first, late])
    assert daily.loc[1] == pytest.approx(0.10)                            # only the first sleeve existed
    assert daily.loc[2] == pytest.approx((11 - 10) / 210)                 # both: +11 and -10 over 110 + 100


def test_summary_drawdown_and_losing_streak():
    daily = pd.Series([0.1, -0.05, -0.05, 0.02, -0.01], index=range(5))
    s = lab.summary(daily)
    assert s["longest_losing_days"] == 2
    assert s["max_drawdown"] == pytest.approx(1 - (1.1 * 0.95 * 0.95) / 1.1)


def test_walk_forward_never_chooses_with_the_fold_itself():
    days = np.arange(lab.START // DAY, lab.END // DAY)
    before_first_fold = days < lab.FOLDS[0][0] // DAY
    good_early = pd.Series(np.where(before_first_fold, 0.002, -0.002) + 1e-4 * np.sin(days), index=days)
    good_late = pd.Series(np.where(before_first_fold, -0.002, 0.002) + 1e-4 * np.cos(days), index=days)
    result = lab.walk_forward([lab.Run({"p": "early"}, good_early), lab.Run({"p": "late"}, good_late)])
    assert result["chosen_per_fold"][0] == {"p": "early"}                  # chosen on what came before the fold
    first_fold = lab.window(result["walk_forward_daily"], *lab.FOLDS[0])
    assert first_fold.sum() < 0                                             # and it lost there, honestly


def test_the_liquidity_filter_uses_only_what_was_known(snapshot):
    df = lab.frame("ALTUSDT", snapshot)

    def flags(frame_):
        daily = frame_["quote_volume"].groupby(frame_.index // DAY).sum()
        median = daily.rolling(30, min_periods=30).median().shift(1)
        return pd.Series(median.reindex(frame_.index // DAY).to_numpy() >= 1e9, index=frame_.index)
    assert flags(df).equals(lab.liquid("ALTUSDT", min_daily_quote=1e9, snapshot=snapshot))
    cuts = [int(df.index[24 * 40]), int(df.index[24 * 60])]
    assert violations(flags, df, cuts) == []


def test_every_configuration_is_appended_to_the_registry(tmp_path):
    path = tmp_path / "trials.jsonl"
    lab.record([{"idea": "a"}], path)
    lab.record([{"idea": "b"}], path)
    assert [json.loads(line)["idea"] for line in path.read_text().splitlines()] == ["a", "b"]
