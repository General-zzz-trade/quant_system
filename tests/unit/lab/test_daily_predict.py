"""The daily BTC/ETH prediction report (lab/daily_predict.py; owner 2026-11-18 "先做BTC和ETH，直接就在此对话推送"):
frozen code and models only, the day's readiness, the sizing rule, the append-only log and the text. Synthetic data."""
from __future__ import annotations

import gzip
import hashlib

import numpy as np
import pandas as pd
import pytest

from data.downloads import perps_feed as pf
from lab import daily_predict as dp
from lab import panel_study as ps

H, DAY = ps.H, ps.DAY
T0 = 1_700_006_400_000                                   # 2023-11-15 00:00 UTC


def test_changed_frozen_code_is_named():
    current = {"a.py": b"x", "b.py": b"y"}
    frozen = {"a.py": b"x", "b.py": b"z"}
    assert dp.code_changes(("a.py", "b.py"), current.__getitem__, frozen.__getitem__) == ["b.py"]


def test_a_model_file_that_is_not_the_frozen_one_is_refused(tmp_path):
    (tmp_path / "m.json").write_text("{}")
    spec = {"models": {"trend_14 logistic": {"m.json": hashlib.sha256(b"{}").hexdigest()}}}
    assert dp.check_models(spec, tmp_path) == []
    (tmp_path / "m.json").write_text('{"changed": 1}')
    assert dp.check_models(spec, tmp_path) == ["m.json"]


def test_the_stop_is_a_multiple_of_the_predicted_volatility_and_the_size_risks_a_fixed_share():
    out = dp.sizing(0.02, capital=1000.0, risk=0.01, min_notional=100.0)
    assert out["stop_pct"] == pytest.approx(0.05)                            # 2.5 x 2 %
    assert out["notional"] == pytest.approx(200.0) and not out["capped"] and not out["below_minimum"]
    small = dp.sizing(0.02, capital=150.0, risk=0.01, min_notional=100.0)
    assert small["notional"] == pytest.approx(30.0) and small["below_minimum"]
    calm = dp.sizing(0.001, capital=150.0, risk=0.01, min_notional=5.0)
    assert calm["notional"] == pytest.approx(300.0) and calm["capped"]       # at most 2x the capital


def test_a_day_that_is_not_whole_gives_reasons_not_predictions():
    t = T0 + 10 * DAY
    state = {"last_update": {"now_ms": t + H, "failed": [], "trading": [f"S{k}" for k in range(100)]}}
    assert dp.problems(state, t, {"BTCUSDT": t, "ETHUSDT": t}, universe=100) == []
    stale = {"last_update": {**state["last_update"], "now_ms": t - H}}
    assert dp.problems(stale, t, {"BTCUSDT": t, "ETHUSDT": t}, universe=100)
    failing = {"last_update": {**state["last_update"], "failed": [f"S{k}" for k in range(6)]}}
    assert dp.problems(failing, t, {"BTCUSDT": t, "ETHUSDT": t}, universe=100)
    assert dp.problems(state, t, {"BTCUSDT": t, "ETHUSDT": t - DAY}, universe=100)
    assert dp.problems(state, t, {"BTCUSDT": t, "ETHUSDT": t}, universe=40)


def test_the_log_is_append_only_chained_and_one_line_per_day(tmp_path):
    log = tmp_path / "log.jsonl"
    first = dp.append_log(log, {"t": 1, "x": 0.5})
    second = dp.append_log(log, {"t": 2, "x": 0.6})
    assert first["prev"] == "genesis" and second["prev"] == hashlib.sha256(
        log.read_text().splitlines()[0].encode()).hexdigest()
    with pytest.raises(ValueError):
        dp.append_log(log, {"t": 2, "x": 0.7})
    assert dp.verify_log(log) == []
    lines = log.read_text().splitlines()
    log.write_text(lines[0].replace("0.5", "0.9") + "\n" + lines[1] + "\n")
    assert dp.verify_log(log) == [2]


def _feed(tmp_path, days=520):
    """BTC, ETH and three others: random walks with every column a snapshot has."""
    feed = tmp_path / "feed"
    feed.mkdir()
    rng = np.random.default_rng(4)
    hours = T0 - (days - 10) * DAY + np.arange(days * 24, dtype=np.int64) * H
    for k, s in enumerate(("BTCUSDT", "ETHUSDT", "AUSDT", "BUSDT", "CUSDT")):
        close = 100 * (k + 1) * np.exp(np.cumsum(rng.normal(0, 0.006 * (k + 1), len(hours))))
        rows = [f"{t},{c},{c * 1.002},{c * 0.998},{c},{10.0},{c * 10 * (5 - k)},{7},{5.0},{c * 5}"
                for t, c in zip(hours, close)]
        (feed / f"{s}_1h.csv.gz").write_bytes(gzip.compress((pf.BAR_HEADER + "\n" + "\n".join(rows) + "\n").encode()))
        settle = hours[::8]
        frows = [f"{t},{r}" for t, r in zip(settle, rng.normal(0.0001, 0.0002, len(settle)))]
        (feed / f"{s}_funding.csv.gz").write_bytes(
            gzip.compress((pf.FUNDING_HEADER + "\n" + "\n".join(frows) + "\n").encode()))
    return feed


def _models(tmp_path):
    rng = np.random.default_rng(5)
    cols = list(ps.PANEL_FEATURES) + ["log_rv_30"]
    x = pd.DataFrame(rng.normal(size=(400, len(cols))), columns=cols)
    t = np.repeat(T0 + np.arange(40, dtype=np.int64) * DAY, 10)
    trend = (x[cols[0]] + rng.normal(size=400) > 1).astype(float).to_numpy()
    vol = np.log(0.02) + 0.1 * x[cols[1]].to_numpy()
    models = {}
    for name, kind, model, y, columns in (("trend_14 logistic", "binary", "logistic", trend, cols),
                                          ("trend_14 lgbm", "binary", "lgbm", trend, cols),
                                          ("fwd_vol_7 ridge", "regression", "ridge", vol, cols),
                                          ("fwd_vol_7 lgbm", "regression", "lgbm", vol, cols),
                                          ("fwd_vol_7 naive", "regression", "ridge", vol, ["log_rv_30"])):
        path = ps.save_frozen(ps.fit_final(x[columns], y, t, kind=kind, model=model),
                              tmp_path / "frozen" / name.replace(" ", "_"))
        models[name] = path
    return models


def test_a_day_s_report_has_both_coins_from_the_frozen_models(tmp_path):
    feed = _feed(tmp_path)
    t = T0
    report = dp.predict_day(feed, t, _models(tmp_path), calibration=None)
    assert report["t"] == t and report["universe"] == 5
    for coin in dp.COINS:
        row = report["coins"][coin]
        assert 0 < row["trend_logistic"] < 1 and 0 < row["trend_lgbm"] < 1
        assert 0 <= row["trend_rank"] <= 1
        assert row["vol_pred"] == pytest.approx(np.exp(np.mean([row["log_vol_ridge"], row["log_vol_lgbm"]])))
        assert row["close"] > 0 and np.isfinite(row["ret_40"]) and np.isfinite(row["rv_30"])
        assert row["last_day"] == t


def test_the_text_says_what_it_does_not_predict_and_shows_reasons_instead_of_numbers():
    coin = {"close": 70000.0, "trend_logistic": 0.08, "trend_lgbm": 0.06, "trend_rank": 0.1, "tercile": None,
            "vol_pred": 0.02, "vol_naive": 0.021, "rv_30": 0.019, "ret_40": 0.05, "funding_last": 0.0001,
            "funding_3d": 0.0009, "last_day": T0}
    report = {"t": T0, "universe": 98, "coins": {"BTCUSDT": coin, "ETHUSDT": coin},
              "sizing": {"BTCUSDT": dp.sizing(0.02, capital=150.0, risk=0.01, min_notional=100.0),
                         "ETHUSDT": dp.sizing(0.02, capital=150.0, risk=0.01, min_notional=20.0)},
              "capital": 150.0, "risk": 0.01, "problems": []}
    text = dp.render(report)
    assert "不预测涨跌" in text and "BTC" in text and "低于最小下单额" in text
    refused = dp.render({"t": T0, "problems": ["BTCUSDT 当天的 K 线不全"]})
    assert "不出预测" in refused and "BTCUSDT 当天的 K 线不全" in refused and "趋势概率" not in refused
