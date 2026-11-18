"""lab/panel_forward.py: the six monthly forward looks of the frozen panel models (research/protocols/
2026-11-17-panel-forward.md). Synthetic snapshot (12 random-walk contracts), models fit on random rows, a frozen spec
and a look registry in a temporary directory; no network, no real data.

- Refusals: a look not registered or not passed, a registry without the protocol's six looks, changed frozen code
  or model files, a snapshot with missing hours, a thin universe.
- A look measures every frozen model pooled, per segment (the segments partition the decision days) and on BTC/ETH;
  only look 6 judges, by the protocol's rule; bars from the end on change nothing; new TradFi contracts are left out.
"""
from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from data.downloads import perps_feed as feed_format
from lab import daily_predict as dp
from lab import forward
from lab import panel_forward as pfw
from lab import panel_study as ps

H, DAY = ps.H, ps.DAY
T0 = 1_700_006_400_000                                   # 2023-11-15 00:00 UTC: the synthetic forward start
ENDS = [T0 + k * DAY for k in (30, 40, 50, 60, 70, 80)]
SYMBOLS = ["BTCUSDT", "ETHUSDT"] + [f"S{k:02d}USDT" for k in range(10)]
NAMES = ("trend_14 logistic", "trend_14 lgbm", "fwd_vol_7 ridge", "fwd_vol_7 lgbm", "fwd_vol_7 naive",
         "fwd_vol_14 ridge", "fwd_vol_14 lgbm", "fwd_vol_14 naive", "ret_14_scaled ridge", "ret_14_scaled lgbm")
SAME = (lambda p: (dp.ROOT / p).read_bytes(),) * 2


@pytest.fixture(autouse=True)
def _synthetic_forward_start(monkeypatch):
    monkeypatch.setattr(forward, "FORWARD_START", T0)


def _day(t):
    return pd.Timestamp(t, unit="ms").strftime("%Y-%m-%d")


def _snapshot(path, *, later=1.0, hole=None):
    """Hourly bars from 450 days before T0 to T0 + 90 days; ``later`` scales prices from ENDS[0] on, ``hole`` drops
    ETHUSDT's bar at that hour."""
    path.mkdir()
    rng = np.random.default_rng(4)
    hours = T0 - 450 * DAY + np.arange(540 * 24, dtype=np.int64) * H
    for k, s in enumerate(SYMBOLS):
        close = 100 * (k + 1) * np.exp(np.cumsum(rng.normal(0, 0.004 + 0.001 * k, len(hours))))
        close = np.where(hours >= ENDS[0], close * later, close)
        rows = [f"{t},{c},{c * 1.002},{c * 0.998},{c},{10.0},{c * 10 * (20 - k)},{7},{5.0},{c * 5}"
                for t, c in zip(hours, close) if not (s == "ETHUSDT" and t == hole)]
        (path / f"{s}_1h.csv.gz").write_bytes(gzip.compress(
            (feed_format.BAR_HEADER + "\n" + "\n".join(rows) + "\n").encode()))
        settle = hours[::8]
        frows = [f"{t},{r}" for t, r in zip(settle, rng.normal(0.0001, 0.0002, len(settle)))]
        (path / f"{s}_funding.csv.gz").write_bytes(
            gzip.compress((feed_format.FUNDING_HEADER + "\n" + "\n".join(frows) + "\n").encode()))
    return path


def _spec(tmp_path, *, ends=ENDS):
    rng = np.random.default_rng(5)
    cols = list(ps.PANEL_FEATURES) + ["log_rv_30"]
    x = pd.DataFrame(rng.normal(size=(400, len(cols))), columns=cols)
    t = np.repeat(T0 - 100 * DAY + np.arange(40, dtype=np.int64) * DAY, 10)
    trend = (x[cols[0]] + rng.normal(size=400) > 1).astype(float).to_numpy()
    vol = np.log(0.02) + 0.1 * x[cols[1]].to_numpy()
    models = {}
    for name in NAMES:
        target, model = name.split(" ")
        kind = "binary" if target == "trend_14" else "regression"
        columns = ["log_rv_30"] if model == "naive" else cols
        path = ps.save_frozen(ps.fit_final(x[columns], trend if kind == "binary" else vol, t, kind=kind,
                                           model="ridge" if model == "naive" else model),
                              tmp_path / "frozen" / name.replace(" ", "_"))
        files = [path] + ([path.with_name(path.name[:-len(".model.txt")] + ".meta.json")]
                          if path.name.endswith(".model.txt") else [])
        models[name] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    spec = {"kind": "panel_forward", "passed": False, "registered": True, "study": "panel_test:frozen",
            "code_commit": "frozen", "features": cols, "models": models}
    path = tmp_path / "forward_spec.json"
    path.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    looks = tmp_path / "looks.jsonl"
    looks.write_text("".join(json.dumps({"study": "panel_test:frozen", "spec": str(path), "end": _day(e),
                                         "spec_sha256": forward.spec_digest(path),
                                         "registered": _day(T0 - DAY)}) + "\n" for e in ends))
    return path, looks


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("panel_forward")
    snapshot = _snapshot(root / "snap")
    spec, looks = _spec(root)
    return {"root": root, "snapshot": snapshot, "spec": spec, "looks": looks}


def _look(world, out, *, end=ENDS[0], data=None, tradfi=(), now=None, code=SAME, min_universe=10, looks=None,
          spec=None):
    return pfw.look(end=_day(end), data=data or world["snapshot"], out=out, tradfi=tradfi,
                    spec_path=spec or world["spec"], looks=looks or world["looks"],
                    now_ms=now if now is not None else end + 2 * DAY, code=code, min_universe=min_universe)


# -- the pieces -------------------------------------------------------------------------------------------------
def test_the_segments_partition_the_decision_days_from_the_forward_start():
    segs = pfw.segments(ENDS[:3], 14)
    assert segs[0][0] == T0 and all(a[1] == b[0] for a, b in zip(segs, segs[1:]))
    assert [b for _, b in segs] == [e - 14 * DAY for e in ENDS[:3]]


def test_the_measures_are_the_development_ones():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y, p = (rng.random(200) > 0.6).astype(float), rng.random(200)
    t = np.repeat(np.arange(20, dtype=np.int64) * DAY, 10)
    assert pfw.score("trend_14", y, p, None, t) == pytest.approx(roc_auc_score(y, p))
    assert pfw.score("trend_14", np.ones(5), p[:5], None, t[:5]) is None                # one class: no AUC
    v, q = rng.normal(size=200), rng.normal(size=200)
    r2 = lambda a, b: 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)  # noqa: E731
    assert pfw.score("fwd_vol_7", v, p, q, t) == pytest.approx(r2(v, p) - r2(v, q))
    assert pfw.score("ret_14_scaled", v, p, None, t) == pytest.approx(ps.daily_rank_ic(t, p, v).mean())
    assert pfw.score("ret_14_scaled", v[:9], p[:9], None, t[:9]) is None                # no day of 10 contracts


def test_look_six_judges_by_the_protocols_rule():
    good = {"all": 0.61, "segments": [0.6, 0.55, 0.51, 0.52, 0.4, None]}
    assert pfw.verdict("trend_14", good) is True
    assert pfw.verdict("trend_14", {**good, "all": 0.59}) is False                       # pooled below 0.60
    assert pfw.verdict("trend_14", {**good, "segments": [0.6, 0.55, 0.51, 0.5, 0.4, None]}) is False   # 3 of 6
    assert pfw.verdict("fwd_vol_7", {"all": 0.01, "segments": [0.1, 0.1, 0.1, 0.1, -1, -1]}) is True
    assert pfw.verdict("fwd_vol_7", {"all": 0.0, "segments": [0.1] * 6}) is False        # the gain must be above 0
    assert pfw.verdict("fwd_vol_14", good) is None and pfw.verdict("ret_14_scaled", good) is None


# -- refusals ---------------------------------------------------------------------------------------------------
def test_a_look_not_registered_or_not_passed_is_refused(world, tmp_path):
    with pytest.raises(PermissionError):
        _look(world, tmp_path / "out", end=ENDS[0] + DAY)
    with pytest.raises(PermissionError):
        _look(world, tmp_path / "out", now=ENDS[0] + DAY - 1)
    assert not (tmp_path / "out").exists()


def test_a_registry_without_six_looks_changed_code_or_models_are_refused(world, tmp_path):
    spec, looks = _spec(tmp_path, ends=ENDS[:5])
    with pytest.raises(ValueError, match="5 looks are registered"):
        _look(world, tmp_path / "out", spec=spec, looks=looks)
    with pytest.raises(ValueError, match="panel_study.py"):
        _look(world, tmp_path / "out", code=(lambda p: b"now", lambda p: b"frozen"))
    (tmp_path / "m").mkdir()
    spec, looks = _spec(tmp_path / "m")
    (tmp_path / "m" / "frozen" / "trend_14_logistic.json").write_text("{}")
    with pytest.raises(ValueError, match="trend_14_logistic.json"):
        _look(world, tmp_path / "out", spec=spec, looks=looks)
    assert not (tmp_path / "out").exists()


def test_missing_hours_or_a_thin_universe_are_refused(world, tmp_path):
    holed = _snapshot(tmp_path / "holed", hole=T0 + 5 * DAY + 7 * H)
    with pytest.raises(ValueError, match="ETHUSDT: 1 hours"):
        _look(world, tmp_path / "out", data=holed)
    with pytest.raises(ValueError, match="12 contracts"):
        _look(world, tmp_path / "out", min_universe=13)


# -- looks ------------------------------------------------------------------------------------------------------
def test_the_first_look_measures_every_model_on_one_segment_and_judges_nothing(world, tmp_path):
    r = _look(world, tmp_path / "out")
    assert (r["look"], r["of"], r["judged"]) == (1, 6, False)
    assert sorted(r["models"]) == sorted(n for n in NAMES if not n.endswith("naive"))
    for name, m in r["models"].items():
        assert len(m["segments"]) == 1 and m["segments"][0] == m["all"] and "holds" not in m
        assert m["rows"] == 12 * (30 - pfw.HORIZON[m["target"]])                         # t < end - label days
        if m["target"].startswith("fwd_vol"):                         # the gain over the frozen naive forecast
            assert isinstance(m["all"], float) and isinstance(m["btc_eth"], float)
            assert m["btc_eth"] != m["all"]                                              # 2 of the 12 contracts
        if name.endswith(("ridge", "logistic")):          # the random-fit LightGBM models predict a constant here
            assert isinstance(m["all"], float)
    assert r["models"]["ret_14_scaled ridge"]["btc_eth"] is None                         # no day of 10 contracts
    assert r["universe"] == [{"days": 23, "mean": 12.0, "min": 12}]
    assert json.loads((tmp_path / "out" / "result.json").read_text()) == r
    assert "第 1/6 次" in (tmp_path / "out" / "README.md").read_text()
    with pytest.raises(FileExistsError):
        _look(world, tmp_path / "out")


def test_the_sixth_look_has_six_segments_and_a_verdict_for_the_judged_models(world, tmp_path):
    r = _look(world, tmp_path / "out", end=ENDS[5])
    assert (r["look"], r["judged"]) == (6, True)
    for name, m in r["models"].items():
        assert len(m["segments"]) == 6
        assert m["holds"] == pfw.verdict(m["target"], m)
        assert (m["holds"] is None) == (m["target"] not in pfw.JUDGED)
        assert m["rows"] == 12 * (80 - pfw.HORIZON[m["target"]])


def test_bars_from_the_end_on_change_nothing(world, tmp_path):
    other = _snapshot(tmp_path / "other", later=0.5)
    first = _look(world, tmp_path / "a")
    second = _look(world, tmp_path / "b", data=other)
    assert {k: v for k, v in first.items() if k != "data"} == {k: v for k, v in second.items() if k != "data"}
    assert dp._panel_run().LIMIT == ENDS[0]           # the frozen code reads bars only to the end (the protocol)


def test_contracts_the_exchange_now_calls_tradfi_are_left_out_and_listed(world, tmp_path):
    r = _look(world, tmp_path / "out", tradfi=["S09USDT"])
    assert r["tradfi_left_out_new"] == ["S09USDT"] and r["universe"][0]["mean"] == 11.0
