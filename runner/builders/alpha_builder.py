"""Builder and balance helpers for alpha_main coordinator construction."""
from __future__ import annotations

import bisect
import csv
import logging
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from _quant_hotpath import RustInferenceBridge

if TYPE_CHECKING:
    from data.oi_cache import BinanceOICache

from decision.modules.alpha import AlphaDecisionModule
from decision.signals.alpha_signal import EnsemblePredictor, SignalDiscretizer
from decision.sizing.adaptive import AdaptivePositionSizer
from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.feature_hook import FeatureComputeHook
from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter
from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
from execution.adapters.okx.execution_adapter import OkxExecutionAdapter
from execution.safety.limits import OrderLimiter, OrderLimitsConfig
from strategy.config import (
    SYMBOL_CONFIG, LEVERAGE_LADDER, OKX_LEVERAGE, BINANCE_LEVERAGE,
)

logger = logging.getLogger(__name__)

MODEL_BASE = Path("models_v8")
DATA_DIR = Path("data_files")


# ── CSV cursor: loads full history so each bar gets its own time-appropriate value ──

class CsvCursor:
    """Stateful cursor over a time-sorted CSV — returns the latest value <= bar_ts.

    This solves the z-score NaN problem: instead of feeding the same latest CSV
    value on every warmup bar (giving std=0 → z=NaN), each bar gets the
    historically correct value, providing variance to the z-score buffer.
    """

    def __init__(self, path: Path, ts_col: str, val_col: str,
                 ts_unit: str = "ms") -> None:
        """Load *path* and index (timestamp → value) pairs sorted by timestamp.

        ts_unit: "ms" (epoch millis), "s" (epoch seconds), or "date" (YYYY-MM-DD).
        """
        self._timestamps: List[int] = []   # epoch-ms, sorted
        self._values: List[float] = []
        self._fallback: float = math.nan
        if not path.exists():
            return
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                rows: List[Tuple[int, float]] = []
                for row in reader:
                    ts_raw = row.get(ts_col, "")
                    val_raw = row.get(val_col, "")
                    if not ts_raw or not val_raw:
                        continue
                    try:
                        val = float(val_raw)
                    except (ValueError, TypeError):
                        continue
                    if math.isnan(val):
                        continue
                    ts_ms = self._parse_ts(ts_raw, ts_unit)
                    if ts_ms is not None:
                        rows.append((ts_ms, val))
                rows.sort(key=lambda r: r[0])
                self._timestamps = [r[0] for r in rows]
                self._values = [r[1] for r in rows]
                if self._values:
                    self._fallback = self._values[-1]
        except Exception:
            logger.warning("CsvCursor: failed to load %s", path, exc_info=True)

    @staticmethod
    def _parse_ts(raw: str, unit: str) -> Optional[int]:
        """Parse a timestamp string to epoch-milliseconds."""
        try:
            if unit == "ms":
                return int(float(raw))
            if unit == "s":
                return int(float(raw) * 1000)
            if unit == "date":
                # "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
                from datetime import datetime as _dt, timezone as _tz
                for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
                    try:
                        d = _dt.strptime(raw.strip()[:19], fmt).replace(tzinfo=_tz.utc)
                        return int(d.timestamp() * 1000)
                    except ValueError:
                        continue
                return None
            # ISO format fallback
            from datetime import datetime as _dt, timezone as _tz
            d = _dt.fromisoformat(raw.replace("Z", "+00:00"))
            return int(d.timestamp() * 1000)
        except Exception:
            return None

    def get(self, bar_ts_ms: int) -> float:
        """Return the latest value whose timestamp <= *bar_ts_ms*.

        If bar_ts_ms is 0 (not set), returns the latest value in the CSV
        (live-trading fallback, same as the old behavior).
        """
        if not self._timestamps:
            return self._fallback
        if bar_ts_ms <= 0:
            return self._fallback
        idx = bisect.bisect_right(self._timestamps, bar_ts_ms) - 1
        if idx < 0:
            return self._values[0]
        return self._values[idx]

    @property
    def loaded(self) -> bool:
        return len(self._timestamps) > 0


class CsvDictCursor:
    """Like CsvCursor but returns a dict of multiple value columns per row."""

    def __init__(self, path: Path, ts_col: str, val_cols: dict[str, str],
                 ts_unit: str = "ms") -> None:
        """val_cols: {output_key: csv_column_name}."""
        self._timestamps: List[int] = []
        self._rows: List[dict[str, float]] = []
        self._fallback: dict[str, float] = {k: math.nan for k in val_cols}
        if not path.exists():
            return
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                entries: List[Tuple[int, dict[str, float]]] = []
                for row in reader:
                    ts_raw = row.get(ts_col, "")
                    if not ts_raw:
                        continue
                    ts_ms = CsvCursor._parse_ts(ts_raw, ts_unit)
                    if ts_ms is None:
                        continue
                    vals: dict[str, float] = {}
                    for out_key, csv_col in val_cols.items():
                        try:
                            vals[out_key] = float(row.get(csv_col, "nan"))
                        except (ValueError, TypeError):
                            vals[out_key] = math.nan
                    entries.append((ts_ms, vals))
                entries.sort(key=lambda r: r[0])
                self._timestamps = [e[0] for e in entries]
                self._rows = [e[1] for e in entries]
                if self._rows:
                    self._fallback = self._rows[-1]
        except Exception:
            logger.warning("CsvDictCursor: failed to load %s", path, exc_info=True)

    def get(self, bar_ts_ms: int) -> dict[str, float]:
        if not self._timestamps:
            return dict(self._fallback)
        if bar_ts_ms <= 0:
            return dict(self._fallback)
        idx = bisect.bisect_right(self._timestamps, bar_ts_ms) - 1
        if idx < 0:
            return dict(self._rows[0])
        return dict(self._rows[idx])

    @property
    def loaded(self) -> bool:
        return len(self._timestamps) > 0


# ── Static data source loaders (from CSV files refreshed by data-refresh timer) ──

def _load_latest_csv_value(path: Path, ts_col: str = "timestamp", val_col: str = None) -> float:
    """Load the latest value from a 2-column CSV (timestamp, value)."""
    if not path.exists():
        return math.nan
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            if last_row is None:
                return math.nan
            if val_col:
                return float(last_row[val_col])
            # Auto-detect: use second column
            cols = list(last_row.keys())
            return float(last_row[cols[1]])
    except Exception:
        return math.nan


def _load_latest_macro(path: Path) -> dict:
    """Load latest macro data from macro_daily.csv."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            if last_row is None:
                return {}
            return {
                "dxy": float(last_row.get("dxy", "nan")),
                "spx": float(last_row.get("spx", last_row.get("spy_close", "nan"))),
                "vix": float(last_row.get("vix", "nan")),
                "date": last_row.get("date", ""),
            }
    except Exception:
        return {}


def _load_latest_onchain(path: Path) -> dict:
    """Load latest on-chain metrics."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            return dict(last_row) if last_row else {}
    except Exception:
        return {}


def _build_data_sources(symbol: str, interval: str = "60",
                        oi_cache: Optional["BinanceOICache"] = None) -> dict:
    """Build all external data source callables for FeatureComputeHook.

    Uses CsvCursor objects to load full CSV history so that during warmup
    each bar gets its historically correct value (different per bar),
    preventing z-score buffers from being filled with identical values
    (which would cause std=0 → z-score=NaN).

    The shared _bar_ts[0] is set by FeatureComputeHook before resolving
    sources each bar, so cursors return the value at that bar's timestamp.
    """
    # Shared mutable timestamp — set by feature_hook before each bar
    _bar_ts: list[int] = [0]

    def _set_bar_ts(ts_ms: int) -> None:
        _bar_ts[0] = ts_ms

    sources: dict[str, Any] = {}
    sources["_set_bar_ts"] = _set_bar_ts

    # ── Funding rate (8h intervals, ts in epoch-ms) ──
    funding_path = DATA_DIR / f"{symbol}_funding.csv"
    if funding_path.exists():
        _funding_cursor = CsvCursor(funding_path, "timestamp", "funding_rate", ts_unit="ms")
        if _funding_cursor.loaded:
            sources["funding_rate_source"] = lambda: _funding_cursor.get(_bar_ts[0])
            logger.debug("CsvCursor: funding_rate loaded %d rows", len(_funding_cursor._timestamps))

    # Fix mixed-format OI CSV before loading
    try:
        from scripts.run_full_backtest import _fix_oi_file
        _fix_oi_file(symbol)
    except Exception:
        pass

    # ── Open interest (hourly, ts in epoch-ms) ──
    # CSV cursor as baseline (needed for warmup bars with historical timestamps)
    _oi_csv_cursor: CsvCursor | None = None
    oi_path = DATA_DIR / f"{symbol}_open_interest.csv"
    if oi_path.exists():
        _oi_csv_cursor = CsvCursor(oi_path, "timestamp", "sum_open_interest", ts_unit="ms")
        if not _oi_csv_cursor.loaded:
            _oi_csv_cursor = None

    # Live OI from BinanceOICache (55s refresh) with CSV fallback
    if oi_cache is not None:
        def _oi_live(_cache=oi_cache, _csv=_oi_csv_cursor):
            # During warmup (_bar_ts > 0 and historical), use CSV for correct values
            ts = _bar_ts[0]
            if ts > 0 and _csv is not None:
                import time as _t
                now_ms = int(_t.time() * 1000)
                # If bar is more than 5 minutes old, it's a warmup bar — use CSV
                if now_ms - ts > 300_000:
                    return _csv.get(ts)
            # Live bar: prefer cache
            cached = _cache.get()
            oi_val = cached.get("open_interest", math.nan)
            if not math.isnan(oi_val) and oi_val > 0:
                return oi_val
            # Fallback to CSV latest
            if _csv is not None:
                return _csv.get(ts)
            return math.nan
        sources["oi_source"] = _oi_live
        logger.info("OI source: BinanceOICache (live, 55s) + CSV fallback for %s", symbol)
    elif _oi_csv_cursor is not None:
        sources["oi_source"] = lambda: _oi_csv_cursor.get(_bar_ts[0])

    # ── Long/short ratio (hourly, ts in epoch-ms) ──
    # CSV cursor as baseline
    _ls_csv_cursor: CsvCursor | None = None
    ls_path = DATA_DIR / f"{symbol}_ls_ratio.csv"
    if ls_path.exists():
        _ls_csv_cursor = CsvCursor(ls_path, "timestamp", "long_short_ratio", ts_unit="ms")
        if not _ls_csv_cursor.loaded:
            _ls_csv_cursor = None

    # Live LS ratio from BinanceOICache with CSV fallback
    if oi_cache is not None:
        def _ls_live(_cache=oi_cache, _csv=_ls_csv_cursor):
            ts = _bar_ts[0]
            if ts > 0 and _csv is not None:
                import time as _t
                now_ms = int(_t.time() * 1000)
                if now_ms - ts > 300_000:
                    return _csv.get(ts)
            cached = _cache.get()
            ls_val = cached.get("ls_ratio", math.nan)
            if not math.isnan(ls_val):
                return ls_val
            if _csv is not None:
                return _csv.get(ts)
            return math.nan
        sources["ls_ratio_source"] = _ls_live
        logger.info("LS ratio source: BinanceOICache (live, 55s) + CSV fallback for %s", symbol)
    elif _ls_csv_cursor is not None:
        sources["ls_ratio_source"] = lambda: _ls_csv_cursor.get(_bar_ts[0])

    # ── Spot close (hourly, ts in epoch-ms column "open_time") ──
    spot_path = DATA_DIR / f"{symbol}_spot_1h.csv"
    if spot_path.exists():
        _spot_cursor = CsvCursor(spot_path, "open_time", "close", ts_unit="ms")
        if _spot_cursor.loaded:
            sources["spot_close_source"] = lambda: _spot_cursor.get(_bar_ts[0])

    # ── Fear & Greed Index (daily, ts in epoch-seconds) ──
    fgi_path = DATA_DIR / "fear_greed_index.csv"
    if fgi_path.exists():
        _fgi_cursor = CsvCursor(fgi_path, "timestamp", "value", ts_unit="ms")
        if _fgi_cursor.loaded:
            sources["fgi_source"] = lambda: _fgi_cursor.get(_bar_ts[0])

    # ── Implied volatility (Deribit, ISO timestamps) ──
    iv_path = DATA_DIR / f"{symbol}_deribit_iv.csv"
    if iv_path.exists():
        _iv_cursor = CsvCursor(iv_path, "timestamp", "implied_vol", ts_unit="iso")
        if _iv_cursor.loaded:
            sources["implied_vol_source"] = lambda: _iv_cursor.get(_bar_ts[0])

    # ── Put/call ratio ──
    pcr_path = DATA_DIR / f"{symbol}_deribit_pcr.csv"
    if not pcr_path.exists():
        pcr_path = DATA_DIR / f"{symbol}_deribit_iv.csv"
    if pcr_path.exists():
        _pcr_cursor = CsvCursor(pcr_path, "timestamp", "put_call_ratio", ts_unit="iso")
        if _pcr_cursor.loaded:
            sources["put_call_ratio_source"] = lambda: _pcr_cursor.get(_bar_ts[0])

    # ── On-chain metrics (daily, date column) ──
    sym_lower = symbol.replace("USDT", "").lower()
    onchain_path = DATA_DIR / f"{sym_lower}_onchain_daily.csv"
    if onchain_path.exists():
        _onchain_cursor = CsvDictCursor(
            onchain_path, "date",
            {
                "FlowInExUSD": "exchange_inflow",
                "FlowOutExUSD": "exchange_outflow",
                "SplyExNtv": "exchange_reserve",
                "_netflow": "exchange_netflow",
            },
            ts_unit="date",
        )
        if _onchain_cursor.loaded:
            def _onchain_at_ts(_c=_onchain_cursor):
                row = _c.get(_bar_ts[0])
                flow_in = row.get("FlowInExUSD", math.nan)
                flow_out = row.get("FlowOutExUSD", math.nan)
                netflow = row.pop("_netflow", math.nan)
                # Proxy: AdrActCnt ~ abs(netflow) (activity proportional to net movement)
                row["AdrActCnt"] = abs(netflow) if not math.isnan(netflow) else math.nan
                # Proxy: TxTfrCnt ~ inflow + outflow (total flow volume as tx proxy)
                if not math.isnan(flow_in) and not math.isnan(flow_out):
                    row["TxTfrCnt"] = flow_in + flow_out
                else:
                    row["TxTfrCnt"] = math.nan
                row["HashRate"] = math.nan
                return row
            sources["onchain_source"] = _onchain_at_ts

    # ── Liquidation proxy (hourly, ts in epoch-ms) ──
    liq_path = DATA_DIR / f"{symbol}_liquidation_proxy.csv"
    if liq_path.exists():
        _liq_cursor = CsvDictCursor(
            liq_path, "ts",
            {
                "liq_total_volume": "liq_proxy_volume",
                "liq_buy_volume": "liq_proxy_buy",
                "liq_sell_volume": "liq_proxy_sell",
            },
            ts_unit="ms",
        )
        if _liq_cursor.loaded:
            def _liq_at_ts(_c=_liq_cursor):
                row = _c.get(_bar_ts[0])
                row["liq_count"] = 0.0
                return row
            sources["liquidation_source"] = _liq_at_ts

    # ── Macro (fred_macro.csv or individual ETF files) ──
    macro_path = DATA_DIR / "fred_macro.csv"
    if not macro_path.exists() or macro_path.stat().st_size < 50:
        spy_path = DATA_DIR / "macro" / "SPY_daily.csv"
        vix_path = DATA_DIR / "macro" / "VIX_daily.csv"
        if spy_path.exists():
            _spy_macro_cursor = CsvCursor(spy_path, "date", "close", ts_unit="date")
            _vix_macro_cursor = CsvCursor(vix_path, "date", "close", ts_unit="date") if vix_path.exists() else None

            def _macro_from_etfs():
                result = {"dxy": math.nan, "spx": math.nan, "vix": math.nan, "date": ""}
                result["spx"] = _spy_macro_cursor.get(_bar_ts[0])
                if _vix_macro_cursor is not None:
                    result["vix"] = _vix_macro_cursor.get(_bar_ts[0])
                return result
            sources["macro_source"] = _macro_from_etfs
    else:
        sources["macro_source"] = lambda _p=macro_path: _load_latest_macro(_p)

    # ── Cross-market ETF data (daily, date column) ──
    macro_dir = DATA_DIR / "macro"
    _cm_cursors: dict[str, CsvCursor] = {}
    _etf_map = {
        "spy_close": "SPY_daily.csv",
        "tlt_close": "TLT_daily.csv",
        "uso_close": "USO_daily.csv",
        "xlf_close": "GLD_daily.csv",      # GLD as proxy for XLF
        "ethe_close": "COIN_daily.csv",     # COIN as proxy for ETHE
        "gbtc_vol": "COIN_daily.csv",       # COIN volume as proxy for GBTC vol
    }
    for key, fname in _etf_map.items():
        path = macro_dir / fname
        if path.exists():
            val_col = "volume" if "vol" in key else "close"
            cm = CsvCursor(path, "date", val_col, ts_unit="date")
            if cm.loaded:
                _cm_cursors[key] = cm

    # Treasury 10Y from VIX as proxy (VIX correlates with treasury moves)
    _vix_cm = _cm_cursors.get("_vix")
    vix_path = macro_dir / "VIX_daily.csv"
    if vix_path.exists():
        _vix_cm = CsvCursor(vix_path, "date", "close", ts_unit="date")
        if _vix_cm and _vix_cm.loaded:
            _cm_cursors["treasury_10y"] = _vix_cm  # VIX as treasury proxy

    # USDT dominance — try dedicated CSV, then UUP (US Dollar ETF) as proxy
    _usdt_dom_cm = None
    _usdt_paths = [
        DATA_DIR / "usdt_dominance.csv",
        macro_dir / "usdt_dominance.csv",
    ]
    for _ud_path in _usdt_paths:
        if _ud_path.exists():
            try:
                _usdt_dom_cm = CsvCursor(_ud_path, "date", "value", ts_unit="date")
                if _usdt_dom_cm.loaded:
                    logger.info("Loaded usdt_dominance from %s", _ud_path)
                    break
                _usdt_dom_cm = None
            except Exception:
                _usdt_dom_cm = None
    # Fallback: UUP (US Dollar ETF) as proxy — higher USD → higher USDT dominance
    if _usdt_dom_cm is None:
        uup_path = macro_dir / "UUP_daily.csv"
        if uup_path.exists():
            _usdt_dom_cm = CsvCursor(uup_path, "date", "close", ts_unit="date")
            if _usdt_dom_cm and _usdt_dom_cm.loaded:
                logger.info("Using UUP as usdt_dominance proxy from %s", uup_path)

    if _cm_cursors:
        def _cross_market_at_ts():
            result: dict[str, float] = {}
            ts = _bar_ts[0]
            for key, cursor in _cm_cursors.items():
                if key == "treasury_10y":
                    result["treasury_10y"] = cursor.get(ts)
                elif key != "_vix":
                    result[key] = cursor.get(ts)
            if _usdt_dom_cm is not None and _usdt_dom_cm.loaded:
                result["usdt_dominance"] = _usdt_dom_cm.get(ts)
            return result

        sources["cross_market_source"] = _cross_market_at_ts

    # ── Pre-computed cross-market FEATURES (spy_ret_1d, vix_level, etc.) ──
    # These override Rust push_cross_market() outputs which are 0 (same daily price
    # pushed every hourly bar → return=0). Batch uses cross_market_daily.csv directly.
    cm_feat_path = DATA_DIR / "cross_market_daily.csv"
    if cm_feat_path.exists():
        _cm_feature_cols = {
            "spy_ret_1d": "spy_ret_1d",
            "spy_extreme": "spy_extreme",
            "tlt_ret_5d": "tlt_ret_5d",
            "uso_ret_5d": "uso_ret_5d",
            "xlf_ret_5d": "xlf_ret_5d",
            "treasury_10y_chg_5d": "treasury_10y_chg_5d",
            "vix_level": "vix_level",
            "ethe_ret_1d": "ethe_ret_1d",
            "gbtc_ret_1d": "gbtc_ret_1d",
            "coin_ret_1d": "coin_ret_1d",
            "gld_ret_5d": "gld_ret_5d",
            # V12+ trad-fi risk factors
            "hyg_ret_1d": "hyg_ret_1d",
            "hyg_ret_5d": "hyg_ret_5d",
            "credit_spread_chg": "credit_spread_chg",
            "iwm_ret_1d": "iwm_ret_1d",
            "risk_appetite": "risk_appetite",
            "xlk_ret_1d": "xlk_ret_1d",
            "vix_chg_1d": "vix_chg_1d",
            "vix_chg_5d": "vix_chg_5d",
            "fxi_ret_1d": "fxi_ret_1d",
        }
        _cm_feat_cursor = CsvDictCursor(cm_feat_path, "date", _cm_feature_cols, ts_unit="date")
        if _cm_feat_cursor.loaded:
            sources["cross_market_features_source"] = lambda: _cm_feat_cursor.get(_bar_ts[0])
            logger.info("Loaded cross_market_daily.csv: %d rows, %d features",
                         len(_cm_feat_cursor._timestamps), len(_cm_feature_cols))

    # ── Stablecoin supply (stablecoin_daily.csv) ──
    stablecoin_path = DATA_DIR / "stablecoin_daily.csv"
    if stablecoin_path.exists():
        _stablecoin_cursor = CsvCursor(stablecoin_path, "date", "total_supply", ts_unit="date")
        if _stablecoin_cursor.loaded:
            def _stablecoin_chg_7d():
                ts = _bar_ts[0]
                current = _stablecoin_cursor.get(ts)
                ts_7d = ts - 7 * 86400 * 1000
                old = _stablecoin_cursor.get(ts_7d)
                if old > 0 and not math.isnan(current) and not math.isnan(old):
                    return (current - old) / old
                return 0.0
            sources["stablecoin_source"] = _stablecoin_chg_7d

    # ── Counterpart close prices for dominance features ──
    # BTC runners need ETH close, ETH runners need BTC close (time-synced for warmup)
    if "BTC" in symbol:
        counterpart_path = DATA_DIR / "ETHUSDT_1h.csv"
    else:
        counterpart_path = DATA_DIR / "BTCUSDT_1h.csv"
    if counterpart_path.exists():
        _cp_cursor = CsvCursor(counterpart_path, "open_time", "close", ts_unit="ms")
        if _cp_cursor.loaded:
            counterpart_sym = "ETHUSDT" if "BTC" in symbol else "BTCUSDT"
            sources["counterpart_close_source"] = lambda: (_cp_cursor.get(_bar_ts[0]), counterpart_sym)
            logger.info("Counterpart close source: %s (%d rows) for %s dominance",
                        counterpart_path.name, len(_cp_cursor._timestamps), symbol)

    # ── Taker data (taker_buy_volume, trades, taker_buy_quote_volume from kline CSV) ──
    # Bybit WS/REST klines do NOT provide taker fields, so we load from local CSV.
    # For 4h bars, aggregate 4 consecutive 1h bars.
    if interval == "15":
        taker_csv = DATA_DIR / f"{symbol}_15m.csv"
    else:
        taker_csv = DATA_DIR / f"{symbol}_1h.csv"
    if taker_csv.exists():
        _taker_cursor = CsvDictCursor(
            taker_csv, "open_time",
            {
                "taker_buy_volume": "taker_buy_volume",
                "trades": "trades",
                "taker_buy_quote_volume": "taker_buy_quote_volume",
            },
            ts_unit="ms",
        )
        if _taker_cursor.loaded:
            if interval == "240":
                # 4h bars: aggregate 4 consecutive 1h bars
                # Build aggregated index once at load time
                _agg: dict[int, dict[str, float]] = {}
                for i, ts_ms in enumerate(_taker_cursor._timestamps):
                    ts_sec = ts_ms // 1000
                    hour_of_day = (ts_sec % 86400) // 3600
                    boundary_hour = (hour_of_day // 4) * 4
                    boundary_ms = ((ts_sec // 86400) * 86400 + boundary_hour * 3600) * 1000
                    if boundary_ms not in _agg:
                        _agg[boundary_ms] = {"taker_buy_volume": 0.0, "trades": 0.0, "taker_buy_quote_volume": 0.0}
                    row = _taker_cursor._rows[i]
                    for k in ("taker_buy_volume", "trades", "taker_buy_quote_volume"):
                        v = row.get(k, 0.0)
                        if not math.isnan(v):
                            _agg[boundary_ms][k] += v
                # Build sorted lists for bisect lookup
                _agg_ts = sorted(_agg.keys())
                _agg_vals = [_agg[t] for t in _agg_ts]

                def _taker_4h_lookup():
                    ts = _bar_ts[0]
                    if ts <= 0 or not _agg_ts:
                        return {"taker_buy_volume": 0.0, "trades": 0.0, "taker_buy_quote_volume": 0.0}
                    idx = bisect.bisect_right(_agg_ts, ts) - 1
                    if idx < 0:
                        return dict(_agg_vals[0])
                    return dict(_agg_vals[idx])
                sources["taker_source"] = _taker_4h_lookup
                logger.info("Taker source: %s (4h aggregated, %d entries)", taker_csv.name, len(_agg_ts))
            else:
                def _taker_lookup(_c=_taker_cursor):
                    return _c.get(_bar_ts[0])
                sources["taker_source"] = _taker_lookup
                logger.info("Taker source: %s (%d rows)", taker_csv.name, len(_taker_cursor._timestamps))

    return sources


def get_initial_balance(adapter: Any) -> float:
    """Fetch USDT equity from Bybit adapter for tick processor initialization.

    Returns 0.0 on any failure (tick processor will be initialized with zero
    balance and updated from exchange on first bar).
    """
    try:
        snapshot = adapter.get_balances()
        bal = snapshot.get("USDT")
        if bal is not None:
            return float(bal.total)
    except Exception:
        logger.debug("Could not fetch initial balance for tick processor", exc_info=True)
    return 0.0


def build_coordinator(
    symbol: str,
    runner_key: str,
    model_info: dict,
    adapter: Any,
    dry_run: bool = False,
    oi_cache: Optional["BinanceOICache"] = None,
    circuit_breaker: Any = None,
    kill_switch: Any = None,
) -> tuple[EngineCoordinator, AlphaDecisionModule]:
    """Build a full coordinator pipeline for one runner.

    Returns (coordinator, alpha_module) so callers can wire consensus
    and warmup independently.
    """
    cfg = SYMBOL_CONFIG.get(runner_key, {})
    is_4h = "4h" in runner_key

    # External data sources (CSV files refreshed by data-refresh timer)
    interval = cfg.get("interval", "60")
    data_sources = _build_data_sources(symbol, interval=interval, oi_cache=oi_cache)
    n_sources = len(data_sources)
    logger.info("Data sources for %s: %d connected (%s)",
                runner_key, n_sources, ", ".join(sorted(data_sources.keys())))

    # Feature engine (per-symbol Rust instance created lazily by hook)
    feature_hook = FeatureComputeHook(
        computer=None,
        warmup_bars=cfg.get("warmup", 300 if is_4h else 800),
        **data_sources,
    )

    # Inference bridge for z-score normalization + constraints
    bridge = RustInferenceBridge(
        model_info["zscore_window"],
        model_info["zscore_warmup"],
    )

    # Signal pipeline components
    predictor = EnsemblePredictor(
        model_info["horizon_models"],
        model_info["config"],
    )
    discretizer = SignalDiscretizer(
        bridge,
        symbol=symbol,
        deadzone=model_info["deadzone"],
        min_hold=model_info["min_hold"],
        max_hold=model_info["max_hold"],
        long_only=model_info.get("long_only", False),
    )
    sizer = AdaptivePositionSizer(
        runner_key=runner_key,
        step_size=cfg.get("step", 0.001),
        min_size=cfg.get("size", 0.001),
        max_qty=cfg.get("max_qty", 0),
    )

    # Venue-aware leverage resolution (D13).  OKX and Binance each
    # have their own env-var override so a $400 OKX live account can
    # run 10x while Bybit demo keeps its legacy 3x.  Combined with
    # the micro tier_cap (BTC 0.20, ETH 0.10) this gives effective
    # leverage of (2.0x, 1.0x) — the Sharpe-optimal point from D13.
    venue = getattr(adapter, "venue", "binance")
    if venue == "okx":
        leverage = OKX_LEVERAGE
    elif venue == "binance":
        leverage = BINANCE_LEVERAGE
    else:
        leverage = LEVERAGE_LADDER[0][1] if LEVERAGE_LADDER else 10.0
    logger.info("Leverage resolved for %s venue=%s: %.1fx", runner_key, venue, leverage)

    # Decision module (venue stamps audit log + tags downstream events)
    alpha_module = AlphaDecisionModule(
        symbol=symbol,
        runner_key=runner_key,
        predictor=predictor,
        discretizer=discretizer,
        sizer=sizer,
        leverage=leverage,
        signal_only=is_4h,  # 4h runners publish to consensus only, no orders
        venue=venue,
    )

    # Fetch exchange balance for state store initialization
    balance = get_initial_balance(adapter)
    logger.info("Initial balance for %s: $%.2f", runner_key, balance)

    # RustTickProcessor: PERMANENTLY DISABLED (architecture decision, 2026-03)
    #
    # Why not use it:
    #   tick_processor has its own internal z-score buffer that cannot sync with
    #   the Python-side InferenceBridge used by AlphaDecisionModule.decide().
    #   When enabled, decide() reads z=0 from the Python bridge (empty) while
    #   the tick processor's Rust buffer holds the real z-scores — signals diverge.
    #
    # Why it's not worth fixing:
    #   The Python pipeline already delegates every critical path to Rust:
    #     - RustFeatureEngine push+get:   ~25μs
    #     - EnsemblePredictor predict:    ~34μs  (RustTreePredictor/RustRidgePredictor)
    #     - RustInferenceBridge z-score:   ~2μs
    #     - AdaptivePositionSizer (Rust):  ~1μs
    #     - AlphaDecisionModule.decide():  ~200μs total (incl. Python glue)
    #   vs tick_processor monolithic:      ~80μs
    #   The ~120μs difference does not justify maintaining two signal routing
    #   paths, especially since 10+ external CSV data sources (funding, OI,
    #   on-chain, macro) can only be injected through the Python pipeline.
    tick_proc = None

    # Coordinator config
    coordinator_cfg = CoordinatorConfig(
        symbol_default=symbol,
        symbols=(symbol,),
        currency="USDT",
        feature_hook=feature_hook,
        tick_processor=tick_proc,
        starting_balance=balance,
    )

    # Assemble coordinator
    coordinator = EngineCoordinator(cfg=coordinator_cfg)

    # Attach decision bridge
    decision_bridge = DecisionBridge(
        dispatcher_emit=coordinator.emit,
        modules=[alpha_module],
    )
    coordinator.attach_decision_bridge(decision_bridge)

    # Attach execution bridge (live only)
    if not dry_run:
        venue = getattr(adapter, "venue", "bybit")
        if venue == "binance":
            exec_adapter = BinanceExecutionAdapter(adapter, circuit_breaker=circuit_breaker)
        elif venue == "okx":
            exec_adapter = OkxExecutionAdapter(adapter, circuit_breaker=circuit_breaker)
        else:
            exec_adapter = BybitExecutionAdapter(adapter)

        # Composite risk gate: OrderLimiter + KillSwitch drawdown check
        risk_gate = _build_composite_gate(
            kill_switch=kill_switch,
            venue=getattr(adapter, "venue", "binance"),
        )

        execution_bridge = ExecutionBridge(
            adapter=exec_adapter,
            dispatcher_emit=coordinator.emit,
            risk_gate=risk_gate,
        )
        coordinator.attach_execution_bridge(execution_bridge)

    fast_path = ("RustTickProcessor ENABLED" if tick_proc is not None
                  else "Python pipeline (Rust-accelerated, ~200us/bar)")
    logger.info(
        "Built coordinator: runner_key=%s symbol=%s dry_run=%s warmup=%d path=%s",
        runner_key, symbol, dry_run,
        cfg.get("warmup", 300 if is_4h else 800),
        fast_path,
    )
    return coordinator, alpha_module


# ── OrderLimiter factory ──────────────────────────────────────

class _CompositeRiskGate:
    """Composite risk gate: KillSwitch + OrderLimiter + PortfolioRiskFile.

    ExecutionBridge calls ``risk_gate.check(event)`` and expects an object
    with ``.allowed`` and ``.reason`` attributes.

    Order of checks (fail-fast, cheap → expensive):
      1. KillSwitch  — per-venue daily drawdown cutoff
      2. OrderLimiter — per-venue rate + notional + daily caps
      3. PortfolioRiskFileReader — cross-venue total notional cap
         (fail-open: missing/stale file → allow, log a warning)
    """

    def __init__(
        self,
        limiter: OrderLimiter,
        kill_switch: Any = None,
        portfolio_reader: Any = None,
        venue: str = "binance",
    ) -> None:
        self._limiter = limiter
        self._kill_switch = kill_switch
        self._portfolio_reader = portfolio_reader
        self._venue = venue

    def check(self, event: Any) -> Any:
        from decimal import Decimal as D
        _OK = type("R", (), {"allowed": True, "reason": None})()

        # 1. KillSwitch check (RustKillSwitch.allow_order)
        if self._kill_switch is not None:
            sym = str(getattr(event, "symbol", ""))
            try:
                allowed, reason = self._kill_switch.allow_order(symbol=sym)
                if not allowed:
                    return type("R", (), {"allowed": False, "reason": f"kill_switch: {reason}"})()
            except Exception:
                pass  # KillSwitch failure = allow (fail-open for safety modules)

        # 2. OrderLimiter check (rate + notional)
        qty = getattr(event, "qty", D("0"))
        price = getattr(event, "price", None)
        result = self._limiter.check_order(qty=qty, price=price)
        if not result.allowed:
            return type("R", (), {"allowed": False, "reason": result.detail or result.violated_rule})()

        # 3. Cross-venue portfolio cap (fail-open on missing/stale file)
        if self._portfolio_reader is not None:
            try:
                sym = str(getattr(event, "symbol", ""))
                side = str(getattr(event, "side", "buy")).lower()
                px = float(price) if price is not None else 0.0
                qf = float(qty) if qty is not None else 0.0
                # Only gate new directional exposure — reduce_only / qty=0 skip
                if qf != 0 and px > 0:
                    proposed_notional = abs(qf) * px
                    ok, reason = self._portfolio_reader.check_would_exceed(
                        symbol=sym,
                        venue=self._venue,
                        side=side,
                        proposed_notional_usd=proposed_notional,
                    )
                    if not ok:
                        return type("R", (), {
                            "allowed": False,
                            "reason": f"portfolio_gate: {reason}",
                        })()
            except Exception:
                pass  # fail-open on reader exception

        return _OK


def _build_composite_gate(
    kill_switch: Any = None,
    venue: str = "binance",
) -> _CompositeRiskGate:
    """Build composite risk gate: OrderLimiter + KillSwitch + PortfolioRiskFile."""
    from decimal import Decimal as D
    from runner.gates.portfolio_risk_file import PortfolioRiskFileReader
    from strategy.config import PORTFOLIO_RISK_MAX_STALE_SEC

    cfg = OrderLimitsConfig(
        max_order_notional=D("25000"),    # $25k max per order
        max_daily_orders=200,             # 200 orders/day max
        max_daily_notional=D("500000"),   # $500k daily notional cap
        max_orders_per_second=2.0,        # 2 orders/sec rate limit
    )
    limiter = OrderLimiter(cfg)
    portfolio_reader = PortfolioRiskFileReader(
        max_stale_seconds=float(PORTFOLIO_RISK_MAX_STALE_SEC),
    )
    gate = _CompositeRiskGate(
        limiter,
        kill_switch=kill_switch,
        portfolio_reader=portfolio_reader,
        venue=venue,
    )
    logger.info(
        "CompositeRiskGate initialized: OrderLimiter(max=$25k, daily=$500k, rate=2/s) "
        "+ KillSwitch(%s) + PortfolioRiskFile(venue=%s)",
        "active" if kill_switch is not None else "none",
        venue,
    )
    return gate
