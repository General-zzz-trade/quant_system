# execution/config/load.py
"""Load execution configuration from dict / YAML-like structure."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Mapping, Optional

from execution.config.venue_config import VenueConfig
from execution.config.retry_config import RetryConfig
from execution.config.reconcile_config import ReconcileConfig
from execution.config.mapping_config import (
    MappingConfig, FieldMapping, StatusMapping, SideMapping,
)


def _dec(v: Any) -> Decimal:
    if v is None:
        return Decimal("0")
    return Decimal(str(v))


def load_venue_config(raw: Mapping[str, Any]) -> VenueConfig:
    """从 dict 加载单个交易所配置。"""
    symbols_raw = raw.get("symbols", ())
    symbols = tuple(str(s).upper() for s in symbols_raw) if symbols_raw else ()

    return VenueConfig(
        name=str(raw["name"]).lower(),
        enabled=bool(raw.get("enabled", True)),
        rest_url=str(raw.get("rest_url", "")),
        ws_url=str(raw.get("ws_url", "")),
        testnet=bool(raw.get("testnet", False)),
        api_key_env=str(raw.get("api_key_env", "")),
        api_secret_env=str(raw.get("api_secret_env", "")),
        rest_rate_per_sec=float(raw.get("rest_rate_per_sec", 10.0)),
        ws_rate_per_sec=float(raw.get("ws_rate_per_sec", 5.0)),
        order_rate_per_sec=float(raw.get("order_rate_per_sec", 10.0)),
        burst=float(raw.get("burst", 20.0)),
        symbols=symbols,
        reduce_only_supported=bool(raw.get("reduce_only_supported", True)),
        post_only_supported=bool(raw.get("post_only_supported", True)),
        hedge_mode=bool(raw.get("hedge_mode", False)),
        connect_timeout_sec=float(raw.get("connect_timeout_sec", 10.0)),
        read_timeout_sec=float(raw.get("read_timeout_sec", 5.0)),
    )


def load_retry_config(raw: Mapping[str, Any]) -> RetryConfig:
    """从 dict 加载重试配置。"""
    codes = raw.get("retryable_status_codes", (408, 429, 500, 502, 503, 504))
    return RetryConfig(
        max_attempts=int(raw.get("max_attempts", 3)),
        base_delay_sec=float(raw.get("base_delay_sec", 0.10)),
        max_delay_sec=float(raw.get("max_delay_sec", 2.00)),
        jitter_sec=float(raw.get("jitter_sec", 0.0)),
        retryable_status_codes=tuple(int(c) for c in codes),
    )


def load_reconcile_config(raw: Mapping[str, Any]) -> ReconcileConfig:
    """从 dict 加载对账配置。"""
    return ReconcileConfig(
        interval_sec=float(raw.get("interval_sec", 60.0)),
        qty_tolerance=_dec(raw.get("qty_tolerance", "0.00001")),
        price_tolerance_pct=_dec(raw.get("price_tolerance_pct", "0.001")),
        auto_fix_info=bool(raw.get("auto_fix_info", False)),
        auto_fix_warning=bool(raw.get("auto_fix_warning", False)),
        auto_fix_critical=bool(raw.get("auto_fix_critical", False)),
        halt_on_critical=bool(raw.get("halt_on_critical", True)),
        max_consecutive_failures=int(raw.get("max_consecutive_failures", 3)),
        timeout_sec=float(raw.get("timeout_sec", 30.0)),
    )


def load_execution_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """
    加载完整的执行层配置。

    输入 dict 结构：
    {
        "venues": { "binance": {...}, ... },
        "retry": {...},
        "reconcile": {...},
    }

    返回解析后的配置对象 dict。
    """
    result: Dict[str, Any] = {}

    # venues
    venues_raw = raw.get("venues", {})
    venues: Dict[str, VenueConfig] = {}
    for name, vcfg in venues_raw.items():
        vcfg_with_name = {**vcfg, "name": name}
        venues[name] = load_venue_config(vcfg_with_name)
    result["venues"] = venues

    # retry
    retry_raw = raw.get("retry", {})
    result["retry"] = load_retry_config(retry_raw)

    # reconcile
    reconcile_raw = raw.get("reconcile", {})
    result["reconcile"] = load_reconcile_config(reconcile_raw)

    return result
