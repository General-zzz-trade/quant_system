"""The mainnet pilot's start check (tech lead 2026-11-17: "试运行前一键检查"), run through the account check:

    python3 -m scripts.account_check --pilot --spec <signed spec> [--state-dir /var/lib/quant/state/mainnet]

One list of what a pilot start still lacks, each item passed (✓) or not (✗), then "可以启动" or "还缺 N 项":
- the settings: the runtime's own start checks (``account_settings``), mainnet, the pilot's daily limit of 3 %
  (``MAX_DAILY_DRAWDOWN_PCT=3``; the code's default is 5), the signing key set, the alert file's folder writable;
- the signed spec: its signature (the same HMAC-SHA256 as ``infra.model_signing``, the key held in memory only),
  ``passed`` true (a person marks it), and the runtime's parser;
- the state directory: the runtime's start check for the scope (version included);
- the account, read-only: no position and no open or conditional order anywhere (the strategy ledger is empty before
  the first start, so anything held is outside it), one-way position mode, each spec symbol's leverage and margin type
  as the settings say, and a USDT wallet of at least the strategy capital.

Secrets are never printed. Every request is a GET on the account check's allowlist; nothing is changed anywhere.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

FAPI = "https://fapi.binance.com"
PILOT_DAILY_PCT = Decimal("3")          # owner 2026-11-16: the pilot stops for the day at -3 %


def _item(ok: bool, text: str) -> tuple[bool, str]:
    return ok, text


def settings_items(values: Mapping[str, str]) -> list[tuple[bool, str]]:
    from scripts import account_settings
    _, passed = account_settings.report(values)
    items = [_item(passed, "启动时的配置检查（`account_check --settings` 列出原因）")]
    items.append(_item(values.get("BINANCE_TESTNET", "").strip() == "0", "主网：BINANCE_TESTNET=0"))
    daily = values.get("MAX_DAILY_DRAWDOWN_PCT", "").strip()
    try:
        daily_ok = Decimal(daily) == PILOT_DAILY_PCT
    except InvalidOperation:
        daily_ok = False
    items.append(_item(daily_ok, f"日亏限额 MAX_DAILY_DRAWDOWN_PCT=3（现在：{daily or '没设，代码默认 5'}）"))
    items.append(_item(bool(values.get("QUANT_MODEL_SIGN_KEY", "").strip()), "签名密钥 QUANT_MODEL_SIGN_KEY 已设"))
    alert = values.get("ALERT_FILE", "").strip()
    folder = Path(alert).parent if alert else None
    writable = folder is not None and folder.is_dir() and os.access(folder, os.W_OK)
    items.append(_item(writable, f"告警文件的文件夹能写：{folder if folder else 'ALERT_FILE 没设'}"))
    capital = values.get("STRATEGY_CAPITAL_USDT", "").strip()
    items.append(_item(bool(capital), f"策略资金 STRATEGY_CAPITAL_USDT={capital or '没设'}"))
    return items


def spec_items(path: Path, key: str) -> tuple[list[tuple[bool, str]], list[str]]:
    """The spec's items and its symbols (empty when it cannot be read)."""
    path = Path(path)
    if not path.is_file():
        return [_item(False, f"规格文件不存在：{path}")], []
    data = path.read_bytes()
    sig = path.with_suffix(path.suffix + ".sig")
    signed = bool(key) and sig.is_file() and hmac.compare_digest(
        hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest(), sig.read_text().strip())
    items = [_item(signed, f"规格的签名对得上：{sig.name if sig.is_file() else '没有签名文件'}")]
    try:
        document = json.loads(data)
        from trading.domain.strategy.spec import MAINNET_SYMBOLS, parse_spec
        spec = parse_spec(document)
    except Exception as exc:
        return items + [_item(False, f"规格读不出：{type(exc).__name__}: {exc}"[:200])], []
    items.append(_item(document.get("passed") is True, "规格已被技术负责人标记为通过（passed: true）"))
    other = sorted(set(spec.symbols) - MAINNET_SYMBOLS)
    items.append(_item(not other, f"规格只交易 BTCUSDT 和 ETHUSDT（实盘的硬限制；其他："
                                  f"{', '.join(other) if other else '无'}）"))
    return items, list(spec.symbols)


def state_items(state_dir: Path, scope: str) -> list[tuple[bool, str]]:
    from trading.infrastructure.state_dir import verify
    try:
        verify(Path(state_dir), scope)
    except Exception as exc:
        return [_item(False, f"状态目录没通过启动检查：{exc}"[:300])]
    return [_item(True, f"状态目录通过启动检查：{state_dir}（{scope}）")]


def account_items(client: Any, values: Mapping[str, str], symbols: list[str]) -> list[tuple[bool, str]]:
    items = []
    rows = client.get(FAPI, "/fapi/v2/positionRisk")
    held = [f"{r['symbol']} {r['positionAmt']}" for r in rows if Decimal(str(r.get("positionAmt") or 0)) != 0]
    items.append(_item(not held, f"账户上没有任何仓位（现在：{', '.join(held) if held else '无'}）"))
    orders = client.get(FAPI, "/fapi/v1/openOrders")
    algo = client.get(FAPI, "/fapi/v1/openAlgoOrders")
    algo = algo.get("orders", algo) if isinstance(algo, dict) else algo
    items.append(_item(not orders and not algo, f"没有挂单和条件单（现在：挂单 {len(orders)}，条件单 {len(algo)}）"))
    dual = client.get(FAPI, "/fapi/v1/positionSide/dual").get("dualSidePosition")
    items.append(_item(dual is False, f"单向持仓模式（dualSidePosition={dual}）"))
    want_lev = values.get("BINANCE_EXCHANGE_LEVERAGE", "").strip()
    want_margin = values.get("BINANCE_MARGIN_TYPE", "").strip().lower()
    by_symbol = {r["symbol"]: r for r in rows}
    for symbol in symbols:
        row = by_symbol.get(symbol)
        if row is None:
            items.append(_item(False, f"{symbol}：交易所没有返回它的设置"))
            continue
        lev, margin = str(row.get("leverage")), str(row.get("marginType", "")).lower()
        ok = lev == want_lev and margin == want_margin
        items.append(_item(ok, f"{symbol}：杠杆 {lev} 倍、{margin}（设置要求 {want_lev or '没设'} 倍、"
                               f"{want_margin or '没设'}）"))
    account = client.get(FAPI, "/fapi/v2/account")
    wallet = next((Decimal(str(a.get("walletBalance") or 0)) for a in account.get("assets", [])
                   if a.get("asset") == "USDT"), Decimal(0))
    capital = values.get("STRATEGY_CAPITAL_USDT", "").strip()
    try:
        enough = bool(capital) and wallet >= Decimal(capital)
    except InvalidOperation:
        enough = False
    items.append(_item(enough, f"USDT 余额 {wallet} 不少于策略资金 {capital or '没设'}"))
    return items


def run(values: Mapping[str, str], client: Any, *, spec_path: Path, state_dir: Path) -> tuple[list[str], bool]:
    items = settings_items(values)
    found, symbols = spec_items(spec_path, values.get("QUANT_MODEL_SIGN_KEY", "").strip())
    items += found
    items += state_items(state_dir, values.get("RISK_SCOPE", "").strip())
    try:
        items += account_items(client, values, symbols)
    except Exception as exc:
        items.append(_item(False, f"账户读不出：{type(exc).__name__}"))
    missing = sum(1 for ok, _ in items if not ok)
    lines = ["== 试运行启动前的检查（只读；不显示任何密钥）"]
    lines += [f"{'✓' if ok else '✗'} {text}" for ok, text in items]
    done = "可以启动：每一项都通过了。启动仍然由技术负责人决定。"
    lines.append(done if not missing else f"还缺 {missing} 项，见上面的 ✗。")
    return lines, not missing


__all__ = ["PILOT_DAILY_PCT", "account_items", "run", "settings_items", "spec_items", "state_items"]
