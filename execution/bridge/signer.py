# execution/bridge/signer.py
"""Request signing utilities for venue authentication."""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, Optional, Protocol
from urllib.parse import urlencode

try:
    from _quant_hotpath import rust_hmac_sign as _rust_hmac_sign, rust_hmac_verify as _rust_hmac_verify
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class Signer(Protocol):
    """签名器协议。"""

    def sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """对请求参数签名，返回包含签名的参数。"""
        ...


class HmacSha256Signer:
    """
    HMAC-SHA256 签名器 — 用于 Binance 等交易所。

    自动添加 timestamp 和 signature 字段。
    """

    def __init__(self, secret: str) -> None:
        self._secret = secret.encode("utf-8")

    def sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """对参数进行 HMAC-SHA256 签名。"""
        if _HAS_RUST:
            return self._sign_rust(params)
        return self._sign_py(params)

    def _sign_rust(self, params: Dict[str, Any]) -> Dict[str, Any]:
        items = [(str(k), str(v)) for k, v in params.items()]
        query, sig = _rust_hmac_sign(self._secret.decode("utf-8"), items)
        signed = dict(params)
        # Extract timestamp that Rust may have added
        if "timestamp" not in signed:
            for part in query.split("&"):
                if part.startswith("timestamp="):
                    signed["timestamp"] = int(part.split("=", 1)[1])
                    break
        signed["signature"] = sig
        return signed

    def _sign_py(self, params: Dict[str, Any]) -> Dict[str, Any]:
        signed = dict(params)
        if "timestamp" not in signed:
            signed["timestamp"] = int(time.time() * 1000)
        query_string = urlencode(sorted(signed.items()))
        signature = hmac.new(
            self._secret, query_string.encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        signed["signature"] = signature
        return signed

    @staticmethod
    def verify(
        params: Dict[str, Any],
        secret: str,
    ) -> bool:
        """验证签名是否正确。"""
        p = dict(params)
        sig = p.pop("signature", "")
        query_string = urlencode(sorted(p.items()))
        expected = hmac.new(
            secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(sig, expected)


class NoopSigner:
    """无签名（用于测试/模拟环境）。"""

    def sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return dict(params)
