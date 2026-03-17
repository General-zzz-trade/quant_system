from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from _quant_hotpath import (
    rust_sanitize as _rust_sanitize,
    rust_short_hash as _rust_short_hash,
    rust_make_idempotency_key as _rust_make_idempotency_key,
)


def _sanitize(s: str) -> str:
    return str(_rust_sanitize(s or ""))


def _short_hash(text: str, n: int = 10) -> str:
    return str(_rust_short_hash(text, n))


def make_idempotency_key(*, venue: str, action: str, key: str) -> str:
    """
    稳定幂等键：同输入 => 同输出（用于 retry / reconnect / replay）
    """
    return str(_rust_make_idempotency_key(venue, action, key))


@dataclass(slots=True)
class RequestIdFactory:
    """
    生成交易所 clientOrderId + 内部幂等 key 的统一工厂。

    默认：
    - deterministic=True：相同 logical_id => 相同 client_order_id（机构级重试语义）
    - max_len=36：兼容 Binance 常见限制
    """
    namespace: str = "qsys"
    run_id: str = "local"
    max_len: int = 36
    deterministic: bool = True

    _seq: int = field(default=0, init=False, repr=False)

    def next_nonce(self) -> int:
        self._seq += 1
        return self._seq

    def client_order_id(
        self,
        *,
        strategy: str,
        symbol: str,
        logical_id: Optional[str] = None,
        nonce: Optional[int] = None,
    ) -> str:
        ns = _sanitize(self.namespace).lower()
        run = _sanitize(self.run_id).lower()
        strat = _sanitize(strategy).lower()
        sym = _sanitize(symbol).upper()

        if self.deterministic and logical_id:
            suffix = _short_hash(f"{ns}|{run}|{strat}|{sym}|{logical_id}", n=10)
        else:
            n = nonce if nonce is not None else self.next_nonce()
            suffix = f"n{n:x}"  # hex nonce，短且可读

        # 组装：ns-run-strat-sym-suffix
        raw = f"{ns}-{run}-{strat}-{sym}-{suffix}"
        raw = _sanitize(raw)

        # 过长时：保留 suffix，压缩前缀
        if len(raw) <= self.max_len:
            return raw

        # 强制尾部保留：-suffix
        tail = f"-{suffix}"
        budget = self.max_len - len(tail)
        if budget < 8:
            # 极端情况：直接用 hash 生成短 id
            compact = f"{ns}-{_short_hash(raw, n=max(8, self.max_len - len(ns) - 1))}"
            return compact[: self.max_len]

        head = f"{ns}-{run}-{strat}-{sym}"
        head = _sanitize(head)
        head = head[:budget].rstrip("-")
        return f"{head}{tail}"[: self.max_len]
