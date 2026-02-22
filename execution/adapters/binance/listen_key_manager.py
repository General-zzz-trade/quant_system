# execution/adapters/binance/listen_key_manager.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


class Clock(Protocol):
    def now(self) -> float: ...


@dataclass(slots=True)
class ListenKeyManagerConfig:
    validity_sec: float = 60.0 * 60.0          # 60 min :contentReference[oaicite:7]{index=7}
    renew_margin_sec: float = 5.0 * 60.0       # 提前 5 min keepalive（保守）
    recreate_backoff_sec: float = 1.0          # 重建失败后最小退避


@dataclass(slots=True)
class BinanceUmListenKeyManager:
    client: any
    clock: Clock
    cfg: ListenKeyManagerConfig = field(default_factory=ListenKeyManagerConfig)

    listen_key: Optional[str] = None
    _expires_at: float = 0.0
    _next_allowed_action_at: float = 0.0

    def _set_fresh(self, lk: str) -> None:
        now = self.clock.now()
        self.listen_key = lk
        self._expires_at = now + self.cfg.validity_sec

    def ensure(self) -> str:
        """
        确保有可用 listenKey；必要时创建。
        """
        now = self.clock.now()
        if now < self._next_allowed_action_at:
            # 防抖：避免连续失败导致的 tight loop
            if self.listen_key:
                return self.listen_key
            raise RuntimeError("listenKey unavailable (backoff)")

        if not self.listen_key:
            lk = self.client.create()
            self._set_fresh(lk)
            return lk

        # 如果已经过期，直接重建
        if now >= self._expires_at:
            lk = self.client.create()
            self._set_fresh(lk)
            return lk

        return self.listen_key

    def tick(self) -> Optional[str]:
        """
        周期调用：接近过期则 keepalive；
        keepalive 发现 listenKey 不存在则重建。
        返回最新 listenKey（若发生变化或刚创建），否则 None。
        """
        now = self.clock.now()

        # 没有 key → 创建
        if not self.listen_key:
            try:
                lk = self.ensure()
                return lk
            except Exception:
                self._next_allowed_action_at = now + self.cfg.recreate_backoff_sec
                return None

        # 还没到 renew 窗口
        if now < (self._expires_at - self.cfg.renew_margin_sec):
            return None

        # 到 renew 窗口：keepalive
        try:
            new_lk = self.client.keepalive(self.listen_key)
            # keepalive 也会延长有效期（按 60 分钟）:contentReference[oaicite:8]{index=8}
            self._set_fresh(new_lk)
            return new_lk if new_lk != self.listen_key else new_lk
        except Exception as e:
            # -1125 或类似 → 重建 :contentReference[oaicite:9]{index=9}
            if hasattr(self.client, "is_listen_key_missing_error") and self.client.is_listen_key_missing_error(e):
                try:
                    lk = self.client.create()
                    self._set_fresh(lk)
                    return lk
                except Exception:
                    self._next_allowed_action_at = now + self.cfg.recreate_backoff_sec
                    return None

            # 其他错误：短退避，下次再试（不立刻丢 key）
            self._next_allowed_action_at = now + self.cfg.recreate_backoff_sec
            return None
