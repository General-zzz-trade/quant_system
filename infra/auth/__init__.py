# infra/auth
"""Authentication and API key management."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class APICredentials:
    """API 凭证 (不可变, 不含 secret 的 repr)。"""
    api_key: str
    api_secret: str
    passphrase: str = ""

    def __repr__(self) -> str:
        masked = self.api_key[:4] + "****" if len(self.api_key) > 4 else "****"
        return f"APICredentials(api_key='{masked}')"


def load_credentials(
    *,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    passphrase: Optional[str] = None,
    env_prefix: str = "EXCHANGE",
) -> APICredentials:
    """从参数或环境变量加载凭证。"""
    import os
    key = api_key or os.environ.get(f"{env_prefix}_API_KEY", "")
    secret = api_secret or os.environ.get(f"{env_prefix}_API_SECRET", "")
    pp = passphrase or os.environ.get(f"{env_prefix}_PASSPHRASE", "")
    if not key or not secret:
        raise ValueError(f"missing API credentials (tried env {env_prefix}_API_KEY/SECRET)")
    return APICredentials(api_key=key, api_secret=secret, passphrase=pp)
