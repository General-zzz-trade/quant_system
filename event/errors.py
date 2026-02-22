from __future__ import annotations

from typing import Optional, Any


# ============================================================
# 根异常（制度级）
# ============================================================

class EventError(Exception):
    """
    EventError —— 事件系统统一异常基类（制度级）

    设计铁律：
    - 不持有 event / handler 等运行期对象
    - 只携带“定位信息”（event_id / event_type / handler_name）
    - 不进入 codec / store / replay
    """

    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
        event_id: Optional[str] = None,
        event_type: Optional[Any] = None,   # 仅用于展示，不强依赖 EventType
        handler_name: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.event_id = event_id
        self.event_type = event_type
        self.handler_name = handler_name

    @property
    def fatal(self) -> bool:
        """
        是否为致命错误（制度级判定）
        - 通过异常类型区分，不使用可变字段
        """
        return isinstance(self, EventFatalError)

    def __str__(self) -> str:
        parts = [self.message]

        if self.event_type is not None:
            parts.append(f"event_type={self.event_type}")

        if self.event_id is not None:
            parts.append(f"event_id={self.event_id}")

        if self.handler_name is not None:
            parts.append(f"handler={self.handler_name}")

        if self.cause is not None:
            parts.append(f"cause={self.cause}")

        return " | ".join(parts)


# ============================================================
# 非致命错误（可恢复 / 可跳过）
# ============================================================

class EventSchemaError(EventError):
    """事件结构/协议错误（不可通过重试修复）"""


class SchemaNotFoundError(EventSchemaError):
    """Schema registry lookup failed"""


class EventValidationError(EventError):
    """事件校验失败（字段/取值不合法）"""


class EventSecurityError(EventError):
    """安全/权限相关错误"""


class EventDispatchError(EventError):
    """分发阶段错误（队列/路由/背压等）"""


class EventHandlerError(EventError):
    """处理器执行错误（业务异常包装）"""


class EventRuntimeError(EventError):
    """运行期通用错误（线程/时钟/内部状态）"""


class EventStoreError(EventError):
    """存储/落盘相关错误"""


class EventReplayError(EventError):
    """回放/重演相关错误"""


class EventCheckpointError(EventError):
    """检查点/快照相关错误"""


# ============================================================
# 致命错误（制度级，不可恢复）
# ============================================================

class EventFatalError(EventError):
    """致命错误：应触发系统级中断/降级策略"""


# ============================================================
# 工具函数
# ============================================================

def root_cause(err: BaseException) -> BaseException:
    """
    递归获取最底层 cause（用于 trace / metrics）
    """
    cur = err
    while isinstance(cur, EventError) and cur.cause is not None:
        cur = cur.cause
    return cur


def is_fatal_error(err: BaseException) -> bool:
    """
    判断是否为致命错误（类型级）
    """
    return isinstance(err, EventFatalError)

