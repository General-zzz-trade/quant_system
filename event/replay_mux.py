# event/replay_mux.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from event.checkpoint import Checkpoint, StreamCursor
from event.errors import EventReplayError
from event.replay import EventReplay


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True, slots=True)
class ReplayMaxConfig:
    """
    ReplayMax 行为配置（冻结）

    约定：
    - ReplayMax 不改变 Replay 语义
    - 只负责批量、节流、checkpoint 触发
    """
    batch_size: int = 1000
    flush_interval_s: float = 0.0
    stop_on_error: bool = True
    checkpoint_every: int = 10_000
    name: str = "replay-max"


# ============================================================
# ReplayMax
# ============================================================

class ReplayMax:
    """
    ReplayMax —— Replay 的批量驱动器（机构级）

    冻结约定：
    - 不直接 emit Event
    - 不直接操作 EventRuntime
    - 不推进 Clock
    - 所有语义均委托给 EventReplay
    """

    def __init__(
        self,
        *,
        replay: EventReplay,
        cfg: Optional[ReplayMaxConfig] = None,
        checkpoint: Optional[Checkpoint] = None,
    ) -> None:
        self._replay = replay
        self._cfg = cfg or ReplayMaxConfig()
        self._checkpoint = checkpoint

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run(
        self,
        *,
        stream_id: str,
        items: Iterable[Any],
        start_index: int = 0,
    ) -> None:
        """
        批量驱动 Replay

        参数：
        - stream_id: 数据流标识（用于 checkpoint）
        - items: 可迭代回放对象（如 HistoricalBar）
        - start_index: 从 checkpoint 恢复时的起始位置
        """
        batch = []
        processed = 0
        index = start_index

        for item in items:
            if index < start_index:
                index += 1
                continue

            batch.append(item)
            index += 1

            if len(batch) >= self._cfg.batch_size:
                self._flush_batch(batch)
                processed += len(batch)
                batch.clear()
                self._maybe_checkpoint(stream_id, index)
                self._maybe_sleep()

        # flush remaining
        if batch:
            self._flush_batch(batch)
            processed += len(batch)
            batch.clear()
            self._maybe_checkpoint(stream_id, index)

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------

    def _flush_batch(self, batch: list[Any]) -> None:
        try:
            # Replay 负责语义一致性（emit / clock / header / lifecycle）
            self._replay.replay_bars(batch)
        except Exception as exc:
            if self._cfg.stop_on_error:
                raise EventReplayError(
                    "ReplayMax batch failed",
                    cause=exc,
                ) from exc
            # 非 stop 模式：直接吞掉，继续跑
            # 这是压力回放工具的明确设计选择

    def _maybe_checkpoint(self, stream_id: str, index: int) -> None:
        if not self._checkpoint:
            return

        if index % self._cfg.checkpoint_every != 0:
            return

        cp = Checkpoint(
            run_id=self._checkpoint.run_id,
            name=self._cfg.name,
            stream_cursors=(
                StreamCursor(
                    stream_id=stream_id,
                    cursor={"index": index},
                ),
            ),
        )
        self._checkpoint.save(cp)

    def _maybe_sleep(self) -> None:
        if self._cfg.flush_interval_s > 0:
            time.sleep(self._cfg.flush_interval_s)
