from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.replay import EventReplay, ReplayConfig


class JsonlEventSource:
    """Reads events from a JSON Lines file, yielding decoded dicts.

    Each line must be a JSON object with at minimum an 'event_type' field.
    Compatible with the EventSource protocol (engine/replay.py).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._count: Optional[int] = None

    def __len__(self) -> int:
        if self._count is None:
            with self._path.open("r") as f:
                self._count = sum(1 for line in f if line.strip())
        return self._count

    def __iter__(self) -> Iterator[Any]:
        try:
            from event.codec import decode_event_json
            use_codec = True
        except Exception:
            use_codec = False

        with self._path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if use_codec:
                    try:
                        yield decode_event_json(line)
                        continue
                    except Exception:
                        pass
                # Fallback: yield raw dict
                yield json.loads(line)


def run_replay(
    *,
    event_log_path: Path,
    symbol: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> int:
    """Replay events from a JSONL event log through the engine coordinator.

    Returns the number of events processed.
    """
    source = JsonlEventSource(event_log_path)

    symbol_default = symbol or "BTCUSDT"

    coordinator = EngineCoordinator(
        cfg=CoordinatorConfig(
            symbol_default=symbol_default.upper(),
            currency="USDT",
        )
    )

    coordinator.start()

    replay = EventReplay(
        dispatcher=coordinator._dispatcher,
        source=source,
        config=ReplayConfig(strict_order=False, actor="replay"),
    )

    processed = replay.run()
    coordinator.stop()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "replay_summary.json"
        summary_path.write_text(
            json.dumps({"events_processed": processed, "source": str(event_log_path)}, indent=2)
        )

    print(f"Replayed {processed} events from {event_log_path}")
    return processed
