"""Single writer lease for one account scope on a local persistent disk."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


class ExecutionLease:
    def __init__(self, journal_path: Path, scope: str) -> None:
        key = hashlib.sha256(scope.encode()).hexdigest()[:24]
        path = Path(journal_path).resolve().with_name(f"execution-{key}.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Any = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if path.stat().st_size == 0:
                    self._file.write(b"0")
                    self._file.flush()
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._file.close()
            raise RuntimeError("This account scope already has an execution process on this disk") from exc

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()  # OS releases the lease, including on process death

    def __enter__(self) -> ExecutionLease:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
