"""Test helper: change the paper state a paper directory has saved (S10 part 3).

Since S10 decision 4 the paper state lives in the paper directory's ``state.sqlite``, saved with the order journal;
``paper_state.json`` is a copy that is never read back. A test that needs a saved state of another spec or an old
cursor edits the saved document here, where it used to edit the JSON file. Only the setup changes.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trading.adapters.codec import decode, encode
from trading.adapters.sqlite.schema import STATE_DB


def edit_saved_paper_state(state_dir: Path, change: Callable[[dict[str, Any]], None]) -> None:
    with sqlite3.connect(Path(state_dir) / STATE_DB) as db:
        (text,) = db.execute("SELECT document FROM paper_state WHERE id = 1").fetchone()
        document = json.loads(text, object_hook=decode)
        change(document)
        db.execute("UPDATE paper_state SET document = ? WHERE id = 1", (json.dumps(document, default=encode),))


__all__ = ["edit_saved_paper_state"]
