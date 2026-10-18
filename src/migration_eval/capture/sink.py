"""Where recorded calls go: an append-only JSONL file, one v0 record per line."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from migration_eval.models import CallRecord


def record_to_json(record: CallRecord) -> str:
    # exclude_none keeps lines small; by_alias writes response_format.schema as "schema".
    return record.model_dump_json(exclude_none=True, by_alias=True)


class JsonlSink:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")  # append: restarts keep old records
        self._lock = threading.Lock()

    def write(self, record: CallRecord) -> None:
        line = record_to_json(record)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()  # a crash right after a call should not lose that record

    def close(self) -> None:
        with self._lock:
            self._file.close()
