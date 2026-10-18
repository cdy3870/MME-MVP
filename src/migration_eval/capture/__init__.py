"""Capture: turn provider API traffic into v0 records.

Two entry points share the same provider parsers and the same `normalize` step:

* the recorder (`start()`, `component()`) captures live httpx calls, and
* the importer (`import_jsonl`) normalizes logs a partner already has.
"""

from migration_eval.capture.context import component
from migration_eval.capture.recorder import start, stop

__all__ = ["component", "start", "stop"]
