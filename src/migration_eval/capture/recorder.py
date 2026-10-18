"""The public recorder API: `start()` and `stop()`.

    import migration_eval
    migration_eval.start()          # from here on, provider calls are recorded

    with migration_eval.component("support-triage"):
        client.chat.completions.create(...)
"""

from __future__ import annotations

import os

from migration_eval.capture import httpx_hook
from migration_eval.capture.exchange import Exchange
from migration_eval.capture.normalize import build_record
from migration_eval.capture.sink import JsonlSink

DEFAULT_PATH = "migration_eval_records.jsonl"

_sink: JsonlSink | None = None


def _handle(exchange: Exchange) -> None:
    sink = _sink
    if sink is not None:
        sink.write(build_record(exchange))


def start(path: str | os.PathLike[str] = DEFAULT_PATH) -> None:
    """Record every OpenAI, Anthropic and Gemini call made through httpx to a JSONL file.

    Records are appended, so restarting the app keeps what was captured before.
    """
    global _sink
    stop()
    _sink = JsonlSink(path)
    httpx_hook.install(_handle)


def stop() -> None:
    """Stop recording and restore httpx to its original behaviour."""
    global _sink
    httpx_hook.uninstall()
    if _sink is not None:
        _sink.close()
        _sink = None
