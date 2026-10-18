"""The hand-off between "something observed a model call" and "turn it into a record"."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from migration_eval.capture.providers import Provider


@dataclass
class Exchange:
    """One model call as observed, before any provider-specific parsing.

    The recorder fills this in from live httpx traffic; the importer fills it in from a
    log line. Both then go through `normalize.build_record`, so a call is turned into a
    record the same way no matter how it was captured.
    """

    provider: Provider
    url: str
    started_at: datetime  # timezone-aware
    component: str
    source_type: Literal["recorder", "import_jsonl"]
    request_body: bytes | str | dict[str, Any] | None = None
    response_body: bytes | str | dict[str, Any] | list[Any] | None = None
    status_code: int | None = None
    content_type: str | None = None
    completed_at: datetime | None = None
    total_ms: float | None = None
    ttft_ms: float | None = None  # streamed calls only
    complete: bool = True  # False if a stream was closed or broke before it finished
    model_hint: str | None = None  # used when the request body does not name the model
