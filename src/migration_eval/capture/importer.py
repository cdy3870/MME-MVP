"""Normalize logs a partner already has into v0 records (`migration-eval import jsonl`).

Each line of the input is one JSON object, in one of two shapes:

1. An existing v0 record (it has `schema_version`). It is validated and passed through.
2. A raw exchange: the provider's own request and response bodies plus a little context.

       {
         "provider": "openai",              # or "url": "https://api.openai.com/v1/chat/completions"
         "started_at": "2026-10-12T14:03:21Z",   # ISO 8601, or epoch seconds/milliseconds
         "latency_ms": 888,                 # optional
         "status_code": 200,                # optional; >= 400 is recorded as a failed call
         "model": "gpt-4o-mini",            # optional; needed for Gemini logs without a "url"
         "component": "support-triage",     # optional; overrides --component
         "request": { ...provider request body... },
         "response": { ...provider response body... }   # or the raw SSE text of a stream
       }

Raw exchanges go through the same provider parsers and `build_record` as the live
recorder, so a call produces the same record however it was captured.

Lines that cannot be turned into a record are skipped and reported with a reason; the
import never stops at the first bad line. The exception is a file that mixes schema
versions: that is rejected as a whole (REC-5) before any line is normalized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from migration_eval.capture.context import DEFAULT_COMPONENT
from migration_eval.capture.exchange import Exchange
from migration_eval.capture.file_checks import check_schema_versions
from migration_eval.capture.normalize import build_record
from migration_eval.capture.providers import PROVIDERS, Provider, detect
from migration_eval.models import CallRecord


@dataclass
class ImportResult:
    lines_read: int = 0
    records: list[CallRecord] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)  # (line number, reason)


class _Skip(Exception):
    """This line cannot become a record; the message is the reason shown to the user."""


def _parse_time(value: Any) -> datetime:
    if isinstance(value, bool) or value is None:
        raise _Skip("missing started_at (expected 'started_at' or 'timestamp')")
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value  # milliseconds vs seconds
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            raise _Skip("started_at is not an ISO 8601 timestamp") from None
        # A timestamp with no timezone is read as UTC. Replay only uses gaps between
        # requests, so a constant offset does not change the arrival pattern.
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    raise _Skip("started_at must be a string or a number")


def _provider_for(line: dict[str, Any]) -> Provider:
    name = line.get("provider")
    if name is not None:
        if name not in PROVIDERS:
            raise _Skip(f"unknown provider {name!r} (supported: {', '.join(sorted(PROVIDERS))})")
        return PROVIDERS[name]
    url = line.get("url")
    if isinstance(url, str) and (found := detect(url)):
        return found
    raise _Skip("cannot tell which provider this is from (add 'provider' or a supported 'url')")


def _exchange(line: dict[str, Any], default_component: str) -> Exchange:
    provider = _provider_for(line)
    started_at = _parse_time(line.get("started_at", line.get("timestamp")))
    total_ms = line.get("latency_ms")
    if isinstance(total_ms, bool) or not isinstance(total_ms, (int, float)):
        total_ms = None
    status = line.get("status_code")
    return Exchange(
        provider=provider,
        url=line.get("url") if isinstance(line.get("url"), str) else "",
        started_at=started_at,
        component=line.get("component") or default_component,
        source_type="import_jsonl",
        request_body=line.get("request"),
        response_body=line.get("response"),
        status_code=status if isinstance(status, int) and not isinstance(status, bool) else None,
        completed_at=started_at + timedelta(milliseconds=total_ms) if total_ms is not None else None,
        total_ms=total_ms,
        model_hint=line.get("model") if isinstance(line.get("model"), str) else None,
    )


def _record_for(line: dict[str, Any], default_component: str) -> CallRecord:
    if "schema_version" in line:
        if line["schema_version"] != "v0":
            raise _Skip(f"unsupported schema_version {line['schema_version']!r}")
        try:
            return CallRecord.model_validate(line)
        except ValidationError as exc:
            where = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors()[:3])
            raise _Skip(f"not a valid v0 record (check: {where})") from None
    return build_record(_exchange(line, default_component))


def import_jsonl(path: str | Path, *, component: str | None = None) -> ImportResult:
    """Read `path` and normalize each line. `component` tags lines that do not name one.

    Raises FileRejected if the file mixes schema versions.
    """
    result = ImportResult()
    default_component = component or DEFAULT_COMPONENT

    # Pass 1: parse every line. The version check needs the raw dicts and the whole file
    # at once, so it cannot be done line by line while normalizing.
    parsed: list[tuple[int, dict[str, Any]]] = []
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            result.lines_read += 1
            try:
                line = json.loads(raw)
            except json.JSONDecodeError:
                result.skipped.append((number, "not valid JSON"))
                continue
            if not isinstance(line, dict):
                result.skipped.append((number, "line is not a JSON object"))
                continue
            parsed.append((number, line))

    check_schema_versions(line for _, line in parsed)

    # Pass 2: normalize.
    for number, line in parsed:
        try:
            result.records.append(_record_for(line, default_component))
        except _Skip as skip:
            result.skipped.append((number, str(skip)))

    result.skipped.sort()  # report skipped lines in file order across both passes
    return result
