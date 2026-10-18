"""Turn an `Exchange` into a validated v0 `CallRecord`.

This is where the pieces meet:

    provider parsers (dicts shaped like the schema)
        -> generated Pydantic models (validate against the schema)
        -> CallRecord

`build_record` never raises for bad provider data. A call that cannot be parsed still
becomes a record (identity, timing, whatever parsed) with `metadata.parse_error` saying
why, so the grader (REC-4) can report it instead of the call silently vanishing.
"""

from __future__ import annotations

import json
import uuid
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from pydantic import ValidationError

from migration_eval.capture.exchange import Exchange
from migration_eval.capture.providers import ParsedRequest, ParsedResponse, Unsupported
from migration_eval.capture.providers.base import drop_none, error_response
from migration_eval.capture.sse import parse_sse
from migration_eval.models import CallRecord

# Fields kept when a record has to fall back to identity + timing only.
_FALLBACK_KEYS = (
    "schema_version", "id", "component", "provider", "model", "started_at",
    "completed_at", "latency", "source",
)  # fmt: skip


def tool_version() -> str:
    try:
        return version("migration-eval")
    except PackageNotFoundError:
        return "0.0.0"


def _describe(exc: Exception) -> str:
    """A short, content-free description of a parse failure.

    Exception text can echo request content, so only exceptions whose messages we control
    (or that only name a missing key) are included verbatim.
    """
    if isinstance(exc, (Unsupported, KeyError, IndexError, json.JSONDecodeError)):
        return f"{type(exc).__name__}: {exc}"
    return type(exc).__name__


def _load_json(body: Any) -> Any:
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        return json.loads(body)
    return body


def _parse_request(ex: Exchange, errors: list[str]) -> ParsedRequest | None:
    if ex.request_body is None:
        return None
    try:
        body = _load_json(ex.request_body)
        return ex.provider.parse_request(body, ex.url)
    except Exception as exc:  # noqa: BLE001 - provider data is untrusted
        errors.append(f"request: {_describe(exc)}")
        return None


def _is_sse(body: str, content_type: str | None) -> bool:
    return "text/event-stream" in (content_type or "") or body.lstrip().startswith(
        ("data:", "event:")
    )


def _parse_response(ex: Exchange, errors: list[str]) -> ParsedResponse | None:
    if ex.response_body is None:
        return None
    try:
        body = ex.response_body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            if _is_sse(body, ex.content_type):
                return ex.provider.parse_stream(parse_sse(body))
            body = json.loads(body)
        if isinstance(body, list):  # a non-SSE stream is a JSON array of chunks (Gemini)
            return ex.provider.parse_stream(body)
        return ex.provider.parse_response(body)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"response: {_describe(exc)}")
        return None


def _http_error(ex: Exchange) -> ParsedResponse:
    try:
        body = _load_json(ex.response_body) if ex.response_body is not None else None
    except (ValueError, UnicodeDecodeError):
        body = None
    return ParsedResponse(response=error_response(body, ex.status_code or 0), usage={})


def _validation_summary(exc: ValidationError) -> str:
    # Locations and error kinds only: pydantic's message would include the offending values.
    return ", ".join(
        f"{'.'.join(str(p) for p in err['loc'])} ({err['type']})" for err in exc.errors()[:5]
    )


def build_record(ex: Exchange) -> CallRecord:
    errors: list[str] = []
    request = _parse_request(ex, errors)

    if ex.status_code is not None and ex.status_code >= 400:
        parsed = _http_error(ex)
    else:
        parsed = _parse_response(ex, errors)

    response = parsed.response if parsed else None
    usage = parsed.usage if parsed else {}
    if response is not None and not ex.complete:
        # The stream ended early: keep what arrived, but do not present it as a full answer.
        response = {**response, "status": "error", "error": {"type": "stream_closed_early"}}

    model = (
        (request.model if request else None)
        or ex.model_hint
        or (response or {}).get("model")
        or "unknown"
    )

    record = drop_none(
        {
            "schema_version": "v0",
            "id": f"rec_{uuid.uuid4().hex}",
            "component": ex.component,
            "provider": ex.provider.name,
            "model": model,
            "started_at": ex.started_at,
            "completed_at": ex.completed_at,
            "latency": drop_none({"total_ms": ex.total_ms, "time_to_first_token_ms": ex.ttft_ms})
            or None,
            "usage": usage or None,
            "request": request.request if request else None,
            "response": response,
            "metadata": {"parse_error": "; ".join(errors)} if errors else None,
            "source": {"type": ex.source_type, "tool_version": tool_version()},
        }
    )
    try:
        return CallRecord.model_validate(record)
    except ValidationError as exc:
        # A parser produced something the schema rejects. Fall back to identity and timing,
        # keeping token usage if it is valid so the record can still be graded for speed
        # and cost, and drop it only if usage was the invalid part.
        metadata = {
            "parse_error": "; ".join([*errors, f"schema validation failed: {_validation_summary(exc)}"])
        }
        base = {k: record[k] for k in _FALLBACK_KEYS if k in record}
        base["metadata"] = metadata
        try:
            return CallRecord.model_validate({**base, **({"usage": usage} if usage else {})})
        except ValidationError:
            return CallRecord.model_validate(base)
