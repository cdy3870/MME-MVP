"""Types and helpers shared by the provider parsers.

A provider parser has three jobs, all pure functions over already-decoded JSON:

* `parse_request`: provider request body -> `ModelRequest` fields of the v0 schema
* `parse_response`: provider response body -> `ModelResponse` fields plus token usage
* `parse_stream`: the list of streamed events/chunks -> the same thing as `parse_response`

Parsers return plain dicts shaped like the schema. `normalize.build_record` validates
them with the generated Pydantic models, so a parser bug surfaces as a validation error
instead of a silently malformed record.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class Unsupported(ValueError):
    """The call uses a feature v0 cannot represent faithfully (e.g. a server-side tool).

    Raising instead of dropping the feature keeps a replay from silently testing a
    different request than the one that was recorded.
    """


@dataclass(frozen=True)
class ParsedRequest:
    model: str | None
    request: dict[str, Any]  # shaped like ModelRequest


@dataclass(frozen=True)
class ParsedResponse:
    response: dict[str, Any]  # shaped like ModelResponse
    usage: dict[str, int]  # shaped like Usage


@dataclass(frozen=True)
class Provider:
    name: str  # the value stored in record.provider
    matches: Callable[[str], bool]  # does this URL belong to this provider's generate API?
    parse_request: Callable[[dict[str, Any], str], ParsedRequest]  # (body, url)
    parse_response: Callable[[dict[str, Any]], ParsedResponse]
    parse_stream: Callable[[list[Any]], ParsedResponse]


def drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def tool_arguments(raw: Any) -> dict[str, Any]:
    """Normalize tool-call arguments to `{"arguments": {...}}`.

    Providers send arguments as a JSON string (OpenAI) or an object (Anthropic, Gemini).
    If the model emitted JSON that does not parse into an object, keep the original text
    as `arguments_raw` so the tool-call check (EVL-4) can see it.
    """
    if isinstance(raw, dict):
        return {"arguments": raw}
    if raw is None or raw == "":
        return {"arguments": {}}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"arguments_raw": str(raw)}
    return {"arguments": value} if isinstance(value, dict) else {"arguments_raw": str(raw)}


def split_tool_results(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn one user turn into v0 messages.

    Anthropic and Gemini deliver tool results inside a user turn. v0 gives them their own
    `tool` message, placed first (providers require results to come before other content).
    """
    results = [p for p in parts if p["type"] == "tool_result"]
    rest = [p for p in parts if p["type"] != "tool_result"]
    messages: list[dict[str, Any]] = []
    if results:
        messages.append({"role": "tool", "content": results})
    if rest:
        messages.append({"role": "user", "content": rest})
    return messages


def error_response(body: Any, status_code: int) -> dict[str, Any]:
    """Build a `ModelResponse` for an HTTP error from the provider's error body, if any."""
    detail: dict[str, Any] = {"http_status": status_code}
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        kind = err.get("type") or err.get("status") or err.get("code")
        if kind is not None:
            detail["type"] = str(kind)
        if isinstance(err.get("message"), str):
            detail["message"] = err["message"]
    return {"status": "error", "error": detail}
