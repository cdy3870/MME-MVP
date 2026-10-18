"""Anthropic Messages (`POST /v1/messages`), including streaming."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from migration_eval.capture.providers.base import (
    ParsedRequest,
    ParsedResponse,
    Provider,
    Unsupported,
    drop_none,
    split_tool_results,
    tool_arguments,
)

NAME = "anthropic"

_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

_KNOWN_REQUEST_KEYS = {
    "model", "messages", "system", "temperature", "top_p", "top_k", "max_tokens",
    "stop_sequences", "tools", "tool_choice", "stream",
}  # fmt: skip

# Extended-thinking blocks are model-specific and signed, so they cannot be replayed
# against another model. They are dropped rather than rejected.
_IGNORED_BLOCKS = {"thinking", "redacted_thinking"}


def matches(url: str) -> bool:
    parts = urlsplit(url)
    return parts.hostname == "api.anthropic.com" and parts.path.rstrip("/").endswith("/v1/messages")


# --- request -----------------------------------------------------------------------------


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    texts = []
    for block in content or []:
        if block.get("type") != "text":
            raise Unsupported(f"tool_result content block {block.get('type')!r}")
        texts.append(block["text"])
    return "".join(texts)


def _blocks(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            src = block["source"]
            if src.get("type") == "base64":
                parts.append(
                    {"type": "image", "media_type": src["media_type"], "data": src["data"]}
                )
            elif src.get("type") == "url":
                parts.append({"type": "image", "url": src["url"]})
            else:
                raise Unsupported(f"image source {src.get('type')!r}")
        elif kind == "tool_use":
            parts.append(
                {
                    "type": "tool_call",
                    "id": block["id"],
                    "name": block["name"],
                    **tool_arguments(block.get("input")),
                }
            )
        elif kind == "tool_result":
            result = {
                "type": "tool_result",
                "tool_call_id": block["tool_use_id"],
                "content": _tool_result_text(block.get("content")),
            }
            if block.get("is_error") is not None:
                result["is_error"] = bool(block["is_error"])
            parts.append(result)
        elif kind in _IGNORED_BLOCKS:
            continue
        else:
            raise Unsupported(f"content block type {kind!r}")
    return parts


def _messages(m: dict[str, Any]) -> list[dict[str, Any]]:
    role, content = m["role"], m["content"]
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    parts = _blocks(content)
    if role == "user":
        return split_tool_results(parts)
    return [{"role": role, "content": parts}]


def _system(system: Any) -> list[dict[str, Any]]:
    if not system:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    return [{"role": "system", "content": _blocks(system)}]


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tool in tools:
        if tool.get("type", "custom") != "custom":  # e.g. web_search, which runs server-side
            raise Unsupported(f"tool type {tool['type']!r}")
        out.append(
            drop_none(
                {
                    "name": tool["name"],
                    "description": tool.get("description"),
                    "input_schema": tool.get("input_schema"),
                }
            )
        )
    return out


def _tool_choice(choice: dict[str, Any]) -> Any:
    kind = choice.get("type")
    if kind in ("auto", "none"):
        return kind
    if kind == "any":
        return "required"
    if kind == "tool":
        return {"type": "tool", "name": choice["name"]}
    raise Unsupported(f"tool_choice {kind!r}")


def parse_request(body: dict[str, Any], url: str) -> ParsedRequest:
    params = drop_none(
        {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "max_output_tokens": body.get("max_tokens"),
            "stop": body.get("stop_sequences"),
            "extra": {k: v for k, v in body.items() if k not in _KNOWN_REQUEST_KEYS} or None,
        }
    )
    messages = _system(body.get("system"))
    for m in body.get("messages", []):
        messages.extend(_messages(m))
    request = drop_none(
        {
            "messages": messages,
            "parameters": params or None,
            "tools": _tools(body["tools"]) if body.get("tools") else None,
            "tool_choice": _tool_choice(body["tool_choice"]) if body.get("tool_choice") else None,
            "stream": bool(body.get("stream")),
        }
    )
    return ParsedRequest(model=body.get("model"), request=request)


# --- response ----------------------------------------------------------------------------


def _usage(u: dict[str, Any] | None) -> dict[str, int]:
    u = u or {}
    # Anthropic's input_tokens excludes cached tokens; both are reported separately.
    mapped = {
        "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "cached_input_tokens": u.get("cache_read_input_tokens"),
    }
    return {k: v for k, v in mapped.items() if v is not None}


def _assemble(
    blocks: list[dict[str, Any]],
    *,
    id: str | None,
    model: str | None,
    stop_reason: str | None,
    usage: dict[str, Any] | None,
) -> ParsedResponse:
    text = "".join(b["text"] for b in blocks if b["type"] == "text")
    calls = [
        drop_none({"id": b.get("id"), "name": b["name"], **tool_arguments(b.get("input"))})
        for b in blocks
        if b["type"] == "tool_use"
    ]
    response = drop_none(
        {
            "status": "success",
            "id": id,
            "model": model,
            "text": text or None,
            "tool_calls": calls or None,
            "finish_reason": _STOP_REASONS.get(stop_reason, "other") if stop_reason else None,
        }
    )
    return ParsedResponse(response=response, usage=_usage(usage))


def parse_response(body: dict[str, Any]) -> ParsedResponse:
    return _assemble(
        body.get("content") or [],
        id=body.get("id"),
        model=body.get("model"),
        stop_reason=body.get("stop_reason"),
        usage=body.get("usage"),
    )


def parse_stream(events: list[Any]) -> ParsedResponse:
    """Merge Messages stream events.

    `message_start` carries id, model and input usage. Each content block is opened by
    `content_block_start` and filled by `content_block_delta` events: `text_delta` pieces
    for text, `input_json_delta` pieces (a JSON string in fragments) for tool input.
    `message_delta` carries the stop reason and the final output token count.
    """
    blocks: dict[int, dict[str, Any]] = {}
    tool_json: dict[int, str] = {}
    id = model = stop_reason = None
    usage: dict[str, Any] = {}
    for event in events:
        kind = event.get("type")
        if kind == "message_start":
            message = event["message"]
            id, model = message.get("id"), message.get("model")
            usage.update(message.get("usage") or {})
        elif kind == "content_block_start":
            block = dict(event["content_block"])
            if block["type"] == "text":
                block["text"] = block.get("text", "")
            blocks[event["index"]] = block
        elif kind == "content_block_delta":
            delta, index = event["delta"], event["index"]
            if delta.get("type") == "text_delta":
                blocks[index]["text"] += delta["text"]
            elif delta.get("type") == "input_json_delta":
                tool_json[index] = tool_json.get(index, "") + delta["partial_json"]
        elif kind == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
            usage.update(event.get("usage") or {})
        elif kind == "error":
            err = event.get("error") or {}
            return ParsedResponse(
                response={
                    "status": "error",
                    "error": drop_none({"type": err.get("type"), "message": err.get("message")}),
                },
                usage=_usage(usage),
            )
    for index, raw in tool_json.items():
        blocks[index]["input"] = raw  # a JSON string; tool_arguments() parses it in _assemble
    return _assemble(
        [blocks[i] for i in sorted(blocks)], id=id, model=model, stop_reason=stop_reason, usage=usage
    )


PROVIDER = Provider(
    name=NAME,
    matches=matches,
    parse_request=parse_request,
    parse_response=parse_response,
    parse_stream=parse_stream,
)
