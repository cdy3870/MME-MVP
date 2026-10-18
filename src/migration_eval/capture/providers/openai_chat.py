"""OpenAI Chat Completions (`POST /v1/chat/completions`), including streaming.

Not covered yet: the newer Responses API (`/v1/responses`).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from migration_eval.capture.providers.base import (
    ParsedRequest,
    ParsedResponse,
    Provider,
    Unsupported,
    drop_none,
    tool_arguments,
)

NAME = "openai"

_FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}

# Request keys that are mapped explicitly; everything else lands in parameters.extra.
_KNOWN_REQUEST_KEYS = {
    "model", "messages", "temperature", "top_p", "seed", "max_tokens",
    "max_completion_tokens", "stop", "response_format", "tools", "tool_choice", "stream",
}  # fmt: skip


def matches(url: str) -> bool:
    parts = urlsplit(url)
    return parts.hostname == "api.openai.com" and parts.path.rstrip("/").endswith(
        "/chat/completions"
    )


# --- request -----------------------------------------------------------------------------


def _image(url: str) -> dict[str, Any]:
    if url.startswith("data:") and ";base64," in url:
        header, data = url[len("data:") :].split(";base64,", 1)
        return drop_none({"type": "image", "media_type": header or None, "data": data})
    return {"type": "image", "url": url}


def _content(content: Any) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": part["text"]})
        elif kind == "image_url":
            parts.append(_image(part["image_url"]["url"]))
        else:
            raise Unsupported(f"content part type {kind!r}")
    return parts


def _text_of(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")


def _message(m: dict[str, Any]) -> dict[str, Any]:
    role = m.get("role")
    if role == "developer":  # newer name for the system role
        role = "system"
    if role in ("system", "user"):
        return {"role": role, "content": _content(m.get("content"))}
    if role == "assistant":
        content = _content(m.get("content"))
        calls = m.get("tool_calls") or []
        if not calls:
            return {"role": "assistant", "content": content}
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            if content:
                parts.append({"type": "text", "text": content})
        else:
            parts.extend(content)
        for call in calls:
            fn = call["function"]
            parts.append(
                {
                    "type": "tool_call",
                    "id": call["id"],
                    "name": fn["name"],
                    **tool_arguments(fn.get("arguments")),
                }
            )
        return {"role": "assistant", "content": parts}
    if role == "tool":
        result = {
            "type": "tool_result",
            "tool_call_id": m["tool_call_id"],
            "content": _text_of(_content(m.get("content"))),
        }
        return {"role": "tool", "content": [result]}
    raise Unsupported(f"message role {role!r}")


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tool in tools:
        if tool.get("type") != "function":
            raise Unsupported(f"tool type {tool.get('type')!r}")
        fn = tool["function"]
        out.append(
            drop_none(
                {
                    "name": fn["name"],
                    "description": fn.get("description"),
                    "input_schema": fn.get("parameters"),
                }
            )
        )
    return out


def _tool_choice(choice: Any) -> Any:
    if choice in ("auto", "none", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "tool", "name": choice["function"]["name"]}
    raise Unsupported(f"tool_choice {choice!r}")


def _response_format(rf: dict[str, Any]) -> dict[str, Any]:
    kind = rf.get("type")
    if kind in ("text", "json_object"):
        return {"type": kind}
    if kind == "json_schema":
        return {"type": "json_schema", "schema": rf.get("json_schema", {}).get("schema", {})}
    raise Unsupported(f"response_format {kind!r}")


def parse_request(body: dict[str, Any], url: str) -> ParsedRequest:
    stop = body.get("stop")
    params = drop_none(
        {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "seed": body.get("seed"),
            "max_output_tokens": body.get("max_completion_tokens") or body.get("max_tokens"),
            "stop": [stop] if isinstance(stop, str) else stop,
            "response_format": _response_format(body["response_format"])
            if body.get("response_format")
            else None,
            "extra": {k: v for k, v in body.items() if k not in _KNOWN_REQUEST_KEYS} or None,
        }
    )
    request = drop_none(
        {
            "messages": [_message(m) for m in body.get("messages", [])],
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
    prompt_details = u.get("prompt_tokens_details") or {}
    completion_details = u.get("completion_tokens_details") or {}
    mapped = {
        "input_tokens": u.get("prompt_tokens"),
        "output_tokens": u.get("completion_tokens"),
        "cached_input_tokens": prompt_details.get("cached_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
    }
    return {k: v for k, v in mapped.items() if v is not None}


def _response(
    *,
    id: str | None,
    model: str | None,
    text: str | None,
    calls: list[dict[str, Any]],
    finish_reason: str | None,
) -> dict[str, Any]:
    return drop_none(
        {
            "status": "success",
            "id": id,
            "model": model,
            "text": text,
            "tool_calls": calls or None,
            "finish_reason": _FINISH_REASONS.get(finish_reason or "", "other")
            if finish_reason
            else None,
        }
    )


def parse_response(body: dict[str, Any]) -> ParsedResponse:
    choice = body["choices"][0]
    message = choice.get("message") or {}
    calls = [
        {"id": c["id"], "name": c["function"]["name"], **tool_arguments(c["function"].get("arguments"))}
        for c in message.get("tool_calls") or []
    ]
    content = message.get("content")
    return ParsedResponse(
        response=_response(
            id=body.get("id"),
            model=body.get("model"),
            text=content if isinstance(content, str) else None,
            calls=calls,
            finish_reason=choice.get("finish_reason"),
        ),
        usage=_usage(body.get("usage")),
    )


def parse_stream(chunks: list[Any]) -> ParsedResponse:
    """Merge `chat.completion.chunk` events.

    Text arrives as `delta.content` pieces. Tool calls arrive as fragments keyed by
    `index`: the first fragment carries id and name, later ones append to `arguments`.
    Usage is only present if the caller set `stream_options.include_usage`.
    """
    text: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason = id = model = None
    usage: dict[str, Any] | None = None
    for chunk in chunks:
        id = id or chunk.get("id")
        model = model or chunk.get("model")
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text.append(delta["content"])
            for frag in delta.get("tool_calls") or []:
                call = calls.setdefault(frag.get("index", 0), {"id": None, "name": "", "args": ""})
                call["id"] = frag.get("id") or call["id"]
                fn = frag.get("function") or {}
                call["name"] += fn.get("name") or ""
                call["args"] += fn.get("arguments") or ""
            finish_reason = choice.get("finish_reason") or finish_reason
    tool_calls = [
        drop_none({"id": c["id"], "name": c["name"], **tool_arguments(c["args"])})
        for _, c in sorted(calls.items())
    ]
    return ParsedResponse(
        response=_response(
            id=id,
            model=model,
            text="".join(text) or None,
            calls=tool_calls,
            finish_reason=finish_reason,
        ),
        usage=_usage(usage),
    )


PROVIDER = Provider(
    name=NAME,
    matches=matches,
    parse_request=parse_request,
    parse_response=parse_response,
    parse_stream=parse_stream,
)
