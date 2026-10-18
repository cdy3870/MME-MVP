"""Gemini `generateContent` / `streamGenerateContent` (Gemini API and Vertex AI REST).

Two quirks compared with the other providers:

* the model name is in the URL path (`.../models/gemini-2.0-flash:generateContent`), and
* tool calls have no ids, so a function response is linked back to its call by name.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

from migration_eval.capture.providers.base import (
    ParsedRequest,
    ParsedResponse,
    Provider,
    Unsupported,
    drop_none,
    split_tool_results,
)

NAME = "gemini"

_MODEL_IN_PATH = re.compile(r"models/([^/:]+):(?:stream)?generateContent", re.IGNORECASE)

_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
}

_ROLES = {"user": "user", "model": "assistant"}

# generationConfig keys mapped explicitly; the rest go to parameters.extra.
_KNOWN_GENERATION_KEYS = {
    "temperature", "topP", "topK", "maxOutputTokens", "stopSequences", "seed",
    "responseMimeType", "responseSchema",
}  # fmt: skip
_KNOWN_REQUEST_KEYS = {
    "contents", "systemInstruction", "generationConfig", "tools", "toolConfig",
}  # fmt: skip


def matches(url: str) -> bool:
    parts = urlsplit(url)
    return bool(
        parts.hostname
        and parts.hostname.endswith("googleapis.com")
        and _MODEL_IN_PATH.search(parts.path)
    )


# --- request -----------------------------------------------------------------------------


def _parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in parts:
        if part.get("thought"):  # thinking summaries echoed back in history
            continue
        if "text" in part:
            out.append({"type": "text", "text": part["text"]})
        elif "inlineData" in part:
            data = part["inlineData"]
            if not data["mimeType"].startswith("image/"):
                raise Unsupported(f"inlineData mimeType {data['mimeType']!r}")
            out.append({"type": "image", "media_type": data["mimeType"], "data": data["data"]})
        elif "fileData" in part:
            data = part["fileData"]
            out.append(
                drop_none({"type": "image", "media_type": data.get("mimeType"), "url": data["fileUri"]})
            )
        elif "functionCall" in part:
            call = part["functionCall"]
            out.append(
                drop_none(
                    {
                        "type": "tool_call",
                        "id": call.get("id"),
                        "name": call["name"],
                        "arguments": call.get("args") or {},
                    }
                )
            )
        elif "functionResponse" in part:
            resp = part["functionResponse"]
            out.append(
                {
                    "type": "tool_result",
                    "tool_call_id": resp.get("id") or resp["name"],
                    "content": json.dumps(resp.get("response"), separators=(",", ":")),
                }
            )
        else:
            raise Unsupported(f"content part keys {sorted(part)}")
    return out


def _messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = (body.get("systemInstruction") or {}).get("parts")
    if system:
        messages.append({"role": "system", "content": "".join(p.get("text", "") for p in system)})
    for content in body.get("contents", []):
        role = _ROLES.get(content.get("role") or "user")
        if role is None:
            raise Unsupported(f"content role {content.get('role')!r}")
        parts = _parts(content.get("parts", []))
        messages.extend(split_tool_results(parts) if role == "user" else [{"role": role, "content": parts}])
    return messages


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tool in tools:
        if set(tool) - {"functionDeclarations", "function_declarations"}:
            raise Unsupported(f"tool kinds {sorted(tool)}")
        for fn in tool.get("functionDeclarations") or tool.get("function_declarations") or []:
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


def _tool_choice(config: dict[str, Any]) -> Any:
    fc = config.get("functionCallingConfig") or {}
    mode, names = fc.get("mode", "AUTO"), fc.get("allowedFunctionNames") or []
    if mode == "AUTO":
        return "auto"
    if mode == "NONE":
        return "none"
    if mode == "ANY":
        if not names:
            return "required"
        if len(names) == 1:
            return {"type": "tool", "name": names[0]}
        raise Unsupported("tool choice restricted to several functions")
    raise Unsupported(f"function calling mode {mode!r}")


def _response_format(gen: dict[str, Any]) -> dict[str, Any] | None:
    if gen.get("responseMimeType") != "application/json":
        return None
    if gen.get("responseSchema"):
        return {"type": "json_schema", "schema": gen["responseSchema"]}
    return {"type": "json_object"}


def parse_request(body: dict[str, Any], url: str) -> ParsedRequest:
    gen = body.get("generationConfig") or {}
    extra = {k: v for k, v in gen.items() if k not in _KNOWN_GENERATION_KEYS}
    extra.update({k: v for k, v in body.items() if k not in _KNOWN_REQUEST_KEYS})
    params = drop_none(
        {
            "temperature": gen.get("temperature"),
            "top_p": gen.get("topP"),
            "top_k": gen.get("topK"),
            "max_output_tokens": gen.get("maxOutputTokens"),
            "stop": gen.get("stopSequences"),
            "seed": gen.get("seed"),
            "response_format": _response_format(gen),
            "extra": extra or None,
        }
    )
    path = urlsplit(url).path
    match = _MODEL_IN_PATH.search(path)
    request = drop_none(
        {
            "messages": _messages(body),
            "parameters": params or None,
            "tools": _tools(body["tools"]) if body.get("tools") else None,
            "tool_choice": _tool_choice(body["toolConfig"]) if body.get("toolConfig") else None,
            "stream": "streamgeneratecontent" in path.lower(),
        }
    )
    return ParsedRequest(model=match.group(1) if match else None, request=request)


# --- response ----------------------------------------------------------------------------


def _usage(u: dict[str, Any] | None) -> dict[str, int]:
    u = u or {}
    mapped = {
        "input_tokens": u.get("promptTokenCount"),
        "output_tokens": u.get("candidatesTokenCount"),
        "cached_input_tokens": u.get("cachedContentTokenCount"),
        "reasoning_tokens": u.get("thoughtsTokenCount"),
    }
    return {k: v for k, v in mapped.items() if v is not None}


def _assemble(
    parts: list[dict[str, Any]],
    *,
    finish: str | None,
    blocked: bool,
    usage: dict[str, Any] | None,
    id: str | None,
    model: str | None,
) -> ParsedResponse:
    text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
    calls = [
        drop_none(
            {
                "id": p["functionCall"].get("id"),
                "name": p["functionCall"]["name"],
                "arguments": p["functionCall"].get("args") or {},
            }
        )
        for p in parts
        if "functionCall" in p
    ]
    if calls and finish in (None, "STOP"):
        reason: str | None = "tool_calls"
    elif blocked:
        reason = "content_filter"
    else:
        reason = _FINISH_REASONS.get(finish, "other") if finish else None
    response = drop_none(
        {
            "status": "success",
            "id": id,
            "model": model,
            "text": text or None,
            "tool_calls": calls or None,
            "finish_reason": reason,
        }
    )
    return ParsedResponse(response=response, usage=_usage(usage))


def _blocked(body: dict[str, Any]) -> bool:
    return bool((body.get("promptFeedback") or {}).get("blockReason"))


def parse_response(body: dict[str, Any]) -> ParsedResponse:
    candidate = (body.get("candidates") or [{}])[0]
    return _assemble(
        (candidate.get("content") or {}).get("parts") or [],
        finish=candidate.get("finishReason"),
        blocked=_blocked(body),
        usage=body.get("usageMetadata"),
        id=body.get("responseId"),
        model=body.get("modelVersion"),
    )


def parse_stream(chunks: list[Any]) -> ParsedResponse:
    """Merge streamed `GenerateContentResponse` chunks (an SSE stream or a JSON array).

    Each chunk carries the next slice of parts; the final chunk carries `finishReason`
    and the cumulative `usageMetadata`.
    """
    parts: list[dict[str, Any]] = []
    finish = id = model = usage = None
    blocked = False
    for chunk in chunks:
        candidate = (chunk.get("candidates") or [{}])[0]
        parts.extend((candidate.get("content") or {}).get("parts") or [])
        finish = candidate.get("finishReason") or finish
        usage = chunk.get("usageMetadata") or usage
        id = chunk.get("responseId") or id
        model = chunk.get("modelVersion") or model
        blocked = blocked or _blocked(chunk)
    return _assemble(parts, finish=finish, blocked=blocked, usage=usage, id=id, model=model)


PROVIDER = Provider(
    name=NAME,
    matches=matches,
    parse_request=parse_request,
    parse_response=parse_response,
    parse_stream=parse_stream,
)
