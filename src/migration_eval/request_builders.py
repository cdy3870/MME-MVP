"""Request builders (EVL-10): turn a logged v0 request into the request a candidate receives.

Replay fidelity is the product claim, so a candidate should get what the partner really sent:

    * same provider as the logged traffic: rebuild that provider's request body from the v0
      request and change only the model. "Exact" is proved by a round trip: parse the built
      body with the recorder's `parse_request` and check it equals the original v0 request.
    * different provider: convert with the builder for the target provider. Each builder is
      the inverse of that provider's `parse_request` in `capture/providers/`. Anything the
      target cannot carry is listed in `unconverted`, not dropped silently, and the report
      flags results that used a converted request (so a low score can be read as "the model,
      or the conversion").

`make_complete` is the real `CompleteFn` for the harness: it builds the body, sends it and
turns the answer into a `ReplayResult` using the provider's `parse_response`. It never logs
or stores request headers, and API keys come from the caller (read from the environment).

Open decisions (see the notes on EVL-10):
    * the v0 record keeps no raw body, so anything the parsers drop (e.g. Anthropic thinking
      blocks) cannot be reproduced; "exact" means "exact as far as v0 can say"
    * `CompleteFn` receives no record, so it cannot know which provider the traffic came from;
      `make_complete` takes `source_provider` (a component has one incumbent provider)
    * replays are not streamed: `stream` is built into the body (so the round trip is exact)
      and removed before sending, and the round trip ignores it; time to first token is
      therefore not measured
    * OpenAI's token limit is sent as `max_completion_tokens` (accepted by every current chat
      model); the record does not say which name the partner used
    * Gemini puts some settings in `generationConfig` and some at the top level, and v0 merges
      both into `parameters.extra`; a short list of known `generationConfig` keys decides
    * Anthropic and Gemini have no `seed`, and Anthropic Messages has no `response_format`;
      those are listed as unconverted, which can make EVL-4's schema check fail for a reason
      that is not the model's
    * JSON Schema keywords that Gemini's `parameters` subset rejects are not checked
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from migration_eval.capture.providers import PROVIDERS, Unsupported
from migration_eval.capture.providers.base import drop_none
from migration_eval.models.record import ModelRequest
from migration_eval.replay import CompleteFn, RateLimited, ReplayResult, error_result

DEFAULT_ANTHROPIC_MAX_TOKENS = 4096  # Anthropic requires max_tokens; used when the log has none
ANTHROPIC_VERSION = "2023-06-01"

# generationConfig keys (beyond the ones v0 maps) that belong inside generationConfig.
_GEMINI_GENERATION_EXTRA = {
    "candidateCount", "presencePenalty", "frequencyPenalty", "responseLogprobs", "logprobs",
    "thinkingConfig", "responseModalities", "speechConfig", "mediaResolution", "audioTimestamp",
}  # fmt: skip

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass(frozen=True)
class UnconvertedField:
    path: str  # where in the v0 request, e.g. "parameters.extra.logit_bias"
    reason: str  # why the target provider cannot carry it


@dataclass(frozen=True)
class BuiltRequest:
    provider: str  # the provider the body is for
    model: str
    body: dict[str, Any]  # the JSON body to send
    converted: bool  # False for an exact same-provider rebuild
    unconverted: list[UnconvertedField] = field(default_factory=list)


# --- helpers -----------------------------------------------------------------------------


def _v0(request: ModelRequest) -> dict[str, Any]:
    """The request as plain JSON-shaped dicts, the shape the parsers produce."""
    return request.model_dump(mode="json", by_alias=True, exclude_none=True)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")


def _call_id(part: dict[str, Any]) -> str:
    # Gemini calls have no id and its results are linked by function name, so the name stands in.
    return part.get("id") or part["name"]


def _add_extra(body: dict[str, Any], extra: dict[str, Any] | None, exact: bool, lost: list[UnconvertedField]) -> None:
    for key, value in (extra or {}).items():
        if exact:
            body[key] = value
        else:
            lost.append(UnconvertedField(f"parameters.extra.{key}", "provider-specific setting, not sent to another provider"))


def _lost(lost: list[UnconvertedField], path: str, reason: str) -> None:
    lost.append(UnconvertedField(path, reason))


# --- OpenAI Chat Completions -------------------------------------------------------------


def _openai_parts(parts: list[dict[str, Any]], lost: list[UnconvertedField]) -> list[dict[str, Any]]:
    out = []
    for part in parts:
        if part["type"] == "text":
            out.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image" and (part.get("url") or part.get("data")):
            url = part.get("url") or f"data:{part.get('media_type') or ''};base64,{part['data']}"
            out.append({"type": "image_url", "image_url": {"url": url}})
        else:
            _lost(lost, f"messages.content.{part['type']}", "OpenAI user and system messages cannot carry this part")
    return out


def _openai_message(m: dict[str, Any], lost: list[UnconvertedField]) -> list[dict[str, Any]]:
    role, content = m["role"], m["content"]

    if role == "tool":
        if isinstance(content, str):
            _lost(lost, "messages.content", "a tool message without a tool_call_id")
            return []
        messages = []
        for part in content:
            if part.get("is_error"):
                _lost(lost, "messages.content.tool_result.is_error", "OpenAI tool messages have no error flag")
            messages.append({"role": "tool", "tool_call_id": part["tool_call_id"], "content": part["content"]})
        return messages

    if role == "assistant" and not isinstance(content, str):
        texts = [p["text"] for p in content if p["type"] == "text"]
        calls = [p for p in content if p["type"] == "tool_call"]
        for part in content:
            if part["type"] not in ("text", "tool_call"):
                _lost(lost, f"messages.content.{part['type']}", "OpenAI assistant messages cannot carry this part")
        if not calls:
            return [{"role": "assistant", "content": [{"type": "text", "text": t} for t in texts]}]
        tool_calls = []
        for call in calls:
            arguments = json.dumps(call["arguments"]) if "arguments" in call else call.get("arguments_raw", "")
            tool_calls.append(
                {"id": _call_id(call), "type": "function", "function": {"name": call["name"], "arguments": arguments}}
            )
        return [{"role": "assistant", "content": "".join(texts) or None, "tool_calls": tool_calls}]

    if isinstance(content, str):
        return [{"role": role, "content": content}]
    return [{"role": role, "content": _openai_parts(content, lost)}]


def build_openai_chat(model: str, request: ModelRequest, *, exact: bool) -> BuiltRequest:
    """v0 request -> OpenAI Chat Completions body: messages, tool calls and results, tool
    definitions and tool choice, and parameters (the inverse of `openai_chat.parse_request`).

    With `exact=True` (same provider) `parameters.extra` is passed through as it was logged;
    otherwise it is listed in `unconverted`.
    """
    r = _v0(request)
    lost: list[UnconvertedField] = []
    params = r.get("parameters", {})

    messages: list[dict[str, Any]] = []
    for m in r.get("messages", []):
        messages.extend(_openai_message(m, lost))
    body: dict[str, Any] = {"model": model, "messages": messages}

    for source, target in (("temperature", "temperature"), ("top_p", "top_p"), ("seed", "seed"),
                           ("max_output_tokens", "max_completion_tokens"), ("stop", "stop")):  # fmt: skip
        if source in params:
            body[target] = params[source]
    if "top_k" in params:
        _lost(lost, "parameters.top_k", "OpenAI has no top_k")

    response_format = params.get("response_format")
    if response_format:
        if response_format["type"] == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_format.get("schema", {})},
            }
        else:
            body["response_format"] = {"type": response_format["type"]}

    if r.get("tools"):
        body["tools"] = [
            {
                "type": "function",
                "function": drop_none(
                    {"name": t["name"], "description": t.get("description"), "parameters": t.get("input_schema")}
                ),
            }
            for t in r["tools"]
        ]
    choice = r.get("tool_choice")
    if isinstance(choice, dict):
        body["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    elif choice:
        body["tool_choice"] = choice
    if r.get("stream"):
        body["stream"] = True

    _add_extra(body, params.get("extra"), exact, lost)
    return BuiltRequest("openai", model, body, converted=not exact, unconverted=lost)


# --- Anthropic Messages ------------------------------------------------------------------


def _anthropic_blocks(parts: list[dict[str, Any]], lost: list[UnconvertedField]) -> list[dict[str, Any]]:
    out = []
    for part in parts:
        kind = part["type"]
        if kind == "text":
            out.append({"type": "text", "text": part["text"]})
        elif kind == "image" and part.get("url"):
            out.append({"type": "image", "source": {"type": "url", "url": part["url"]}})
        elif kind == "image" and part.get("data") and part.get("media_type"):
            out.append({"type": "image", "source": {"type": "base64", "media_type": part["media_type"], "data": part["data"]}})
        elif kind == "tool_call":
            if "arguments" not in part:
                _lost(lost, "messages.content.tool_call.arguments_raw", "Anthropic tool input must be a JSON object")
            out.append({"type": "tool_use", "id": _call_id(part), "name": part["name"], "input": part.get("arguments", {})})
        else:
            _lost(lost, f"messages.content.{kind}", "Anthropic cannot carry this part here")
    return out


def build_anthropic_messages(model: str, request: ModelRequest, *, exact: bool) -> BuiltRequest:
    """v0 request -> Anthropic Messages body (the inverse of `anthropic_messages.parse_request`).

    System messages go in the top-level `system`, and Anthropic requires `max_tokens`: when the
    v0 request has none, say so in `unconverted` and pick a default rather than failing.
    """
    r = _v0(request)
    lost: list[UnconvertedField] = []
    params = r.get("parameters", {})

    system: list[Any] = []
    messages: list[dict[str, Any]] = []
    for m in r.get("messages", []):
        role, content = m["role"], m["content"]
        if role == "system":
            system.append(content)
        elif role == "tool":
            blocks = []
            for part in [] if isinstance(content, str) else content:
                block = {"type": "tool_result", "tool_use_id": part["tool_call_id"], "content": part["content"]}
                if "is_error" in part:
                    block["is_error"] = part["is_error"]
                blocks.append(block)
            if blocks:  # consecutive user turns are combined by the API
                messages.append({"role": "user", "content": blocks})
            else:
                _lost(lost, "messages.content", "a tool message without tool results")
        else:
            messages.append({"role": role, "content": content if isinstance(content, str) else _anthropic_blocks(content, lost)})

    body: dict[str, Any] = {"model": model, "messages": messages}
    if system:
        if all(isinstance(s, str) for s in system):
            body["system"] = "\n\n".join(system)
        else:
            blocks = []
            for s in system:
                blocks.extend([{"type": "text", "text": s}] if isinstance(s, str) else _anthropic_blocks(s, lost))
            body["system"] = blocks

    if "max_output_tokens" in params:
        body["max_tokens"] = params["max_output_tokens"]
    else:
        body["max_tokens"] = DEFAULT_ANTHROPIC_MAX_TOKENS
        _lost(lost, "parameters.max_output_tokens", f"Anthropic requires max_tokens; used {DEFAULT_ANTHROPIC_MAX_TOKENS}")
    for source, target in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("stop", "stop_sequences")):
        if source in params:
            body[target] = params[source]
    if "seed" in params:
        _lost(lost, "parameters.seed", "Anthropic has no seed")
    if params.get("response_format", {}).get("type", "text") != "text":
        _lost(lost, "parameters.response_format", "Anthropic Messages has no response_format")

    if r.get("tools"):
        body["tools"] = [
            drop_none(
                {
                    "name": t["name"],
                    "description": t.get("description"),
                    "input_schema": t.get("input_schema") or {"type": "object", "properties": {}},
                }
            )
            for t in r["tools"]
        ]
    choice = r.get("tool_choice")
    if isinstance(choice, dict):
        body["tool_choice"] = {"type": "tool", "name": choice["name"]}
    elif choice:
        body["tool_choice"] = {"type": {"auto": "auto", "none": "none", "required": "any"}[choice]}
    if r.get("stream"):
        body["stream"] = True

    _add_extra(body, params.get("extra"), exact, lost)
    return BuiltRequest("anthropic", model, body, converted=not exact, unconverted=lost)


# --- Gemini generateContent --------------------------------------------------------------


def _gemini_parts(parts: list[dict[str, Any]], names: dict[str, str], lost: list[UnconvertedField]) -> list[dict[str, Any]]:
    out = []
    for part in parts:
        kind = part["type"]
        if kind == "text":
            out.append({"text": part["text"]})
        elif kind == "image" and part.get("data") and part.get("media_type"):
            out.append({"inlineData": {"mimeType": part["media_type"], "data": part["data"]}})
        elif kind == "image" and part.get("url"):
            out.append({"fileData": drop_none({"mimeType": part.get("media_type"), "fileUri": part["url"]})})
        elif kind == "tool_call":
            if "arguments" not in part:
                _lost(lost, "messages.content.tool_call.arguments_raw", "Gemini function arguments must be a JSON object")
            names[_call_id(part)] = part["name"]
            out.append({"functionCall": drop_none({"id": part.get("id"), "name": part["name"], "args": part.get("arguments", {})})})
        elif kind == "tool_result":
            call_id = part["tool_call_id"]
            name = names.get(call_id, call_id)
            try:
                value = json.loads(part["content"])
            except ValueError:
                value = None
            response = {"name": name, "response": value if isinstance(value, dict) else {"result": part["content"]}}
            if call_id != name:
                response["id"] = call_id
            out.append({"functionResponse": response})
        else:
            _lost(lost, f"messages.content.{kind}", "Gemini cannot carry this part")
    return out


def build_gemini_generate(model: str, request: ModelRequest, *, exact: bool) -> BuiltRequest:
    """v0 request -> Gemini generateContent body (the inverse of `gemini_generate.parse_request`).

    The model is not in the body: it is in the URL (`endpoint`).
    """
    r = _v0(request)
    lost: list[UnconvertedField] = []
    params = r.get("parameters", {})

    system: list[str] = []
    contents: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    after_tool = False  # the last message was a tool message; its results share a turn with what follows
    for m in r.get("messages", []):
        role, content = m["role"], m["content"]
        parts_in = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if role == "system":
            system.append(_text_of(content))
            continue
        parts = _gemini_parts(parts_in, names, lost)
        if role == "assistant":
            contents.append({"role": "model", "parts": parts})
            after_tool = False
        else:
            if after_tool and contents:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": "user", "parts": parts})
            after_tool = role == "tool"

    body: dict[str, Any] = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}

    generation: dict[str, Any] = {}
    for source, target in (("temperature", "temperature"), ("top_p", "topP"), ("top_k", "topK"),
                           ("max_output_tokens", "maxOutputTokens"), ("stop", "stopSequences"), ("seed", "seed")):  # fmt: skip
        if source in params:
            generation[target] = params[source]
    response_format = params.get("response_format")
    if response_format and response_format["type"] != "text":
        generation["responseMimeType"] = "application/json"
        if response_format["type"] == "json_schema":
            generation["responseSchema"] = response_format.get("schema", {})

    if r.get("tools"):
        body["tools"] = [
            {
                "functionDeclarations": [
                    drop_none({"name": t["name"], "description": t.get("description"), "parameters": t.get("input_schema")})
                    for t in r["tools"]
                ]
            }
        ]
    choice = r.get("tool_choice")
    if choice:
        if isinstance(choice, dict):
            config = {"mode": "ANY", "allowedFunctionNames": [choice["name"]]}
        else:
            config = {"mode": {"auto": "AUTO", "none": "NONE", "required": "ANY"}[choice]}
        body["toolConfig"] = {"functionCallingConfig": config}

    extra: dict[str, Any] = {}
    _add_extra(extra, params.get("extra"), exact, lost)
    for key, value in extra.items():
        (generation if key in _GEMINI_GENERATION_EXTRA else body)[key] = value
    if generation:
        body["generationConfig"] = generation
    return BuiltRequest("gemini", model, body, converted=not exact, unconverted=lost)


# --- choosing, proving and sending -------------------------------------------------------

_BUILDERS = {"openai": build_openai_chat, "anthropic": build_anthropic_messages, "gemini": build_gemini_generate}


def build_request(source_provider: str, provider: str, model: str, request: ModelRequest) -> BuiltRequest:
    """The request `model` on `provider` should receive for a request logged from `source_provider`.

    Dispatch to the builder for `provider`; `exact` is `source_provider == provider`, and
    `converted` is its opposite. Raise ValueError for a provider with no builder.
    """
    builder = _BUILDERS.get(provider)
    if builder is None:
        raise ValueError(f"no request builder for provider {provider!r}")
    return builder(model, request, exact=source_provider == provider)


def round_trips(built: BuiltRequest, original: ModelRequest) -> bool:
    """True when parsing `built.body` with the provider's `parse_request` gives back `original`.

    This is the proof that a same-provider request is exact. Compare the v0 requests, not the
    raw bodies (field order and defaults differ). `stream` is ignored: replays are not streamed.
    """
    try:
        parsed = PROVIDERS[built.provider].parse_request(built.body, endpoint(built.provider, built.model))
        again = ModelRequest.model_validate(parsed.request)
    except (Unsupported, KeyError, ValueError):
        return False

    def comparable(request: ModelRequest) -> dict[str, Any]:
        return request.model_dump(mode="json", by_alias=True, exclude_none=True, exclude={"stream"})

    return comparable(again) == comparable(original)


def endpoint(provider: str, model: str) -> str:
    """The generate URL for `model` on `provider` (Gemini puts the model in the path)."""
    if provider == "openai":
        return "https://api.openai.com/v1/chat/completions"
    if provider == "anthropic":
        return "https://api.anthropic.com/v1/messages"
    if provider == "gemini":
        return f"{_GEMINI_BASE}/{model}:generateContent"
    raise ValueError(f"no endpoint for provider {provider!r}")


def auth_headers(provider: str, api_key: str) -> dict[str, str]:
    """The headers that carry the key for `provider` (`x-api-key` plus a version for
    Anthropic, a bearer token for OpenAI, `x-goog-api-key` for Gemini). Never log these.
    """
    if provider == "openai":
        return {"Authorization": f"Bearer {api_key}"}
    if provider == "anthropic":
        return {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    if provider == "gemini":
        return {"x-goog-api-key": api_key}
    raise ValueError(f"no auth for provider {provider!r}")


def make_complete(client: httpx.AsyncClient, api_keys: dict[str, str], source_provider: str) -> CompleteFn:
    """The real `CompleteFn`: build with `build_request`, POST, parse, and time it.

    Raise `RateLimited(retry_after)` on a 429; any other failure (including a field that
    cannot be built) comes back as a `ReplayResult` with `error` set. Time only the send to
    the last byte, which is what the schema's `latency.total_ms` means (EVL-7).
    """

    async def complete(provider: str, model: str, request: ModelRequest) -> ReplayResult:
        api_key = api_keys.get(provider)
        if not api_key:
            return error_result(f"no API key for provider {provider!r}")
        try:
            built = build_request(source_provider, provider, model, request)
            url = endpoint(provider, model)
        except (ValueError, KeyError) as exc:
            return error_result(f"could not build the request: {exc}")

        body = {k: v for k, v in built.body.items() if k != "stream"}  # the answer is parsed as non-streamed
        started = time.perf_counter()
        try:
            response = await client.post(url, json=body, headers=auth_headers(provider, api_key))
        except httpx.HTTPError as exc:
            return error_result(f"{type(exc).__name__}: {exc}")
        latency_ms = (time.perf_counter() - started) * 1000

        flags = {"converted": built.converted, "unconverted_fields": tuple(f"{u.path}: {u.reason}" for u in built.unconverted)}

        if response.status_code == 429:
            try:
                retry_after = float(response.headers["retry-after"])
            except (KeyError, ValueError):  # missing, or an HTTP date
                retry_after = None
            raise RateLimited(retry_after)
        if response.status_code >= 400:
            return ReplayResult(None, None, None, None, latency_ms, f"HTTP {response.status_code}: {response.text[:200]}", **flags)

        try:
            parsed = PROVIDERS[provider].parse_response(response.json())
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return ReplayResult(None, None, None, None, latency_ms, f"unreadable response: {type(exc).__name__}", **flags)
        return ReplayResult(
            text=parsed.response.get("text"),
            tool_calls=parsed.response.get("tool_calls"),
            input_tokens=parsed.usage.get("input_tokens"),
            output_tokens=parsed.usage.get("output_tokens"),
            latency_ms=latency_ms,
            error=None,
            **flags,
        )

    return complete


def unconverted_for_report(results: list[ReplayResult]) -> dict[str, Any]:
    """What the report needs to flag converted results: how many requests were converted, and
    each unconverted field ("path: reason") with the number of requests it affected. Empty
    counts for an exact candidate.
    """
    counts: dict[str, int] = {}
    for result in results:
        for item in set(result.unconverted_fields):
            counts[item] = counts.get(item, 0) + 1
    return {
        "requests": len(results),
        "converted_requests": sum(1 for r in results if r.converted),
        "unconverted": [{"field": item, "requests": n} for item, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))],
    }
