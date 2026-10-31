"""Request builders (EVL-10): exact same-provider rebuilds, converted cross-provider requests,
and the real `complete` function."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from typing import Any

import httpx
import pytest

from migration_eval.capture.providers import PROVIDERS
from migration_eval.models.record import ModelRequest
from migration_eval.replay import RateLimited, ReplayResult
from migration_eval.request_builders import (
    auth_headers,
    build_request,
    endpoint,
    make_complete,
    round_trips,
    unconverted_for_report,
)

SCHEMA = {"type": "object", "properties": {"id": {"type": "string"}}}
TOOL_RESULT = '{"status":"shipped"}'  # JSON text, so it survives Gemini's object-shaped responses

OPENAI = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a support bot."},
        {"role": "user", "content": [
            {"type": "text", "text": "Where is order 8841?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
        {"role": "assistant", "content": "Let me look.", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_order", "arguments": '{"id": "8841"}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": TOOL_RESULT},
        {"role": "user", "content": "thanks"},
    ],
    "temperature": 0.2, "top_p": 0.9, "seed": 7, "max_completion_tokens": 300, "stop": ["END"],
    "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}},
    "tools": [{"type": "function", "function": {"name": "get_order", "description": "Find an order", "parameters": SCHEMA}}],
    "tool_choice": {"type": "function", "function": {"name": "get_order"}},
    "logit_bias": {"50256": -100},
}  # fmt: skip

ANTHROPIC = {
    "model": "claude-x", "max_tokens": 300, "system": "You are a support bot.",
    "temperature": 0.2, "top_p": 0.9, "top_k": 40, "stop_sequences": ["END"],
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Where is order 8841?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_order", "input": {"id": "8841"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": TOOL_RESULT},
            {"type": "text", "text": "thanks"},
        ]},
    ],
    "tools": [{"name": "get_order", "description": "Find an order", "input_schema": SCHEMA}],
    "tool_choice": {"type": "tool", "name": "get_order"},
    "metadata": {"user_id": "u1"},
}  # fmt: skip

GEMINI = {
    "systemInstruction": {"parts": [{"text": "You are a support bot."}]},
    "contents": [
        {"role": "user", "parts": [
            {"text": "Where is order 8841?"},
            {"inlineData": {"mimeType": "image/png", "data": "AAAA"}},
        ]},
        {"role": "model", "parts": [
            {"text": "Let me look."},
            {"functionCall": {"name": "get_order", "args": {"id": "8841"}}},
        ]},
        {"role": "user", "parts": [
            {"functionResponse": {"name": "get_order", "response": {"status": "shipped"}}},
            {"text": "thanks"},
        ]},
    ],
    "generationConfig": {
        "temperature": 0.2, "topP": 0.9, "topK": 40, "maxOutputTokens": 300, "stopSequences": ["END"],
        "seed": 7, "responseMimeType": "application/json", "responseSchema": SCHEMA, "candidateCount": 1,
    },
    "tools": [{"functionDeclarations": [{"name": "get_order", "description": "Find an order", "parameters": SCHEMA}]}],
    "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_order"]}},
    "safetySettings": [{"category": "HARM_CATEGORY_X", "threshold": "BLOCK_NONE"}],
}  # fmt: skip

BODIES = {"openai": (OPENAI, "gpt-4o"), "anthropic": (ANTHROPIC, "claude-x"), "gemini": (GEMINI, "gemini-2.0-flash")}


def v0(provider: str, body: dict[str, Any] | None = None) -> ModelRequest:
    original, model = BODIES[provider]
    parsed = PROVIDERS[provider].parse_request(body or original, endpoint(provider, model))
    return ModelRequest.model_validate(parsed.request)


def parse_back(built) -> dict[str, Any]:
    parsed = PROVIDERS[built.provider].parse_request(built.body, endpoint(built.provider, built.model))
    return ModelRequest.model_validate(parsed.request).model_dump(mode="json", by_alias=True, exclude_none=True)


def without_call_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gemini calls have no ids, and its text is always a list of parts; both are the same
    conversation once normalized."""
    out = copy.deepcopy(messages)
    for message in out:
        if isinstance(message["content"], str) and message["role"] in ("user", "assistant"):
            message["content"] = [{"type": "text", "text": message["content"]}]
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part["type"] == "tool_call":
                    part.pop("id", None)
    return out


def paths(built) -> set[str]:
    return {u.path for u in built.unconverted}


# --- same provider: exact -----------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_a_same_provider_request_round_trips_exactly(provider):
    original = v0(provider)
    built = build_request(provider, provider, "swapped-model", original)
    assert round_trips(built, original)
    assert built.converted is False and built.unconverted == []
    assert built.provider == provider and built.model == "swapped-model"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_only_the_model_changes_in_the_body(provider):
    original = v0(provider)
    assert build_request(provider, provider, "swapped-model", original).body["model"] == "swapped-model"


def test_gemini_carries_the_model_in_the_url_not_the_body():
    built = build_request("gemini", "gemini", "gemini-2.5-pro", v0("gemini"))
    assert "model" not in built.body and endpoint("gemini", "gemini-2.5-pro").endswith("models/gemini-2.5-pro:generateContent")


def test_provider_specific_settings_are_passed_through_when_exact():
    assert build_request("openai", "openai", "m", v0("openai")).body["logit_bias"] == {"50256": -100}
    assert build_request("anthropic", "anthropic", "m", v0("anthropic")).body["metadata"] == {"user_id": "u1"}
    gemini = build_request("gemini", "gemini", "m", v0("gemini")).body
    assert gemini["safetySettings"] and gemini["generationConfig"]["candidateCount"] == 1  # each at its own level


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_a_minimal_request_round_trips(provider):
    body = {
        "openai": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        "anthropic": {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
        "gemini": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    }[provider]
    original = v0(provider, body)
    assert round_trips(build_request(provider, provider, "m", original), original)


def test_a_changed_request_fails_the_round_trip():
    original = v0("openai")
    built = build_request("openai", "openai", "m", original)
    built.body["messages"][1]["content"] = "something else"
    assert not round_trips(built, original)


def test_tool_calls_with_unparseable_arguments_keep_their_raw_text():
    body = copy.deepcopy(OPENAI)
    body["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"id": '
    original = v0("openai", body)
    built = build_request("openai", "openai", "m", original)
    assert built.body["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"id": '
    assert round_trips(built, original)


# --- field mapping ------------------------------------------------------------------------


def test_parameters_map_to_each_providers_names():
    original = v0("openai")
    assert {k: build_request("openai", "openai", "m", original).body[k] for k in ("max_completion_tokens", "stop", "seed")} == {
        "max_completion_tokens": 300, "stop": ["END"], "seed": 7,
    }  # fmt: skip
    anthropic = build_request("openai", "anthropic", "m", original).body
    assert anthropic["max_tokens"] == 300 and anthropic["stop_sequences"] == ["END"] and "seed" not in anthropic
    gemini = build_request("openai", "gemini", "m", original).body["generationConfig"]
    assert gemini["maxOutputTokens"] == 300 and gemini["stopSequences"] == ["END"] and gemini["seed"] == 7


@pytest.mark.parametrize("choice, openai, anthropic, gemini", [
    ("auto", "auto", {"type": "auto"}, {"mode": "AUTO"}),
    ("none", "none", {"type": "none"}, {"mode": "NONE"}),
    ("required", "required", {"type": "any"}, {"mode": "ANY"}),
])  # fmt: skip
def test_tool_choice_words_map_to_each_provider(choice, openai, anthropic, gemini):
    request = ModelRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "t"}], "tool_choice": choice}
    )
    assert build_request("openai", "openai", "m", request).body["tool_choice"] == openai
    assert build_request("openai", "anthropic", "m", request).body["tool_choice"] == anthropic
    assert build_request("openai", "gemini", "m", request).body["toolConfig"] == {"functionCallingConfig": gemini}


def test_a_named_tool_choice_maps_to_each_provider():
    request = v0("anthropic")
    assert build_request("anthropic", "openai", "m", request).body["tool_choice"] == {"type": "function", "function": {"name": "get_order"}}
    assert build_request("anthropic", "gemini", "m", request).body["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY", "allowedFunctionNames": ["get_order"],
    }  # fmt: skip


def test_response_formats_map_to_each_provider():
    original = v0("openai")
    assert build_request("openai", "openai", "m", original).body["response_format"]["json_schema"]["schema"] == SCHEMA
    gemini = build_request("openai", "gemini", "m", original).body["generationConfig"]
    assert gemini["responseMimeType"] == "application/json" and gemini["responseSchema"] == SCHEMA
    json_object = ModelRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "parameters": {"response_format": {"type": "json_object"}}}
    )
    assert build_request("openai", "openai", "m", json_object).body["response_format"] == {"type": "json_object"}
    assert "responseSchema" not in build_request("openai", "gemini", "m", json_object).body["generationConfig"]


# --- cross provider: converted ------------------------------------------------------------


@pytest.mark.parametrize("source", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("target", ["openai", "anthropic", "gemini"])
def test_conversion_keeps_the_conversation_and_the_tools(source, target):
    if source == target:
        pytest.skip("same provider is covered by the exact tests")
    original = v0(source)
    built = build_request(source, target, "m", original)
    assert built.converted is True
    after, before = parse_back(built), original.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert without_call_ids(after["messages"]) == without_call_ids(before["messages"])
    assert after["tools"] == before["tools"]
    assert after["tool_choice"] == before["tool_choice"]


def test_a_plain_text_tool_result_becomes_an_object_for_gemini():
    request = ModelRequest.model_validate({
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_call", "id": "call_1", "name": "get_order", "arguments": {}}]},
            {"role": "tool", "content": [{"type": "tool_result", "tool_call_id": "call_1", "content": "shipped"}]},
        ]
    })  # fmt: skip
    parts = build_request("openai", "gemini", "m", request).body["contents"][1]["parts"]
    assert parts == [{"functionResponse": {"name": "get_order", "response": {"result": "shipped"}, "id": "call_1"}}]


def test_gemini_function_responses_are_named_after_the_call_they_answer():
    built = build_request("openai", "gemini", "m", v0("openai"))
    response = built.body["contents"][2]["parts"][0]["functionResponse"]
    assert response["name"] == "get_order" and response["id"] == "call_1"


def test_parallel_tool_results_share_one_gemini_turn_with_the_message_after_them():
    request = ModelRequest.model_validate({
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_call", "id": "a", "name": "f", "arguments": {}},
                {"type": "tool_call", "id": "b", "name": "g", "arguments": {}},
            ]},
            {"role": "tool", "content": [{"type": "tool_result", "tool_call_id": "a", "content": "{}"}]},
            {"role": "tool", "content": [{"type": "tool_result", "tool_call_id": "b", "content": "{}"}]},
            {"role": "user", "content": "thanks"},
        ]
    })  # fmt: skip
    contents = build_request("openai", "gemini", "m", request).body["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert [list(p)[0] for p in contents[2]["parts"]] == ["functionResponse", "functionResponse", "text"]


def test_a_gemini_call_without_an_id_gets_its_name_as_the_id_elsewhere():
    openai = build_request("gemini", "openai", "m", v0("gemini")).body["messages"]
    call = next(m for m in openai if m["role"] == "assistant")["tool_calls"][0]
    result = next(m for m in openai if m["role"] == "tool")
    assert call["id"] == "get_order" and result["tool_call_id"] == "get_order"


# --- what could not be converted ----------------------------------------------------------


def test_openai_to_anthropic_lists_what_anthropic_cannot_carry():
    built = build_request("openai", "anthropic", "m", v0("openai"))
    assert paths(built) == {"parameters.seed", "parameters.response_format", "parameters.extra.logit_bias"}


def test_anthropic_to_openai_lists_top_k_and_provider_settings():
    built = build_request("anthropic", "openai", "m", v0("anthropic"))
    assert paths(built) == {"parameters.top_k", "parameters.extra.metadata"}
    assert "metadata" not in built.body


def test_gemini_to_openai_lists_both_levels_of_provider_settings():
    built = build_request("gemini", "openai", "m", v0("gemini"))
    assert paths(built) == {"parameters.top_k", "parameters.extra.safetySettings", "parameters.extra.candidateCount"}


def test_a_missing_max_tokens_is_filled_in_for_anthropic_and_listed():
    request = ModelRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})
    built = build_request("openai", "anthropic", "m", request)
    assert built.body["max_tokens"] == 4096 and paths(built) == {"parameters.max_output_tokens"}


def test_a_tool_error_flag_is_lost_for_openai_but_kept_for_anthropic():
    request = ModelRequest.model_validate({
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_call", "id": "c", "name": "f", "arguments": {}}]},
            {"role": "tool", "content": [{"type": "tool_result", "tool_call_id": "c", "content": "boom", "is_error": True}]},
        ]
    })  # fmt: skip
    assert "messages.content.tool_result.is_error" in paths(build_request("anthropic", "openai", "m", request))
    kept = build_request("openai", "anthropic", "m", request)
    assert kept.body["messages"][1]["content"][0]["is_error"] is True
    assert "messages.content.tool_result.is_error" not in paths(kept)


def test_arguments_that_are_not_an_object_are_listed_for_anthropic():
    request = v0("openai", {**OPENAI, "messages": [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "[1, 2]"}}]},
    ]})  # fmt: skip
    built = build_request("openai", "anthropic", "m", request)
    assert "messages.content.tool_call.arguments_raw" in paths(built)


def test_a_fully_convertible_request_is_still_flagged_as_converted():
    request = ModelRequest.model_validate({"messages": [{"role": "user", "content": "hi"}], "parameters": {"max_output_tokens": 5}})
    built = build_request("openai", "anthropic", "m", request)
    assert built.converted is True and built.unconverted == []


def test_an_unknown_provider_has_no_builder():
    with pytest.raises(ValueError):
        build_request("openai", "mistral", "m", v0("openai"))


# --- endpoints and headers ----------------------------------------------------------------


def test_every_endpoint_is_recognized_by_that_providers_parser():
    for name in ("openai", "anthropic", "gemini"):
        assert PROVIDERS[name].matches(endpoint(name, "some-model"))


def test_auth_headers_per_provider():
    assert auth_headers("openai", "k") == {"Authorization": "Bearer k"}
    assert auth_headers("anthropic", "k")["x-api-key"] == "k" and "anthropic-version" in auth_headers("anthropic", "k")
    assert auth_headers("gemini", "k") == {"x-goog-api-key": "k"}
    with pytest.raises(ValueError):
        auth_headers("mistral", "k")
    with pytest.raises(ValueError):
        endpoint("mistral", "m")


# --- make_complete ------------------------------------------------------------------------

REPLIES = {
    "openai": {"id": "x", "model": "gpt", "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    "anthropic": {"id": "m", "model": "c", "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                  "usage": {"input_tokens": 10, "output_tokens": 5}},
    "gemini": {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
               "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}},
}  # fmt: skip
KEYS = {"openai": "sk-openai", "anthropic": "sk-anthropic", "gemini": "sk-gemini"}


def call(handler, provider="anthropic", source="anthropic", request=None, keys=KEYS) -> tuple[Any, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as client:
            complete = make_complete(client, keys, source)
            return await complete(provider, "target-model", request or v0(source))

    try:
        return asyncio.run(run()), seen
    except RateLimited as exc:
        return exc, seen


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_a_successful_call_is_sent_to_the_right_place_and_parsed(provider):
    result, seen = call(lambda req: httpx.Response(200, json=REPLIES[provider]), provider=provider, source=provider)
    assert isinstance(result, ReplayResult) and result.error is None
    assert (result.text, result.input_tokens, result.output_tokens) == ("hi", 10, 5)
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.converted is False and result.unconverted_fields == ()
    assert str(seen[0].url) == endpoint(provider, "target-model")
    assert KEYS[provider] in " ".join(seen[0].headers.values())


def test_the_key_goes_to_the_matching_provider_only():
    _, seen = call(lambda req: httpx.Response(200, json=REPLIES["anthropic"]))
    assert "sk-openai" not in str(seen[0].headers) and "sk-gemini" not in str(seen[0].headers)


def test_the_body_is_not_streamed_even_if_the_log_was():
    streamed = v0("openai", {**OPENAI, "stream": True})
    assert build_request("openai", "openai", "m", streamed).body["stream"] is True
    _, seen = call(lambda req: httpx.Response(200, json=REPLIES["openai"]), "openai", "openai", streamed)
    assert "stream" not in json.loads(seen[0].content)


def test_a_converted_call_reports_what_it_lost():
    result, _ = call(lambda req: httpx.Response(200, json=REPLIES["anthropic"]), "anthropic", "openai", v0("openai"))
    assert result.converted is True
    assert "parameters.seed: Anthropic has no seed" in result.unconverted_fields


def test_a_429_is_raised_with_the_retry_after():
    exc, _ = call(lambda req: httpx.Response(429, headers={"retry-after": "7"}))
    assert isinstance(exc, RateLimited) and exc.retry_after == 7.0


@pytest.mark.parametrize("headers", [{}, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}])
def test_a_429_without_a_usable_retry_after_still_raises(headers):
    exc, _ = call(lambda req: httpx.Response(429, headers=headers))
    assert isinstance(exc, RateLimited) and exc.retry_after is None


def test_an_http_error_becomes_an_error_result_with_a_trimmed_message():
    result, _ = call(lambda req: httpx.Response(500, text="x" * 1000))
    assert result.error.startswith("HTTP 500: ") and len(result.error) < 300 and result.latency_ms is not None


def test_a_network_failure_becomes_an_error_result():
    def boom(req):
        raise httpx.ConnectError("no route")

    result, _ = call(boom)
    assert "ConnectError" in result.error and result.text is None


def test_an_unreadable_success_response_becomes_an_error_result():
    result, _ = call(lambda req: httpx.Response(200, text="<html>not json</html>"))
    assert result.error.startswith("unreadable response")


def test_a_missing_key_is_an_error_and_nothing_is_sent():
    result, seen = call(lambda req: httpx.Response(200, json=REPLIES["anthropic"]), keys={})
    assert "no API key" in result.error and seen == []


def test_a_provider_with_no_builder_is_an_error_result():
    result, seen = call(lambda req: httpx.Response(200), provider="mistral", keys={"mistral": "k"})
    assert "could not build" in result.error and seen == []


def test_the_api_key_never_appears_in_an_error_result():
    result, _ = call(lambda req: httpx.Response(401, text="bad key"))
    assert "sk-" not in result.error


# --- unconverted_for_report ---------------------------------------------------------------


def test_the_report_counts_converted_requests_and_each_lost_field():
    def r(converted=False, fields=()):
        return ReplayResult("t", None, 1, 1, 1.0, None, converted, tuple(fields))

    summary = unconverted_for_report([r(True, ["a: x", "b: y"]), r(True, ["a: x"]), r()])
    assert summary == {
        "requests": 3, "converted_requests": 2,
        "unconverted": [{"field": "a: x", "requests": 2}, {"field": "b: y", "requests": 1}],
    }  # fmt: skip


def test_an_exact_candidate_has_nothing_to_flag():
    assert unconverted_for_report([ReplayResult("t", None, 1, 1, 1.0, None)])["converted_requests"] == 0
    assert unconverted_for_report([])["unconverted"] == []
