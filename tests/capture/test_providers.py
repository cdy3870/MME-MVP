"""Provider parsers: provider JSON in, v0-shaped dicts out. No httpx involved."""

from __future__ import annotations

import pytest

from migration_eval.capture.providers import Unsupported, anthropic_messages as anthropic
from migration_eval.capture.providers import gemini_generate as gemini
from migration_eval.capture.providers import openai_chat as openai
from migration_eval.capture.sse import parse_sse

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_STREAM_URL = GEMINI_URL.replace(":generateContent", ":streamGenerateContent?alt=sse")


# --- OpenAI ------------------------------------------------------------------------------


def test_openai_request_maps_messages_tools_and_params():
    body = {
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_completion_tokens": 256,
        "stop": "END",
        "frequency_penalty": 0.5,
        "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
        "tool_choice": {"type": "function", "function": {"name": "get_order"}},
        "tools": [
            {"type": "function", "function": {"name": "get_order", "description": "d", "parameters": {"type": "object"}}}
        ],
        "messages": [
            {"role": "developer", "content": "be brief"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_order", "arguments": '{"id": "8841"}'}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "shipped"},
        ],
    }
    parsed = openai.parse_request(body, "https://api.openai.com/v1/chat/completions")

    assert parsed.model == "gpt-4o"
    req = parsed.request
    assert req["messages"][0] == {"role": "system", "content": "be brief"}  # developer -> system
    assert req["messages"][1]["content"][1] == {"type": "image", "media_type": "image/png", "data": "AAAA"}
    assert req["messages"][2]["content"] == [
        {"type": "tool_call", "id": "c1", "name": "get_order", "arguments": {"id": "8841"}}
    ]
    assert req["messages"][3] == {
        "role": "tool",
        "content": [{"type": "tool_result", "tool_call_id": "c1", "content": "shipped"}],
    }
    assert req["tools"] == [{"name": "get_order", "description": "d", "input_schema": {"type": "object"}}]
    assert req["tool_choice"] == {"type": "tool", "name": "get_order"}
    assert req["parameters"] == {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "stop": ["END"],
        "response_format": {"type": "json_schema", "schema": {"type": "object"}},
        "extra": {"frequency_penalty": 0.5},
    }


def test_openai_rejects_server_side_tools_instead_of_dropping_them():
    body = {"model": "gpt-4o", "messages": [], "tools": [{"type": "web_search"}]}
    with pytest.raises(Unsupported):
        openai.parse_request(body, "")


def test_openai_response_text_and_usage():
    body = {
        "id": "chatcmpl-1",
        "model": "gpt-4o-2024-08-06",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }
    parsed = openai.parse_response(body)
    assert parsed.response == {
        "status": "success", "id": "chatcmpl-1", "model": "gpt-4o-2024-08-06",
        "text": "hi", "finish_reason": "stop",
    }  # fmt: skip
    assert parsed.usage == {"input_tokens": 10, "output_tokens": 3, "cached_input_tokens": 4, "reasoning_tokens": 1}


def test_openai_tool_call_with_invalid_json_keeps_raw_arguments():
    body = {
        "choices": [
            {
                "message": {"content": None, "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{oops"}}]},
                "finish_reason": "tool_calls",
            }
        ]
    }
    call = openai.parse_response(body).response["tool_calls"][0]
    assert call == {"id": "c1", "name": "f", "arguments_raw": "{oops"}


def test_openai_stream_merges_text_and_tool_call_fragments():
    sse = (
        'data: {"id":"c1","model":"gpt-4o","choices":[{"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
        'data: {"id":"c1","choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"id":"c1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"get_order","arguments":"{\\"id\\":"}}]}}]}\n\n'
        'data: {"id":"c1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"8841\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        'data: {"id":"c1","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":5}}\n\n'
        "data: [DONE]\n\n"
    )
    parsed = openai.parse_stream(parse_sse(sse))
    assert parsed.response["text"] == "Hello"
    assert parsed.response["tool_calls"] == [{"id": "t1", "name": "get_order", "arguments": {"id": "8841"}}]
    assert parsed.response["finish_reason"] == "tool_calls"
    assert parsed.response["model"] == "gpt-4o"
    assert parsed.usage == {"input_tokens": 7, "output_tokens": 5}


# --- Anthropic ---------------------------------------------------------------------------


def test_anthropic_request_splits_tool_results_and_drops_thinking():
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "system": "You help.",
        "tool_choice": {"type": "any"},
        "tools": [{"name": "cancel", "description": "d", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": "cancel 8841"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "tool_use", "id": "tu1", "name": "cancel", "input": {"order_id": "8841"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "thanks"},
                    {"type": "tool_result", "tool_use_id": "tu1", "content": [{"type": "text", "text": "done"}]},
                ],
            },
        ],
    }
    req = anthropic.parse_request(body, "").request
    assert req["messages"][0] == {"role": "system", "content": "You help."}
    assert req["messages"][2]["content"] == [
        {"type": "tool_call", "id": "tu1", "name": "cancel", "arguments": {"order_id": "8841"}}
    ]
    # the tool result is its own message and comes first, ahead of the user's text
    assert req["messages"][3] == {
        "role": "tool",
        "content": [{"type": "tool_result", "tool_call_id": "tu1", "content": "done"}],
    }
    assert req["messages"][4] == {"role": "user", "content": [{"type": "text", "text": "thanks"}]}
    assert req["tool_choice"] == "required"
    assert req["parameters"] == {"max_output_tokens": 1024}


def test_anthropic_rejects_server_side_tools():
    with pytest.raises(Unsupported):
        anthropic.parse_request({"model": "m", "messages": [], "tools": [{"type": "web_search_20250305", "name": "web_search"}]}, "")


def test_anthropic_response_with_tool_use_and_cache_usage():
    body = {
        "id": "msg_1",
        "model": "claude-sonnet-4-5",
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "tu1", "name": "cancel", "input": {"order_id": "8841"}},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 900},
    }
    parsed = anthropic.parse_response(body)
    assert parsed.response["text"] == "Checking."
    assert parsed.response["tool_calls"] == [{"id": "tu1", "name": "cancel", "arguments": {"order_id": "8841"}}]
    assert parsed.response["finish_reason"] == "tool_calls"
    assert parsed.usage == {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 900}


def test_anthropic_stream_merges_blocks():
    sse = (
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","model":"claude-x","usage":{"input_tokens":50,"output_tokens":1}}}\n\n'
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me"}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" check."}}\n\n'
        'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu1","name":"cancel","input":{}}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"order"}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"_id\\":\\"8841\\"}"}}\n\n'
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":42}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    parsed = anthropic.parse_stream(parse_sse(sse))
    assert parsed.response["text"] == "Let me check."
    assert parsed.response["tool_calls"] == [{"id": "tu1", "name": "cancel", "arguments": {"order_id": "8841"}}]
    assert parsed.response["id"] == "msg_1" and parsed.response["finish_reason"] == "tool_calls"
    assert parsed.usage == {"input_tokens": 50, "output_tokens": 42}  # the final count wins


def test_anthropic_stream_error_event_becomes_error_response():
    sse = 'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
    parsed = anthropic.parse_stream(parse_sse(sse))
    assert parsed.response == {"status": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}


# --- Gemini ------------------------------------------------------------------------------


def test_gemini_request_takes_model_from_url_and_links_function_response_by_name():
    body = {
        "systemInstruction": {"parts": [{"text": "Be terse."}]},
        "generationConfig": {"temperature": 0.1, "topP": 0.9, "maxOutputTokens": 128, "responseMimeType": "application/json", "candidateCount": 1},
        "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_order"]}},
        "tools": [{"functionDeclarations": [{"name": "get_order", "description": "d", "parameters": {"type": "OBJECT"}}]}],
        "safetySettings": [{"category": "x"}],
        "contents": [
            {"role": "user", "parts": [{"text": "status of 8841?"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "get_order", "args": {"id": "8841"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "get_order", "response": {"status": "shipped"}}}]},
        ],
    }
    parsed = gemini.parse_request(body, GEMINI_URL)
    assert parsed.model == "gemini-2.0-flash"
    req = parsed.request
    assert req["messages"][0] == {"role": "system", "content": "Be terse."}
    assert req["messages"][2]["content"] == [{"type": "tool_call", "name": "get_order", "arguments": {"id": "8841"}}]
    assert req["messages"][3] == {
        "role": "tool",
        "content": [{"type": "tool_result", "tool_call_id": "get_order", "content": '{"status":"shipped"}'}],
    }
    assert req["tool_choice"] == {"type": "tool", "name": "get_order"}
    assert req["parameters"]["response_format"] == {"type": "json_object"}
    assert req["parameters"]["extra"] == {"candidateCount": 1, "safetySettings": [{"category": "x"}]}
    assert req["stream"] is False
    assert gemini.parse_request(body, GEMINI_STREAM_URL).request["stream"] is True


def test_gemini_response_function_call_finishes_as_tool_calls():
    body = {
        "candidates": [{"content": {"parts": [{"functionCall": {"name": "get_order", "args": {"id": "8841"}}}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 8, "thoughtsTokenCount": 12},
        "modelVersion": "gemini-2.0-flash-001",
        "responseId": "r1",
    }
    parsed = gemini.parse_response(body)
    assert parsed.response["tool_calls"] == [{"name": "get_order", "arguments": {"id": "8841"}}]
    assert parsed.response["finish_reason"] == "tool_calls"
    assert parsed.usage == {"input_tokens": 30, "output_tokens": 8, "reasoning_tokens": 12}


def test_gemini_stream_merges_chunks_and_skips_thought_parts():
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
        {
            "candidates": [{"content": {"parts": [{"text": "lo"}]}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
            "modelVersion": "gemini-2.0-flash-001",
        },
    ]
    parsed = gemini.parse_stream(chunks)
    assert parsed.response["text"] == "Hello"
    assert parsed.response["finish_reason"] == "length"
    assert parsed.usage == {"input_tokens": 5, "output_tokens": 2}


def test_gemini_prompt_blocked_is_a_content_filter_finish():
    parsed = gemini.parse_response({"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 4}})
    assert parsed.response == {"status": "success", "finish_reason": "content_filter"}


def test_url_matching():
    assert openai.matches("https://api.openai.com/v1/chat/completions")
    assert not openai.matches("https://api.openai.com/v1/embeddings")
    assert anthropic.matches("https://api.anthropic.com/v1/messages")
    assert not anthropic.matches("https://api.anthropic.com/v1/messages/batches")
    assert gemini.matches(GEMINI_URL) and gemini.matches(GEMINI_STREAM_URL)
    assert not gemini.matches("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:countTokens")
