"""The live recorder: real httpx clients over a fake transport, so no network is used."""

from __future__ import annotations

import asyncio
import gzip
import logging

import httpx
import pytest

import migration_eval
from migration_eval.capture import httpx_hook

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_STREAM_URL = GEMINI_URL.replace(":generateContent", ":streamGenerateContent?alt=sse")

OPENAI_BODY = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
OPENAI_REPLY = {
    "id": "chatcmpl-1",
    "model": "gpt-4o-mini-2024-07-18",
    "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 9, "completion_tokens": 2},
}
OPENAI_SSE = (
    'data: {"id":"c1","model":"gpt-4o-mini","choices":[{"delta":{"content":"Hel"}}]}\n\n'
    'data: {"id":"c1","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
    'data: {"id":"c1","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2}}\n\n'
    "data: [DONE]\n\n"
)


@pytest.fixture
def records_path(tmp_path):
    path = tmp_path / "records.jsonl"
    migration_eval.start(path)
    yield path
    migration_eval.stop()


def sync_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def chunked(data: bytes, size: int = 25):
    """Deliver a body in small pieces, splitting events mid-line like a real network stream.

    This matters: httpx reads a `content=bytes` response up front, so it would look like a
    non-streaming response by the time the recorder sees it. An iterator stays a stream.
    """
    return (data[i : i + size] for i in range(0, len(data), size))


def sse_response(text: str, **headers) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream", **headers},
        content=chunked(text.encode()),
    )


# --- non-streaming -----------------------------------------------------------------------


def test_records_a_call_and_the_app_still_gets_its_response(records_path, read_records):
    with sync_client(lambda request: httpx.Response(200, json=OPENAI_REPLY)) as client:
        response = client.post(OPENAI_URL, json=OPENAI_BODY)

    assert response.json() == OPENAI_REPLY  # untouched

    (record,) = read_records(records_path)
    assert record["provider"] == "openai"
    assert record["model"] == "gpt-4o-mini"
    assert record["component"] == "default"
    assert record["request"]["messages"] == [{"role": "user", "content": "hi"}]
    assert record["response"]["text"] == "hello"
    assert record["usage"] == {"input_tokens": 9, "output_tokens": 2}
    assert record["latency"]["total_ms"] >= 0
    assert "time_to_first_token_ms" not in record["latency"]
    assert record["source"]["type"] == "recorder"


def test_component_tags_calls_and_nests(records_path, read_records):
    with sync_client(lambda request: httpx.Response(200, json=OPENAI_REPLY)) as client:
        client.post(OPENAI_URL, json=OPENAI_BODY)
        with migration_eval.component("triage"):
            client.post(OPENAI_URL, json=OPENAI_BODY)
            with migration_eval.component("summarizer"):
                client.post(OPENAI_URL, json=OPENAI_BODY)
            client.post(OPENAI_URL, json=OPENAI_BODY)

    assert [r["component"] for r in read_records(records_path)] == [
        "default", "triage", "summarizer", "triage",
    ]  # fmt: skip


def test_component_follows_async_tasks(records_path, read_records):
    async def call(name: str) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=OPENAI_REPLY))) as client:
            with migration_eval.component(name):
                await asyncio.sleep(0)  # let the other task run in between
                await client.post(OPENAI_URL, json=OPENAI_BODY)

    async def main() -> None:
        await asyncio.gather(call("a"), call("b"))

    asyncio.run(main())
    assert sorted(r["component"] for r in read_records(records_path)) == ["a", "b"]


def test_ignores_other_hosts_and_non_post_requests(records_path):
    with sync_client(lambda request: httpx.Response(200, json={})) as client:
        client.post("https://example.com/v1/chat/completions", json=OPENAI_BODY)
        client.post("https://api.openai.com/v1/embeddings", json={"input": "x"})
        client.get(OPENAI_URL)
    assert records_path.read_text() == ""


def test_http_error_is_recorded_as_a_failed_call(records_path, read_records):
    body = {"error": {"type": "rate_limit_error", "message": "slow down"}}
    with sync_client(lambda request: httpx.Response(429, json=body)) as client:
        client.post(OPENAI_URL, json=OPENAI_BODY)

    (record,) = read_records(records_path)
    assert record["response"] == {
        "status": "error",
        "error": {"http_status": 429, "type": "rate_limit_error", "message": "slow down"},
    }


def test_api_keys_are_never_recorded(records_path):
    with sync_client(lambda request: httpx.Response(200, json=OPENAI_REPLY)) as client:
        client.post(OPENAI_URL, json=OPENAI_BODY, headers={"Authorization": "Bearer sk-super-secret"})
    assert "sk-super-secret" not in records_path.read_text()


def test_unparseable_call_is_kept_with_a_reason_and_does_not_break_the_app(records_path, read_records):
    body = {**OPENAI_BODY, "tools": [{"type": "web_search"}]}  # a tool v0 cannot represent
    with sync_client(lambda request: httpx.Response(200, json=OPENAI_REPLY)) as client:
        response = client.post(OPENAI_URL, json=body)

    assert response.status_code == 200
    (record,) = read_records(records_path)
    assert "request" not in record  # not recorded, rather than recorded wrongly
    assert "web_search" in record["metadata"]["parse_error"]
    assert record["response"]["text"] == "hello"  # the rest still parsed


def test_recorder_failures_never_reach_the_app(records_path, monkeypatch, caplog):
    def boom(exchange):
        raise RuntimeError("secret prompt text must not be logged")

    monkeypatch.setattr("migration_eval.capture.recorder.build_record", boom)
    with caplog.at_level(logging.WARNING, logger="migration_eval"):
        with sync_client(lambda request: httpx.Response(200, json=OPENAI_REPLY)) as client:
            response = client.post(OPENAI_URL, json=OPENAI_BODY)

    assert response.json() == OPENAI_REPLY
    assert "RuntimeError" in caplog.text
    assert "secret prompt text" not in caplog.text  # class name only, never content


def test_gemini_non_streaming(records_path, read_records):
    reply = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
    }
    body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    with sync_client(lambda request: httpx.Response(200, json=reply)) as client:
        client.post(GEMINI_URL, json=body)

    (record,) = read_records(records_path)
    assert (record["provider"], record["model"]) == ("gemini", "gemini-2.0-flash")
    assert record["response"]["text"] == "ok"


# --- streaming ---------------------------------------------------------------------------


def test_streamed_response_is_recorded_when_the_stream_ends(records_path, read_records):
    with sync_client(lambda request: sse_response(OPENAI_SSE)) as client:
        with client.stream("POST", OPENAI_URL, json={**OPENAI_BODY, "stream": True}) as response:
            lines = list(response.iter_lines())

    # the caller sees the stream exactly as sent
    assert lines[0].startswith("data: ") and lines[-2] == "data: [DONE]"

    (record,) = read_records(records_path)
    assert record["request"]["stream"] is True
    assert record["response"] == {
        "status": "success", "id": "c1", "model": "gpt-4o-mini", "text": "Hello", "finish_reason": "stop",
    }  # fmt: skip
    assert record["usage"] == {"input_tokens": 9, "output_tokens": 2}
    assert 0 <= record["latency"]["time_to_first_token_ms"] <= record["latency"]["total_ms"]


def test_stream_closed_early_is_marked_incomplete(records_path, read_records):
    def handler(request):
        def chunks():
            yield b'data: {"id":"c1","choices":[{"delta":{"content":"Hel"}}]}\n\n'
            yield b'data: {"id":"c1","choices":[{"delta":{"content":"lo"}}]}\n\n'

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chunks())

    with sync_client(handler) as client:
        with client.stream("POST", OPENAI_URL, json={**OPENAI_BODY, "stream": True}) as response:
            next(response.iter_bytes())  # read one chunk, then walk away

    (record,) = read_records(records_path)
    assert record["response"]["status"] == "error"
    assert record["response"]["error"] == {"type": "stream_closed_early"}
    assert record["response"]["text"] == "Hel"  # what arrived is kept


def test_gzip_encoded_stream_is_decoded(records_path, read_records):
    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
            content=chunked(gzip.compress(OPENAI_SSE.encode())),
        )

    with sync_client(handler) as client:
        with client.stream("POST", OPENAI_URL, json={**OPENAI_BODY, "stream": True}) as response:
            list(response.iter_lines())

    (record,) = read_records(records_path)
    assert record["response"]["text"] == "Hello"
    # first-chunk timing only exists on the streaming path, so this proves that path ran
    assert "time_to_first_token_ms" in record["latency"]


def test_async_streaming_anthropic(records_path, read_records):
    sse = (
        'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-x","usage":{"input_tokens":5}}}\n\n'
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
    )

    async def achunks():  # an AsyncClient needs an async byte stream
        for piece in chunked(sse.encode()):
            yield piece

    async def main() -> list[str]:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=achunks()
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            body = {"model": "claude-x", "max_tokens": 10, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
            async with client.stream("POST", ANTHROPIC_URL, json=body) as response:
                return [line async for line in response.aiter_lines()]

    lines = asyncio.run(main())
    assert any("text_delta" in line for line in lines)

    (record,) = read_records(records_path)
    assert (record["provider"], record["response"]["text"]) == ("anthropic", "Hi")
    assert record["usage"] == {"input_tokens": 5, "output_tokens": 1}
    assert "time_to_first_token_ms" in record["latency"]


def test_gemini_sse_stream(records_path, read_records):
    sse = (
        'data: {"candidates":[{"content":{"parts":[{"text":"He"}]}}]}\n\n'
        'data: {"candidates":[{"content":{"parts":[{"text":"llo"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2}}\n\n'
    )
    body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    with sync_client(lambda request: sse_response(sse)) as client:
        with client.stream("POST", GEMINI_STREAM_URL, json=body) as response:
            list(response.iter_lines())

    (record,) = read_records(records_path)
    assert record["request"]["stream"] is True
    assert record["response"]["text"] == "Hello"


# --- lifecycle ---------------------------------------------------------------------------


def test_stop_restores_httpx_and_stops_recording(tmp_path):
    original_send = httpx.Client.send
    path = tmp_path / "r.jsonl"
    migration_eval.start(path)
    assert httpx.Client.send is not original_send
    migration_eval.stop()
    assert httpx.Client.send is original_send

    with sync_client(lambda request: httpx.Response(200, json=OPENAI_REPLY)) as client:
        client.post(OPENAI_URL, json=OPENAI_BODY)
    assert path.read_text() == ""


def test_calling_start_twice_does_not_record_twice(tmp_path, read_records):
    path = tmp_path / "r.jsonl"
    migration_eval.start(path)
    migration_eval.start(path)
    try:
        with sync_client(lambda request: httpx.Response(200, json=OPENAI_REPLY)) as client:
            client.post(OPENAI_URL, json=OPENAI_BODY)
    finally:
        migration_eval.stop()
    assert len(read_records(path)) == 1


def test_records_are_appended_across_restarts(tmp_path, read_records):
    path = tmp_path / "r.jsonl"
    for _ in range(2):
        migration_eval.start(path)
        with sync_client(lambda request: httpx.Response(200, json=OPENAI_REPLY)) as client:
            client.post(OPENAI_URL, json=OPENAI_BODY)
        migration_eval.stop()
    assert len(read_records(path)) == 2


@pytest.fixture(autouse=True)
def _always_uninstall():
    yield
    httpx_hook.uninstall()
