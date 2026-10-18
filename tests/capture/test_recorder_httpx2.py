"""The recorder must also see calls made through httpx2, which current OpenAI and Anthropic
SDKs use for their default client. httpx2 has the same API as httpx, but it is a separate
package, so patching httpx alone would record nothing for an app using the SDK defaults."""

from __future__ import annotations

import asyncio

import pytest

import migration_eval
from migration_eval.capture import httpx_hook

httpx2 = pytest.importorskip("httpx2")
import httpx  # noqa: E402

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
BODY = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
REPLY = {
    "id": "chatcmpl-1",
    "model": "gpt-4o-mini-2024-07-18",
    "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 9, "completion_tokens": 2},
}
SSE = (
    'data: {"id":"c1","model":"gpt-4o-mini","choices":[{"delta":{"content":"Hel"}}]}\n\n'
    'data: {"id":"c1","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def chunked(text: str, size: int = 25):
    data = text.encode()
    return (data[i : i + size] for i in range(0, len(data), size))


@pytest.fixture
def records_path(tmp_path):
    path = tmp_path / "records.jsonl"
    migration_eval.start(path)
    yield path
    migration_eval.stop()


@pytest.fixture(autouse=True)
def _always_uninstall():
    yield
    httpx_hook.uninstall()


def test_non_streaming_call(records_path, read_records):
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=REPLY))
    with httpx2.Client(transport=transport) as client:
        response = client.post(OPENAI_URL, json=BODY)

    assert response.json() == REPLY
    (record,) = read_records(records_path)
    assert (record["provider"], record["response"]["text"]) == ("openai", "hello")


def test_calls_through_a_client_subclass_are_recorded(records_path, read_records):
    """The SDKs wrap httpx2.Client in their own subclass; it must inherit the patch."""

    class SdkStyleClient(httpx2.Client):
        pass

    transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=REPLY))
    with SdkStyleClient(transport=transport) as client:
        client.post(OPENAI_URL, json=BODY)
    assert len(read_records(records_path)) == 1


def test_streaming_call(records_path, read_records):
    def handler(request):
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=chunked(SSE))

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        with client.stream("POST", OPENAI_URL, json={**BODY, "stream": True}) as response:
            lines = list(response.iter_lines())

    assert lines[-2] == "data: [DONE]"  # the caller still sees the whole stream
    (record,) = read_records(records_path)
    assert record["response"]["text"] == "Hello"
    assert "time_to_first_token_ms" in record["latency"]  # proves the stream path ran


def test_async_streaming_call(records_path, read_records):
    async def achunks():
        for piece in chunked(SSE):
            yield piece

    def handler(request):
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=achunks())

    async def main() -> None:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
            async with client.stream("POST", OPENAI_URL, json={**BODY, "stream": True}) as response:
                [line async for line in response.aiter_lines()]

    asyncio.run(main())
    (record,) = read_records(records_path)
    assert record["response"]["text"] == "Hello"
    assert "time_to_first_token_ms" in record["latency"]


def test_one_call_is_recorded_once_even_with_both_libraries_patched(records_path, read_records):
    """Each library has its own Client class, so a call is observed by exactly one patch."""
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=REPLY))) as c:
        c.post(OPENAI_URL, json=BODY)
    with httpx2.Client(transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json=REPLY))) as c:
        c.post(OPENAI_URL, json=BODY)
    assert len(read_records(records_path)) == 2


def test_stop_restores_both_libraries(tmp_path):
    originals = (httpx.Client.send, httpx.AsyncClient.send, httpx2.Client.send, httpx2.AsyncClient.send)
    migration_eval.start(tmp_path / "r.jsonl")
    assert httpx2.Client.send is not originals[2]
    migration_eval.stop()
    assert (httpx.Client.send, httpx.AsyncClient.send, httpx2.Client.send, httpx2.AsyncClient.send) == originals
