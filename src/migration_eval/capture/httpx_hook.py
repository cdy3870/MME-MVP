"""Observe provider calls made through httpx (and its successor, httpx2).

The OpenAI, Anthropic and Google Python SDKs all send requests with `Client.send` or
`AsyncClient.send` of an httpx-family library, so wrapping those two methods sees every
call without touching the SDKs. Current OpenAI and Anthropic SDKs build their default
client on `httpx2`, older ones on `httpx`, so both are patched whenever they are
installed (the two have the same API). Requests to anything that is not a known provider
generate endpoint pass straight through untouched.

Two cases:

* Non-streaming: `send` has already read the body when it returns, so the full
  response is available immediately.
* Streaming: `send` returns before the body is read. We replace `response.stream` with a
  tee that copies each chunk as the caller consumes it and reports the call when the
  stream closes. The caller sees exactly the same bytes, in the same order.

Observation must never break the application: every failure here is logged (class name
only, never content) and swallowed. Request headers (including API keys) are never read.
"""

from __future__ import annotations

import functools
import importlib
import logging
import time
import zlib
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime, timezone
from types import ModuleType
from typing import Any

from migration_eval.capture.context import current_component
from migration_eval.capture.exchange import Exchange
from migration_eval.capture.providers import Provider, detect

logger = logging.getLogger("migration_eval")

Handler = Callable[[Exchange], None]

# HTTP libraries with the httpx API. Whichever are installed get patched.
LIBRARY_NAMES = ("httpx", "httpx2")

# library name -> (original Client.send, original AsyncClient.send), while installed
_originals: dict[str, tuple[Any, Any]] = {}


def installed_libraries() -> list[ModuleType]:
    libraries = []
    for name in LIBRARY_NAMES:
        try:
            libraries.append(importlib.import_module(name))
        except ImportError:
            continue
    return libraries


def _decode(raw: bytes, encoding: str) -> bytes | None:
    """Undo Content-Encoding on a streamed body. None means "can't", so the body is skipped."""
    encoding = encoding.strip().lower()
    if encoding in ("", "identity"):
        return raw
    if encoding == "gzip":
        return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)  # tolerates truncation
    if encoding == "deflate":
        try:
            return zlib.decompressobj().decompress(raw)
        except zlib.error:
            return zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)
    return None


class _Pending:
    """What we know about one in-flight call, captured when the request is sent."""

    def __init__(self, provider: Provider, request: Any, library: ModuleType) -> None:
        self.provider = provider
        self.url = str(request.url)
        try:
            self.request_body: bytes | None = request.content
        except library.RequestNotRead:  # the body is a generator we must not consume
            self.request_body = None
        # Read now: by the time a stream finishes we may be in a different context.
        self.component = current_component()
        self.started_at = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()
        self.ttft_ms: float | None = None
        self.chunks: list[bytes] = []
        self.finished = False

    def on_chunk(self, chunk: bytes) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = (time.perf_counter() - self._t0) * 1000
        self.chunks.append(chunk)

    def exchange(self, response: Any, body: bytes | None, complete: bool) -> Exchange:
        return Exchange(
            provider=self.provider,
            url=self.url,
            started_at=self.started_at,
            component=self.component,
            source_type="recorder",
            request_body=self.request_body,
            response_body=body,
            status_code=response.status_code,
            content_type=response.headers.get("content-type"),
            completed_at=datetime.now(timezone.utc),
            total_ms=(time.perf_counter() - self._t0) * 1000,
            ttft_ms=self.ttft_ms,
            complete=complete,
        )


def _emit(handler: Handler, exchange: Exchange) -> None:
    try:
        handler(exchange)
    except Exception as exc:  # noqa: BLE001
        logger.warning("migration_eval could not record a call: %s", type(exc).__name__)


def _finish_stream(pending: _Pending, response: Any, exhausted: bool, handler: Handler) -> None:
    if pending.finished:
        return
    pending.finished = True
    try:
        body = _decode(b"".join(pending.chunks), response.headers.get("content-encoding", ""))
        exchange = pending.exchange(response, body, complete=exhausted)
    except Exception as exc:  # noqa: BLE001
        logger.warning("migration_eval could not record a call: %s", type(exc).__name__)
        return
    _emit(handler, exchange)


@functools.cache
def _tee_classes(library: ModuleType) -> tuple[type, type]:
    """Stream wrappers built on this library's own base classes.

    httpx and httpx2 each define their own `SyncByteStream`/`AsyncByteStream`, and a
    client type-checks the stream it is given, so each library needs its own subclasses.
    """

    class SyncTee(library.SyncByteStream):  # type: ignore[name-defined,misc]
        def __init__(self, inner: Any, pending: _Pending, response: Any, handler: Handler) -> None:
            self._inner, self._pending, self._response, self._handler = inner, pending, response, handler
            self._exhausted = False

        def __iter__(self) -> Iterator[bytes]:
            for chunk in self._inner:
                self._pending.on_chunk(chunk)
                yield chunk
            self._exhausted = True

        def close(self) -> None:
            try:
                self._inner.close()
            finally:
                _finish_stream(self._pending, self._response, self._exhausted, self._handler)

    class AsyncTee(library.AsyncByteStream):  # type: ignore[name-defined,misc]
        def __init__(self, inner: Any, pending: _Pending, response: Any, handler: Handler) -> None:
            self._inner, self._pending, self._response, self._handler = inner, pending, response, handler
            self._exhausted = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            async for chunk in self._inner:
                self._pending.on_chunk(chunk)
                yield chunk
            self._exhausted = True

        async def aclose(self) -> None:
            try:
                await self._inner.aclose()
            finally:
                _finish_stream(self._pending, self._response, self._exhausted, self._handler)

    return SyncTee, AsyncTee


def _begin(request: Any, library: ModuleType) -> _Pending | None:
    if request.method != "POST":
        return None
    provider = detect(str(request.url))
    if provider is None:
        return None
    try:
        return _Pending(provider, request, library)
    except Exception as exc:  # noqa: BLE001
        logger.warning("migration_eval could not record a call: %s", type(exc).__name__)
        return None


def _observe(
    pending: _Pending, response: Any, handler: Handler, library: ModuleType, *, is_async: bool
) -> None:
    try:
        if response.is_stream_consumed:  # send() already read it: the non-streaming case
            _emit(handler, pending.exchange(response, response.content, complete=True))
        else:
            sync_tee, async_tee = _tee_classes(library)
            tee = async_tee if is_async else sync_tee
            response.stream = tee(response.stream, pending, response, handler)
    except Exception as exc:  # noqa: BLE001
        logger.warning("migration_eval could not record a call: %s", type(exc).__name__)


def _patch(library: ModuleType, handler: Handler) -> None:
    sync_send, async_send = library.Client.send, library.AsyncClient.send

    @functools.wraps(sync_send)
    def send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        pending = _begin(request, library)
        response = sync_send(self, request, *args, **kwargs)
        if pending is not None:
            _observe(pending, response, handler, library, is_async=False)
        return response

    @functools.wraps(async_send)
    async def asend(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        pending = _begin(request, library)
        response = await async_send(self, request, *args, **kwargs)
        if pending is not None:
            _observe(pending, response, handler, library, is_async=True)
        return response

    library.Client.send = send
    library.AsyncClient.send = asend
    _originals[library.__name__] = (sync_send, async_send)


def install(handler: Handler) -> None:
    """Start reporting every provider call to `handler`. Safe to call twice."""
    uninstall()
    for library in installed_libraries():
        _patch(library, handler)


def uninstall() -> None:
    for name, (sync_send, async_send) in list(_originals.items()):
        library = importlib.import_module(name)
        library.Client.send = sync_send
        library.AsyncClient.send = async_send
    _originals.clear()
