"""Replay (EVL-3): send logged requests to a model at the logged pace, within rate limits.

The model-calling function is passed in (`complete`), not imported here. The real one builds
the provider's request (EVL-10) and sends it over HTTP; tests pass a fake. The clock and the
sleep are passed in too, so pacing can be tested without waiting.

Do not call `migration_eval.start()` anywhere in the harness: it would record the replays.

Open decisions (see the notes on EVL-3):
    * the speed factor: replaying a week of traffic in real time would take a week, so the
      logged gaps are divided by `speed`; how the caller picks it (NFR-8: a 200-request
      smoke run in under 30 minutes) is not decided
    * where the partner's rate limits come from (the ERD names them but not the source)
    * a crude `max_calls` stands in for the spend meter (RUN-5) until that exists; a request
      counts as one call however many times it is retried
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from migration_eval.models import CallRecord
from migration_eval.models.record import ModelRequest
from migration_eval.profile import context_tokens


@dataclass(frozen=True)
class ReplayResult:
    """What one replayed call produced."""

    text: str | None
    tool_calls: list[dict[str, Any]] | None  # v0-shaped: {"name", "arguments", ...}
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float | None
    error: str | None  # None on success


# Sends one request to a model: (provider, model, request) -> result. It reports a 429 by
# raising RateLimited; any other failure comes back as a ReplayResult with `error` set.
CompleteFn = Callable[[str, str, ModelRequest], Awaitable[ReplayResult]]

WINDOW_SECONDS = 60.0  # the rate limits are per minute
MAX_BACKOFF_SECONDS = 30.0


class RateLimited(Exception):
    """The provider answered 429. `retry_after` is the seconds it asked us to wait, if it said."""

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("rate limited")
        self.retry_after = retry_after


def error_result(message: str) -> ReplayResult:
    return ReplayResult(None, None, None, None, None, message)


def select_sample(records: list[CallRecord], n: int, seed: int) -> list[CallRecord]:
    """Pick `n` of the records at random, repeatably for a given seed.

    Return them in their original time order (by `started_at`), because replay uses the
    gaps between them. If there are `n` or fewer, return them all. Every candidate must be
    replayed on the same sample, so the caller selects once and reuses it.
    """
    # A private generator: seeding the global one would change randomness for the whole process.
    chosen = list(records) if len(records) <= n else random.Random(seed).sample(records, n)
    return sorted(chosen, key=lambda record: record.started_at)


def arrival_offsets(records: list[CallRecord], speed: float = 1.0) -> list[float]:
    """Seconds after the first request at which each request should be sent.

    The first offset is 0. Offsets follow the logged `started_at` gaps divided by `speed`
    (2.0 replays twice as fast). Raise ValueError if `speed` is not positive. The records
    must be in time order, as `select_sample` returns them.
    """
    if speed <= 0:
        raise ValueError("speed must be positive")
    if not records:
        return []

    first = records[0].started_at
    return [(record.started_at - first).total_seconds() / speed for record in records]


class RateLimiter:
    """Allows at most `requests_per_minute` requests and `tokens_per_minute` tokens a minute.

    Either limit may be None for "no limit". A sliding window over the last minute is enough.
    `now` and `sleep` are injected so tests do not wait.
    """

    def __init__(
        self,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        now: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        for limit in (requests_per_minute, tokens_per_minute):
            if limit is not None and limit <= 0:
                raise ValueError("a rate limit must be positive (use None for no limit)")
        self._requests_per_minute = requests_per_minute
        self._tokens_per_minute = tokens_per_minute
        self._now = now or time.monotonic  # monotonic: the wall clock can jump
        self._sleep = sleep or asyncio.sleep
        self._window: deque[tuple[float, int]] = deque()  # (time sent, tokens) in the last minute
        self._lock = asyncio.Lock()  # one waiter at a time, so two tasks cannot both claim the same room

    async def acquire(self, tokens: int = 0) -> None:
        """Wait until a request using about `tokens` tokens fits within both limits, then claim it."""
        if self._tokens_per_minute is not None:
            tokens = min(tokens, self._tokens_per_minute)  # an oversized request must not wait forever

        async with self._lock:
            while True:
                now = self._now()
                while self._window and self._window[0][0] <= now - WINDOW_SECONDS:
                    self._window.popleft()

                requests_ok = self._requests_per_minute is None or len(self._window) < self._requests_per_minute
                used_tokens = sum(t for _, t in self._window)
                tokens_ok = self._tokens_per_minute is None or used_tokens + tokens <= self._tokens_per_minute
                if requests_ok and tokens_ok:
                    self._window.append((now, tokens))
                    return

                # Room appears when the oldest request leaves the window.
                await self._sleep(max(self._window[0][0] + WINDOW_SECONDS - now, 0.0))


async def call_with_retry(
    complete: CompleteFn,
    provider: str,
    model: str,
    request: ModelRequest,
    max_retries: int = 3,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> ReplayResult:
    """Call `complete`; on RateLimited wait (the provider's `retry_after`, else a growing
    backoff) and try again, up to `max_retries` times.

    When retries run out, return a ReplayResult whose `error` says it was rate limited,
    rather than raising, so one throttled request does not abort the whole replay.
    """
    if max_retries < 0:
        raise ValueError("max_retries must not be negative")
    sleep = sleep or asyncio.sleep

    attempt = 0
    while True:
        try:
            return await complete(provider, model, request)
        except RateLimited as limited:
            if attempt >= max_retries:
                return error_result(f"rate limited after {max_retries} retries")
            wait = limited.retry_after if limited.retry_after is not None else min(2.0**attempt, MAX_BACKOFF_SECONDS)
            await sleep(wait)
            attempt += 1


async def replay(
    records: list[CallRecord],
    provider: str,
    model: str,
    complete: CompleteFn,
    *,
    speed: float = 1.0,
    limiter: RateLimiter | None = None,
    max_calls: int | None = None,
    now: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> list[ReplayResult]:
    """Replay each record's request against `provider`/`model`, one result per record, in order.

    Each request is sent at its `arrival_offsets` time or as soon as the limiter allows,
    whichever is later, using `call_with_retry`. Records without a request cannot be
    replayed: give them a ReplayResult with an `error` instead of skipping them, so the
    result list lines up with `records`. Stop with an error result for the rest once
    `max_calls` calls have been made.
    """
    now = now or time.monotonic
    sleep = sleep or asyncio.sleep
    offsets = arrival_offsets(records, speed)
    start = now()

    async def send(record: CallRecord) -> ReplayResult:
        if limiter is not None:
            await limiter.acquire(context_tokens(record) or 0)
        return await call_with_retry(complete, provider, model, record.request, sleep=sleep)

    results: list[ReplayResult | None] = [None] * len(records)
    tasks: dict[int, asyncio.Task[ReplayResult]] = {}
    calls = 0

    for index, (record, offset) in enumerate(zip(records, offsets)):
        if record.request is None:
            results[index] = error_result("record has no request")
            continue
        if max_calls is not None and calls >= max_calls:
            results[index] = error_result("call limit reached")
            continue
        calls += 1

        wait = start + offset - now()
        if wait > 0:
            await sleep(wait)
        # A task, not an await: a slow response must not delay the next request, or the
        # logged arrival pattern is lost.
        tasks[index] = asyncio.create_task(send(record))

    try:
        finished = await asyncio.gather(*tasks.values())
    except BaseException:
        for task in tasks.values():
            task.cancel()
        raise
    for index, result in zip(tasks, finished):
        results[index] = result

    return results  # type: ignore[return-value]  # every slot is filled by now
