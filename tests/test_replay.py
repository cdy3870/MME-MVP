"""Replay (EVL-3): sampling, pacing, rate limiting, retries."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone

import pytest

from migration_eval.models import CallRecord
from migration_eval.replay import (
    RateLimited,
    RateLimiter,
    ReplayResult,
    arrival_offsets,
    call_with_retry,
    replay,
    select_sample,
)

T0 = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
OK = ReplayResult("ok", None, 10, 5, 100.0, None)


def rec(seconds: float, id: str | None = None, request: bool = True) -> CallRecord:
    data = {
        "schema_version": "v0", "id": id or f"r{seconds}", "component": "c", "provider": "openai",
        "model": "m", "started_at": (T0 + timedelta(seconds=seconds)).isoformat(),
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }  # fmt: skip
    if request:
        data["request"] = {"messages": [{"role": "user", "content": f"hello {seconds}"}]}
    return CallRecord.model_validate(data)


class FakeClock:
    """Time that only moves when someone sleeps. Sleeping yields first so tasks that were just
    started run at the current time, then advances the clock."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        await asyncio.sleep(0)
        self.t += seconds


# --- select_sample ------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_sample():
    records = [rec(i) for i in range(50)]
    assert [r.id for r in select_sample(records, 10, seed=1)] == [r.id for r in select_sample(records, 10, seed=1)]


def test_different_seeds_give_different_samples():
    records = [rec(i) for i in range(50)]
    assert {r.id for r in select_sample(records, 10, seed=1)} != {r.id for r in select_sample(records, 10, seed=2)}


def test_the_sample_is_returned_in_time_order():
    records = [rec(i) for i in range(50)]
    shuffled = records[::-1]
    sample = select_sample(shuffled, 10, seed=3)
    assert [r.started_at for r in sample] == sorted(r.started_at for r in sample)
    assert len(sample) == 10


def test_asking_for_more_than_exist_returns_them_all_in_time_order():
    records = [rec(5), rec(1), rec(3)]
    assert [r.id for r in select_sample(records, 10, seed=0)] == ["r1", "r3", "r5"]
    assert select_sample([], 5, seed=0) == []


def test_sampling_does_not_disturb_the_global_random_state():
    random.seed(1234)
    expected = random.random()
    random.seed(1234)
    select_sample([rec(i) for i in range(50)], 10, seed=99)
    assert random.random() == expected


# --- arrival_offsets ----------------------------------------------------------------------


def test_offsets_follow_the_logged_gaps_starting_at_zero():
    assert arrival_offsets([rec(10), rec(12), rec(15.5)]) == [0, 2, 5.5]


def test_speed_compresses_the_gaps():
    assert arrival_offsets([rec(0), rec(10), rec(20)], speed=2.0) == [0, 5, 10]


def test_no_records_gives_no_offsets():
    assert arrival_offsets([]) == []


@pytest.mark.parametrize("speed", [0, -1])
def test_a_speed_that_is_not_positive_is_an_error(speed):
    with pytest.raises(ValueError, match="positive"):
        arrival_offsets([rec(0)], speed=speed)


# --- RateLimiter --------------------------------------------------------------------------


def limiter(clock: FakeClock, **kwargs) -> RateLimiter:
    return RateLimiter(now=clock.now, sleep=clock.sleep, **kwargs)


def test_requests_within_the_limit_do_not_wait():
    clock = FakeClock()
    lim = limiter(clock, requests_per_minute=3)

    async def go():
        for _ in range(3):
            await lim.acquire()

    asyncio.run(go())
    assert clock.sleeps == [] and clock.t == 0


def test_a_request_over_the_limit_waits_until_the_oldest_leaves_the_window():
    clock = FakeClock()
    lim = limiter(clock, requests_per_minute=2)

    async def go():
        await lim.acquire()  # t=0
        clock.t = 10
        await lim.acquire()  # t=10
        await lim.acquire()  # must wait until the t=0 request is a minute old

    asyncio.run(go())
    assert clock.t == 60 and clock.sleeps == [50]


def test_the_token_limit_is_enforced_separately():
    clock = FakeClock()
    lim = limiter(clock, tokens_per_minute=1000)

    async def go():
        await lim.acquire(600)
        await lim.acquire(300)  # 900 used: fits
        await lim.acquire(200)  # 1100: waits for the first to leave

    asyncio.run(go())
    assert clock.t == 60


def test_no_limits_never_wait():
    clock = FakeClock()
    lim = limiter(clock)

    async def go():
        for _ in range(100):
            await lim.acquire(10**9)

    asyncio.run(go())
    assert clock.sleeps == []


def test_a_request_bigger_than_the_whole_token_limit_does_not_wait_forever():
    clock = FakeClock()
    lim = limiter(clock, tokens_per_minute=100)
    asyncio.run(lim.acquire(10_000))  # would never fit if taken literally
    assert clock.sleeps == []


@pytest.mark.parametrize("kwargs", [{"requests_per_minute": 0}, {"tokens_per_minute": -5}])
def test_a_limit_that_is_not_positive_is_an_error(kwargs):
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(**kwargs)


def test_overlapping_requests_share_the_limit():
    clock = FakeClock()
    lim = limiter(clock, requests_per_minute=1)
    sent: list[float] = []

    async def one():
        await lim.acquire()
        sent.append(clock.t)

    async def go():
        await asyncio.gather(one(), one(), one())

    asyncio.run(go())
    assert sent == [0, 60, 120]


# --- call_with_retry ----------------------------------------------------------------------


def scripted(*outcomes):
    """A fake `complete` that raises or returns each outcome in turn, counting calls."""
    calls = {"n": 0}

    async def complete(provider, model, request):
        outcome = outcomes[min(calls["n"], len(outcomes) - 1)]
        calls["n"] += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return complete, calls


def test_a_successful_call_is_returned_untouched_without_retrying():
    complete, calls = scripted(OK)
    clock = FakeClock()
    assert asyncio.run(call_with_retry(complete, "openai", "m", None, sleep=clock.sleep)) is OK
    assert calls["n"] == 1 and clock.sleeps == []


def test_a_rate_limited_call_is_retried_with_a_growing_backoff():
    complete, calls = scripted(RateLimited(), RateLimited(), OK)
    clock = FakeClock()
    assert asyncio.run(call_with_retry(complete, "openai", "m", None, sleep=clock.sleep)) is OK
    assert calls["n"] == 3 and clock.sleeps == [1.0, 2.0]


def test_the_providers_retry_after_beats_the_backoff():
    complete, _ = scripted(RateLimited(retry_after=7.5), OK)
    clock = FakeClock()
    asyncio.run(call_with_retry(complete, "openai", "m", None, sleep=clock.sleep))
    assert clock.sleeps == [7.5]


def test_the_backoff_is_capped():
    complete, _ = scripted(*([RateLimited()] * 9), OK)
    clock = FakeClock()
    asyncio.run(call_with_retry(complete, "openai", "m", None, max_retries=9, sleep=clock.sleep))
    assert max(clock.sleeps) == 30.0


def test_running_out_of_retries_returns_an_error_result_instead_of_raising():
    complete, calls = scripted(RateLimited())
    result = asyncio.run(call_with_retry(complete, "openai", "m", None, max_retries=2, sleep=FakeClock().sleep))
    assert calls["n"] == 3  # the first try plus two retries
    assert result.error == "rate limited after 2 retries" and result.text is None


def test_a_failure_that_is_not_a_429_comes_back_as_the_result_with_no_retry():
    failed = ReplayResult(None, None, None, None, 50.0, "server error")
    complete, calls = scripted(failed)
    assert asyncio.run(call_with_retry(complete, "openai", "m", None, sleep=FakeClock().sleep)) is failed
    assert calls["n"] == 1


def test_an_unexpected_exception_is_not_swallowed():
    complete, _ = scripted(RuntimeError("a bug"))
    with pytest.raises(RuntimeError, match="a bug"):
        asyncio.run(call_with_retry(complete, "openai", "m", None, sleep=FakeClock().sleep))


# --- replay -------------------------------------------------------------------------------


def recording(clock: FakeClock):
    """A fake `complete` that logs when each request was sent, and answers with its own text."""
    log: list[tuple[float, str]] = []

    async def complete(provider, model, request):
        text = request.messages[0].content
        log.append((clock.t, text))
        return ReplayResult(text, None, 1, 1, 1.0, None)

    return complete, log


def test_requests_are_sent_at_their_logged_times():
    clock = FakeClock()
    complete, log = recording(clock)
    records = [rec(0), rec(2), rec(7)]
    asyncio.run(replay(records, "openai", "m", complete, now=clock.now, sleep=clock.sleep))
    assert [t for t, _ in log] == [0, 2, 7]


def test_speed_scales_the_send_times():
    clock = FakeClock()
    complete, log = recording(clock)
    asyncio.run(replay([rec(0), rec(10), rec(30)], "openai", "m", complete, speed=10.0, now=clock.now, sleep=clock.sleep))
    assert [t for t, _ in log] == [0, 1, 3]


def test_results_line_up_with_the_records_in_order():
    clock = FakeClock()
    complete, _ = recording(clock)
    records = [rec(0), rec(1), rec(2)]
    results = asyncio.run(replay(records, "openai", "m", complete, now=clock.now, sleep=clock.sleep))
    assert [r.text for r in results] == ["hello 0", "hello 1", "hello 2"]


def test_a_record_without_a_request_gets_an_error_in_its_place():
    clock = FakeClock()
    complete, log = recording(clock)
    records = [rec(0), rec(1, request=False), rec(2)]
    results = asyncio.run(replay(records, "openai", "m", complete, now=clock.now, sleep=clock.sleep))
    assert [r.error for r in results] == [None, "record has no request", None]
    assert len(log) == 2  # nothing was sent for it


def test_the_call_limit_stops_sending_and_marks_the_rest():
    clock = FakeClock()
    complete, log = recording(clock)
    results = asyncio.run(replay([rec(i) for i in range(5)], "openai", "m", complete, max_calls=2, now=clock.now, sleep=clock.sleep))
    assert len(log) == 2
    assert [r.error for r in results] == [None, None, "call limit reached", "call limit reached", "call limit reached"]


def test_the_rate_limiter_delays_requests_that_would_exceed_the_limit():
    clock = FakeClock()
    complete, log = recording(clock)
    lim = RateLimiter(requests_per_minute=1, now=clock.now, sleep=clock.sleep)
    asyncio.run(replay([rec(0), rec(1)], "openai", "m", complete, limiter=lim, now=clock.now, sleep=clock.sleep))
    assert [t for t, _ in log] == [0, 60]  # the second was due at 1 but waited for room


def test_a_throttled_request_does_not_abort_the_others():
    calls = {"n": 0}

    async def complete(provider, model, request):
        calls["n"] += 1
        if "hello 1" in request.messages[0].content:
            raise RateLimited()
        return OK

    clock = FakeClock()
    results = asyncio.run(replay([rec(0), rec(1), rec(2)], "openai", "m", complete, now=clock.now, sleep=clock.sleep))
    assert [r.error for r in results] == [None, "rate limited after 3 retries", None]


def test_a_slow_response_does_not_delay_the_next_request():
    """Real time, scaled down: sends are 50ms apart and each response takes 300ms."""
    sent: list[float] = []

    async def slow(provider, model, request):
        sent.append(asyncio.get_running_loop().time())
        await asyncio.sleep(0.3)
        return OK

    asyncio.run(replay([rec(0), rec(1), rec(2)], "openai", "m", slow, speed=20.0))
    gaps = [b - a for a, b in zip(sent, sent[1:])]
    assert all(0.03 < gap < 0.2 for gap in gaps), gaps  # about 50ms, not 300ms


def test_nothing_to_replay_returns_nothing():
    clock = FakeClock()
    complete, log = recording(clock)
    assert asyncio.run(replay([], "openai", "m", complete, now=clock.now, sleep=clock.sleep)) == []
    assert log == []
