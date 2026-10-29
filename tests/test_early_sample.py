"""Early sample (EVL-2): run a candidate in batches and stop when it is clearly worse than the noise floor."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from migration_eval.early_sample import (
    MAX_REQUESTS,
    count_agreements,
    proportion_interval,
    run_candidate,
    should_stop,
)
from migration_eval.models import CallRecord
from migration_eval.replay import ReplayResult

T0 = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)


def result(text="ok", error=None) -> ReplayResult:
    return ReplayResult(text, None, 10, 5, 100.0, error)


def rec(i: int) -> CallRecord:
    return CallRecord.model_validate(
        {
            "schema_version": "v0", "id": f"r{i}", "component": "c", "provider": "openai", "model": "m",
            "started_at": (T0 + timedelta(seconds=i)).isoformat(),
            "request": {"messages": [{"role": "user", "content": f"q{i}"}]},
        }  # fmt: skip
    )


# --- proportion_interval ------------------------------------------------------------------


def test_interval_matches_a_known_wilson_value():
    # 80 of 100 at 95%: the textbook Wilson interval is about (0.7112, 0.8666).
    lower, upper = proportion_interval(80, 100, confidence=0.95)
    assert lower == pytest.approx(0.7112, abs=1e-3) and upper == pytest.approx(0.8666, abs=1e-3)


def test_interval_contains_the_observed_proportion_and_stays_in_0_1():
    for agreed in range(0, 51):
        lower, upper = proportion_interval(agreed, 50)
        assert 0 <= lower <= agreed / 50 <= upper <= 1


def test_all_agreeing_has_an_upper_bound_of_exactly_one():
    assert proportion_interval(50, 50)[1] == 1.0
    assert proportion_interval(0, 50)[0] == 0.0


def test_more_data_gives_a_narrower_interval():
    small = proportion_interval(40, 50)
    large = proportion_interval(160, 200)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_higher_confidence_gives_a_wider_interval():
    narrow = proportion_interval(40, 50, confidence=0.90)
    wide = proportion_interval(40, 50, confidence=0.99)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


@pytest.mark.parametrize("agreed, n", [(1, 0), (-1, 10), (11, 10)])
def test_interval_rejects_impossible_counts(agreed, n):
    with pytest.raises(ValueError):
        proportion_interval(agreed, n)


# --- should_stop --------------------------------------------------------------------------


def test_never_stops_before_the_minimum_even_if_every_request_disagreed():
    assert not should_stop(0, 49, noise_floor=0.95)
    assert should_stop(0, 50, noise_floor=0.95)


def test_stops_when_even_the_upper_bound_is_below_the_floor():
    assert should_stop(20, 50, noise_floor=0.9)  # 40% agreement, upper bound well under 0.9


def test_does_not_stop_when_the_interval_still_reaches_the_floor():
    assert not should_stop(44, 50, noise_floor=0.9)  # 88% agreement: plausibly at the floor


def test_a_candidate_that_always_agrees_is_never_stopped_even_at_a_floor_of_one():
    assert not should_stop(50, 50, noise_floor=1.0)
    assert not should_stop(200, 200, noise_floor=1.0)


def test_stops_returns_a_real_bool_not_none():
    assert should_stop(50, 50, noise_floor=0.5) is False
    assert should_stop(10, 20, noise_floor=0.5) is False  # below the minimum


# --- count_agreements ---------------------------------------------------------------------


def test_counts_a_request_when_most_incumbent_runs_agree():
    candidate = [result("a"), result("b"), result("c")]
    # Three runs, three requests. Request 0: all agree. Request 1: 2 of 3. Request 2: 1 of 3.
    runs = [
        [result("a"), result("b"), result("c")],
        [result("a"), result("b"), result("X")],
        [result("a"), result("Y"), result("Z")],
    ]
    assert count_agreements(candidate, runs) == 2


def test_an_even_split_is_not_a_majority():
    assert count_agreements([result("a")], [[result("a")], [result("b")]]) == 0


def test_runs_are_indexed_by_run_then_request():
    # If the nesting were read the other way round (request, then run) this would crash or miscount.
    candidate = [result("a"), result("b")]
    runs = [[result("a"), result("b")]] * 3
    assert count_agreements(candidate, runs) == 2


def test_a_failed_candidate_result_never_agrees():
    assert count_agreements([result(error="boom")], [[result("ok")]] * 3) == 0


def test_a_custom_comparator_is_used():
    assert count_agreements([result("a")], [[result("zzz")]] * 3, comparator=lambda a, b: True) == 1


def test_no_candidate_results_is_zero():
    assert count_agreements([], [[], [], []]) == 0


# --- run_candidate ------------------------------------------------------------------------


def make_sample(n: int) -> list[CallRecord]:
    return [rec(i) for i in range(n)]


def incumbent(n: int, runs: int = 3) -> list[list[ReplayResult]]:
    return [[result(f"answer {i}") for i in range(n)] for _ in range(runs)]


def candidate_model(agree: bool | list[bool]):
    """A fake model that gives the incumbent's answer (or a different one) for each request."""
    calls: list[str] = []

    async def complete(provider, model, request):
        question = request.messages[0].content
        calls.append(question)
        i = int(question[1:])
        ok = agree if isinstance(agree, bool) else agree[i]
        return result(f"answer {i}" if ok else "something else")

    return complete, calls


def run(sample, floor, complete, **kwargs):
    return asyncio.run(
        run_candidate("openai", "cand", sample, incumbent(len(sample)), floor, complete, speed=1000.0, **kwargs)
    )


def test_a_matching_candidate_runs_every_request_and_is_not_stopped():
    complete, calls = candidate_model(True)
    outcome = run(make_sample(100), 0.9, complete)
    assert outcome.requests_run == 100 and len(outcome.results) == 100 and len(calls) == 100
    assert outcome.agreement == 1.0 and outcome.stopped_early is False
    assert (outcome.provider, outcome.model) == ("openai", "cand")


def test_a_clearly_worse_candidate_stops_at_the_first_look_after_the_minimum():
    complete, calls = candidate_model(False)
    outcome = run(make_sample(200), 0.9, complete)
    # Batches of 25: the first check allowed is after 50 requests, and it stops there.
    assert outcome.stopped_early is True and outcome.requests_run == 50 and len(calls) == 50
    assert outcome.agreement == 0.0


def test_it_does_not_stop_before_min_requests_even_when_every_request_disagrees():
    complete, _ = candidate_model(False)
    outcome = run(make_sample(200), 0.9, complete, batch_size=10, min_requests=30)
    assert outcome.requests_run == 30 and outcome.stopped_early is True


def test_it_never_runs_more_than_max_requests():
    complete, calls = candidate_model(True)
    outcome = run(make_sample(300), 0.9, complete)
    assert MAX_REQUESTS == 200 and outcome.requests_run == 200 and len(calls) == 200


def test_a_sample_that_is_not_a_multiple_of_the_batch_size_still_runs_every_request():
    complete, calls = candidate_model(True)
    outcome = run(make_sample(60), 0.9, complete, batch_size=25)
    assert outcome.requests_run == 60 and len(calls) == 60  # 25 + 25 + 10


def test_results_line_up_with_the_first_requests_of_the_sample():
    complete, calls = candidate_model(True)
    outcome = run(make_sample(60), 0.9, complete, batch_size=25)
    assert calls == [f"q{i}" for i in range(60)]
    assert [r.text for r in outcome.results] == [f"answer {i}" for i in range(60)]


def test_agreement_is_tracked_across_batches_not_just_the_last_one():
    # The first 30 requests agree, the rest do not: 30 of 100 overall.
    complete, _ = candidate_model([i < 30 for i in range(100)])
    outcome = run(make_sample(100), 0.0, complete)  # a floor of 0 means it never stops
    assert outcome.requests_run == 100 and outcome.agreement == pytest.approx(0.30)


def test_a_sample_smaller_than_one_batch_runs_once():
    complete, calls = candidate_model(True)
    outcome = run(make_sample(10), 0.9, complete)
    assert outcome.requests_run == 10 and outcome.stopped_early is False


def test_an_empty_sample_gives_no_agreement():
    complete, calls = candidate_model(True)
    outcome = run([], 0.9, complete)
    assert outcome.requests_run == 0 and outcome.agreement is None and outcome.results == [] and calls == []


def test_replay_options_are_passed_through():
    complete, _ = candidate_model(True)
    outcome = run(make_sample(5), 0.9, complete, max_calls=2)
    assert [r.error for r in outcome.results] == [None, None] + ["call limit reached"] * 3


def test_a_custom_comparator_decides_agreement():
    complete, _ = candidate_model(False)
    outcome = run(make_sample(30), 0.9, complete, comparator=lambda a, b: True)
    assert outcome.agreement == 1.0
