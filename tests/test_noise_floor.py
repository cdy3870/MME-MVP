"""Noise floor (EVL-1): replaying the incumbent and measuring its agreement with itself."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from migration_eval.models import CallRecord
from migration_eval.noise_floor import NOISE_FLOOR_RUNS, noise_floor, replay_incumbent, same_output
from migration_eval.replay import ReplayResult

T0 = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)


def result(text="ok", calls=None, error=None) -> ReplayResult:
    return ReplayResult(text, calls, 10, 5, 100.0, error)


def rec(i: int) -> CallRecord:
    return CallRecord.model_validate(
        {
            "schema_version": "v0", "id": f"r{i}", "component": "c", "provider": "openai", "model": "m",
            "started_at": (T0 + timedelta(seconds=i)).isoformat(),
            "request": {"messages": [{"role": "user", "content": f"q{i}"}]},
        }  # fmt: skip
    )


# --- same_output --------------------------------------------------------------------------


def test_equal_text_agrees_even_with_different_whitespace():
    assert same_output(result("hello"), result("  hello \n"))


def test_different_text_disagrees():
    assert not same_output(result("hello"), result("goodbye"))


def test_tool_calls_are_compared_by_name_and_arguments_not_by_id():
    a = result(None, [{"id": "toolu_1", "name": "get_order", "arguments": {"id": "8841"}}])
    b = result(None, [{"id": "toolu_2", "name": "get_order", "arguments": {"id": "8841"}}])
    assert same_output(a, b)


def test_a_different_tool_or_different_arguments_disagrees():
    base = result(None, [{"name": "get_order", "arguments": {"id": "8841"}}])
    assert not same_output(base, result(None, [{"name": "cancel_order", "arguments": {"id": "8841"}}]))
    assert not same_output(base, result(None, [{"name": "get_order", "arguments": {"id": "9999"}}]))
    assert not same_output(base, result(None))


def test_an_answer_with_no_text_does_not_crash():
    assert same_output(result(None), result(None))
    assert not same_output(result(None), result("something"))


@pytest.mark.parametrize("a, b", [(result(error="boom"), result("ok")), (result("ok"), result(error="boom")), (result(error="x"), result(error="x"))])
def test_a_result_with_an_error_agrees_with_nothing_even_another_error(a, b):
    assert not same_output(a, b)


# --- replay_incumbent ---------------------------------------------------------------------


def recording():
    log: list[str] = []

    async def complete(provider, model, request):
        log.append(request.messages[0].content)
        return result(f"answer to {request.messages[0].content}")

    return complete, log


def run_incumbent(sample, complete, **kwargs):
    return asyncio.run(replay_incumbent(sample, "openai", "m", complete, speed=1000.0, **kwargs))


def test_it_runs_the_sample_three_times_by_default():
    complete, log = recording()
    runs = run_incumbent([rec(0), rec(1)], complete)
    assert NOISE_FLOOR_RUNS == 3 and len(runs) == 3 and all(len(run) == 2 for run in runs)
    assert len(log) == 6


def test_each_run_lines_up_with_the_sample():
    complete, _ = recording()
    runs = run_incumbent([rec(0), rec(1)], complete, runs=2)
    assert [[r.text for r in run] for run in runs] == [["answer to q0", "answer to q1"]] * 2


def test_runs_go_one_after_another_not_interleaved():
    complete, log = recording()
    run_incumbent([rec(0), rec(1)], complete, runs=2)
    assert log == ["q0", "q1", "q0", "q1"]  # not q0, q0, q1, q1


def test_replay_options_are_passed_through_to_replay():
    complete, _ = recording()
    runs = run_incumbent([rec(0), rec(1), rec(2)], complete, runs=2, max_calls=1)
    assert [[r.error for r in run] for run in runs] == [[None, "call limit reached", "call limit reached"]] * 2


def test_zero_runs_gives_no_runs():
    complete, log = recording()
    assert run_incumbent([rec(0)], complete, runs=0) == [] and log == []


# --- noise_floor --------------------------------------------------------------------------


def test_a_model_that_always_agrees_with_itself_has_a_floor_of_one():
    run = [result("a"), result("b")]
    assert noise_floor([run, run, run]) == 1.0


def test_a_model_that_never_agrees_has_a_floor_of_zero():
    assert noise_floor([[result("a")], [result("b")], [result("c")]]) == 0.0


def test_every_pair_of_runs_is_compared_for_every_request():
    # Two requests, three runs: 3 pairs each, so 6 comparisons. In request 0, run 3 differs,
    # which breaks pairs (1,3) and (2,3) but not (1,2); request 1 is fully stable.
    runs = [
        [result("x"), result("y")],
        [result("x"), result("y")],
        [result("DIFFERENT"), result("y")],
    ]
    assert noise_floor(runs) == pytest.approx(4 / 6)


def test_two_runs_give_one_comparison_per_request():
    assert noise_floor([[result("a"), result("b")], [result("a"), result("DIFFERENT")]]) == 0.5


def test_a_custom_comparator_is_used_and_receives_results_not_runs():
    seen = []

    def comparator(a, b):
        seen.append((a.text, b.text))
        return True

    assert noise_floor([[result("a1"), result("a2")], [result("b1"), result("b2")]], comparator) == 1.0
    assert seen == [("a1", "b1"), ("a2", "b2")]


def test_failed_replays_count_as_disagreement():
    runs = [[result("a"), result("b")], [result("a"), result(error="rate limited")]]
    assert noise_floor(runs) == 0.5


@pytest.mark.parametrize("runs", [[], [[result("a")]], [[], []], [[], [], []]])
def test_nothing_to_compare_is_none_not_a_crash(runs):
    assert noise_floor(runs) is None


def test_runs_of_different_lengths_are_an_error():
    with pytest.raises(ValueError, match="same number"):
        noise_floor([[result("a")], [result("a"), result("b")]])


def test_end_to_end_a_stable_fake_model_has_a_floor_of_one():
    complete, _ = recording()
    runs = run_incumbent([rec(0), rec(1), rec(2)], complete)
    assert noise_floor(runs) == 1.0


def test_end_to_end_a_flaky_fake_model_has_a_floor_below_one():
    counter = {"n": 0}

    async def flaky(provider, model, request):
        counter["n"] += 1
        return result(f"v{counter['n'] % 3}")  # a cycle of 3 does not line up with 2 requests per run

    runs = run_incumbent([rec(0), rec(1)], flaky)
    assert noise_floor(runs) < 1
