"""Grading (REC-4): what each record can be used for, and why."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from migration_eval.capture.grading import (
    REPLAY_GRADE_TARGET,
    Grade,
    GradeResult,
    Reason,
    format_summary,
    grade,
    summarize,
)
from migration_eval.models import CallRecord

EXAMPLES = Path(__file__).resolve().parents[2] / "schema" / "examples" / "valid"


def example(name: str, **changes) -> CallRecord:
    data = json.loads((EXAMPLES / f"{name}.json").read_text())
    data.update(changes)
    return CallRecord.model_validate(data)


def test_a_complete_record_is_replay_grade():
    assert grade(example("replay_grade_chat")) == GradeResult(Grade.REPLAY_GRADE)


def test_a_tool_call_only_answer_is_still_replay_grade():
    result = grade(example("replay_grade_tool_use_streamed"))  # text is null, has tool_calls
    assert result.grade == Grade.REPLAY_GRADE


def test_timing_and_tokens_without_content_is_speed_and_cost_only():
    # This example has no request or response at all, so it must not fall into "insufficient".
    result = grade(example("speed_and_cost_only"))
    assert result.grade == Grade.SPEED_AND_COST_ONLY
    assert result.reasons == (Reason.REQUEST_MISSING, Reason.RESPONSE_MISSING)


def test_a_failed_call_is_insufficient_even_with_latency():
    result = grade(example("failed_call"))  # carries latency but the call failed
    assert result.grade == Grade.INSUFFICIENT
    assert Reason.RESPONSE_ERROR in result.reasons


def test_a_failed_call_stays_insufficient_even_if_it_has_tokens_and_latency():
    record = example("failed_call", usage={"input_tokens": 5, "output_tokens": 0})
    assert grade(record).grade == Grade.INSUFFICIENT


def test_a_stream_that_ended_early_says_so():
    record = example(
        "replay_grade_chat",
        response={"status": "error", "text": "Hel", "error": {"type": "stream_closed_early"}},
    )
    result = grade(record)
    assert result.grade == Grade.INSUFFICIENT
    assert Reason.STREAM_INCOMPLETE in result.reasons
    assert Reason.RESPONSE_ERROR not in result.reasons


def test_a_successful_response_with_no_text_or_tool_calls_is_not_replay_grade():
    record = example("replay_grade_chat", response={"status": "success", "finish_reason": "content_filter"})
    assert Reason.RESPONSE_EMPTY in grade(record).reasons


def test_empty_message_list_counts_as_no_request():
    record = example("replay_grade_chat", request={"messages": []})
    assert Reason.REQUEST_MISSING in grade(record).reasons


def test_a_record_with_nothing_usable_lists_every_reason():
    record = CallRecord.model_validate(
        {
            "schema_version": "v0", "id": "r", "component": "c", "provider": "openai",
            "model": "m", "started_at": "2026-10-12T14:03:21Z",
        }  # fmt: skip
    )
    result = grade(record)
    assert result.grade == Grade.INSUFFICIENT
    assert set(result.reasons) == {
        Reason.REQUEST_MISSING, Reason.RESPONSE_MISSING, Reason.LATENCY_MISSING, Reason.USAGE_MISSING,
    }  # fmt: skip


def test_a_parse_error_from_capture_is_named_as_a_reason():
    record = example(
        "speed_and_cost_only", metadata={"parse_error": "request: Unsupported: tool type 'web_search'"}
    )
    assert Reason.PARSE_ERROR in grade(record).reasons


# --- summary -----------------------------------------------------------------------------


def results(*grades: Grade) -> list[GradeResult]:
    return [GradeResult(g) for g in grades]


def test_summary_counts_and_percentages():
    summary = summarize(
        results(Grade.REPLAY_GRADE, Grade.REPLAY_GRADE, Grade.SPEED_AND_COST_ONLY, Grade.INSUFFICIENT)
    )
    assert summary.total == 4
    assert summary.counts == {
        Grade.REPLAY_GRADE: 2, Grade.SPEED_AND_COST_ONLY: 1, Grade.INSUFFICIENT: 1,
    }  # fmt: skip
    assert summary.replay_grade_fraction == 0.5


@pytest.mark.parametrize(
    ("replay", "other", "meets"),
    [(9, 1, True), (8, 2, False), (10, 0, True), (89, 11, False)],
)
def test_the_ninety_percent_target(replay, other, meets):
    summary = summarize(results(*[Grade.REPLAY_GRADE] * replay, *[Grade.INSUFFICIENT] * other))
    assert summary.meets_target is meets
    assert REPLAY_GRADE_TARGET == 0.9


def test_summary_of_nothing_does_not_divide_by_zero():
    summary = summarize([])
    assert (summary.total, summary.replay_grade_fraction, summary.meets_target) == (0, 0.0, False)
    assert "Replay-grade" in format_summary(summary)


def test_summary_tallies_reasons_across_records():
    summary = summarize(
        [
            GradeResult(Grade.INSUFFICIENT, (Reason.RESPONSE_ERROR,)),
            GradeResult(Grade.INSUFFICIENT, (Reason.RESPONSE_ERROR, Reason.USAGE_MISSING)),
            GradeResult(Grade.REPLAY_GRADE),
        ]
    )
    assert summary.reasons == {Reason.RESPONSE_ERROR: 2, Reason.USAGE_MISSING: 1}


def test_formatted_summary_reads_well():
    summary = summarize(
        [GradeResult(Grade.REPLAY_GRADE)] * 3
        + [GradeResult(Grade.INSUFFICIENT, (Reason.RESPONSE_ERROR,))] * 2
    )
    text = format_summary(summary)
    assert "Replay-grade" in text and "60.0%" in text and "below the 90% target" in text
    assert "2 x response_error: the provider call failed" in text
