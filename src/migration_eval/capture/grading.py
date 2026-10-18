"""Grade each record by what it can be used for (REC-4).

    replay-grade         enough to replay the request and compare the answer
    speed and cost only  no usable content, but latency and token counts are known
    insufficient         neither

The grade is derived from the record each time rather than stored in it, so the rules can
change without re-recording anything. Each result carries the reasons the record fell
short, so the validation summary can say exactly what is missing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from migration_eval.models import CallRecord

# The share of records that must be replay-grade (ERD success measure: at least 90%).
REPLAY_GRADE_TARGET = 0.9


class Grade(StrEnum):
    REPLAY_GRADE = "replay_grade"
    SPEED_AND_COST_ONLY = "speed_and_cost_only"
    INSUFFICIENT = "insufficient"


class Reason(StrEnum):
    """Why a record is not replay-grade (and, if insufficient, not usable for speed/cost)."""

    REQUEST_MISSING = "request_missing"
    RESPONSE_MISSING = "response_missing"
    RESPONSE_ERROR = "response_error"
    STREAM_INCOMPLETE = "stream_incomplete"
    RESPONSE_EMPTY = "response_empty"
    PARSE_ERROR = "parse_error"
    LATENCY_MISSING = "latency_missing"
    USAGE_MISSING = "usage_missing"


REASON_TEXT: dict[Reason, str] = {
    Reason.REQUEST_MISSING: "no request messages were captured",
    Reason.RESPONSE_MISSING: "no response was captured",
    Reason.RESPONSE_ERROR: "the provider call failed",
    Reason.STREAM_INCOMPLETE: "the stream ended before the response was complete",
    Reason.RESPONSE_EMPTY: "the response has no text and no tool calls",
    Reason.PARSE_ERROR: "part of the call could not be parsed",
    Reason.LATENCY_MISSING: "no latency was recorded",
    Reason.USAGE_MISSING: "no input/output token counts were recorded",
}


@dataclass(frozen=True)
class GradeResult:
    grade: Grade
    # Why it is not replay-grade; empty for replay-grade records.
    reasons: tuple[Reason, ...] = ()


def _replay_problems(record: CallRecord) -> list[Reason]:
    """What stops this record from being replay-grade. Empty means it is."""
    problems: list[Reason] = []

    if record.request is None or not record.request.messages:
        problems.append(Reason.REQUEST_MISSING)

    response = record.response
    if response is None:
        problems.append(Reason.RESPONSE_MISSING)
    elif response.status != "success":
        stopped_early = response.error is not None and response.error.type == "stream_closed_early"
        problems.append(Reason.STREAM_INCOMPLETE if stopped_early else Reason.RESPONSE_ERROR)
    elif not response.text and not response.tool_calls:
        # A tool-call-only answer has no text but is still a real answer.
        problems.append(Reason.RESPONSE_EMPTY)

    return problems


def _speed_and_cost_problems(record: CallRecord) -> list[Reason]:
    """What stops this record from being usable for speed and cost. Empty means it is."""
    problems: list[Reason] = []

    if record.latency is None or record.latency.total_ms is None:
        problems.append(Reason.LATENCY_MISSING)

    usage = record.usage
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        problems.append(Reason.USAGE_MISSING)

    return problems


def grade(record: CallRecord) -> GradeResult:
    replay_problems = _replay_problems(record)
    if not replay_problems:
        return GradeResult(Grade.REPLAY_GRADE)

    reasons = list(replay_problems)
    if record.metadata and "parse_error" in record.metadata:
        reasons.append(Reason.PARSE_ERROR)

    # A failed call's latency and tokens say nothing about how the model performs, so a
    # failure is insufficient even if it happens to carry timing.
    failed = record.response is not None and record.response.status != "success"
    speed_problems = _speed_and_cost_problems(record)

    if not failed and not speed_problems:
        return GradeResult(Grade.SPEED_AND_COST_ONLY, tuple(reasons))
    return GradeResult(Grade.INSUFFICIENT, tuple(reasons + speed_problems))


@dataclass(frozen=True)
class Summary:
    total: int
    counts: dict[Grade, int]
    # How many records each reason applies to (a record can have several).
    reasons: Counter[Reason] = field(default_factory=Counter)

    @property
    def replay_grade_fraction(self) -> float:
        return self.counts[Grade.REPLAY_GRADE] / self.total if self.total else 0.0

    @property
    def meets_target(self) -> bool:
        return self.replay_grade_fraction >= REPLAY_GRADE_TARGET


def summarize(results: Iterable[GradeResult]) -> Summary:
    counts = {g: 0 for g in Grade}
    reasons: Counter[Reason] = Counter()
    total = 0
    for result in results:
        total += 1
        counts[result.grade] += 1
        reasons.update(result.reasons)
    return Summary(total=total, counts=counts, reasons=reasons)


_LABELS = {
    Grade.REPLAY_GRADE: "Replay-grade",
    Grade.SPEED_AND_COST_ONLY: "Speed and cost only",
    Grade.INSUFFICIENT: "Insufficient",
}


def format_summary(summary: Summary) -> str:
    """The validation summary printed after an import."""
    lines = []
    for g in Grade:
        count = summary.counts[g]
        share = count / summary.total * 100 if summary.total else 0.0
        line = f"{_LABELS[g]:<20} {count:>5}  ({share:5.1f}%)"
        if g is Grade.REPLAY_GRADE and summary.total:
            line += "  meets the 90% target" if summary.meets_target else "  below the 90% target"
        lines.append(line)
    if summary.reasons:
        lines.append("Why records fell short:")
        for reason, n in summary.reasons.most_common():
            lines.append(f"  {n:>5} x {reason.value}: {REASON_TEXT[reason]}")
    return "\n".join(lines)
