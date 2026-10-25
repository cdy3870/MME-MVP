"""Profile: what one component's recorded traffic looks like (the input to candidate filtering).

The ERD derives a profile from an upload: volume, token lengths, tool use, request
categories. CAT-3 only needs two facts (how big the largest call is, and whether tools are
used) plus which model the traffic already uses, so this starts there. Grow it as later
stages need more, rather than guessing now.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from migration_eval.models import CallRecord


@dataclass(frozen=True)
class Incumbent:
    """The model the traffic currently runs on."""

    provider: str
    model: str


@dataclass(frozen=True)
class Profile:
    component: str
    record_count: int  # records for this component; 0 when none
    # The largest input + output token count of any single call, or None when no record
    # has both counts. This is how much context a candidate needs.
    max_context_tokens: int | None
    uses_tools: bool  # any call defined tools, made tool calls, or carried tool results
    incumbent: Incumbent | None  # None when there are no records


def records_for_component(records: list[CallRecord], component: str) -> list[CallRecord]:
    """The records tagged with `component`, in their original order."""
    filtered_records = []
    for record in records:
        if record.component is not None and record.component == component:
            filtered_records.append(record)

    return filtered_records


def context_tokens(record: CallRecord) -> int | None:
    """Input plus output tokens for one call; None unless both counts are recorded.

    Tokenizers differ between providers, so this is the incumbent's count, not an exact
    figure for a candidate. Applying a safety margin is the caller's decision (see
    `candidates.check_context`).
    """
    usage = record.usage
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return None

    return usage.input_tokens + usage.output_tokens


def record_uses_tools(record: CallRecord) -> bool:
    """True if the call defined tools (`request.tools`), received a tool result or earlier
    tool call in its messages, or answered with tool calls (`response.tool_calls`).

    Messages are either a string or a list of parts; only the list form can hold tool parts.
    """
    request, response = record.request, record.response

    if request and request.tools:
        return True
    if response and response.tool_calls:
        return True
    if request and request.messages:
        for message in request.messages:
            if isinstance(message.content, list) and any(
                part.root.type in ("tool_call", "tool_result") for part in message.content
            ):
                return True

    return False


def find_incumbent(records: list[CallRecord]) -> Incumbent | None:
    """The (provider, model) used by the most records; None for no records.

    Break ties deterministically, e.g. by provider then model name, so the result does not
    depend on record order. Open question: what if a component really does use several
    models?
    """
    if not records:
        return None

    counts = Counter((record.provider, record.model) for record in records)
    # Most records first; ties go to the alphabetically first (provider, model).
    provider, model = min(counts, key=lambda pair: (-counts[pair], pair))
    return Incumbent(provider=provider, model=model)


def build_profile(records: list[CallRecord], component: str) -> Profile:
    """Profile one component: filter with `records_for_component`, then fill in each field.

    `max_context_tokens` is the maximum of `context_tokens` over records that have it.
    """
    filtered_records = records_for_component(records, component)
    token_counts = [t for r in filtered_records if (t := context_tokens(r)) is not None]

    return Profile(
        component=component,
        record_count=len(filtered_records),
        max_context_tokens=max(token_counts, default=None),
        uses_tools=any(record_uses_tools(r) for r in filtered_records),
        incumbent=find_incumbent(filtered_records),
    )
