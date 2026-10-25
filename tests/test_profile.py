"""Profile: what a component's recorded traffic looks like."""

from __future__ import annotations

import json
from pathlib import Path

from migration_eval.models import CallRecord
from migration_eval.profile import (
    Incumbent,
    build_profile,
    context_tokens,
    find_incumbent,
    record_uses_tools,
    records_for_component,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "schema" / "examples" / "valid"


def load(name: str) -> CallRecord:
    return CallRecord.model_validate(json.loads((EXAMPLES / f"{name}.json").read_text()))


def variant(record: CallRecord, **changes) -> CallRecord:
    data = record.model_dump(mode="json", exclude_none=True, by_alias=True)
    return CallRecord.model_validate({**data, **changes})


CHAT = load("replay_grade_chat")  # support-triage, openai/gpt-4o-mini, 412 + 37 tokens
TOOLS = load("replay_grade_tool_use_streamed")  # order-assistant, anthropic, tool call + tool result
NO_RESPONSE = load("speed_and_cost_only")  # summarizer, no request or response
FAILED = load("failed_call")  # summarizer, a failed call without token counts


def test_records_for_component_filters_and_keeps_order():
    assert [r.id for r in records_for_component([CHAT, TOOLS, NO_RESPONSE, FAILED], "summarizer")] == [
        "rec_0003",
        "rec_0004",
    ]


def test_context_tokens_is_input_plus_output():
    assert context_tokens(CHAT) == 412 + 37


def test_context_tokens_is_none_without_both_counts():
    assert context_tokens(FAILED) is None
    assert context_tokens(variant(CHAT, usage={"input_tokens": 5})) is None


def test_tool_use_is_found_in_the_response_the_definitions_and_the_messages():
    assert record_uses_tools(TOOLS) is True  # tool call in the response
    defined_only = variant(CHAT, request={"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "f"}]})
    assert record_uses_tools(defined_only) is True
    tool_result = {"role": "tool", "content": [{"type": "tool_result", "tool_call_id": "a", "content": "ok"}]}
    assert record_uses_tools(variant(CHAT, request={"messages": [tool_result]})) is True


def test_a_plain_call_or_one_without_a_response_does_not_use_tools():
    assert record_uses_tools(CHAT) is False
    assert record_uses_tools(NO_RESPONSE) is False


def test_incumbent_is_the_most_used_model_and_none_for_no_records():
    a = variant(CHAT, provider="openai", model="gpt-4o")
    b = variant(CHAT, provider="anthropic", model="claude-x")
    assert find_incumbent([]) is None
    assert find_incumbent([a, b, b]) == Incumbent("anthropic", "claude-x")


def test_an_incumbent_tie_does_not_depend_on_record_order():
    a = variant(CHAT, provider="openai", model="gpt-4o")
    b = variant(CHAT, provider="anthropic", model="claude-x")
    assert find_incumbent([a, b]) == find_incumbent([b, a]) == Incumbent("anthropic", "claude-x")


def test_build_profile_uses_only_the_components_own_records():
    profile = build_profile([CHAT, TOOLS, NO_RESPONSE, FAILED], "support-triage")
    assert (profile.component, profile.record_count) == ("support-triage", 1)
    assert profile.max_context_tokens == 449
    assert profile.uses_tools is False  # the tool use belongs to another component
    assert profile.incumbent == Incumbent("openai", "gpt-4o-mini")


def test_build_profile_takes_the_largest_call_and_detects_tools():
    profile = build_profile([CHAT, TOOLS], "order-assistant")
    assert profile.max_context_tokens == 1180 + 96
    assert profile.uses_tools is True


def test_build_profile_for_a_component_with_no_records():
    profile = build_profile([CHAT], "nothing-here")
    assert (profile.record_count, profile.max_context_tokens, profile.uses_tools, profile.incumbent) == (
        0,
        None,
        False,
        None,
    )


def test_build_profile_with_records_but_no_token_counts():
    assert build_profile([FAILED], "summarizer").max_context_tokens is None
