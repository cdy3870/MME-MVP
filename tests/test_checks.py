"""Deterministic checks (EVL-4): output schema and tool calls, run before any judge call."""

from __future__ import annotations

from migration_eval.checks import (
    CheckResult,
    check_output_schema,
    check_tool_calls,
    first_failure,
    run_checks,
)
from migration_eval.models.record import ModelRequest
from migration_eval.replay import ReplayResult

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
TOOLS = [{"name": "get_order", "input_schema": {"type": "object"}}]


def request(response_format=None, tools=None) -> ModelRequest:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    if response_format:
        body["parameters"] = {"response_format": response_format}
    if tools:
        body["tools"] = tools
    return ModelRequest.model_validate(body)


def schema_request() -> ModelRequest:
    return request({"type": "json_schema", "schema": SCHEMA})


def result(text="ok", calls=None, error=None) -> ReplayResult:
    return ReplayResult(text, calls, 10, 5, 100.0, error)


def call(name="get_order", **arguments) -> dict:
    return {"id": "x", "name": name, "arguments": arguments}


# --- output schema ------------------------------------------------------------------------


def test_valid_json_matching_the_schema_passes():
    assert check_output_schema(schema_request(), result('{"n": 3}')).passed


def test_json_that_breaks_the_schema_fails():
    outcome = check_output_schema(schema_request(), result('{"n": "three"}'))
    assert outcome.check == "output_schema" and not outcome.passed


def test_text_that_is_not_json_fails_instead_of_crashing():
    assert not check_output_schema(schema_request(), result("not json")).passed
    assert not check_output_schema(schema_request(), result(None)).passed


def test_no_schema_means_not_applicable():
    assert check_output_schema(request(), result("anything")) is None
    assert check_output_schema(request({"type": "json_object"}), result("anything")) is None


# --- tool calls ---------------------------------------------------------------------------


def test_neither_calling_a_tool_is_not_applicable():
    assert check_tool_calls(request(tools=TOOLS), result(), result()) is None
    assert check_tool_calls(request(), result(), result()) is None


def test_same_tool_and_arguments_passes_even_with_different_ids():
    a = [{"id": "1", "name": "get_order", "arguments": {"id": "8841"}}]
    b = [{"id": "2", "name": "get_order", "arguments": {"id": "8841"}}]
    assert check_tool_calls(request(tools=TOOLS), result(None, a), result(None, b)).passed


def test_a_tool_the_request_did_not_define_fails():
    outcome = check_tool_calls(request(tools=TOOLS), result(None, [call()]), result(None, [call("delete_all")]))
    assert not outcome.passed and "delete_all" in outcome.detail


def test_different_arguments_fail():
    outcome = check_tool_calls(request(tools=TOOLS), result(None, [call(id="1")]), result(None, [call(id="2")]))
    assert not outcome.passed


def test_the_candidate_calling_no_tool_when_the_incumbent_did_fails():
    assert not check_tool_calls(request(tools=TOOLS), result(None, [call(id="1")]), result("text")).passed


def test_the_candidate_calling_a_tool_when_the_incumbent_did_not_fails():
    assert not check_tool_calls(request(tools=TOOLS), result("text"), result(None, [call(id="1")])).passed


def test_a_tool_call_with_no_tools_defined_fails_without_crashing():
    assert not check_tool_calls(request(), result(None, [call()]), result(None, [call()])).passed


# --- run_checks and first_failure ---------------------------------------------------------


def test_run_checks_skips_checks_that_do_not_apply():
    assert run_checks(request(), result("a"), result("b")) == []


def test_run_checks_returns_every_applicable_check_in_order():
    req = request({"type": "json_schema", "schema": SCHEMA}, TOOLS)
    out = run_checks(req, result('{"n": 1}', [call(id="1")]), result('{"n": 1}', [call(id="1")]))
    assert [c.check for c in out] == ["output_schema", "tool_calls"] and all(c.passed for c in out)


def test_a_candidate_with_an_error_fails_without_running_the_others():
    out = run_checks(schema_request(), result('{"n": 1}'), result(error="HTTP 500"))
    assert len(out) == 1 and not out[0].passed and out[0].detail == "HTTP 500"


def test_first_failure_is_the_first_failed_check():
    checks = [CheckResult("output_schema", True), CheckResult("tool_calls", False, "bad"), CheckResult("x", False)]
    assert first_failure(checks).check == "tool_calls"


def test_first_failure_is_none_when_everything_passed_or_nothing_ran():
    assert first_failure([CheckResult("output_schema", True)]) is None
    assert first_failure([]) is None
