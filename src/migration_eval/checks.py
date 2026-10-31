"""Deterministic checks (EVL-4): cheap, exact tests that run before any judge call.

A candidate's output is checked against the request and the incumbent's output. A failed
check settles the pair as "different" without paying for the judge (EVL-5); a pair that
passes every applicable check goes on to the judge.

A check returns None when it does not apply (a request with no response schema has nothing to
validate), so "not applicable" is never confused with "passed".

Open decisions (see the notes on EVL-4):
    * how refusals are detected: phrase matching is cheap but misses wording; a provider's own
      refusal signal (`finish_reason`) is better where it exists, and `ReplayResult` does not
      carry it yet
    * `jsonschema` is a dev dependency today; it becomes a runtime one
    * a failed check on the incumbent's own output (e.g. it also refused) should not count
      against the candidate

Every function here is a stub: implement them one at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from jsonschema import ValidationError, validate

from migration_eval.models.record import ModelRequest
from migration_eval.replay import ReplayResult


@dataclass(frozen=True)
class CheckResult:
    check: str  # "output_schema", "tool_calls" or "refusal"
    passed: bool
    detail: str | None = None  # why it failed, for the report


def check_output_schema(request: ModelRequest, candidate: ReplayResult) -> CheckResult | None:
    """If the request asked for `json_schema` output, the candidate's text must parse as JSON
    and validate against that schema. None when the request had no schema.
    """

    response_format = request.parameters.response_format if request.parameters else None

    if response_format is None or response_format.type != "json_schema":
        return None

    try:
        validate(instance=json.loads(candidate.text or ""), schema=response_format.schema_)
    except (ValueError, ValidationError) as exc:  # ValueError covers JSON that does not parse
        return CheckResult(check="output_schema", passed=False, detail=f"Schema validation failed: {exc}")
    return CheckResult(check="output_schema", passed=True, detail="Schema validation succeeded")


def check_tool_calls(request: ModelRequest, incumbent: ReplayResult, candidate: ReplayResult) -> CheckResult | None:
    """The candidate calls only tools the request defined, and the same tools with the same
    arguments as the incumbent (names and arguments, not ids). None when neither called a tool.
    """

    cand_tool_calls = candidate.tool_calls or []
    inc_tool_calls = incumbent.tool_calls or []

    if not cand_tool_calls and not inc_tool_calls:
        return None

    tool_names = [tool.name for tool in request.tools or []]
    for tool in cand_tool_calls:
        if tool["name"] not in tool_names:
            return CheckResult(check="tool_calls", passed=False, detail=f"Invalid tool name: {tool['name']}")

    def names_and_arguments(calls):  # ids differ on every call, so leave them out
        return [(c.get("name"), c.get("arguments"), c.get("arguments_raw")) for c in calls]

    if names_and_arguments(cand_tool_calls) != names_and_arguments(inc_tool_calls):
        return CheckResult(check="tool_calls", passed=False, detail="Tool calls differ from the incumbent's")
    return CheckResult(check="tool_calls", passed=True, detail="Valid tool use")

def run_checks(request: ModelRequest, incumbent: ReplayResult, candidate: ReplayResult) -> list[CheckResult]:
    """Every applicable check, in order: output schema, and tool calls. Skip the ones that
    return None. A candidate with an `error` fails without running the others.
    """

    if candidate.error is not None:
        return [CheckResult(check="error", passed=False, detail=candidate.error)]

    results = [check_output_schema(request, candidate), check_tool_calls(request, incumbent, candidate)]
    return [result for result in results if result is not None]


def first_failure(checks: list[CheckResult]) -> CheckResult | None:
    """The first failed check, or None when all passed (so the judge should run)."""

    for result in checks:
        if not result.passed:
            return result

    return None
