"""The report schema, its generated models, and the example reports the frontend is built on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from migration_eval.early_sample import CandidateRun
from migration_eval.models import Report
from migration_eval.verdict import Agreement, Thresholds, assign_verdict, explain_verdict

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schema" / "report.v0.schema.json").read_text())
EXAMPLES = ROOT / "schema" / "examples" / "report"
VALID = sorted((EXAMPLES / "valid").glob("*.json"))
INVALID = sorted((EXAMPLES / "invalid").glob("*.json"))
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=Draft202012Validator.FORMAT_CHECKER)

FULL = json.loads((EXAMPLES / "valid" / "support_bot_all_verdicts.json").read_text())


def dump(report: Report) -> dict:
    return report.model_dump(mode="json", exclude_none=False, by_alias=True)


def test_schema_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(SCHEMA)


def test_examples_are_present():
    assert VALID and INVALID


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_valid_example_accepted_by_schema_and_model(path: Path):
    data = json.loads(path.read_text())
    assert not list(VALIDATOR.iter_errors(data))
    Report.model_validate(data)


@pytest.mark.parametrize("path", INVALID, ids=lambda p: p.name)
def test_invalid_example_rejected_by_schema_and_model(path: Path):
    data = json.loads(path.read_text())
    assert list(VALIDATOR.iter_errors(data)), "schema should reject this example"
    with pytest.raises(ValidationError):
        Report.model_validate(data)


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_round_trip_keeps_the_report_unchanged(path: Path):
    data = json.loads(path.read_text())
    out = dump(Report.model_validate(data))
    assert not list(VALIDATOR.iter_errors(out))
    assert out["candidates"] == data["candidates"] and out["noise_floor"] == data["noise_floor"]


def test_the_full_example_shows_every_verdict_and_a_dropped_candidate():
    assert {c["verdict"] for c in FULL["candidates"]} == {
        "safe_to_swap", "swap_with_caveats", "do_not_swap", "insufficient_data",
    }  # fmt: skip
    assert FULL["dropped_candidates"] and any(c["stopped_early"] for c in FULL["candidates"])
    assert any(c["conversion"]["converted_requests"] for c in FULL["candidates"])


# --- the examples agree with the evaluator --------------------------------------------------


def agreement_of(data: dict) -> Agreement:
    return Agreement(**data)


@pytest.mark.parametrize("candidate", FULL["candidates"], ids=lambda c: c["model"])
def test_each_verdict_and_explanation_is_what_the_evaluator_would_produce(candidate):
    thresholds = Thresholds(**FULL["thresholds"])
    floor = FULL["noise_floor"]["value"]
    overall = agreement_of(candidate["agreement"])
    by_category = {c["category"]: agreement_of(c["agreement"]) for c in candidate["by_category"]}
    run = CandidateRun(candidate["provider"], candidate["model"], [], overall.rate, candidate["requests_run"], candidate["stopped_early"])
    converted = candidate["conversion"]["converted_requests"] > 0

    verdict = assign_verdict(run, overall, by_category, floor, thresholds, used_converted_requests=converted)
    assert candidate["verdict"] == verdict.name.lower()
    assert candidate["explanation"] == explain_verdict(
        verdict, run, overall, floor, thresholds, by_category=by_category, used_converted_requests=converted
    )


@pytest.mark.parametrize("candidate", FULL["candidates"], ids=lambda c: c["model"])
def test_the_numbers_in_each_candidate_add_up(candidate):
    overall = candidate["agreement"]
    assert overall["n"] == candidate["requests_run"] == sum(c["agreement"]["n"] for c in candidate["by_category"])
    assert overall["agreed"] == sum(c["agreement"]["agreed"] for c in candidate["by_category"])
    assert overall["rate"] == pytest.approx(overall["agreed"] / overall["n"])
    assert overall["ci_low"] <= overall["rate"] <= overall["ci_high"]

    conversion = candidate["conversion"]
    assert conversion["requests"] == candidate["requests_run"] and conversion["converted_requests"] <= conversion["requests"]
    assert all(u["requests"] <= conversion["converted_requests"] for u in conversion["unconverted"])

    disagreements = overall["n"] - overall["agreed"]
    assert sum(f["requests"] for f in candidate["check_failures"]) <= disagreements  # a failed check is a disagreement

    assert candidate["cost"]["total_usd"] == pytest.approx(candidate["cost"]["per_request_usd"] * candidate["requests_run"])
    assert candidate["latency"]["n"] + candidate["latency"]["failed"] <= candidate["requests_run"]


def test_a_stopped_candidate_ran_fewer_requests_than_the_sample():
    stopped = [c for c in FULL["candidates"] if c["stopped_early"]]
    assert stopped and all(c["requests_run"] < FULL["sample"]["requests_replayed"] for c in stopped)


def test_no_example_carries_prompt_or_output_text():
    """A report holds aggregates only; the schema forbids extra fields, so text cannot sneak in."""
    for path in VALID:
        text = path.read_text()
        assert '"messages"' not in text and '"prompt"' not in text and '"output"' not in text
