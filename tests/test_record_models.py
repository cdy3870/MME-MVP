"""The generated Pydantic models must agree with the JSON Schema they come from."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from migration_eval.models import CallRecord

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "record.v0.schema.json"
EXAMPLES = ROOT / "schema" / "examples"

VALID = sorted((EXAMPLES / "valid").glob("*.json"))
INVALID = sorted((EXAMPLES / "invalid").glob("*.json"))

SCHEMA = json.loads(SCHEMA_PATH.read_text())
# The format checker is what makes "format": "date-time" actually enforced.
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=Draft202012Validator.FORMAT_CHECKER)


def dump(record: CallRecord) -> dict:
    # by_alias: ResponseFormat.schema_ is aliased to "schema" to avoid clashing with BaseModel.
    return record.model_dump(mode="json", exclude_none=True, by_alias=True)


def test_schema_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(SCHEMA)


def test_examples_are_present():
    assert VALID and INVALID


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_valid_example_accepted_by_schema_and_model(path: Path):
    data = json.loads(path.read_text())
    assert not list(VALIDATOR.iter_errors(data))
    CallRecord.model_validate(data)


@pytest.mark.parametrize("path", INVALID, ids=lambda p: p.name)
def test_invalid_example_rejected_by_schema_and_model(path: Path):
    data = json.loads(path.read_text())
    assert list(VALIDATOR.iter_errors(data)), "schema should reject this example"
    with pytest.raises(ValidationError):
        CallRecord.model_validate(data)


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_round_trip_stays_schema_valid(path: Path):
    record = CallRecord.model_validate_json(path.read_text())
    out = dump(record)
    assert not list(VALIDATOR.iter_errors(out))
    assert CallRecord.model_validate(out) == record


def test_response_format_schema_key_survives_round_trip():
    data = json.loads((EXAMPLES / "valid" / "replay_grade_chat.json").read_text())
    data["request"]["parameters"]["response_format"] = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"label": {"type": "string"}}},
    }
    record = CallRecord.model_validate(data)
    out = dump(record)
    assert out["request"]["parameters"]["response_format"]["schema"] == {
        "type": "object",
        "properties": {"label": {"type": "string"}},
    }
    assert not list(VALIDATOR.iter_errors(out))


def test_naive_timestamp_rejected():
    data = json.loads((EXAMPLES / "valid" / "replay_grade_chat.json").read_text())
    data["started_at"] = "2026-10-12T14:03:21"  # no timezone
    with pytest.raises(ValidationError):
        CallRecord.model_validate(data)


def test_generated_files_match_schema():
    """Fails if schema/record.v0.schema.json changed without regenerating the outputs."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_models.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout
