from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schema" / "record.v0.schema.json").read_text())
_VALIDATOR = Draft202012Validator(SCHEMA, format_checker=Draft202012Validator.FORMAT_CHECKER)


@pytest.fixture
def read_records():
    """Read a JSONL file and check every line against the JSON Schema, not just our models."""

    def read(path: Path) -> list[dict]:
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        for record in records:
            errors = list(_VALIDATOR.iter_errors(record))
            assert not errors, [e.message for e in errors]
        return records

    return read
