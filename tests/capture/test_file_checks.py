"""Whole-file rejection (REC-5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from migration_eval.capture.file_checks import FileRejected, check_replay_grade, check_schema_versions
from migration_eval.capture.grading import Grade, GradeResult, Reason, summarize
from migration_eval.cli import app

GOOD = {
    "provider": "openai",
    "started_at": "2026-10-12T14:03:21Z",
    "latency_ms": 100,
    "request": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    "response": {
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2},
    },
}
FAILED = {**GOOD, "status_code": 429, "response": {"error": {"type": "rate_limit_error"}}}


def write(tmp_path: Path, *lines) -> Path:
    path = tmp_path / "logs.jsonl"
    path.write_text("\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines) + "\n")
    return path


# --- check_schema_versions ---------------------------------------------------------------


def test_one_version_is_fine():
    check_schema_versions([{"schema_version": "v0"}, {"schema_version": "v0"}])


def test_no_lines_is_not_a_version_mismatch():
    check_schema_versions([])


def test_raw_lines_count_as_v0():
    check_schema_versions([{"schema_version": "v0"}, {"provider": "openai"}])


def test_mixed_versions_are_rejected_and_each_version_is_named_with_its_count():
    lines = [{"schema_version": "v0"}] * 3 + [{"schema_version": "v1"}]
    with pytest.raises(FileRejected) as error:
        check_schema_versions(lines)
    message = str(error.value)
    assert "v0 (3 lines)" in message and "v1 (1 lines)" in message
    assert "Re-export" in message  # says how to fix it


def test_a_raw_line_mixed_with_a_v1_record_is_also_mixed():
    with pytest.raises(FileRejected):
        check_schema_versions([{"provider": "openai"}, {"schema_version": "v1"}])


def test_an_odd_version_value_does_not_crash_the_check():
    with pytest.raises(FileRejected):
        check_schema_versions([{"schema_version": "v0"}, {"schema_version": ["v0"]}])


# --- check_replay_grade ------------------------------------------------------------------


def test_one_replay_grade_record_is_enough():
    check_replay_grade(summarize([GradeResult(Grade.REPLAY_GRADE), GradeResult(Grade.INSUFFICIENT)]))


def test_no_replay_grade_records_is_rejected_with_what_is_missing():
    summary = summarize(
        [
            GradeResult(Grade.INSUFFICIENT, (Reason.RESPONSE_ERROR, Reason.USAGE_MISSING)),
            GradeResult(Grade.SPEED_AND_COST_ONLY, (Reason.REQUEST_MISSING,)),
        ]
    )
    with pytest.raises(FileRejected) as error:
        check_replay_grade(summary)
    message = str(error.value)
    assert "none of the 2 records are replay-grade" in message
    assert "1 x response_error: the provider call failed" in message
    assert "1 x request_missing" in message


def test_an_empty_summary_is_rejected():
    with pytest.raises(FileRejected, match="no records"):
        check_replay_grade(summarize([]))


# --- the CLI -----------------------------------------------------------------------------


def invoke(path: Path, *args: str):
    return CliRunner().invoke(app, ["import", "jsonl", str(path), *args])


def test_cli_rejects_a_file_with_mixed_versions_and_writes_nothing(tmp_path):
    v0 = json.loads((Path(__file__).resolve().parents[2] / "schema/examples/valid/replay_grade_chat.json").read_text())
    path = write(tmp_path, v0, {**v0, "schema_version": "v1"})
    result = invoke(path)

    assert result.exit_code == 1
    assert "Rejected: this file mixes schema versions" in result.output
    assert "v0 (1 lines)" in result.output and "v1 (1 lines)" in result.output
    assert not (tmp_path / "logs.v0.jsonl").exists()


def test_cli_rejects_a_file_with_no_replay_grade_records_and_explains_why(tmp_path):
    path = write(tmp_path, FAILED, FAILED, "{not json")
    result = invoke(path)

    assert result.exit_code == 1
    assert "Rejected: none of the 2 records are replay-grade" in result.output
    assert "2 x response_error" in result.output
    assert "line 3: not valid JSON" in result.output  # skipped lines are still reported
    assert "Nothing was written" in result.output
    assert not (tmp_path / "logs.v0.jsonl").exists()


def test_cli_does_not_write_output_for_a_rejected_file_even_with_out(tmp_path):
    out = tmp_path / "custom.jsonl"
    result = invoke(write(tmp_path, FAILED), "--out", str(out))
    assert result.exit_code == 1 and not out.exists()


def test_cli_accepts_a_mostly_bad_file_that_still_has_one_replay_grade_record(tmp_path):
    result = invoke(write(tmp_path, GOOD, FAILED, FAILED))

    assert result.exit_code == 0, result.output
    assert "below the 90% target" in result.output  # a warning, not a rejection
    assert len((tmp_path / "logs.v0.jsonl").read_text().splitlines()) == 3  # every record is kept


def test_cli_still_explains_an_empty_import(tmp_path):
    result = invoke(write(tmp_path, "{not json"))
    assert result.exit_code == 1
    assert "No records were imported" in result.output
