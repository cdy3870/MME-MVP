"""`migration-eval export` end to end, through the real CLI."""

from __future__ import annotations

import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typer.testing import CliRunner

from migration_eval.cli import app

NOW = datetime.now(timezone.utc)
PROMPT = "SECRET PROMPT TEXT"


def good(id: str, hours_ago: float = 1, component: str = "support", **changes) -> dict:
    """A replay-grade record: it has a request and a successful response."""
    return {
        "schema_version": "v0",
        "id": id,
        "component": component,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "started_at": (NOW - timedelta(hours=hours_ago)).isoformat(),
        "latency": {"total_ms": 100},
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "request": {"messages": [{"role": "user", "content": PROMPT}]},
        "response": {"status": "success", "text": "hello", "finish_reason": "stop"},
        **changes,
    }


def failed(id: str, hours_ago: float = 1) -> dict:
    return good(id, hours_ago, response={"status": "error", "error": {"type": "rate_limit_error", "http_status": 429}})


def write(tmp_path: Path, *lines) -> Path:
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines) + "\n")
    return path


def run(records: Path, *args: str):
    return CliRunner().invoke(app, ["export", "--records", str(records), *args])


def members(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive, "r:gz") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


# --- success -----------------------------------------------------------------------------


def test_export_writes_the_archive_and_prints_the_summary_first(tmp_path):
    records = write(tmp_path, good("a", component="triage"), good("b", component="summarizer"), good("c", component="summarizer"))
    out = tmp_path / "export.tar.gz"
    result = run(records, "--out", str(out))

    assert result.exit_code == 0, result.output
    assert "Records:     3" in result.output
    assert "summarizer  2" in result.output and "triage      1" in result.output
    assert "File size:" in result.output and "Time range:" in result.output
    assert result.output.index("Records:") < result.output.index(f"Wrote {out}")  # summary, then write
    assert list(members(out)) == ["manifest.json", "fingerprint.json", "records.jsonl"]


def test_the_size_in_the_summary_is_the_size_of_the_file_written(tmp_path):
    out = tmp_path / "export.tar.gz"
    result = run(write(tmp_path, good("a")), "--out", str(out))
    size = out.stat().st_size
    shown = next(l for l in result.output.splitlines() if l.startswith("File size:"))
    expected = f"{size} bytes" if size < 1024 else f"{size / 1024:.1f} KB"
    assert expected in shown


def test_the_archive_has_full_content_and_a_fingerprint_without_it(tmp_path):
    out = tmp_path / "export.tar.gz"
    run(write(tmp_path, good("a")), "--out", str(out))
    contents = members(out)

    assert json.loads(contents["manifest.json"]) == {"schema_version": "v0", "record_count": 1}
    assert PROMPT.encode() in contents["records.jsonl"]  # not redacted
    assert PROMPT.encode() not in contents["fingerprint.json"]


def test_since_keeps_only_records_in_the_window(tmp_path):
    records = write(tmp_path, good("old", hours_ago=24 * 10), good("recent", hours_ago=2))
    out = tmp_path / "export.tar.gz"
    result = run(records, "--since", "7d", "--out", str(out))

    assert "Records:     1" in result.output
    assert json.loads(members(out)["manifest.json"])["record_count"] == 1


def test_since_defaults_to_seven_days(tmp_path):
    records = write(tmp_path, good("old", hours_ago=24 * 8), good("recent", hours_ago=24 * 6))
    out = tmp_path / "export.tar.gz"
    run(records, "--out", str(out))
    assert json.loads(members(out)["manifest.json"])["record_count"] == 1


def test_hours_are_supported(tmp_path):
    records = write(tmp_path, good("old", hours_ago=30), good("recent", hours_ago=2))
    out = tmp_path / "export.tar.gz"
    run(records, "--since", "24h", "--out", str(out))
    assert json.loads(members(out)["manifest.json"])["record_count"] == 1


def test_the_default_file_name_is_dated_and_written_to_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = write(tmp_path, good("a"))
    result = run(records)

    expected = f"migration-eval-export-{datetime.now(timezone.utc):%Y-%m-%d}.tar.gz"
    assert result.exit_code == 0, result.output
    assert (tmp_path / expected).exists()
    assert f"Wrote {expected}" in result.output


def test_the_output_folder_is_created_if_missing(tmp_path):
    out = tmp_path / "exports" / "nested" / "export.tar.gz"
    result = run(write(tmp_path, good("a")), "--out", str(out))
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_a_mostly_unusable_file_still_exports_if_one_record_can_be_replayed(tmp_path):
    out = tmp_path / "export.tar.gz"
    result = run(write(tmp_path, good("ok"), failed("x"), failed("y")), "--out", str(out))
    assert result.exit_code == 0, result.output
    assert json.loads(members(out)["manifest.json"])["record_count"] == 3  # nothing is dropped


# --- rejections: exit 1, nothing written -------------------------------------------------


def assert_rejected(result, out: Path, message: str):
    assert result.exit_code == 1, result.output
    assert message in result.output
    assert not out.exists()


def test_a_file_with_nothing_replayable_is_rejected_with_reasons(tmp_path):
    out = tmp_path / "export.tar.gz"
    result = run(write(tmp_path, failed("a"), failed("b")), "--out", str(out))
    assert_rejected(result, out, "Rejected: none of the 2 records are replay-grade")
    assert "2 x response_error" in result.output and "Nothing was written." in result.output


def test_a_file_mixing_schema_versions_is_rejected(tmp_path):
    out = tmp_path / "export.tar.gz"
    result = run(write(tmp_path, good("a"), good("b", schema_version="v1")), "--out", str(out))
    assert_rejected(result, out, "mixes schema versions: v0 (1 lines), v1 (1 lines)")


def test_a_bad_line_is_rejected_with_its_line_number(tmp_path):
    out = tmp_path / "export.tar.gz"
    result = run(write(tmp_path, good("a"), "{not json"), "--out", str(out))
    assert_rejected(result, out, "line 2 is not valid JSON")


def test_an_invalid_record_names_the_line_and_field(tmp_path):
    out = tmp_path / "export.tar.gz"
    result = run(write(tmp_path, good("a", usage={"input_tokens": -1})), "--out", str(out))
    assert_rejected(result, out, "line 1 is not a valid v0 record (check: usage.input_tokens)")


def test_an_empty_window_says_so_and_writes_nothing(tmp_path):
    out = tmp_path / "export.tar.gz"
    result = run(write(tmp_path, good("old", hours_ago=24 * 30)), "--since", "7d", "--out", str(out))
    assert_rejected(result, out, "No records in the last 7d")


def test_an_empty_file_writes_nothing(tmp_path):
    out = tmp_path / "export.tar.gz"
    records = tmp_path / "records.jsonl"
    records.write_text("")
    assert_rejected(run(records, "--out", str(out)), out, "No records in the last 7d")


def test_rejection_leaves_no_file_even_in_a_new_folder(tmp_path):
    out = tmp_path / "newdir" / "export.tar.gz"
    result = run(write(tmp_path, failed("a")), "--out", str(out))
    assert result.exit_code == 1
    assert not out.exists()


# --- bad arguments: exit 2, with a usage message ---------------------------------------------


def test_a_bad_since_value_is_a_usage_error_naming_the_option(tmp_path):
    out = tmp_path / "export.tar.gz"
    result = run(write(tmp_path, good("a")), "--since", "7w", "--out", str(out))
    assert result.exit_code == 2
    assert "--since" in result.output and "Invalid format" in result.output
    assert not out.exists()


def test_a_zero_or_negative_since_is_a_usage_error(tmp_path):
    for value in ("0d", "-3d"):
        result = run(write(tmp_path, good("a")), "--since", value)
        assert result.exit_code == 2 and "greater than 0" in result.output


def test_a_missing_records_file_is_a_usage_error_not_a_traceback(tmp_path):
    result = run(tmp_path / "nope.jsonl")
    assert result.exit_code == 2
    assert "does not exist" in result.output
    assert "Traceback" not in result.output


def test_the_default_records_file_is_the_recorders_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no migration_eval_records.jsonl here
    result = CliRunner().invoke(app, ["export"])
    assert result.exit_code == 2 and "migration_eval_records.jsonl" in result.output


def test_export_appears_in_help():
    result = CliRunner().invoke(app, ["--help"])
    assert "export" in result.output
