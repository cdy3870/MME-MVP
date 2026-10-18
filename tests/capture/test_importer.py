"""`import jsonl`: existing logs in, v0 records out."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from migration_eval.capture.importer import import_jsonl
from migration_eval.cli import app

ROOT = Path(__file__).resolve().parents[2]
VALID_EXAMPLE = json.loads((ROOT / "schema/examples/valid/replay_grade_chat.json").read_text())

OPENAI_LINE = {
    "provider": "openai",
    "started_at": "2026-10-12T14:03:21.114Z",
    "latency_ms": 888,
    "request": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    "response": {
        "id": "chatcmpl-1",
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2},
    },
}


def write(tmp_path: Path, *lines) -> Path:
    path = tmp_path / "logs.jsonl"
    path.write_text("\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines) + "\n")
    return path


def test_raw_exchange_becomes_a_record_with_timing_and_component(tmp_path):
    result = import_jsonl(write(tmp_path, OPENAI_LINE), component="support")

    (record,) = result.records
    assert result.skipped == []
    assert (record.provider, record.model, record.component) == ("openai", "gpt-4o-mini", "support")
    assert record.response.text == "hello"
    assert record.usage.input_tokens == 9
    assert record.latency.total_ms == 888
    assert (record.completed_at - record.started_at).total_seconds() == 0.888
    assert record.source.type == "import_jsonl"


def test_line_component_overrides_the_default(tmp_path):
    result = import_jsonl(write(tmp_path, {**OPENAI_LINE, "component": "triage"}), component="support")
    assert result.records[0].component == "triage"


def test_provider_can_be_inferred_from_the_url(tmp_path):
    line = {k: v for k, v in OPENAI_LINE.items() if k != "provider"}
    line["url"] = "https://api.openai.com/v1/chat/completions"
    assert import_jsonl(write(tmp_path, line)).records[0].provider == "openai"


def test_gemini_log_without_a_url_needs_a_model(tmp_path):
    line = {
        "provider": "gemini",
        "started_at": 1760277801,  # epoch seconds
        "model": "gemini-2.0-flash",
        "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        "response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]},
    }
    (record,) = import_jsonl(write(tmp_path, line)).records
    assert record.model == "gemini-2.0-flash"
    assert record.response.text == "ok"


def test_streamed_response_stored_as_raw_sse_text(tmp_path):
    sse = 'data: {"id":"c1","choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    (record,) = import_jsonl(write(tmp_path, {**OPENAI_LINE, "response": sse})).records
    assert record.response.text == "Hi"


def test_failed_call_is_kept_as_an_error_record(tmp_path):
    line = {**OPENAI_LINE, "status_code": 500, "response": {"error": {"type": "server_error", "message": "boom"}}}
    (record,) = import_jsonl(write(tmp_path, line)).records
    assert record.response.status == "error"
    assert record.response.error.http_status == 500


def test_timestamps_accept_epoch_milliseconds_and_naive_iso(tmp_path):
    ms = {**OPENAI_LINE, "started_at": 1760277801114}
    naive = {**OPENAI_LINE, "started_at": "2026-10-12T14:03:21"}
    a, b = import_jsonl(write(tmp_path, ms, naive)).records
    assert a.started_at.tzinfo is not None and b.started_at.tzinfo is not None


def test_existing_v0_records_pass_through_and_bad_ones_are_skipped(tmp_path):
    bad_version = {**VALID_EXAMPLE, "schema_version": "v1"}
    bad_record = {**VALID_EXAMPLE, "usage": {"input_tokens": -1}}
    result = import_jsonl(write(tmp_path, VALID_EXAMPLE, bad_version, bad_record))

    assert [r.id for r in result.records] == ["rec_0001"]
    assert dict(result.skipped) == {
        2: "unsupported schema_version 'v1'",
        3: "not a valid v0 record (check: usage.input_tokens)",
    }


def test_bad_lines_are_skipped_with_reasons_and_the_import_continues(tmp_path):
    path = write(
        tmp_path,
        "{not json",
        "[1, 2]",
        {**OPENAI_LINE, "provider": "mistral"},
        {k: v for k, v in OPENAI_LINE.items() if k != "provider"},
        {k: v for k, v in OPENAI_LINE.items() if k != "started_at"},
        OPENAI_LINE,
    )
    result = import_jsonl(path)

    assert len(result.records) == 1  # the last, good line
    reasons = dict(result.skipped)
    assert reasons[1] == "not valid JSON"
    assert reasons[2] == "line is not a JSON object"
    assert "unknown provider 'mistral'" in reasons[3]
    assert "cannot tell which provider" in reasons[4]
    assert "missing started_at" in reasons[5]


def test_unparseable_request_is_imported_with_a_parse_error(tmp_path):
    line = {**OPENAI_LINE, "request": {"model": "gpt-4o", "messages": [], "tools": [{"type": "web_search"}]}}
    (record,) = import_jsonl(write(tmp_path, line)).records
    assert record.request is None
    assert "web_search" in record.metadata["parse_error"]


def test_blank_lines_are_ignored(tmp_path):
    result = import_jsonl(write(tmp_path, OPENAI_LINE, "", "   ", OPENAI_LINE))
    assert (result.lines_read, len(result.records)) == (2, 2)


# --- CLI ---------------------------------------------------------------------------------


def test_cli_writes_v0_jsonl_and_reports_skipped_lines(tmp_path):
    path = write(tmp_path, OPENAI_LINE, "{not json", OPENAI_LINE)
    result = CliRunner().invoke(app, ["import", "jsonl", str(path), "--component", "support"])

    assert result.exit_code == 0, result.output
    assert "Read 3 lines: 2 imported, 1 skipped." in result.output
    assert "line 2: not valid JSON" in result.output
    out = tmp_path / "logs.v0.jsonl"
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["component"] for r in written] == ["support", "support"]
    assert all(r["schema_version"] == "v0" for r in written)


def test_cli_exits_nonzero_and_writes_nothing_when_nothing_imports(tmp_path):
    path = write(tmp_path, "{not json")
    out = tmp_path / "custom.jsonl"
    result = CliRunner().invoke(app, ["import", "jsonl", str(path), "--out", str(out)])

    assert result.exit_code == 1
    assert "No records were imported" in result.output
    assert not out.exists()
