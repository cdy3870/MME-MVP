"""Export (EXP-1, EXP-2)."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from migration_eval.capture.file_checks import FileRejected
from migration_eval.capture.sink import record_to_json
from migration_eval.export import (
    MAX_COMPONENTS_SHOWN,
    ExportSummary,
    build_archive,
    build_fingerprint,
    build_manifest,
    default_export_path,
    format_export_summary,
    load_records,
    parse_since,
    summarize_export,
)
from migration_eval.models import CallRecord

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def record(id: str, started_at: datetime, **changes) -> dict:
    return {
        "schema_version": "v0",
        "id": id,
        "component": "support",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "started_at": started_at.isoformat(),
        **changes,
    }


def write(tmp_path: Path, *lines) -> Path:
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines) + "\n")
    return path


# --- parse_since -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("7d", timedelta(days=7)), ("24h", timedelta(hours=24)), ("1d", timedelta(days=1))],
)
def test_parse_since_accepts_days_and_hours(value, expected):
    assert parse_since(value) == expected


@pytest.mark.parametrize("value", ["7", "d7", "7w", "7D", " 7d", "7 d", "", "1.5d", "7dd"])
def test_parse_since_rejects_bad_formats(value):
    with pytest.raises(ValueError, match="Invalid format"):
        parse_since(value)


@pytest.mark.parametrize("value", ["0d", "0h", "-3d"])
def test_parse_since_rejects_zero_and_negative(value):
    with pytest.raises(ValueError, match="greater than 0"):
        parse_since(value)


def test_parse_since_rejects_absurdly_large_windows_with_a_clear_error():
    with pytest.raises(ValueError, match="too large"):
        parse_since("99999999999d")


# --- load_records ------------------------------------------------------------------------


def test_keeps_records_inside_the_window_and_drops_older_ones(tmp_path):
    path = write(
        tmp_path,
        record("old", NOW - timedelta(days=10)),
        record("recent", NOW - timedelta(days=2)),
        record("just_now", NOW),
    )
    records = load_records(path, since=timedelta(days=7), now=NOW)
    assert [r.id for r in records] == ["recent", "just_now"]


def test_the_cutoff_itself_is_inside_the_window(tmp_path):
    path = write(tmp_path, record("edge", NOW - timedelta(days=7)))
    assert [r.id for r in load_records(path, since=timedelta(days=7), now=NOW)] == ["edge"]


def test_since_none_keeps_everything(tmp_path):
    path = write(tmp_path, record("a", NOW - timedelta(days=900)), record("b", NOW))
    assert [r.id for r in load_records(path, since=None, now=NOW)] == ["a", "b"]


def test_defaults_do_not_crash_and_now_defaults_to_the_current_time(tmp_path):
    just_now = datetime.now(timezone.utc)
    path = write(tmp_path, record("fresh", just_now))
    assert [r.id for r in load_records(path)] == ["fresh"]
    assert [r.id for r in load_records(path, since=timedelta(hours=1))] == ["fresh"]


def test_returns_validated_records_in_file_order(tmp_path):
    path = write(tmp_path, record("b", NOW), record("a", NOW - timedelta(hours=1)))
    records = load_records(path, now=NOW)
    assert [r.id for r in records] == ["b", "a"]
    assert records[0].component == "support"  # a real CallRecord, not a raw dict


def test_blank_lines_are_ignored(tmp_path):
    path = write(tmp_path, "", record("a", NOW), "   ", record("b", NOW), "")
    assert [r.id for r in load_records(path, now=NOW)] == ["a", "b"]


def test_an_empty_window_returns_no_records_instead_of_failing(tmp_path):
    path = write(tmp_path, record("old", NOW - timedelta(days=30)))
    assert load_records(path, since=timedelta(days=7), now=NOW) == []


def test_an_empty_file_returns_no_records(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert load_records(path, now=NOW) == []


def test_mixed_schema_versions_are_rejected_even_when_the_odd_line_is_outside_the_window(tmp_path):
    path = write(
        tmp_path,
        record("recent", NOW - timedelta(days=1)),
        record("old_v1", NOW - timedelta(days=30), schema_version="v1"),
    )
    with pytest.raises(FileRejected, match=r"mixes schema versions: v0 \(1 lines\), v1 \(1 lines\)"):
        load_records(path, since=timedelta(days=7), now=NOW)


def test_invalid_json_is_an_error_naming_the_line_not_a_silent_skip(tmp_path):
    path = write(tmp_path, record("a", NOW), "{not json", record("b", NOW))
    with pytest.raises(FileRejected, match="line 2 is not valid JSON"):
        load_records(path, now=NOW)


def test_a_line_that_is_not_an_object_is_an_error(tmp_path):
    with pytest.raises(FileRejected, match="line 1 is not a JSON object"):
        load_records(write(tmp_path, "[1, 2]"), now=NOW)


def test_an_invalid_v0_record_is_an_error_naming_the_line_and_field(tmp_path):
    path = write(tmp_path, record("a", NOW), record("b", NOW, usage={"input_tokens": -1}))
    with pytest.raises(FileRejected, match=r"line 2 is not a valid v0 record \(check: usage.input_tokens\)"):
        load_records(path, now=NOW)


def test_an_invalid_record_outside_the_window_is_still_an_error(tmp_path):
    """Nothing is silently dropped: a bad line is reported even if it would be filtered out."""
    path = write(tmp_path, record("old_bad", NOW - timedelta(days=30), model=""), record("ok", NOW))
    with pytest.raises(FileRejected, match="line 1"):
        load_records(path, since=timedelta(days=7), now=NOW)


def test_a_naive_now_is_rejected_with_a_clear_message(tmp_path):
    path = write(tmp_path, record("a", NOW))
    with pytest.raises(ValueError, match="timezone-aware"):
        load_records(path, since=timedelta(days=1), now=datetime(2026, 10, 21, 12, 0))


def test_a_window_too_large_to_subtract_keeps_everything(tmp_path):
    path = write(tmp_path, record("a", NOW))
    assert len(load_records(path, since=timedelta(days=999_999_999), now=NOW)) == 1


def test_a_missing_file_raises_the_normal_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_records(tmp_path / "nope.jsonl", now=NOW)


# --- build_manifest, build_fingerprint, build_archive, default_export_path ------------------

PROMPT, ANSWER, TOOL_ARG = "SECRET PROMPT TEXT", "SECRET ANSWER TEXT", "SECRET TOOL ARGUMENT"


def full_record(id: str, component: str = "support") -> CallRecord:
    """A record with real content in it, so leaks into the fingerprint can be detected."""
    return CallRecord.model_validate(
        record(
            id,
            NOW,
            component=component,
            latency={"total_ms": 100},
            usage={"input_tokens": 10, "output_tokens": 5},
            metadata={"category": "classification"},
            request={
                "messages": [{"role": "user", "content": PROMPT}],
                "parameters": {"temperature": 0, "max_output_tokens": 16},
                "stream": False,
            },
            response={
                "status": "success",
                "model": "gpt-4o-mini-2024-07-18",
                "text": ANSWER,
                "tool_calls": [{"name": "lookup", "arguments": {"q": TOOL_ARG}}],
                "finish_reason": "stop",
            },
        )
    )


def read_archive(data: bytes) -> dict[str, bytes]:
    """The archive's members in order (dicts keep insertion order)."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


def test_manifest_holds_the_schema_version_and_record_count():
    assert build_manifest([full_record("a"), full_record("b"), full_record("c")]) == {
        "schema_version": "v0",
        "record_count": 3,
    }


def test_manifest_of_no_records():
    assert build_manifest([]) == {"schema_version": "v0", "record_count": 0}


def test_fingerprint_never_contains_the_model_inputs_or_outputs():
    fingerprint = build_fingerprint(b"records", [full_record("a")])
    text = json.dumps(fingerprint)
    assert PROMPT not in text and ANSWER not in text and TOOL_ARG not in text

    (entry,) = fingerprint["fingerprints"]
    assert "messages" not in entry["request"]
    assert "text" not in entry["response"] and "tool_calls" not in entry["response"]


def test_fingerprint_keeps_everything_else():
    (entry,) = build_fingerprint(b"records", [full_record("a")])["fingerprints"]
    assert entry["id"] == "a" and entry["component"] == "support"
    assert entry["provider"] == "openai" and entry["model"] == "gpt-4o-mini"
    assert entry["started_at"].startswith("2026-10-21")
    assert entry["latency"] == {"total_ms": 100.0}
    assert entry["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert entry["metadata"] == {"category": "classification"}
    assert entry["request"]["parameters"] == {"temperature": 0.0, "max_output_tokens": 16}
    assert entry["request"]["stream"] is False
    assert entry["response"] == {
        "status": "success", "model": "gpt-4o-mini-2024-07-18", "finish_reason": "stop",
    }  # fmt: skip


def test_fingerprint_hash_and_count():
    fingerprint = build_fingerprint(b"some records", [full_record("a"), full_record("b")])
    assert fingerprint["records_sha256"] == hashlib.sha256(b"some records").hexdigest()
    assert fingerprint["record_count"] == 2
    assert [f["id"] for f in fingerprint["fingerprints"]] == ["a", "b"]


def test_fingerprint_hash_changes_when_the_records_change():
    a = build_fingerprint(b"records", [])["records_sha256"]
    b = build_fingerprint(b"recordz", [])["records_sha256"]
    assert a != b


def test_archive_members_are_in_order_with_the_manifest_first():
    members = read_archive(build_archive([full_record("a")]))
    assert list(members) == ["manifest.json", "fingerprint.json", "records.jsonl"]


def test_archive_manifest_is_readable_and_correct():
    records = [full_record("a"), full_record("b")]
    manifest = json.loads(read_archive(build_archive(records))["manifest.json"])
    assert manifest == {"schema_version": "v0", "record_count": 2}


def test_archive_round_trips_the_full_records():
    records = [full_record("a"), full_record("b", component="triage")]
    lines = read_archive(build_archive(records))["records.jsonl"].decode().splitlines()
    assert [CallRecord.model_validate_json(line) for line in lines] == records


def test_archive_records_match_what_the_recorder_and_importer_write():
    records = [full_record("a")]
    expected = (record_to_json(records[0]) + "\n").encode()
    assert read_archive(build_archive(records))["records.jsonl"] == expected


def test_archive_keeps_full_content_even_though_the_fingerprint_does_not():
    members = read_archive(build_archive([full_record("a")]))
    assert PROMPT.encode() in members["records.jsonl"]  # the export is not redacted
    assert PROMPT.encode() not in members["fingerprint.json"]


def test_archive_fingerprint_hash_matches_the_records_in_the_archive():
    members = read_archive(build_archive([full_record("a"), full_record("b")]))
    fingerprint = json.loads(members["fingerprint.json"])
    assert fingerprint["records_sha256"] == hashlib.sha256(members["records.jsonl"]).hexdigest()


def test_archive_of_no_records_is_still_valid():
    members = read_archive(build_archive([]))
    assert json.loads(members["manifest.json"])["record_count"] == 0
    assert members["records.jsonl"] == b""
    assert json.loads(members["fingerprint.json"])["fingerprints"] == []


def test_archive_is_gzipped_tar_bytes():
    data = build_archive([full_record("a")])
    assert isinstance(data, bytes) and data[:2] == b"\x1f\x8b"  # gzip magic number


def test_default_export_path_is_a_path_named_for_the_date():
    assert default_export_path(NOW) == Path("migration-eval-export-2026-10-21.tar.gz")
    assert isinstance(default_export_path(NOW), Path)


def test_default_export_path_without_now_uses_todays_date():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert default_export_path().name == f"migration-eval-export-{today}.tar.gz"


# --- summarize_export and format_export_summary (EXP-2) ------------------------------------


def at(hours_ago: float, id: str = "r", component: str = "support") -> CallRecord:
    return CallRecord.model_validate(record(id, NOW - timedelta(hours=hours_ago), component=component))


def test_summary_counts_records_and_components_largest_first():
    records = [at(1, "a", "triage"), at(2, "b", "summarizer"), at(3, "c", "summarizer"), at(4, "d", "summarizer")]
    summary = summarize_export(records, archive_size=2048)

    assert summary.record_count == 4
    assert summary.components == {"summarizer": 3, "triage": 1}
    assert list(summary.components) == ["summarizer", "triage"]
    assert summary.size_bytes == 2048


def test_summary_time_range_is_the_earliest_and_latest_even_when_records_are_out_of_order():
    records = [at(5, "mid"), at(1, "last"), at(30, "first"), at(2, "other")]
    summary = summarize_export(records, archive_size=1)
    assert summary.first_started_at == NOW - timedelta(hours=30)
    assert summary.last_started_at == NOW - timedelta(hours=1)


def test_summary_of_one_record_has_the_same_first_and_last():
    summary = summarize_export([at(2)], archive_size=1)
    assert summary.first_started_at == summary.last_started_at == NOW - timedelta(hours=2)


def test_summary_of_no_records():
    summary = summarize_export([], archive_size=100)
    assert (summary.record_count, summary.components) == (0, {})
    assert summary.first_started_at is None and summary.last_started_at is None


def summary_of(*records: CallRecord, size: int = 1500) -> ExportSummary:
    return summarize_export(list(records), size)


def test_formatted_summary_shows_count_components_time_range_and_size():
    text = format_export_summary(
        summary_of(at(48, "a", "triage"), at(24, "b", "summarizer"), at(1, "c", "summarizer"), size=1500)
    )
    lines = text.splitlines()
    assert lines[0] == "Records:     3"
    assert lines[1] == "Time range:  2026-10-19 12:00 UTC to 2026-10-21 11:00 UTC"
    assert lines[2] == "File size:   1.5 KB"
    assert lines[3] == "Components:"
    assert lines[4:] == ["  summarizer  2", "  triage      1"]  # largest first, counts aligned


def test_time_range_is_shown_in_utc_whatever_offset_the_records_use():
    plus_two = timezone(timedelta(hours=2))
    record_in_cest = CallRecord.model_validate(
        record("a", datetime(2026, 10, 21, 14, 30, tzinfo=plus_two))  # 12:30 UTC
    )
    assert "2026-10-21 12:30 UTC to 2026-10-21 12:30 UTC" in format_export_summary(summary_of(record_in_cest))


def test_formatted_summary_for_no_records_does_not_crash():
    assert format_export_summary(summarize_export([], archive_size=0)) == "No records in this export."


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0 bytes"),
        (1, "1 byte"),
        (1023, "1023 bytes"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024 * 1024 - 1, "1.0 MB"),  # never shows "1024.0 KB"
        (1024 * 1024, "1.0 MB"),
        (5 * 1024 * 1024 + 512 * 1024, "5.5 MB"),
        (500 * 1024 * 1024, "500.0 MB"),
        (1024**3, "1.0 GB"),
    ],
)
def test_file_size_is_shown_in_a_readable_unit(size, expected):
    text = format_export_summary(summary_of(at(1), size=size))
    assert f"File size:   {expected}" in text.splitlines()


def test_a_long_component_list_is_capped_with_a_count_of_the_rest():
    records = [at(1, f"r{i}", f"component-{i:02d}") for i in range(MAX_COMPONENTS_SHOWN + 3)]
    lines = format_export_summary(summary_of(*records)).splitlines()
    component_lines = lines[lines.index("Components:") + 1 :]
    assert len(component_lines) == MAX_COMPONENTS_SHOWN + 1  # the shown ones plus the "more" line
    assert component_lines[-1] == "  ... and 3 more"


def test_components_at_exactly_the_cap_are_all_shown():
    records = [at(1, f"r{i}", f"component-{i:02d}") for i in range(MAX_COMPONENTS_SHOWN)]
    text = format_export_summary(summary_of(*records))
    assert "more" not in text
