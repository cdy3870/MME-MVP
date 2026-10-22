"""Export (EXP-1, EXP-2): pack recorded calls into one archive for upload.

The CLI neither redacts nor encrypts: both are out of the MVP. The file is protected in
transit by TLS (EXP-4), at rest by the partner's KMS key (EXP-3), and by sending prompts
only to allow-listed providers.

Archive layout, one .tar.gz with the members in this order:

    manifest.json     {"schema_version": "v0", "record_count": N}   read first by the API (EXP-5)
    fingerprint.json  see build_fingerprint
    records.jsonl     one v0 record per line

EXP-2 (the summary shown before the file is written) is summarize_export and
format_export_summary.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from migration_eval.capture.file_checks import FileRejected, check_schema_versions
from migration_eval.capture.sink import record_to_json
from migration_eval.models import CallRecord

SCHEMA_VERSION = "v0"


@dataclass(frozen=True)
class ExportSummary:
    """What the partner sees before the file is written (EXP-2)."""

    record_count: int
    components: dict[str, int]  # component name -> number of records
    first_started_at: datetime | None  # None when there are no records
    last_started_at: datetime | None
    size_bytes: int  # size of the archive that will be written


# --- EXP-1: build the export ---------------------------------------------------------------


def parse_since(value: str) -> timedelta:
    """Turn a window like "7d" or "24h" into a timedelta.

    Support `d` (days) and `h` (hours). Raise ValueError with a clear message for anything
    else, e.g. "7", "d7", "0d" or a negative number.
    """

    match = re.fullmatch(r"(-?\d+)([dh])", value)

    if not match:
        raise ValueError(
            f"Invalid format: '{value}'. Expected a positive integer followed by 'd' or 'h' (e.g., '7d', '24h')."
        )

    amount_str, unit = match.groups()
    amount = int(amount_str)

    if amount <= 0:
        raise ValueError(
            f"Invalid value: '{value}'. The duration must be a positive number greater than 0."
        )

    try:
        if unit == "d":
            return timedelta(days=amount)
        return timedelta(hours=amount)  # unit == "h"
    except OverflowError:
        raise ValueError(f"Invalid value: '{value}'. The duration is too large.") from None


def load_records(
    path: Path,
    since: timedelta | None = None,
    now: datetime | None = None,
) -> list[CallRecord]:
    """Read a v0 JSONL file and return the records inside the window.

    Keep records with `started_at >= now - since`. `now` defaults to the current time in
    UTC; it is a parameter so tests do not depend on the clock. `since=None` keeps all.

    Check the raw lines with `check_schema_versions` before validating them as CallRecords
    (REC-5), and raise FileRejected for a file that mixes versions. Lines that are not valid
    v0 records are an error here, not skipped: an export should not silently drop data.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        # Record timestamps are timezone-aware, so comparing against a naive time would fail.
        raise ValueError("`now` must be timezone-aware")

    try:
        cutoff = None if since is None else now - since
    except OverflowError:
        cutoff = None  # the window reaches back before year 1, so everything is inside it

    # Pass 1: parse every line. The version check needs the raw dicts and the whole file,
    # including lines outside the window, so it cannot be done while filtering.
    parsed: list[tuple[int, dict[str, Any]]] = []
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                line = json.loads(raw)
            except json.JSONDecodeError:
                raise FileRejected(f"line {number} is not valid JSON.") from None
            if not isinstance(line, dict):
                raise FileRejected(f"line {number} is not a JSON object.")
            parsed.append((number, line))

    check_schema_versions(line for _, line in parsed)

    # Pass 2: validate each line as a v0 record and keep the ones inside the window.
    records: list[CallRecord] = []
    for number, line in parsed:
        try:
            record = CallRecord.model_validate(line)
        except ValidationError as exc:
            where = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors()[:3])
            raise FileRejected(f"line {number} is not a valid v0 record (check: {where}).") from None
        if cutoff is None or record.started_at >= cutoff:
            records.append(record)
    return records


def build_manifest(records: list[CallRecord]) -> dict[str, Any]:
    """The manifest the API reads before accepting an upload (EXP-5).

    Per the ERD it holds the schema version and the record count: `{"schema_version": "v0",
    "record_count": len(records)}`.
    """

    return {"schema_version": SCHEMA_VERSION, "record_count": len(records)}


def build_fingerprint(records_jsonl: bytes, records: list[CallRecord]) -> dict[str, Any]:
    """Everything about the records except the model's inputs and outputs.

    Rule: per record, keep every field except `request.messages` (the input) and
    `response.text` / `response.tool_calls` (the output). With the generated models:

        record.model_dump(mode="json", exclude_none=True, by_alias=True,
                          exclude={"request": {"messages"}, "response": {"text", "tool_calls"}})

    Shape: `{"records_sha256": <sha256 of records_jsonl>, "record_count": N,
    "fingerprints": [<one such dict per record>]}`.

    Still to decide (see the notes on EXP-1): whether `request.tools` (tool definitions, which
    can describe the business), `request.parameters.extra`, `response_format.schema` and
    provider error messages count as inputs/outputs, and so should also be excluded.
    """
    records_sha256 = hashlib.sha256(records_jsonl).hexdigest()
    fingerprints = [
        record.model_dump(
            mode="json",
            exclude_none=True,
            by_alias=True,
            exclude={"request": {"messages"}, "response": {"text", "tool_calls"}},
        )
        for record in records
    ]
    return {
        "records_sha256": records_sha256,
        "record_count": len(records),
        "fingerprints": fingerprints,
    }


def build_archive(records: list[CallRecord]) -> bytes:
    """Pack the records into the in-memory .tar.gz described in the module docstring.

    Serialize each record with `record_to_json` from `capture.sink`, so the file matches
    what the recorder and importer write. Return bytes rather than writing a file: EXP-2
    needs the real size before anything touches the disk.
    """

    records_jsonl = "".join(record_to_json(r) + "\n" for r in records).encode("utf-8")
    manifest = build_manifest(records)
    fingerprint = build_fingerprint(records_jsonl, records)

    manifest_json = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    fingerprint_json = json.dumps(fingerprint, sort_keys=True, indent=2).encode("utf-8")

    # The manifest goes first: the API reads it without unpacking the rest (EXP-5).
    files = [
        ("manifest.json", manifest_json),
        ("fingerprint.json", fingerprint_json),
        ("records.jsonl", records_jsonl),
    ]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()  # after the `with`: the gzip footer is written when it closes


def default_export_path(now: datetime | None = None) -> Path:
    """Where `export` writes when `--out` is not given, e.g. `migration-eval-export-2026-10-21.tar.gz`."""
    if now is None:
        now = datetime.now(timezone.utc)
    return Path(f"migration-eval-export-{now:%Y-%m-%d}.tar.gz")


# --- EXP-2: the summary shown before writing -----------------------------------------------


MAX_COMPONENTS_SHOWN = 10


def summarize_export(records: list[CallRecord], archive_size: int) -> ExportSummary:
    """Count the records, count them per component, and find the first and last `started_at`."""
    # most_common() orders components by record count, largest first.
    components = dict(Counter(r.component for r in records).most_common())
    started = [r.started_at for r in records]
    return ExportSummary(
        record_count=len(records),
        components=components,
        first_started_at=min(started, default=None),
        last_started_at=max(started, default=None),
        size_bytes=archive_size,
    )


def _format_size(size: int) -> str:
    """A human-readable size: bytes, then KB, MB and GB in steps of 1024."""
    if size < 1024:
        return f"{size} byte" if size == 1 else f"{size} bytes"
    value = float(size)
    for unit in ("KB", "MB", "GB"):
        value /= 1024
        # Compare the rounded value, so 1,048,575 bytes shows as "1.0 MB", not "1024.0 KB".
        if round(value, 1) < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
    raise AssertionError("unreachable")


def _format_time(moment: datetime) -> str:
    # Shown in UTC whatever the record's own offset was, so the range is unambiguous.
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def format_export_summary(summary: ExportSummary) -> str:
    """Render the summary for the terminal: record count, components, time range and file size.

    Show the size in a human-readable unit (KB or MB). Handle the no-records case.
    """
    if summary.record_count == 0:
        return "No records in this export."

    lines = [
        f"Records:     {summary.record_count}",
        f"Time range:  {_format_time(summary.first_started_at)} to {_format_time(summary.last_started_at)}",
        f"File size:   {_format_size(summary.size_bytes)}",
        "Components:",
    ]
    shown = list(summary.components.items())[:MAX_COMPONENTS_SHOWN]
    width = max(len(name) for name, _ in shown)
    lines.extend(f"  {name:<{width}}  {count}" for name, count in shown)
    hidden = len(summary.components) - len(shown)
    if hidden > 0:
        lines.append(f"  ... and {hidden} more")
    return "\n".join(lines)
