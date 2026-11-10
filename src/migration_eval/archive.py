"""Reading an upload (EXP-5): the inverse of `export.build_archive`.

The server never trusts an uploaded file. It reads the archive, checks it, and only then keeps
its records. The layout it expects is the one `export.py` writes, as one .tar.gz with exactly
these members, manifest first:

    manifest.json     {"schema_version": "v0", "record_count": N}
    fingerprint.json  {"records_sha256": ..., "record_count": N, "fingerprints": [...]}
    records.jsonl     one v0 record per line

A rejected archive raises `FileRejected` (from `capture.file_checks`) with a message the user can
act on, and nothing is kept.

Open decisions (see the notes on EXP-5):
    * the size limits. `MAX_ARCHIVE_BYTES` caps the upload and `MAX_UNPACKED_BYTES` caps what it
      expands to, so a small archive cannot unpack into something huge. Both are placeholders.
    * whether the fingerprint should be checked against the records beyond the hash (it was built
      to let a partner confirm what they sent without the server keeping prompts)
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from migration_eval.capture.file_checks import FileRejected, check_replay_grade, check_schema_versions
from migration_eval.capture.grading import Grade, grade, summarize
from migration_eval.models import CallRecord
from migration_eval.profile import Incumbent, find_incumbent

MAX_ARCHIVE_BYTES = 100 * 1024 * 1024  # placeholder
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024  # placeholder
EXPECTED_MEMBERS = ("manifest.json", "fingerprint.json", "records.jsonl")


@dataclass(frozen=True)
class LoadedArchive:
    manifest: dict[str, Any]
    fingerprint: dict[str, Any]
    records: list[CallRecord]


@dataclass(frozen=True)
class ComponentSummary:
    """What the upload page shows for one component, so the user can pick one to evaluate."""

    name: str
    records: int
    replay_grade: int  # how many of those records can be replayed (see capture/grading.py)
    incumbent: Incumbent | None  # the model its traffic was logged from; None when it cannot be told


def _read_members(data: bytes) -> dict[str, bytes]:
    """Checks 1 to 3: the archive's shape and size. Returns each member's bytes by name."""
    if len(data) > MAX_ARCHIVE_BYTES:
        raise FileRejected(f"the file is larger than the {MAX_ARCHIVE_BYTES // (1024 * 1024)} MB limit.")

    members: dict[str, bytes] = {}
    unpacked = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for info in tar:  # streamed: nothing is unpacked to disk
                if not info.isfile() or info.name not in EXPECTED_MEMBERS or info.name in members:
                    raise FileRejected(
                        f"the archive has an unexpected entry {info.name!r}; it must hold exactly "
                        f"{', '.join(EXPECTED_MEMBERS)}. Re-export it with `migration-eval export`."
                    )
                unpacked += info.size
                if unpacked > MAX_UNPACKED_BYTES:
                    raise FileRejected("the archive expands to more than the size limit.")
                handle = tar.extractfile(info)
                members[info.name] = handle.read(info.size) if handle else b""  # type: ignore[union-attr]
    except (tarfile.TarError, EOFError, OSError):  # not gzip, not tar, or cut off
        raise FileRejected("this is not a valid export archive. Create one with `migration-eval export`.") from None

    if tuple(members) != EXPECTED_MEMBERS:
        missing = [name for name in EXPECTED_MEMBERS if name not in members]
        problem = f"is missing {', '.join(missing)}" if missing else "has its files in the wrong order (the manifest must come first)"
        raise FileRejected(f"the archive {problem}. Re-export it with `migration-eval export`.")
    return members


def _json_object(name: str, raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except ValueError:
        raise FileRejected(f"{name} is not valid JSON.") from None
    if not isinstance(value, dict):
        raise FileRejected(f"{name} is not a JSON object.")
    return value


def read_archive(data: bytes) -> LoadedArchive:
    """Check and unpack an uploaded archive. Raise `FileRejected` when any check fails; the checks, in order:

      1. `len(data) <= MAX_ARCHIVE_BYTES`
      2. it is a valid gzip'd tar, and its members are exactly `EXPECTED_MEMBERS`, in that order,
         plain files with plain names (no directories, links, absolute paths or `..`)
      3. reading the members does not pass `MAX_UNPACKED_BYTES` in total (stop reading, do not
         unpack first and measure after)
      4. the manifest has `schema_version == "v0"` and an integer `record_count`
      5. every line of records.jsonl is a valid v0 record (reuse `export.load_records`' checks:
         `check_schema_versions`, then `CallRecord.model_validate`), and the count matches the manifest
      6. `fingerprint["records_sha256"]` equals the SHA-256 of the records.jsonl bytes
      7. at least one record is replay-grade (`file_checks.check_replay_grade`)
    """
    members = _read_members(data)  # checks 1 to 3

    manifest = _json_object("manifest.json", members["manifest.json"])  # check 4
    if manifest.get("schema_version") != "v0":
        raise FileRejected(f"the archive's schema version is {manifest.get('schema_version')!r}; only v0 is supported.")
    count = manifest.get("record_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise FileRejected("manifest.json has no valid record_count.")

    fingerprint = _json_object("fingerprint.json", members["fingerprint.json"])

    records_bytes = members["records.jsonl"]  # check 5
    try:
        lines = [(n, json.loads(raw)) for n, raw in enumerate(records_bytes.decode("utf-8").splitlines(), start=1) if raw.strip()]
    except (ValueError, UnicodeDecodeError) as exc:
        raise FileRejected(f"records.jsonl could not be read as JSON lines ({type(exc).__name__}).") from None
    for number, line in lines:
        if not isinstance(line, dict):
            raise FileRejected(f"records.jsonl line {number} is not a JSON object.")
    check_schema_versions(line for _, line in lines)

    records: list[CallRecord] = []
    for number, line in lines:
        try:
            records.append(CallRecord.model_validate(line))
        except ValidationError as exc:
            where = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors()[:3])
            raise FileRejected(f"records.jsonl line {number} is not a valid v0 record (check: {where}).") from None
    if len(records) != count:
        raise FileRejected(f"the manifest says {count} records but the file holds {len(records)}.")

    if fingerprint.get("records_sha256") != hashlib.sha256(records_bytes).hexdigest():  # check 6
        raise FileRejected("the records do not match the fingerprint, so the file was changed after it was exported.")

    check_replay_grade(summarize(grade(r) for r in records))  # check 7
    return LoadedArchive(manifest=manifest, fingerprint=fingerprint, records=records)


def component_summaries(records: list[CallRecord]) -> list[ComponentSummary]:
    """One summary per component, largest first: the record count, how many are replay-grade
    (`grading.grade`), and the incumbent (`profile.find_incumbent` over that component's records).
    """
    by_component: dict[str, list[CallRecord]] = {}
    for record in records:
        by_component.setdefault(record.component, []).append(record)

    summaries = [
        ComponentSummary(
            name=name,
            records=len(group),
            replay_grade=sum(1 for r in group if grade(r).grade == Grade.REPLAY_GRADE),
            incumbent=find_incumbent(group),
        )
        for name, group in by_component.items()
    ]
    return sorted(summaries, key=lambda s: (-s.records, s.name))
