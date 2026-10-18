"""Whole-file checks (REC-5).

Unlike grading, which judges one record at a time, these decide whether a file can be
used at all. When one fails the file is rejected: nothing is written and the message says
what is wrong and how to fix it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from migration_eval.capture.grading import REASON_TEXT, Grade, Summary

# Lines with no schema_version are raw log exchanges that the importer normalizes into v0.
IMPLICIT_VERSION = "v0"


class FileRejected(Exception):
    """The whole file is unusable; the message says why and how to fix it."""


def check_schema_versions(lines: Iterable[dict[str, Any]]) -> None:
    """Raise FileRejected if the lines disagree on schema_version.

    This takes the raw parsed JSON, not CallRecords: a record with another version would
    fail model validation before it could be counted, and every validated CallRecord is
    "v0" by construction, so a check on records could never fail.
    """
    versions = Counter(str(line.get("schema_version", IMPLICIT_VERSION)) for line in lines)
    if len(versions) <= 1:
        return
    found = ", ".join(f"{version} ({count} lines)" for version, count in versions.most_common())
    raise FileRejected(
        f"this file mixes schema versions: {found}. "
        "Re-export it so every record has the same schema_version."
    )


def check_replay_grade(summary: Summary) -> None:
    """Raise FileRejected if the file has no replay-grade records."""
    if summary.total == 0:
        raise FileRejected("the file contains no records.")
    if summary.counts[Grade.REPLAY_GRADE] > 0:
        return

    message = [
        f"none of the {summary.total} records are replay-grade, so no quality verdict is possible.",
        "What is missing:",
    ]
    for reason, count in summary.reasons.most_common():
        message.append(f"  {count} x {reason.value}: {REASON_TEXT[reason]}")
    raise FileRejected("\n".join(message))
