"""The `migration-eval` command line."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer

from migration_eval.capture.file_checks import FileRejected, check_replay_grade
from migration_eval.capture.grading import format_summary, grade, summarize
from migration_eval.capture.importer import ImportResult, import_jsonl
from migration_eval.capture.sink import record_to_json

app = typer.Typer(
    name="migration-eval",
    help="Replay your own traffic against candidate models.",
    no_args_is_help=True,
)
import_app = typer.Typer(help="Normalize existing logs into v0 records.", no_args_is_help=True)
app.add_typer(import_app, name="import")

MAX_REASONS_SHOWN = 10


@import_app.command("jsonl")
def import_jsonl_command(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="JSONL log file."),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Where to write v0 records. Default: <input>.v0.jsonl"
    ),
    component: str | None = typer.Option(
        None, "--component", "-c", help="Component name for lines that do not set one."
    ),
) -> None:
    """Normalize a JSONL log into v0 records.

    A file that mixes schema versions, or has no replay-grade records, is rejected:
    nothing is written and the exit code is 1.
    """
    try:
        result = import_jsonl(path, component=component)
    except FileRejected as rejected:  # mixed schema versions
        _reject(rejected)

    typer.echo(f"Read {result.lines_read} lines: {len(result.records)} imported, {len(result.skipped)} skipped.")

    if not result.records:
        _echo_skipped(result)
        typer.echo("No records were imported, so nothing was written.", err=True)
        raise typer.Exit(code=1)

    summary = summarize(grade(r) for r in result.records)
    try:
        check_replay_grade(summary)
    except FileRejected as rejected:  # nothing in the file can be replayed
        _echo_skipped(result)
        _reject(rejected)

    typer.echo(format_summary(summary))
    _echo_skipped(result)

    # Written last, so a rejected file never leaves output behind.
    out = out or path.with_name(f"{path.stem}.v0.jsonl")
    out.write_text("".join(record_to_json(r) + "\n" for r in result.records), encoding="utf-8")
    typer.echo(f"Wrote {out}")


def _reject(rejected: FileRejected) -> NoReturn:
    typer.echo(f"Rejected: {rejected}", err=True)
    typer.echo("Nothing was written.", err=True)
    raise typer.Exit(code=1)


def _echo_skipped(result: ImportResult) -> None:
    for number, reason in result.skipped[:MAX_REASONS_SHOWN]:
        typer.echo(f"  line {number}: {reason}", err=True)
    if len(result.skipped) > MAX_REASONS_SHOWN:
        typer.echo(f"  ... and {len(result.skipped) - MAX_REASONS_SHOWN} more", err=True)
