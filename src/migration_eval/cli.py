"""The `migration-eval` command line."""

from __future__ import annotations

from pathlib import Path

import typer

from migration_eval.capture.importer import import_jsonl
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
    """Normalize a JSONL log into v0 records."""
    result = import_jsonl(path, component=component)
    out = out or path.with_name(f"{path.stem}.v0.jsonl")
    if result.records:
        out.write_text("".join(record_to_json(r) + "\n" for r in result.records), encoding="utf-8")

    typer.echo(f"Read {result.lines_read} lines: {len(result.records)} imported, {len(result.skipped)} skipped.")
    for number, reason in result.skipped[:MAX_REASONS_SHOWN]:
        typer.echo(f"  line {number}: {reason}", err=True)
    if len(result.skipped) > MAX_REASONS_SHOWN:
        typer.echo(f"  ... and {len(result.skipped) - MAX_REASONS_SHOWN} more", err=True)

    if not result.records:
        typer.echo("No records were imported, so nothing was written.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote {out}")
