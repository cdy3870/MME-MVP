"""The `migration-eval` command line."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer

from migration_eval.capture.file_checks import FileRejected, check_replay_grade
from migration_eval.capture.grading import format_summary, grade, summarize
from migration_eval.capture.importer import ImportResult, import_jsonl
from migration_eval.capture.recorder import DEFAULT_PATH as DEFAULT_RECORDS_PATH
from migration_eval.capture.sink import record_to_json
from migration_eval.export import (
    build_archive,
    default_export_path,
    format_export_summary,
    load_records,
    parse_since,
    summarize_export,
)
from migration_eval.gpu.profile import DEFAULT_PROFILE_PATH
from migration_eval.gpu.samples import DEFAULT_DURATION_S, DEFAULT_INTERVAL_S

app = typer.Typer(
    name="migration-eval",
    help="Replay your own traffic against candidate models.",
    no_args_is_help=True,
)
import_app = typer.Typer(help="Normalize existing logs into v0 records.", no_args_is_help=True)
app.add_typer(import_app, name="import")
gpu_app = typer.Typer(help="Read GPU utilization and topology for the GPU report.", no_args_is_help=True)
app.add_typer(gpu_app, name="gpu")

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


@app.command("export")
def export_command(
    since: str = typer.Option("7d", "--since", help="Only export records from this window, e.g. 7d or 24h."),
    records: Path = typer.Option(
        Path(DEFAULT_RECORDS_PATH),
        "--records",
        exists=True,
        dir_okay=False,
        readable=True,
        help="v0 JSONL file to export (the recorder's output or an import).",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Where to write the export. Default: migration-eval-export-<date>.tar.gz"
    ),
) -> None:
    """Pack recorded calls into one file for upload, after showing a summary.

    The export is rejected, and nothing is written, if the file mixes schema versions or
    nothing in the window can be replayed.
    """
    try:
        window = parse_since(since)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--since") from None

    try:
        loaded = load_records(records, window)
    except FileRejected as rejected:  # bad line or mixed schema versions
        _reject(rejected)

    if not loaded:
        typer.echo(f"No records in the last {since}, so there is nothing to export.", err=True)
        raise typer.Exit(code=1)

    try:
        check_replay_grade(summarize(grade(r) for r in loaded))
    except FileRejected as rejected:  # nothing in the window can be replayed
        _reject(rejected)

    archive = build_archive(loaded)
    typer.echo(format_export_summary(summarize_export(loaded, len(archive))))

    # Written last, so a rejected export never leaves a file behind.
    out = out or default_export_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(archive)
    typer.echo(f"Wrote {out}")


@gpu_app.command("capture")
def gpu_capture_command(
    duration: float = typer.Option(
        DEFAULT_DURATION_S, "--duration", min=1, help="How long to sample, in seconds."
    ),
    interval: float = typer.Option(
        DEFAULT_INTERVAL_S, "--interval", min=0.1, help="Seconds between samples."
    ),
    out: Path = typer.Option(DEFAULT_PROFILE_PATH, "--out", "-o", help="Where to write the profile."),
) -> None:
    """Sample this machine's GPUs with nvidia-smi and write gpu_profile.json (GPU-1).

    Run it on the GPU machine during normal traffic. `export` includes the profile.
    """
    # TODO(GPU-1): wire the pieces together:
    #   1. find_nvidia_smi() + make_runner() + check_available(); on NvidiaSmiError echo the
    #      message to stderr and exit 1.
    #   2. take_snapshot(run); echo GPU count, model and driver version.
    #   3. resolve_fields(run), then collect(run, interval, duration, on_tick=<progress line>).
    #   4. Build GpuProfile (hostname = socket.gethostname(), settings.interrupted if Ctrl-C
    #      cut the run short), write_profile(profile, out), echo "Wrote <out>".
    #   GPU-2 (idle warning) hooks in after step 3.
    raise NotImplementedError("gpu capture is not implemented yet (MOD-121)")


def _reject(rejected: FileRejected) -> NoReturn:
    typer.echo(f"Rejected: {rejected}", err=True)
    typer.echo("Nothing was written.", err=True)
    raise typer.Exit(code=1)


def _echo_skipped(result: ImportResult) -> None:
    for number, reason in result.skipped[:MAX_REASONS_SHOWN]:
        typer.echo(f"  line {number}: {reason}", err=True)
    if len(result.skipped) > MAX_REASONS_SHOWN:
        typer.echo(f"  ... and {len(result.skipped) - MAX_REASONS_SHOWN} more", err=True)
