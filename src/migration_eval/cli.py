"""The `migration-eval` command line."""

from __future__ import annotations

import socket
from collections import Counter
from datetime import datetime, timezone
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
from migration_eval.gpu.assessment import assess
from migration_eval.gpu.commands import run_command
from migration_eval.gpu.cpu_samples import COMPUTE_APPS_ARGS, CpuSampler, parse_compute_apps, whole_seconds
from migration_eval.gpu.host import take_host_snapshot
from migration_eval.gpu.idle import check_idle, format_idle_warning
from migration_eval.gpu.importer import PARTNER_COMMANDS, import_files, missing_pieces
from migration_eval.gpu.nvidia_smi import (
    NvidiaSmiError,
    check_available,
    find_nvidia_smi,
    make_runner,
)
from migration_eval.gpu.profile import (
    DEFAULT_PROFILE_PATH,
    CaptureSettings,
    GpuProfile,
    read_profile,
    write_profile,
)
from migration_eval.gpu.serving import collect_serving_settings
from migration_eval.gpu.samples import (
    DEFAULT_DURATION_S,
    DEFAULT_INTERVAL_S,
    collect,
    planned_ticks,
    resolve_fields,
)
from migration_eval.gpu.snapshot import take_snapshot

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
    no_input: bool = typer.Option(
        False, "--no-input", help="Don't ask about the serving engine when no vLLM process is found."
    ),
) -> None:
    """Sample this machine's GPUs with nvidia-smi and write gpu_profile.json (GPU-1).

    Run it on the GPU machine during normal traffic. Ctrl-C stops early and still writes what
    was collected. `export` includes the profile.
    """
    try:
        ticks = planned_ticks(interval, duration)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--duration / --interval") from None

    try:
        run = make_runner(find_nvidia_smi())
        check_available(run)
        snapshot = take_snapshot(run)
        fields = resolve_fields(run)
    except NvidiaSmiError as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo("Nothing was written.", err=True)
        raise typer.Exit(code=1) from None

    models = Counter(gpu.name for gpu in snapshot.gpus)
    typer.echo(
        f"Found {len(snapshot.gpus)} GPU(s): "
        + ", ".join(f"{count} x {name}" for name, count in models.items())
        + f" (driver {snapshot.driver_version})"
    )
    if snapshot.topology is None:
        typer.echo("Warning: could not read `nvidia-smi topo -m`; topology will be missing.", err=True)

    host = take_host_snapshot(run_command)
    not_found = [c.argv[0] for c in host.commands if c.status == "missing"]
    if not_found:
        typer.echo(f"Not installed here, skipped: {', '.join(not_found)}.")

    serving = collect_serving_settings(run_command, typer.prompt, interactive=not no_input)
    for s in serving:
        where = f" (pid {s.pid})" if s.pid else ""
        typer.echo(f"Serving: {s.engine}{where}" + (f", model {s.model}" if s.model else ""))

    if whole_seconds(interval) != interval:
        typer.echo(f"Note: CPU sampling (mpstat, pidstat) uses whole seconds, so every {whole_seconds(interval)}s.")
    typer.echo(f"Sampling every {interval:g}s for {duration:g}s ({ticks} samples). Ctrl-C stops early.")

    def progress(tick: int, total: int) -> None:
        typer.echo(f"\r  {tick}/{total}", nl=False)

    cpu_ticks = max(1, int(duration // whole_seconds(interval)))
    sampler = CpuSampler(interval, cpu_ticks)
    sampler.start()
    started_at = datetime.now(timezone.utc)
    result = collect(run, interval, duration, fields=fields, on_tick=progress)
    ended_at = datetime.now(timezone.utc)
    typer.echo("")

    try:
        gpu_pids = parse_compute_apps(run(list(COMPUTE_APPS_ARGS)))
    except NvidiaSmiError:
        gpu_pids = frozenset()
    cpu_samples, process_samples, cpu_commands = sampler.stop(gpu_pids)
    host.commands.extend(cpu_commands)

    if result.interrupted:
        typer.echo(f"Stopped early after {result.ticks_taken} of {ticks} samples.")
    if result.ticks_failed:
        typer.echo(
            f"Warning: {result.ticks_failed} sample(s) failed and were skipped. Last error: {result.last_error}",
            err=True,
        )
    if not result.samples:
        typer.echo("No samples were collected, so nothing was written.", err=True)
        raise typer.Exit(code=1)
    idle_warning = check_idle(result.samples)
    if idle_warning is not None:
        typer.echo(format_idle_warning(idle_warning), err=True)

    profile = GpuProfile(
        hostname=socket.gethostname(),
        settings=CaptureSettings(
            source="capture",
            interval_s=interval,
            duration_s=duration,
            started_at=started_at,
            ended_at=ended_at,
            interrupted=result.interrupted,
            ticks_planned=ticks,
            ticks_failed=result.ticks_failed,
        ),
        snapshot=snapshot,
        samples=result.samples,
        host=host,
        cpu_samples=cpu_samples,
        process_samples=process_samples,
        serving=serving,
        idle_warning=idle_warning,
    )
    write_profile(profile, out)
    typer.echo(f"Wrote {out}")


@gpu_app.command("import")
def gpu_import_command(
    files: list[Path] = typer.Argument(
        None, exists=True, dir_okay=False, readable=True, help="Command output files the partner sent."
    ),
    out: Path = typer.Option(DEFAULT_PROFILE_PATH, "--out", "-o", help="Where to write the profile."),
    hostname: str | None = typer.Option(None, "--hostname", help="The GPU machine's name, for the report."),
    print_commands: bool = typer.Option(
        False, "--print-commands", help="Print the commands the partner should run, then exit."
    ),
) -> None:
    """Build gpu_profile.json from nvidia-smi (and CPU/NUMA) output run by hand (GPU-3).

    For partners who can't install this CLI on the GPU machine. Run with --print-commands to
    get the exact commands to send them.
    """
    if print_commands:
        typer.echo(PARTNER_COMMANDS)
        return
    if not files:
        raise typer.BadParameter("give the files to import, or --print-commands", param_hint="FILES")
    try:
        profile = import_files(files, hostname)
    except FileRejected as rejected:
        _reject(rejected)

    models = Counter(gpu.name for gpu in profile.snapshot.gpus)
    settings = profile.settings
    typer.echo(
        f"Found {len(profile.snapshot.gpus)} GPU(s): "
        + ", ".join(f"{count} x {name}" for name, count in models.items())
        + f" (driver {profile.snapshot.driver_version})"
    )
    typer.echo(
        f"{len(profile.samples)} samples from {settings.started_at:%Y-%m-%d %H:%M:%S} to "
        f"{settings.ended_at:%H:%M:%S}, every {settings.interval_s:g}s."
    )
    missing = missing_pieces(profile)
    if missing:
        typer.echo(f"Not included: {', '.join(missing)}.")

    profile.idle_warning = check_idle(profile.samples)
    if profile.idle_warning is not None:
        typer.echo(format_idle_warning(profile.idle_warning), err=True)

    write_profile(profile, out)
    typer.echo(f"Wrote {out}")


@gpu_app.command("assess")
def gpu_assess_command(
    profile_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="gpu_profile.json"),
) -> None:
    """Print the GPU performance assessment for a profile, as JSON (GPU-5; for development).

    The report writer will render the same assessment into report.json / report.html.
    """
    try:
        profile = read_profile(profile_path)
    except FileRejected as rejected:
        _reject(rejected)
    typer.echo(assess(profile).model_dump_json(indent=2))


def _reject(rejected: FileRejected) -> NoReturn:
    typer.echo(f"Rejected: {rejected}", err=True)
    typer.echo("Nothing was written.", err=True)
    raise typer.Exit(code=1)


def _echo_skipped(result: ImportResult) -> None:
    for number, reason in result.skipped[:MAX_REASONS_SHOWN]:
        typer.echo(f"  line {number}: {reason}", err=True)
    if len(result.skipped) > MAX_REASONS_SHOWN:
        typer.echo(f"  ... and {len(result.skipped) - MAX_REASONS_SHOWN} more", err=True)
