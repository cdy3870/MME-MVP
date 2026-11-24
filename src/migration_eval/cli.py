"""The `migration-eval` command line."""

from __future__ import annotations

import asyncio
import os
import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

import httpx
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
from migration_eval.gpu.commands import run_command as run_system_command
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
from migration_eval.gpu.prices import DEFAULT_PRICES_PATH, load_prices
from migration_eval.gpu.report import build_gpu_report
from migration_eval.gpu.serving import collect_serving_settings
from migration_eval.gpu.samples import (
    DEFAULT_DURATION_S,
    DEFAULT_INTERVAL_S,
    collect,
    planned_ticks,
    resolve_fields,
)
from migration_eval.gpu.snapshot import take_snapshot
from migration_eval.gpu.vllm_metrics import (
    REQUEST_TIMEOUT_S as VLLM_TIMEOUT_S,
    VllmMetricsSampler,
    apply_versions,
    http_fetch,
    plan_sources,
    probe,
    settings_from_metrics,
)
from migration_eval.keys import MissingKeys, read_keys
from migration_eval.models import CallRecord, Report
from migration_eval.profile import build_profile, records_for_component
from migration_eval.request_builders import make_complete
from migration_eval.run import TIER_SAMPLE_SIZE, RunConfig, RunError, run_evaluation, write_report

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

TIERS = ("smoke", "full")
REQUEST_TIMEOUT_SECONDS = 60  # one stuck call must not hang the whole run


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
    vllm_url: list[str] = typer.Option(
        [], "--vllm-url", help="Read vLLM's /metrics here instead of the address in its launch flags (e.g. a container's mapped port). Repeat for several servers."
    ),
    no_vllm_metrics: bool = typer.Option(
        False, "--no-vllm-metrics", help="Read vLLM's launch flags but don't call its /metrics or /version."
    ),
) -> None:
    """Sample this machine's GPUs with nvidia-smi and write gpu_profile.json (GPU-1).

    Run it on the GPU machine during normal traffic. Ctrl-C stops early and still writes what
    was collected. `export` includes the profile.

    When vLLM is running, its launch flags are read from the process list and its /metrics is
    sampled on the same interval as the GPUs (GPU-14). If a server can't be reached the capture
    carries on without its metrics, and the report says so.
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

    host = take_host_snapshot(run_system_command)
    not_found = [c.argv[0] for c in host.commands if c.status == "missing"]
    if not_found:
        typer.echo(f"Not installed here, skipped: {', '.join(not_found)}.")

    # With --vllm-url the server is known even when `ps` can't see it, so don't ask GPU-4's questions.
    serving = collect_serving_settings(run_system_command, typer.prompt, interactive=not no_input and not vllm_url)
    vllm_sources = plan_sources(serving, vllm_url, enabled=not no_vllm_metrics)
    vllm_client = httpx.Client(timeout=VLLM_TIMEOUT_S)
    fetch = http_fetch(vllm_client)
    for source in vllm_sources:
        probe(source, fetch, lambda: datetime.now(timezone.utc))
    apply_versions(serving, vllm_sources)
    if vllm_url and not any(s.source == "vllm-process" for s in serving):
        serving.extend(settings_from_metrics(source) for source in vllm_sources if source.status == "ok")
    for s in serving:
        where = f" (pid {s.pid})" if s.pid else ""
        version = f" {s.version}" if s.version else ""
        typer.echo(f"Serving: {s.engine}{version}{where}" + (f", model {s.model}" if s.model else ""))
    for source in vllm_sources:
        if source.status == "ok":
            typer.echo(f"vLLM metrics: sampling {source.url}/metrics")
        elif source.status == "unreachable":
            typer.echo(f"Warning: vLLM metrics at {source.url}/metrics couldn't be read ({source.error}); continuing without them. "
                       "If the server listens elsewhere, pass --vllm-url.", err=True)  # fmt: skip

    if whole_seconds(interval) != interval:
        typer.echo(f"Note: CPU sampling (mpstat, pidstat) uses whole seconds, so every {whole_seconds(interval)}s.")
    typer.echo(f"Sampling every {interval:g}s for {duration:g}s ({ticks} samples). Ctrl-C stops early.")

    def progress(tick: int, total: int) -> None:
        typer.echo(f"\r  {tick}/{total}", nl=False)

    cpu_ticks = max(1, int(duration // whole_seconds(interval)))
    sampler = CpuSampler(interval, cpu_ticks)
    sampler.start()
    vllm_sampler = VllmMetricsSampler(vllm_sources, interval, fetch)
    vllm_sampler.start()
    started_at = datetime.now(timezone.utc)
    try:
        result = collect(run, interval, duration, fields=fields, on_tick=progress)
    finally:
        vllm_sampler.stop()
        vllm_client.close()
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
        vllm_metrics=vllm_sources,
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


@gpu_app.command("report")
def gpu_report_command(
    profile_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="gpu_profile.json"),
    prices_path: Path = typer.Option(
        DEFAULT_PRICES_PATH, "--prices", help="GPU price list (GPU-10); costs are skipped when it's missing."
    ),
    records_path: Path | None = typer.Option(
        None, "--records", exists=True, dir_okay=False, help="Recorder logs (v0 JSONL), for cost per million tokens."
    ),
    out: Path | None = typer.Option(None, "--out", help="Write the report JSON here instead of printing it."),
) -> None:
    """The GPU report as JSON: assessment, recommendations, costs and checklist (GPU-5 to GPU-9)."""
    try:
        profile = read_profile(profile_path)
        prices = load_prices(prices_path) if prices_path.exists() else None
        records = load_records(records_path) if records_path is not None else None
    except FileRejected as rejected:
        _reject(rejected)
    if prices is None:
        typer.echo(f"No price list at {prices_path}: costs are left out.", err=True)
    text = build_gpu_report(profile, prices, records).model_dump_json(indent=2)
    if out is None:
        typer.echo(text)
    else:
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"Wrote {out}")


@app.command("run")
def run_command(
    records: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="v0 JSONL file."),
    component: str = typer.Option(..., "--component", "-c", help="The component to evaluate."),
    tier: str = typer.Option("smoke", "--tier", help="smoke (a small sample) or full."),
    cap: float = typer.Option(..., "--cap", help="The most to spend on provider calls, in USD."),
    allow: list[str] = typer.Option(
        [], "--allow", help="A provider candidates may be called on. Repeat it. None means none allowed."
    ),
    catalog: Path = typer.Option(Path("catalog/catalog.yaml"), "--catalog", exists=True, dir_okay=False),
    cost_cap: float | None = typer.Option(None, "--cost-cap", help="Drop candidates predicted above this per month."),
    out: Path = typer.Option(Path("report.json"), "--out", "-o", help="Where to write the report."),
    speed: float = typer.Option(
        1.0, "--speed", help="1 replays at the logged pace, so logs spread over days take days. Higher compresses the gaps; latency is only comparable at 1."
    ),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before sending prompts to providers."),
) -> None:
    """Replay a component's logged calls against candidate models and write a report.

    This sends the logged prompts to every provider you allow. Nothing is sent until you
    confirm (or pass --yes). Provider API keys are read from the environment.
    """
    # Steps, in order, each one stopping with exit code 1 and a clear message on failure:
    #   1. check --tier is one of TIERS (raise typer.BadParameter otherwise)

    if tier not in TIERS:
        raise typer.BadParameter(f"must be one of: {', '.join(TIERS)}", param_hint="--tier")
    if cap <= 0:
        raise typer.BadParameter("must be more than 0", param_hint="--cap")
    if speed <= 0:
        raise typer.BadParameter("must be more than 0", param_hint="--speed")
    if not allow:  # deny-all is the default, and `run_evaluation` would refuse it anyway, after asking
        raise typer.BadParameter(
            "no provider is allowed, so no candidate could be called. Pass --allow <provider> for each one.",
            param_hint="--allow",
        )

    #   2. load the records with `load_records` (no time window); handle FileRejected with `_reject`

    try:
        loaded = load_records(records)
    except FileRejected as rejected:  # bad line or mixed schema versions
        _reject(rejected)

    component_records = records_for_component(loaded, component)
    if not component_records:
        typer.echo(f"No records for component {component!r} in {records}.", err=True)
        raise typer.Exit(code=1)

    #   3. read the keys with `_read_provider_keys(allow)`

    # The baseline replays the current model, so its provider is contacted too, allowed or not.
    incumbent = build_profile(component_records, component).incumbent
    providers = sorted(set(allow) | ({incumbent.provider} if incumbent else set()))
    provider_keys = _read_provider_keys(providers)  # before asking, so nobody approves a run that cannot start

    #   4. build a `RunConfig` from the options (`allow_list=frozenset(allow)`)

    config = RunConfig(component=component, tier=tier, spending_cap_usd=cap, allow_list=frozenset(allow), cost_cap_usd=cost_cap, replay_speed=speed)

    #   5. ask for confirmation with `_confirm_send` unless --yes; exit when the answer is no

    if not yes:
        prompts = min(len(component_records), TIER_SAMPLE_SIZE[config.tier])
        confirmed = _confirm_send(component, prompts, providers, cap)

        if not confirmed:
            typer.echo("Cancelled. Nothing was sent.", err=True)
            raise typer.Exit(code=1)

    #   6. run `_run_async(...)` with `asyncio.run`; catch RunError and print its message

    try:
        report = asyncio.run(_run_async(loaded, catalog, config, provider_keys))
    except RunError as exc:
        typer.echo(f"Run failed: {exc}", err=True)
        raise typer.Exit(code=1) from None

    #   7. `write_report(report, out)`, then print `format_run_summary(report)` and the path

    # Written last, so a failed run never leaves a report behind.
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, out)
    typer.echo(format_run_summary(report))
    typer.echo(f"Wrote {out}")


def _read_provider_keys(allowed: list[str]) -> dict[str, str]:
    """The API keys for the allowed providers, read from the environment (see `keys.py`).

    A missing key or an unsupported provider becomes a usage error that names the variable, never its value.
    """
    try:
        return read_keys(allowed)
    except MissingKeys as exc:
        raise typer.BadParameter(f"set {', '.join(exc.variables)} in the environment (the value is never printed)") from None
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--allow") from None


def _confirm_send(component: str, record_count: int, providers: list[str], cap: float) -> bool:
    """Tell the user what is about to leave their machine and ask whether to go on, e.g.
    "This will send up to 50 prompts from 'support' to openai, anthropic. Spending is capped at
    $1.00. Continue?". Return their answer; the caller exits when it is no.
    """
    noun = "prompt" if record_count == 1 else "prompts"
    return typer.confirm(
        f"This will send up to {record_count} {noun} from {component!r} to {', '.join(providers)}, "
        f"each one to several models. Spending is capped at ${cap:,.2f}. Continue?",
        default=False,
    )


async def _run_async(
    records: list[CallRecord], catalog: Path, config: RunConfig, keys: dict[str, str]
) -> Report:
    """Open one HTTP client for the whole run and call `run_evaluation` with it:
    `make_complete(client, keys, source)` is the factory it needs. Print each stage as it starts.
    Use a timeout (60 seconds is a starting point) so one stuck call cannot hang the run.
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        return await run_evaluation(
            records,
            catalog,
            config,
            lambda source: make_complete(client, keys, source),
            on_stage=lambda stage: typer.echo(f"  {stage}"),
        )


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def format_run_summary(report: Report) -> str:
    """The result in a few lines for the terminal: the noise floor, then one line per candidate
    (`provider/model`, its verdict, agreement and interval) with its explanation indented under
    it, and the dropped candidates' count. A report with no candidates says so.
    """
    floor = report.noise_floor.value
    lines = [
        f"{report.component}: the current model ({report.incumbent.provider}/{report.incumbent.model}) "
        + ("could not be measured against itself." if floor is None else f"agrees with itself {_percent(floor)} of the time (the noise floor).")
    ]

    if not report.candidates:
        lines.append("No candidates were replayed.")
    for c in report.candidates:
        a = c.agreement
        verdict = c.verdict.root.replace("_", " ").capitalize()
        lines.append(
            f"{c.provider}/{c.model}: {verdict}. Agreement {_percent(a.rate)} "
            f"({_percent(a.ci_low)} to {_percent(a.ci_high)}) over {a.n} requests."
        )
        lines.append(f"    {c.explanation}")

    if report.errors:
        lines.append("Calls failed:")
        lines.extend(f"  {e.where}: {e.count} x {e.message}" for e in report.errors)

    dropped = len(report.dropped_candidates)
    if dropped:
        lines.append(f"{dropped} {'model was' if dropped == 1 else 'models were'} not tested; the report says why.")
    return "\n".join(lines)


@app.command("serve")
def serve_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Address to listen on. Keep it on this machine."),
    port: int = typer.Option(8000, "--port"),
    data_dir: Path = typer.Option(
        Path(".migration-eval-data"), "--data-dir", help="Where uploads, runs and reports are kept (owner-only)."
    ),
    catalog: Path = typer.Option(Path("catalog/catalog.yaml"), "--catalog", exists=True, dir_okay=False),
    allow_remote: bool = typer.Option(False, "--allow-remote", help="Listen on a non-local address anyway."),
    executor: str = typer.Option("local", "--executor", help="local (runs in this process) or batch (AWS Batch jobs, see infra/terraform)."),
    partner: str = typer.Option("test", "--partner", help="With --executor batch: the partner whose job definition, bucket prefix and key to use."),
    queue: str = typer.Option("mme-runs", "--queue", help="With --executor batch: the job queue."),
    job_definition: str | None = typer.Option(None, "--job-definition", help="With --executor batch: default mme-worker-<partner>."),
    region: str | None = typer.Option(None, "--region", help="With --executor batch: AWS region (default: your AWS config)."),
    run_timeout: int = typer.Option(3600, "--run-timeout", help="With --executor batch: seconds a run may take before Batch stops it (min 60)."),
) -> None:
    """Start the local server and UI: upload an export, run an evaluation, read the report.

    Provider API keys are read from the environment, never from the UI. The server has no login,
    so it refuses any address but this machine's unless you pass --allow-remote.

    With --executor batch, each run is an AWS Batch job on the infrastructure in infra/terraform:
    the records go to the partner's S3 prefix under its KMS key, keys come from the partner's
    Secrets Manager secret, and the server needs AWS credentials with the orchestrator policy.
    """
    if executor not in ("local", "batch"):
        raise typer.BadParameter("must be local or batch", param_hint="--executor")
    if run_timeout < 60:
        raise typer.BadParameter("must be at least 60 (the Batch minimum)", param_hint="--run-timeout")
    import ipaddress

    import uvicorn

    from migration_eval.keys import key_status
    from migration_eval.server.app import create_app

    try:
        is_local = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:  # a name other than localhost, or not an address
        is_local = False
    if not is_local:
        if not allow_remote:
            typer.echo(f"Refusing to listen on {host!r}: the server has no login. Use --allow-remote to do it anyway.", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Warning: {host!r} is reachable from other machines and the server has no login.", err=True)

    batch = None
    if executor == "batch":
        from migration_eval.server.batch import BatchRuns, BatchUnavailable

        try:
            batch = BatchRuns.connect(partner, queue, job_definition, region, run_timeout)
        except BatchUnavailable as exc:
            typer.echo(f"Cannot use AWS Batch: {exc}", err=True)
            raise typer.Exit(code=1) from None
        s = batch.settings
        typer.echo(f"Runs go to AWS Batch: queue {s.queue}, job definition {s.job_definition}, results in s3://{s.bucket}/{s.runs_prefix}/")

    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    app_ = create_app(data_dir, catalog, frontend_dist=dist if dist.exists() else None, batch=batch)

    if batch is not None:
        typer.echo(f"Provider keys come from the Secrets Manager secret {batch.settings.keys_secret_id or '(none set in the job definition)'}")
    else:
        configured = [name for name, ok in key_status().items() if ok]
        typer.echo(f"Keys set for: {', '.join(configured) if configured else 'no provider (set OPENAI_API_KEY, ANTHROPIC_API_KEY or GEMINI_API_KEY)'}")
    typer.echo(f"Data is kept in {data_dir.resolve()}")
    typer.echo(f"Open http://{host}:{port}/?live=1" if dist.exists() else f"API at http://{host}:{port}/api/docs (build the frontend to get the UI: cd frontend && npm run build)")
    uvicorn.run(app_, host=host, port=port, log_level="warning")


def _reject(rejected: FileRejected) -> NoReturn:
    typer.echo(f"Rejected: {rejected}", err=True)
    typer.echo("Nothing was written.", err=True)
    raise typer.Exit(code=1)


def _echo_skipped(result: ImportResult) -> None:
    for number, reason in result.skipped[:MAX_REASONS_SHOWN]:
        typer.echo(f"  line {number}: {reason}", err=True)
    if len(result.skipped) > MAX_REASONS_SHOWN:
        typer.echo(f"  ... and {len(result.skipped) - MAX_REASONS_SHOWN} more", err=True)
