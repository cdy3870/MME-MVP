"""The `migration-eval` command line."""

from __future__ import annotations

import asyncio
import os
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

MAX_REASONS_SHOWN = 10

# The environment variable that holds each provider's API key. Keys are read from here and
# never from a file, and are never printed.
KEY_VARIABLES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
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

    config = RunConfig(component=component, tier=tier, spending_cap_usd=cap, allow_list=frozenset(allow), cost_cap_usd=cost_cap)

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
    """The API keys for the allowed providers, read from `KEY_VARIABLES`' environment variables.

    Raise typer.BadParameter naming the variable (never its value) when an allowed provider has no
    key, or when a provider is not one we have a builder for. Return only the allowed providers'
    keys, so a key for a provider that is not allowed is never even loaded into the run.
    """

    unknown = [p for p in allowed if p not in KEY_VARIABLES]
    if unknown:
        raise typer.BadParameter(
            f"no support for provider {', '.join(map(repr, unknown))} (supported: {', '.join(sorted(KEY_VARIABLES))})",
            param_hint="--allow",
        )

    provider_keys = {}
    missing = []
    for provider in allowed:
        value = os.environ.get(KEY_VARIABLES[provider], "").strip()
        if value:
            provider_keys[provider] = value
        else:
            missing.append(KEY_VARIABLES[provider])
    if missing:
        raise typer.BadParameter(f"set {', '.join(missing)} in the environment (the value is never printed)")

    return provider_keys


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

    dropped = len(report.dropped_candidates)
    if dropped:
        lines.append(f"{dropped} {'model was' if dropped == 1 else 'models were'} not tested; the report says why.")
    return "\n".join(lines)


def _reject(rejected: FileRejected) -> NoReturn:
    typer.echo(f"Rejected: {rejected}", err=True)
    typer.echo("Nothing was written.", err=True)
    raise typer.Exit(code=1)


def _echo_skipped(result: ImportResult) -> None:
    for number, reason in result.skipped[:MAX_REASONS_SHOWN]:
        typer.echo(f"  line {number}: {reason}", err=True)
    if len(result.skipped) > MAX_REASONS_SHOWN:
        typer.echo(f"  ... and {len(result.skipped) - MAX_REASONS_SHOWN} more", err=True)
