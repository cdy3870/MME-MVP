"""Running evaluations in the background and tracking them (RUN-1, RUN-4, RUN-6).

`RunManager.start` checks the request, saves a `queued` run and starts `_execute` as an asyncio
task; the task calls `run.run_evaluation` and records each stage as it begins. The browser polls
`status`. Cancelling sets a flag that `run_evaluation` reads through its `should_cancel` hook.

Checks `start` makes before anything is sent, each raising `RequestRejected` with a message the
user can act on (the same rules as the CLI's `run` command):
    * `request.confirmed` is true
    * the upload and the component exist, and the component has replay-grade records
    * every provider in `allow` is one we support, and the allowed providers PLUS the current
      model's provider all have a key (`keys.read_keys`); the baseline replays the current model
    * no other run is running (one at a time: runs share the providers' rate limits and one spend
      cap each; an open decision)

Open decisions (see the notes on RUN-4 and RUN-5):
    * per-candidate progress (how many of its requests have run) needs an `on_progress` hook on
      `run_evaluation`; the stage names and the live spend are what the UI gets today
    * `cancel` today only stops new calls and leaves the in-flight ones to finish; RUN-6 wants the
      run halted within 30 seconds
    * a restart loses running tasks, so state saved as `running` or `queued` is stale after one
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from migration_eval.keys import KEY_VARIABLES, MissingKeys, key_status, read_keys
from migration_eval.models import Report
from migration_eval.request_builders import make_complete
from migration_eval.run import CompleteFactory, RunConfig, RunError, SpendMeter, run_evaluation
from migration_eval.server.models import ProviderStatus, RunRequest, RunStatus, RunSummary
from migration_eval.server.store import NotFound, Store

log = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 60  # one stuck call must not hang the run
ACTIVE_STATES = ("queued", "running", "cancelling")

# Builds the `CompleteFactory` for one run from the HTTP client and the keys that run may use.
# The real one wraps `request_builders.make_complete`; tests pass one that returns a fake model.
FactoryBuilder = Callable[[httpx.AsyncClient, dict[str, str]], CompleteFactory]


def real_factory_builder(client: httpx.AsyncClient, keys: dict[str, str]) -> CompleteFactory:
    return lambda source_provider: make_complete(client, keys, source_provider)


class RequestRejected(Exception):
    """The request cannot be accepted. `status_code` is the HTTP status to answer with: 400 for a
    bad request, 404 for an unknown upload or run, 409 for a conflict such as another run in progress."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class _Live:
    """What exists only while a run is going: its task, its spend meter and the stop flag."""

    meter: SpendMeter
    upload_id: str
    task: asyncio.Task[None] | None = None
    cancel_requested: bool = False
    stages_done: list[str] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunManager:
    def __init__(
        self,
        store: Store,
        catalog_path: Path | str,
        factory_builder: FactoryBuilder = real_factory_builder,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """`environ` defaults to `os.environ` and is where the keys are read from."""
        self._store = store
        self._catalog = Path(catalog_path)
        self._factory_builder = factory_builder
        self._environ = os.environ if environ is None else environ
        self._live: dict[str, _Live] = {}
        self._mark_stale_runs_failed()

    def _mark_stale_runs_failed(self) -> None:
        """A run saved as active has no process behind it after a restart, so it cannot finish."""
        for run in self._store.list_runs():
            if run.state in ACTIVE_STATES:
                self._store.save_run(
                    run.model_copy(update={"state": "failed", "finished_at": _now(), "error": "The server stopped while this run was in progress."})
                )

    def providers(self) -> list[ProviderStatus]:
        """Every supported provider and whether its key is set (`keys.key_status`); no values."""
        configured = key_status(self._environ)
        return [ProviderStatus(name=name, key_variable=variable, key_configured=configured[name]) for name, variable in KEY_VARIABLES.items()]

    async def start(self, request: RunRequest) -> RunStatus:
        """Check the request (see the module docstring), save a `queued` run and start it."""
        if not request.confirmed:
            raise RequestRejected("The run was not confirmed. Confirm what will be sent, and to which providers, to start it.")

        try:
            upload = self._store.get_upload(request.upload_id)
        except NotFound:
            raise RequestRejected("That upload does not exist (it may have been deleted).", 404) from None

        component = next((c for c in upload.components if c.name == request.component), None)
        if component is None:
            raise RequestRejected(f"The upload has no component {request.component!r}.")
        if component.replay_grade == 0:
            raise RequestRejected(f"No record of component {request.component!r} can be replayed.")

        # The baseline replays the current model, so its provider is contacted too, allowed or not.
        providers = list(dict.fromkeys(request.allow))
        contacted = sorted(set(providers) | ({component.incumbent_provider} if component.incumbent_provider else set()))
        try:
            keys = read_keys(contacted, self._environ)
        except MissingKeys as exc:
            raise RequestRejected(f"Set {', '.join(exc.variables)} in the environment where the server runs (the value is never shown).") from None
        except ValueError as exc:
            raise RequestRejected(str(exc)) from None

        if any(live.task is not None and not live.task.done() for live in self._live.values()):
            raise RequestRejected("Another run is in progress. Wait for it to finish, or stop it.", 409)

        run_id = uuid.uuid4().hex
        status = RunStatus(
            id=run_id, upload_id=request.upload_id, component=request.component, state="queued",
            started_at=_now(), spending_cap_usd=request.spending_cap_usd,
        )
        self._store.save_run(status)
        live = _Live(meter=SpendMeter(request.spending_cap_usd), upload_id=request.upload_id)
        self._live[run_id] = live
        live.task = asyncio.get_running_loop().create_task(self._execute(run_id, request.model_copy(update={"allow": providers}), keys))
        return status

    def _save(self, run_id: str, **changes: object) -> RunStatus:
        status = self._store.get_run(run_id).model_copy(update=changes)
        self._store.save_run(status)
        return status

    async def _execute(self, run_id: str, request: RunRequest, keys: dict[str, str]) -> None:
        """The background task. Open one `httpx.AsyncClient` (60 second timeout), build the config
        (`run.RunConfig`) and call `run_evaluation` with `on_stage` updating the saved status and
        `should_cancel` reading the cancel flag. On success save the report and mark the run `done`
        (or `cancelled` if it was). On `run.RunError` mark it `failed` with that message; on any
        other exception mark it `failed` with a generic message and log the details on the server,
        so an internal error never reaches the browser.
        """
        live = self._live[run_id]

        def on_stage(stage: str) -> None:
            previous = self._store.get_run(run_id).stage
            if previous is not None and previous not in live.stages_done:
                live.stages_done.append(previous)
            state = "cancelling" if live.cancel_requested else "running"
            self._save(run_id, state=state, stage=stage, stages_done=list(live.stages_done), spent_usd=live.meter.spent_usd)

        try:
            records = self._store.load_records(request.upload_id)
            config = RunConfig(
                component=request.component,
                tier=request.tier,
                spending_cap_usd=request.spending_cap_usd,
                allow_list=frozenset(request.allow),
                cost_cap_usd=request.cost_cap_usd,
                replay_speed=request.replay_speed,
            )
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                report = await run_evaluation(
                    records,
                    self._catalog,
                    config,
                    self._factory_builder(client, keys),
                    on_stage=on_stage,
                    should_cancel=lambda: live.cancel_requested,
                    meter=live.meter,
                )
            self._store.save_report(run_id, report)
            last = self._store.get_run(run_id).stage
            done = [*live.stages_done, last] if last and last not in live.stages_done else list(live.stages_done)
            self._save(
                run_id, state="cancelled" if live.cancel_requested else "done", stages_done=done, stage=None,
                spent_usd=live.meter.spent_usd, finished_at=_now(), has_report=True,
            )  # fmt: skip
        except RunError as exc:
            self._save(run_id, state="failed", error=str(exc), spent_usd=live.meter.spent_usd, finished_at=_now())
        except asyncio.CancelledError:
            self._save(run_id, state="failed", error="The server stopped while this run was in progress.", finished_at=_now())
            raise
        except Exception:
            log.exception("run %s failed", run_id)  # the details stay in the server log
            self._save(run_id, state="failed", error="Something went wrong while running. See the server log for details.", spent_usd=live.meter.spent_usd, finished_at=_now())
        finally:
            self._live.pop(run_id, None)

    def status(self, run_id: str) -> RunStatus:
        """Raise `RequestRejected` (404) for an unknown run."""
        try:
            status = self._store.get_run(run_id)
        except NotFound:
            raise RequestRejected("That run does not exist.", 404) from None
        live = self._live.get(run_id)
        if live is not None:  # the spend is read live; it is only written to disk at stage changes
            status = status.model_copy(update={"spent_usd": live.meter.spent_usd})
        return status

    def cancel(self, run_id: str) -> RunStatus:
        """Ask a running run to stop (state `cancelling`, then `cancelled` once it has). Cancelling a
        finished run is a 409."""
        status = self.status(run_id)
        live = self._live.get(run_id)
        if live is None or status.state not in ACTIVE_STATES:
            raise RequestRejected("That run has already finished.", 409)
        live.cancel_requested = True
        return self._save(run_id, state="cancelling")

    def summaries(self) -> list[RunSummary]:
        """Past runs, newest first, with the verdict counts for finished ones."""
        rows: list[RunSummary] = []
        for run in self._store.list_runs():
            verdicts: dict[str, int] | None = None
            if run.has_report:
                try:
                    counts: dict[str, int] = {}
                    for candidate in self._store.get_report(run.id).candidates:
                        counts[candidate.verdict.root] = counts.get(candidate.verdict.root, 0) + 1
                    verdicts = counts
                except (NotFound, ValueError):
                    verdicts = None
            rows.append(RunSummary(id=run.id, component=run.component, state=run.state, started_at=run.started_at, verdicts=verdicts))
        return rows

    def report(self, run_id: str) -> Report:
        """The finished report. Raise `RequestRejected` (404) while there is none."""
        try:
            return self._store.get_report(run_id)
        except NotFound:
            raise RequestRejected("That run has no report (yet).", 404) from None

    def delete_run(self, run_id: str) -> None:
        """Remove a finished run and its report. A run still going must be stopped first (409)."""
        if self.status(run_id).state in ACTIVE_STATES:
            raise RequestRejected("Stop the run before deleting it.", 409)
        self._store.delete_run(run_id)

    def delete_upload(self, upload_id: str) -> None:
        """Remove an upload, unless a run that is going is using it (409)."""
        if any(live.upload_id == upload_id for live in self._live.values()):
            raise RequestRejected("A run is using this upload. Stop it first.", 409)
        try:
            self._store.delete_upload(upload_id)
        except NotFound:
            raise RequestRejected("That upload does not exist.", 404) from None

    def shutdown(self) -> None:
        """Cancel running tasks when the server stops."""
        for live in list(self._live.values()):
            if live.task is not None and not live.task.done():
                live.task.cancel()
