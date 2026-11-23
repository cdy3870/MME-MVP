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

On AWS Batch (MOD-135; `RunManager(..., batch=BatchRuns(...))`, `serve --executor batch`):
    * `start` makes the same checks, except that keys live in the partner's Secrets Manager secret,
      which the server can't read: it checks the secret's `mme:key-variables` tag when there is
      one, and otherwise leaves the check to the worker. Then it puts the upload's records in S3
      and submits a job (`server/batch.py`); the run is saved with its `batch_job_id`.
    * `status` refreshes an active run from AWS (throttled): Batch's job state, the worker's
      progress.json while it runs, then status.json and report.json once it has ended. Batch's
      waiting and starting states show as the stages "waiting for a machine" and "starting the
      worker", ahead of the run's own stages.
    * `cancel` calls TerminateJob; the worker stops starting candidates and writes its report.
    * a server restart loses nothing: the job keeps running in AWS and the next `status` picks
      it up again. Only runs without a job id are marked failed at startup.

Open decisions (see the notes on RUN-4 and RUN-5):
    * progress is the stage, the spend, and the requests done of the stage's total
      (`run_evaluation(on_progress=...)`); there is no time estimate for the stage yet
    * `cancel` today only stops new calls and leaves the in-flight ones to finish; RUN-6 wants the
      run halted within 30 seconds
    * a restart loses running tasks, so state saved as `running` or `queued` is stale after one
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
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
from migration_eval.server.batch import BATCH_WAITING, STARTING_STAGE, WAITING_STAGE, BatchRuns, BatchUnavailable, JobView, Throttle
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
    calls_done: int | None = None  # the current stage's replayed requests, read live like the spend
    calls_total: int | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _secret_name(secret_id: str | None) -> str | None:
    """`mme/keys/test` from its ARN (…:secret:mme/keys/test-AbCdEf), for messages."""
    if not secret_id:
        return None
    name = secret_id.split(":secret:", 1)[-1]
    return name[:-7] if ":secret:" in secret_id and re.fullmatch(r".+-[A-Za-z0-9]{6}", name) else name


class RunManager:
    def __init__(
        self,
        store: Store,
        catalog_path: Path | str,
        factory_builder: FactoryBuilder = real_factory_builder,
        environ: Mapping[str, str] | None = None,
        batch: BatchRuns | None = None,
    ) -> None:
        """`environ` defaults to `os.environ` and is where the keys are read from. With `batch`,
        runs execute as AWS Batch jobs instead of in this process (see the module docstring)."""
        self._store = store
        self._catalog = Path(catalog_path)
        self._factory_builder = factory_builder
        self._environ = os.environ if environ is None else environ
        self._live: dict[str, _Live] = {}
        self._batch = batch
        self._throttle = Throttle(batch.min_poll_seconds if batch is not None else 0.0)
        self._refresh_lock = threading.Lock()  # status() runs in FastAPI's thread pool
        self._mark_stale_runs_failed()

    def _mark_stale_runs_failed(self) -> None:
        """A local run saved as active has no process behind it after a restart, so it cannot
        finish. A Batch run keeps going in AWS and is left for `status` to pick up again."""
        for run in self._store.list_runs():
            if run.state in ACTIVE_STATES and run.batch_job_id is None:
                self._store.save_run(
                    run.model_copy(update={"state": "failed", "finished_at": _now(), "error": "The server stopped while this run was in progress."})
                )

    def providers(self) -> list[ProviderStatus]:
        """Every supported provider and whether its key is set (`keys.key_status`); no values.
        On Batch: from the secret's `mme:key-variables` tag, or all assumed set when it has none."""
        if self._batch is not None:
            known = self._batch.key_variables()
            secret = _secret_name(self._batch.settings.keys_secret_id)
            return [
                ProviderStatus(
                    name=name, key_variable=variable, key_configured=known is None or variable in known,
                    key_source="secret", key_secret=secret,
                )
                for name, variable in KEY_VARIABLES.items()
            ]  # fmt: skip
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
        if self._batch is not None:
            return await self._start_batch(request, providers, contacted)
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

    async def _start_batch(self, request: RunRequest, providers: list[str], contacted: list[str]) -> RunStatus:
        """Check the keys against the secret's tag (when it has one), put the records in S3 and
        submit the job. An AWS failure leaves a failed run behind and answers 502 with the reason."""
        assert self._batch is not None
        unsupported = [p for p in contacted if p not in KEY_VARIABLES]
        if unsupported:
            raise RequestRejected(f"Unsupported provider: {', '.join(unsupported)}. Supported: {', '.join(KEY_VARIABLES)}.")
        known = await asyncio.to_thread(self._batch.key_variables)
        if known is not None:
            missing = [KEY_VARIABLES[p] for p in contacted if KEY_VARIABLES[p] not in known]
            if missing:
                secret = _secret_name(self._batch.settings.keys_secret_id)
                raise RequestRejected(f"Add {', '.join(missing)} to the {secret} secret (the value is never shown).")

        # Bring active Batch runs up to date first, so one that ended in AWS doesn't block this one.
        for run in self._store.list_runs():
            if run.state in ACTIVE_STATES and run.batch_job_id:
                await asyncio.to_thread(self._refresh, run, True)
        if any(run.state in ACTIVE_STATES for run in self._store.list_runs()):
            raise RequestRejected("Another run is in progress. Wait for it to finish, or stop it.", 409)

        run_id = uuid.uuid4().hex
        self._store.save_run(
            RunStatus(
                id=run_id, upload_id=request.upload_id, component=request.component, state="queued", stage=WAITING_STAGE,
                started_at=_now(), spending_cap_usd=request.spending_cap_usd,
            )
        )  # fmt: skip
        try:
            records = await asyncio.to_thread(self._store.read_records_bytes, request.upload_id)
            records_uri = await asyncio.to_thread(self._batch.ensure_upload, request.upload_id, records)
            job_id = await asyncio.to_thread(
                lambda: self._batch.submit(  # type: ignore[union-attr]
                    run_id, records_uri=records_uri, component=request.component, allow=providers, tier=request.tier,
                    spending_cap_usd=request.spending_cap_usd, replay_speed=request.replay_speed, cost_cap_usd=request.cost_cap_usd,
                )  # fmt: skip
            )
        except BatchUnavailable as exc:
            self._save(run_id, state="failed", stage=None, error=str(exc), finished_at=_now())
            raise RequestRejected(str(exc), 502) from None
        return self._save(run_id, batch_job_id=job_id)

    def _refresh(self, status: RunStatus, force: bool = False) -> RunStatus:
        """Bring an active Batch run up to date from AWS, at most every `min_poll_seconds` unless
        forced. An AWS error is logged and the last saved state is returned; the UI keeps polling."""
        if self._batch is None or status.batch_job_id is None or status.state not in ACTIVE_STATES:
            return status
        if not self._throttle.due(status.id) and not force:
            return status
        with self._refresh_lock:
            current = self._store.get_run(status.id)  # another thread may have finished it meanwhile
            if current.state not in ACTIVE_STATES or current.batch_job_id is None:
                return current
            try:
                view = self._batch.poll(current.batch_job_id, current.id)
                changes = self._changes_from(current, view)
            except BatchUnavailable as exc:
                log.warning("could not refresh run %s: %s", current.id, exc)
                return current
            return self._save(current.id, **changes)

    def _changes_from(self, current: RunStatus, view: JobView) -> dict[str, object]:
        """The new state of `current` given what the poll saw (see the module docstring)."""
        assert self._batch is not None
        cancelling = current.state == "cancelling"
        machine = [WAITING_STAGE, STARTING_STAGE]
        progress = view.progress or {}
        no_calls = {"calls_done": None, "calls_total": None}
        if view.batch_status in BATCH_WAITING:
            return {"state": "cancelling" if cancelling else "queued", "stage": WAITING_STAGE, "stages_done": []} | no_calls
        if view.batch_status == "STARTING":
            return {"state": "cancelling" if cancelling else "running", "stage": STARTING_STAGE, "stages_done": [WAITING_STAGE]} | no_calls
        if view.batch_status == "RUNNING":
            done, total = progress.get("calls_done"), progress.get("calls_total")
            counted = isinstance(done, int) and isinstance(total, int)
            changes: dict[str, object] = {
                "state": "cancelling" if cancelling else "running",
                "calls_done": done if counted else None,
                "calls_total": total if counted else None,
            }
            if progress.get("stage"):
                changes |= {"stage": progress["stage"], "stages_done": machine + list(progress.get("stages_done") or [])}
            else:  # the container is up but the worker hasn't announced a stage yet
                changes |= {"stage": STARTING_STAGE, "stages_done": [WAITING_STAGE]}
            if isinstance(progress.get("spent_usd"), (int, float)):
                changes["spent_usd"] = float(progress["spent_usd"])
            return changes

        # Ended: SUCCEEDED, FAILED (which includes a stop), or UNKNOWN (Batch forgot the job).
        worker = view.worker_status or {}
        spent = worker.get("spent_usd", progress.get("spent_usd", current.spent_usd))
        base: dict[str, object] = {
            "stage": None,
            "stages_done": machine + list(worker.get("stages_done") or progress.get("stages_done") or []),
            "spent_usd": float(spent) if isinstance(spent, (int, float)) else None,
            "finished_at": _now(),
            **no_calls,
        }
        report = None
        if worker.get("has_report") or view.batch_status == "SUCCEEDED":
            try:
                report = self._batch.fetch_report(current.id)
            except ValueError:  # not a valid report
                log.exception("run %s: unreadable report.json", current.id)
                return base | {"state": "failed", "error": "The worker's report could not be read. See the server log."}
        if report is not None:
            self._store.save_report(current.id, report)
        stopped = cancelling or worker.get("status") == "stopped"
        if report is not None:
            return base | {"state": "cancelled" if stopped else "done", "has_report": True}
        if stopped:
            return base | {"state": "cancelled"}
        return base | {"state": "failed", "error": self._failure_message(view)}

    def _failure_message(self, view: JobView) -> str:
        assert self._batch is not None
        worker = view.worker_status or {}
        where = self._batch.log_location(view)
        log_hint = f" The worker's log is in CloudWatch: {where}." if where else ""
        if worker.get("error_kind") in ("setup", "run") and worker.get("error"):
            return f"The run failed: {worker['error']}."
        if worker:
            return "Something went wrong in the worker." + log_hint
        if view.batch_status == "UNKNOWN":
            return "AWS Batch no longer has this job, and the worker left no status."
        return f"The worker did not finish: {view.reason or 'AWS Batch gave no reason'}." + log_hint

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
            live.calls_done = live.calls_total = None  # a stage that replays sets them again
            state = "cancelling" if live.cancel_requested else "running"
            self._save(run_id, state=state, stage=stage, stages_done=list(live.stages_done), spent_usd=live.meter.spent_usd)

        def on_progress(done: int, total: int) -> None:
            live.calls_done, live.calls_total = done, total

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
                    on_progress=on_progress,
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
        if status.batch_job_id is not None:
            return self._refresh(status)
        live = self._live.get(run_id)
        if live is not None:  # the spend and calls are read live; only stage changes are written to disk
            status = status.model_copy(update={"spent_usd": live.meter.spent_usd, "calls_done": live.calls_done, "calls_total": live.calls_total})
        return status

    def cancel(self, run_id: str) -> RunStatus:
        """Ask a running run to stop (state `cancelling`, then `cancelled` once it has). Cancelling a
        finished run is a 409."""
        status = self.status(run_id)
        if status.batch_job_id is not None:
            if status.state not in ACTIVE_STATES:
                raise RequestRejected("That run has already finished.", 409)
            if status.state == "cancelling":
                return status
            if self._batch is None:
                raise RequestRejected("This run is on AWS Batch; restart the server with --executor batch to stop it.", 409)
            try:
                self._batch.terminate(status.batch_job_id)
            except BatchUnavailable as exc:
                raise RequestRejected(str(exc), 502) from None
            return self._save(run_id, state="cancelling")
        live = self._live.get(run_id)
        if live is None or status.state not in ACTIVE_STATES:
            raise RequestRejected("That run has already finished.", 409)
        live.cancel_requested = True
        return self._save(run_id, state="cancelling")

    def summaries(self) -> list[RunSummary]:
        """Past runs, newest first, with the verdict counts for finished ones."""
        for run in self._store.list_runs():  # active Batch runs are only updated when someone looks
            if run.state in ACTIVE_STATES and run.batch_job_id is not None:
                self._refresh(run)
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
        status = self.status(run_id)
        if status.state in ACTIVE_STATES:
            raise RequestRejected("Stop the run before deleting it.", 409)
        if status.batch_job_id is not None and self._batch is not None:
            try:  # best effort: S3 lifecycle rules expire partner data anyway
                self._batch.delete_run_outputs(run_id)
            except BatchUnavailable as exc:
                log.warning("run %s: S3 files not deleted: %s", run_id, exc)
        self._store.delete_run(run_id)

    def delete_upload(self, upload_id: str) -> None:
        """Remove an upload, unless a run that is going is using it (409)."""
        if any(live.upload_id == upload_id for live in self._live.values()) or any(
            run.upload_id == upload_id and run.state in ACTIVE_STATES and run.batch_job_id for run in self._store.list_runs()
        ):
            raise RequestRejected("A run is using this upload. Stop it first.", 409)
        try:
            self._store.delete_upload(upload_id)
        except NotFound:
            raise RequestRejected("That upload does not exist.", 404) from None
        if self._batch is not None:
            try:
                self._batch.delete_upload(upload_id)
            except BatchUnavailable as exc:
                log.warning("upload %s: S3 copy not deleted: %s", upload_id, exc)

    def shutdown(self) -> None:
        """Cancel running tasks when the server stops."""
        for live in list(self._live.values()):
            if live.task is not None and not live.task.done():
                live.task.cancel()
