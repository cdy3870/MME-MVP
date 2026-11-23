"""Entrypoint for the run worker image: one evaluation run per container (MOD-135, RUN-3).

    1. read the provider keys (KEYS_SECRET_ID, or already in the environment)
    2. download the records (v0 JSONL) from RECORDS_URI into WORK_DIR
    3. call `run.run_evaluation` for COMPONENT, writing progress.json (stage, finished stages,
       spend so far, requests done of the stage's total) for the server to show: at every stage,
       as requests finish (at most every PROGRESS_MIN_SECONDS) and every PROGRESS_EVERY_SECONDS
    4. upload report.json and status.json to s3://RESULT_BUCKET/RESULT_PREFIX/RUN_ID/
    5. delete WORK_DIR, so no partner data is left in the container (RUN-7)

Environment:
    RUN_ID, RECORDS_URI (s3://bucket/key), COMPONENT, CAP_USD, ALLOW (comma-separated providers),
    RESULT_BUCKET, RESULT_PREFIX; optional TIER (default smoke), SPEED (default 1), COST_CAP_USD,
    CATALOG (default catalog/catalog.yaml), WORK_DIR (default /work).

    Provider keys (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY) come from one of:
      * KEYS_SECRET_ID: a Secrets Manager secret holding a JSON object of those variables, read
        with the job role (the Terraform setup, ACC-2)
      * already in the environment, injected by the job definition's `secrets`
    Never as plain SubmitJob overrides, which show up in DescribeJobs.

    RESULT_KMS_KEY_ID, when set, encrypts everything the worker writes with that KMS key (the
    partner's; the Terraform bucket policy refuses partner-prefix writes without it).

Stopping: SIGTERM (TerminateJob, a timeout) sets the stop flag that `run_evaluation` checks, so no
new candidate starts and the report is still written, with unfinished candidates marked
"insufficient data" (RUN-6). ECS kills the container 30 seconds after SIGTERM, so a candidate
with many calls in flight may not finish in time.

status.json: status (completed | stopped | failed), has_report, spent_usd, and on failure
`error` with `error_kind`: "setup" or "run" (a message the user can act on) or "internal"
(details only in the worker log).

Exit code: 0 completed or stopped, 1 the run failed (setup or run error), 2 an internal error.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import httpx

from migration_eval.capture.file_checks import FileRejected
from migration_eval.export import load_records
from migration_eval.keys import KEY_VARIABLES, MissingKeys, read_keys
from migration_eval.profile import build_profile, records_for_component
from migration_eval.request_builders import make_complete
from migration_eval.run import CompleteFactory, RunConfig, RunError, SpendMeter, run_evaluation, write_report

REQUIRED = ("RUN_ID", "RECORDS_URI", "COMPONENT", "CAP_USD", "ALLOW", "RESULT_BUCKET", "RESULT_PREFIX")
REQUEST_TIMEOUT_SECONDS = 60  # one stuck call must not hang the run
PROGRESS_EVERY_SECONDS = 10  # spend changes between stages; refresh progress.json this often
PROGRESS_MIN_SECONDS = 2.0  # request counts change constantly; write them at most this often

FactoryBuilder = Callable[[httpx.AsyncClient, dict[str, str]], CompleteFactory]


def real_factory_builder(client: httpx.AsyncClient, keys: dict[str, str]) -> CompleteFactory:
    return lambda source_provider: make_complete(client, keys, source_provider)


class SetupError(Exception):
    """Something the user can fix before re-running (keys, records, settings)."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise SetupError(f"RECORDS_URI is not an s3:// URI: {uri}")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not bucket or not key:
        raise SetupError(f"RECORDS_URI should be s3://bucket/key, got {uri}")
    return bucket, key


def load_provider_keys(secrets: Any, secret_id: str) -> dict[str, str]:
    """The provider key variables in the secret. Errors never include the secret's value."""
    try:
        value = secrets.get_secret_value(SecretId=secret_id)["SecretString"]
    except Exception as exc:
        raise SetupError(f"could not read the provider-key secret {secret_id}: {type(exc).__name__}") from None
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        raise SetupError(f"the provider-key secret {secret_id} is not a JSON object like {{\"ANTHROPIC_API_KEY\": \"...\"}}")
    variables = set(KEY_VARIABLES.values())
    return {name: v for name, v in data.items() if name in variables and isinstance(v, str) and v.strip()}


class Outputs:
    """Writes the run's files under RESULT_PREFIX/RUN_ID/, encrypted with the partner's key."""

    def __init__(self, s3: Any, env: Mapping[str, str]) -> None:
        self._s3 = s3
        self.bucket = env["RESULT_BUCKET"]
        self.prefix = f"{env['RESULT_PREFIX'].strip('/')}/{env['RUN_ID']}"
        kms_key = env.get("RESULT_KMS_KEY_ID")
        self._sse = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key} if kms_key else {}

    def put_json(self, name: str, data: dict[str, Any]) -> None:
        self._s3.put_object(
            Bucket=self.bucket, Key=f"{self.prefix}/{name}", Body=json.dumps(data, indent=2).encode(),
            ContentType="application/json", **self._sse,
        )  # fmt: skip

    def upload(self, name: str, path: Path) -> str:
        self._s3.upload_file(str(path), self.bucket, f"{self.prefix}/{name}", ExtraArgs=self._sse or None)
        return f"s3://{self.bucket}/{self.prefix}/{name}"


class Progress:
    """progress.json: the current stage, the finished ones and the spend. A failed write is logged
    and skipped; progress must never stop the run."""

    def __init__(self, outputs: Outputs, meter: SpendMeter, clock: Callable[[], float] = time.monotonic) -> None:
        self._outputs = outputs
        self._meter = meter
        self._clock = clock
        self._lock = threading.Lock()
        self._last_write = float("-inf")
        self.stage: str | None = None
        self.stages_done: list[str] = []
        self.calls_done: int | None = None
        self.calls_total: int | None = None

    def on_stage(self, stage: str) -> None:
        if self.stage is not None and self.stage not in self.stages_done:
            self.stages_done.append(self.stage)
        self.stage = stage
        self.calls_done = self.calls_total = None  # a stage that replays sets them again
        print(f"worker: stage {stage}", flush=True)
        self.write()

    def on_progress(self, done: int, total: int) -> None:
        """Requests finished in this stage. Written when the stage starts or ends, and otherwise
        at most every PROGRESS_MIN_SECONDS."""
        self.calls_done, self.calls_total = done, total
        if done in (0, total) or self._clock() - self._last_write >= PROGRESS_MIN_SECONDS:
            self.write()

    def write(self) -> None:
        data = {
            "stage": self.stage, "stages_done": list(self.stages_done), "spent_usd": self._meter.spent_usd,
            "calls_done": self.calls_done, "calls_total": self.calls_total, "updated_at": now(),
        }  # fmt: skip
        self._last_write = self._clock()
        with self._lock:
            try:
                self._outputs.put_json("progress.json", data)
            except Exception as exc:
                print(f"worker: could not write progress.json: {type(exc).__name__}: {exc}", flush=True)

    def finished_stages(self) -> list[str]:
        last = [self.stage] if self.stage is not None and self.stage not in self.stages_done else []
        return [*self.stages_done, *last]


async def _evaluate(
    env: Mapping[str, str],
    records_path: Path,
    key_env: Mapping[str, str],
    meter: SpendMeter,
    progress: Progress,
    factory_builder: FactoryBuilder,
    should_stop: Callable[[], bool],
):
    """Load the records, check the keys and run the evaluation. Raises SetupError or RunError."""
    try:
        records = load_records(records_path)
    except FileRejected as exc:
        raise SetupError(f"the records file was rejected: {exc}") from None
    component = env["COMPONENT"]
    component_records = records_for_component(records, component)
    if not component_records:
        raise SetupError(f"the records have no component {component!r}")

    allow = list(dict.fromkeys(p.strip() for p in env["ALLOW"].split(",") if p.strip()))
    incumbent = build_profile(component_records, component).incumbent
    contacted = sorted(set(allow) | ({incumbent.provider} if incumbent else set()))
    try:
        keys = read_keys(contacted, key_env)
    except MissingKeys as exc:
        where = f"the provider-key secret {env['KEYS_SECRET_ID']}" if env.get("KEYS_SECRET_ID") else "the worker's environment"
        raise SetupError(f"missing provider keys: add {', '.join(exc.variables)} to {where}") from None
    except ValueError as exc:
        raise SetupError(str(exc)) from None

    try:
        config = RunConfig(
            component=component,
            tier=env.get("TIER", "smoke"),
            spending_cap_usd=float(env["CAP_USD"]),
            allow_list=frozenset(allow),
            cost_cap_usd=float(env["COST_CAP_USD"]) if env.get("COST_CAP_USD") else None,
            replay_speed=float(env.get("SPEED", "1")),
        )
    except ValueError as exc:
        raise SetupError(f"bad run settings: {exc}") from None

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(PROGRESS_EVERY_SECONDS)
            await asyncio.to_thread(progress.write)

    beat = asyncio.create_task(heartbeat())
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            return await run_evaluation(
                records,
                Path(env.get("CATALOG", "catalog/catalog.yaml")),
                config,
                factory_builder(client, keys),
                on_stage=progress.on_stage,
                should_cancel=should_stop,
                meter=meter,
                on_progress=progress.on_progress,
            )
    finally:
        beat.cancel()


def run_worker(
    env: Mapping[str, str],
    s3: Any,
    secrets: Any,
    factory_builder: FactoryBuilder = real_factory_builder,
    should_stop: Callable[[], bool] = lambda: False,
) -> int:
    """Do one run; see the module docstring. Returns the exit code."""
    missing = [name for name in REQUIRED if not env.get(name)]
    if missing:
        print(f"worker: missing environment variables: {', '.join(missing)}", flush=True)
        return 2

    outputs = Outputs(s3, env)
    work = Path(env.get("WORK_DIR", "/work")) / env["RUN_ID"]
    records_path, report_path = work / "records.jsonl", work / "report.json"
    meter = SpendMeter(float(env["CAP_USD"])) if _is_number(env["CAP_USD"]) else SpendMeter(0.0)
    progress = Progress(outputs, meter)
    status: dict[str, Any] = {
        "run_id": env["RUN_ID"],
        "batch_job_id": env.get("AWS_BATCH_JOB_ID"),
        "attempt": int(env.get("AWS_BATCH_JOB_ATTEMPT", "1")),
        "started_at": now(),
        "has_report": False,
    }
    exit_code = 0
    try:
        key_env: dict[str, str] = dict(env)
        if env.get("KEYS_SECRET_ID"):
            keys = load_provider_keys(secrets, env["KEYS_SECRET_ID"])
            key_env.update(keys)
            print(f"worker: provider keys loaded: {', '.join(sorted(keys)) or 'none'}", flush=True)
        if not _is_number(env["CAP_USD"]):
            raise SetupError(f"CAP_USD is not a number: {env['CAP_USD']!r}")

        work.mkdir(parents=True, exist_ok=True)
        bucket, key = split_s3(env["RECORDS_URI"])
        try:
            s3.download_file(bucket, key, str(records_path))
        except Exception as exc:
            raise SetupError(f"could not read {env['RECORDS_URI']}: {type(exc).__name__}") from None
        print(f"worker: downloaded {env['RECORDS_URI']} ({records_path.stat().st_size} bytes)", flush=True)

        report = asyncio.run(_evaluate(env, records_path, key_env, meter, progress, factory_builder, should_stop))
        write_report(report, report_path)
        status["report"] = outputs.upload("report.json", report_path)
        status["has_report"] = True
        status["status"] = "stopped" if should_stop() else "completed"
        print(f"worker: uploaded {status['report']}", flush=True)
    except (SetupError, RunError) as exc:
        kind = "setup" if isinstance(exc, SetupError) else "run"
        print(f"worker: failed ({kind}): {exc}", flush=True)
        status.update(status="failed", error=str(exc), error_kind=kind)
        exit_code = 1
    except Exception as exc:  # details stay in the log; the server shows a generic message
        print(f"worker: internal error: {type(exc).__name__}: {exc}", flush=True)
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", error_kind="internal")
        exit_code = 2
    finally:
        shutil.rmtree(work, ignore_errors=True)

    status.update(
        work_dir_deleted=not work.exists(), spent_usd=meter.spent_usd,
        stages_done=progress.finished_stages(), finished_at=now(),
    )  # fmt: skip
    try:
        outputs.put_json("status.json", status)
    except Exception as exc:
        print(f"worker: could not write status.json: {type(exc).__name__}: {exc}", flush=True)
        return exit_code or 2
    print(f"worker: {status['status']} (exit {exit_code})", flush=True)
    return exit_code


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def main() -> int:
    stop = threading.Event()

    def on_term(signum: int, frame: object) -> None:
        print("worker: SIGTERM received, stopping after the calls in flight", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, on_term)
    return run_worker(os.environ, boto3.client("s3"), boto3.client("secretsmanager"), should_stop=stop.is_set)


if __name__ == "__main__":
    sys.exit(main())
