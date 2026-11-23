"""Running evaluations as AWS Batch jobs (MOD-135), for `migration-eval serve --executor batch`.

The infrastructure comes from infra/terraform: a job queue and one job definition per partner.
The job definition fixes the partner's bucket prefix, KMS key and provider-key secret; this module
reads them from it once at startup, then for each run:

    ensure_upload   write the upload's records to s3://<bucket>/partners/<p>/uploads/<upload id>.jsonl
                    under the partner's key (once per upload; later runs reuse it)
    submit          SubmitJob with the run's settings as environment overrides, and its time limit
    poll            DescribeJobs, plus the worker's progress.json, and status.json once it has ended
    fetch_report    report.json from partners/<p>/runs/<run id>/
    terminate       TerminateJob (the worker gets SIGTERM and writes what it finished)

Every AWS failure becomes `BatchUnavailable` with a message for the user (the AWS error code and
message, never a secret). The server needs the `orchestrator` policy from Terraform.

Polling is throttled to one AWS refresh per run every `min_poll_seconds`, since the UI polls the
server every second.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from migration_eval.models import Report

WAITING_STAGE = "waiting for a machine"  # SUBMITTED, PENDING, RUNNABLE: usually an instance starting
STARTING_STAGE = "starting the worker"  # STARTING: pulling the image, starting the container
BATCH_ACTIVE = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")
BATCH_WAITING = ("SUBMITTED", "PENDING", "RUNNABLE")
KEY_VARIABLES_TAG = "mme:key-variables"  # on the secret: which key variables it holds, comma-separated


class BatchUnavailable(Exception):
    """An AWS call failed. The message is safe to show the user."""


def _aws_error(action: str, exc: Exception) -> BatchUnavailable:
    response = getattr(exc, "response", None)
    if isinstance(response, dict) and "Error" in response:
        error = response["Error"]
        return BatchUnavailable(f"{action} failed: {error.get('Code', 'Error')}: {error.get('Message', '')}".rstrip(": "))
    return BatchUnavailable(f"{action} failed: {type(exc).__name__}: {exc}")


def _error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code"))
    return None


@dataclass(frozen=True)
class BatchSettings:
    partner: str
    queue: str
    job_definition: str  # name:revision, resolved at startup
    bucket: str
    runs_prefix: str  # partners/<p>/runs
    uploads_prefix: str  # partners/<p>/uploads
    kms_key_arn: str | None
    keys_secret_id: str | None
    log_group: str | None
    timeout_seconds: int = 3600


@dataclass(frozen=True)
class JobView:
    """What one poll saw."""

    batch_status: str
    reason: str | None  # Batch's statusReason, or the container's reason (e.g. an image pull error)
    progress: dict[str, Any] | None  # progress.json, once the worker has written one
    worker_status: dict[str, Any] | None  # status.json, read only once the job has ended
    log_stream: str | None


class BatchRuns:
    def __init__(self, settings: BatchSettings, batch: Any, s3: Any, secrets: Any, min_poll_seconds: float = 3.0) -> None:
        self.settings = settings
        self._batch = batch
        self._s3 = s3
        self._secrets = secrets
        self.min_poll_seconds = min_poll_seconds
        self._uploaded: set[str] = set()

    # ---- setup -----------------------------------------------------------------------------

    @classmethod
    def connect(
        cls, partner: str, queue: str, job_definition: str | None = None, region: str | None = None, timeout_seconds: int = 3600
    ) -> BatchRuns:
        """Open AWS clients and read the partner's job definition. Raises BatchUnavailable."""
        try:
            import boto3
        except ImportError:
            raise BatchUnavailable("boto3 is not installed: pip install -e '.[aws]'") from None
        try:
            session = boto3.Session(region_name=region)
            batch = session.client("batch")
            settings = cls.read_settings(batch, partner, queue, job_definition or f"mme-worker-{partner}", timeout_seconds)
            return cls(settings, batch, session.client("s3"), session.client("secretsmanager"))
        except BatchUnavailable:
            raise
        except Exception as exc:
            raise _aws_error("Connecting to AWS Batch", exc) from None

    @staticmethod
    def read_settings(batch: Any, partner: str, queue: str, job_definition: str, timeout_seconds: int = 3600) -> BatchSettings:
        """The partner's settings from the latest active revision of its job definition."""
        name, _, revision = job_definition.partition(":")
        try:
            found = batch.describe_job_definitions(jobDefinitionName=name, status="ACTIVE")["jobDefinitions"]
        except Exception as exc:
            raise _aws_error("Reading the job definition", exc) from None
        if revision:
            found = [d for d in found if str(d["revision"]) == revision]
        if not found:
            raise BatchUnavailable(f"No active job definition {job_definition!r}. Run terraform apply in infra/terraform, or check --partner.")
        latest = max(found, key=lambda d: d["revision"])
        container = latest.get("containerProperties", {})
        env = {e["name"]: e["value"] for e in container.get("environment", [])}
        if "RESULT_BUCKET" not in env or "RESULT_PREFIX" not in env:
            raise BatchUnavailable(f"{name}:{latest['revision']} sets no RESULT_BUCKET / RESULT_PREFIX; use the job definition Terraform made.")
        runs_prefix = env["RESULT_PREFIX"].strip("/")
        log_options = (container.get("logConfiguration") or {}).get("options") or {}
        return BatchSettings(
            partner=partner,
            queue=queue,
            job_definition=f"{name}:{latest['revision']}",
            bucket=env["RESULT_BUCKET"],
            runs_prefix=runs_prefix,
            uploads_prefix=f"{runs_prefix.rsplit('/', 1)[0]}/uploads" if "/" in runs_prefix else "uploads",
            kms_key_arn=env.get("RESULT_KMS_KEY_ID"),
            keys_secret_id=env.get("KEYS_SECRET_ID"),
            log_group=log_options.get("awslogs-group"),
            timeout_seconds=timeout_seconds,
        )

    # ---- keys ------------------------------------------------------------------------------

    def key_variables(self) -> set[str] | None:
        """The key variables the partner's secret holds, from its `mme:key-variables` tag. None when
        the tag is missing or the secret can't be described: the server can't read the secret's
        value (by design), so the worker checks the keys when the run starts."""
        if not self.settings.keys_secret_id:
            return None
        try:
            tags = self._secrets.describe_secret(SecretId=self.settings.keys_secret_id).get("Tags", [])
        except Exception:
            return None
        value = next((t["Value"] for t in tags if t.get("Key") == KEY_VARIABLES_TAG), None)
        return None if value is None else {v.strip() for v in value.split(",") if v.strip()}

    # ---- runs ------------------------------------------------------------------------------

    def _sse(self) -> dict[str, str]:
        return {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": self.settings.kms_key_arn} if self.settings.kms_key_arn else {}

    def _upload_key(self, upload_id: str) -> str:
        return f"{self.settings.uploads_prefix}/{upload_id}.jsonl"

    def _run_key(self, run_id: str, name: str) -> str:
        return f"{self.settings.runs_prefix}/{run_id}/{name}"

    def ensure_upload(self, upload_id: str, records_jsonl: bytes) -> str:
        """Put the upload's records in S3 under the partner's key, once; return the s3:// URI."""
        key = self._upload_key(upload_id)
        uri = f"s3://{self.settings.bucket}/{key}"
        if upload_id in self._uploaded:
            return uri
        try:
            self._s3.head_object(Bucket=self.settings.bucket, Key=key)
        except Exception as exc:
            if _error_code(exc) not in ("404", "NoSuchKey", "NotFound"):
                raise _aws_error("Checking the upload in S3", exc) from None
            try:
                self._s3.put_object(Bucket=self.settings.bucket, Key=key, Body=records_jsonl, ContentType="application/x-ndjson", **self._sse())
            except Exception as exc2:
                raise _aws_error("Uploading the records to S3", exc2) from None
        self._uploaded.add(upload_id)
        return uri

    def submit(
        self,
        run_id: str,
        *,
        records_uri: str,
        component: str,
        allow: Iterable[str],
        tier: str,
        spending_cap_usd: float,
        replay_speed: float,
        cost_cap_usd: float | None = None,
    ) -> str:
        """Submit the run; return the Batch job id."""
        env = {
            "RUN_ID": run_id,
            "RECORDS_URI": records_uri,
            "COMPONENT": component,
            "ALLOW": ",".join(allow),
            "TIER": tier,
            "CAP_USD": repr(float(spending_cap_usd)),
            "SPEED": repr(float(replay_speed)),
        }
        if cost_cap_usd is not None:
            env["COST_CAP_USD"] = repr(float(cost_cap_usd))
        try:
            response = self._batch.submit_job(
                jobName=f"mme-{self.settings.partner}-{run_id[:12]}",
                jobQueue=self.settings.queue,
                jobDefinition=self.settings.job_definition,
                containerOverrides={"environment": [{"name": k, "value": v} for k, v in env.items()]},
                timeout={"attemptDurationSeconds": self.settings.timeout_seconds},
                retryStrategy={"attempts": 1},  # a retry would repeat paid calls with a fresh cap (RUN-4, RUN-5)
                tags={"run_id": run_id, "partner": self.settings.partner},
                propagateTags=True,
            )
        except Exception as exc:
            raise _aws_error("Submitting the run to AWS Batch", exc) from None
        return str(response["jobId"])

    def poll(self, job_id: str, run_id: str) -> JobView:
        try:
            jobs = self._batch.describe_jobs(jobs=[job_id])["jobs"]
        except Exception as exc:
            raise _aws_error("Reading the job from AWS Batch", exc) from None
        if not jobs:  # Batch forgets jobs some days after they end
            return JobView("UNKNOWN", "AWS Batch no longer has this job.", None, self._read_json(run_id, "status.json"), None)
        job = jobs[0]
        container = job.get("container") or {}
        status = str(job["status"])
        reason = container.get("reason") or job.get("statusReason")
        progress = self._read_json(run_id, "progress.json") if status in ("RUNNING", "SUCCEEDED", "FAILED") else None
        worker_status = self._read_json(run_id, "status.json") if status not in BATCH_ACTIVE else None
        return JobView(status, reason, progress, worker_status, container.get("logStreamName"))

    def _read_json(self, run_id: str, name: str) -> dict[str, Any] | None:
        try:
            body = self._s3.get_object(Bucket=self.settings.bucket, Key=self._run_key(run_id, name))["Body"].read()
        except Exception as exc:
            if _error_code(exc) in ("NoSuchKey", "404", "NotFound"):
                return None
            raise _aws_error(f"Reading {name} from S3", exc) from None
        try:
            data = json.loads(body)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def fetch_report(self, run_id: str) -> Report | None:
        try:
            body = self._s3.get_object(Bucket=self.settings.bucket, Key=self._run_key(run_id, "report.json"))["Body"].read()
        except Exception as exc:
            if _error_code(exc) in ("NoSuchKey", "404", "NotFound"):
                return None
            raise _aws_error("Reading the report from S3", exc) from None
        return Report.model_validate_json(body)

    def terminate(self, job_id: str) -> None:
        try:
            self._batch.terminate_job(jobId=job_id, reason="Stopped from the UI")
        except Exception as exc:
            raise _aws_error("Stopping the job", exc) from None

    def log_location(self, view: JobView) -> str | None:
        if self.settings.log_group and view.log_stream:
            return f"{self.settings.log_group} / {view.log_stream}"
        return None

    # ---- deleting --------------------------------------------------------------------------

    def delete_run_outputs(self, run_id: str) -> None:
        self._delete_prefix(f"{self.settings.runs_prefix}/{run_id}/")

    def delete_upload(self, upload_id: str) -> None:
        try:
            self._s3.delete_object(Bucket=self.settings.bucket, Key=self._upload_key(upload_id))
        except Exception as exc:
            raise _aws_error("Deleting the upload from S3", exc) from None
        self._uploaded.discard(upload_id)

    def _delete_prefix(self, prefix: str) -> None:
        try:
            listed = self._s3.list_objects_v2(Bucket=self.settings.bucket, Prefix=prefix).get("Contents", [])
            if listed:
                self._s3.delete_objects(Bucket=self.settings.bucket, Delete={"Objects": [{"Key": o["Key"]} for o in listed]})
        except Exception as exc:
            raise _aws_error("Deleting the run's files from S3", exc) from None


class Throttle:
    """At most one refresh per key every `seconds`."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._last: dict[str, float] = {}

    def due(self, key: str) -> bool:
        current = time.monotonic()
        if current - self._last.get(key, float("-inf")) < self.seconds:
            return False
        self._last[key] = current
        return True
