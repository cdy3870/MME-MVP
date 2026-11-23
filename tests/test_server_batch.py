"""Runs on AWS Batch through the server (MOD-135): the UI's routes with `create_app(..., batch=...)`.

AWS is faked: `FakeBatch`, `FakeS3` and `FakeSecrets` answer like boto3 clients, and each test moves
the Batch job through its states and writes the worker's files the way docker/worker.py does.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from migration_eval.server.app import create_app
from migration_eval.server.batch import STARTING_STAGE, WAITING_STAGE, BatchRuns, BatchSettings, BatchUnavailable
from migration_eval.server.store import Store

# tests/ has no __init__.py files, so pytest puts this directory on sys.path (prepend mode).
from test_run import factory, records

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog" / "catalog.yaml"
REPORT_JSON = (ROOT / "schema" / "examples" / "report" / "valid" / "support_bot_all_verdicts.json").read_bytes()
KMS = "arn:aws:kms:us-east-1:111122223333:key/test"
SECRET = "arn:aws:secretsmanager:us-east-1:111122223333:secret:mme/keys/test-AbCdEf"


class ClientError(Exception):
    """Shaped like botocore's: the code and message are in `response["Error"]`."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": message or f"{code} happened"}}


class FakeBatch:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.submitted: list[dict[str, Any]] = []
        self.terminated: list[str] = []
        self.describe_calls = 0
        self.fail_submit: ClientError | None = None

    def submit_job(self, **kwargs: Any) -> dict[str, str]:
        if self.fail_submit:
            raise self.fail_submit
        job_id = f"job-{len(self.submitted) + 1}"
        self.submitted.append(kwargs)
        self.jobs[job_id] = {"status": "SUBMITTED"}
        return {"jobId": job_id}

    def describe_jobs(self, jobs: list[str]) -> dict[str, Any]:
        self.describe_calls += 1
        return {"jobs": [{"jobId": j, **self.jobs[j]} for j in jobs if j in self.jobs]}

    def terminate_job(self, jobId: str, reason: str) -> None:
        self.terminated.append(jobId)

    def describe_job_definitions(self, jobDefinitionName: str, status: str) -> dict[str, Any]:
        env = [
            {"name": "RESULT_BUCKET", "value": "mme-runs-1"},
            {"name": "RESULT_PREFIX", "value": "partners/test/runs"},
            {"name": "RESULT_KMS_KEY_ID", "value": KMS},
            {"name": "KEYS_SECRET_ID", "value": SECRET},
        ]
        log = {"logDriver": "awslogs", "options": {"awslogs-group": "/mme/run-workers"}}
        return {
            "jobDefinitions": [
                {"revision": 2, "containerProperties": {"environment": env[:1]}},  # no RESULT_PREFIX
                {"revision": 3, "containerProperties": {"environment": env, "logConfiguration": log}},
            ]
        }


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, Any]]] = {}
        self.puts: list[str] = []

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise ClientError("404", "Not Found")
        return {}

    def put_object(self, Bucket: str, Key: str, Body: bytes, **extra: Any) -> None:
        self.puts.append(Key)
        self.objects[Key] = (Body, extra)

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise ClientError("NoSuchKey", "The specified key does not exist.")
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def list_objects_v2(self, Bucket: str, Prefix: str) -> dict[str, Any]:
        return {"Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)]}

    def delete_objects(self, Bucket: str, Delete: dict[str, Any]) -> None:
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.objects.pop(Key, None)

    # what the worker writes
    def worker_writes(self, run_id: str, name: str, data: dict[str, Any] | bytes) -> None:
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.objects[f"partners/test/runs/{run_id}/{name}"] = (body, {})


class FakeSecrets:
    def __init__(self, key_variables: str | None = None) -> None:
        self.tags = [] if key_variables is None else [{"Key": "mme:key-variables", "Value": key_variables}]

    def describe_secret(self, SecretId: str) -> dict[str, Any]:
        return {"Name": "mme/keys/test", "Tags": self.tags}


class Aws:
    def __init__(self, key_variables: str | None = None, min_poll_seconds: float = 0.0) -> None:
        self.batch, self.s3, self.secrets = FakeBatch(), FakeS3(), FakeSecrets(key_variables)
        settings = BatchRuns.read_settings(self.batch, "test", "mme-runs", "mme-worker-test", timeout_seconds=1800)
        self.runs = BatchRuns(settings, self.batch, self.s3, self.secrets, min_poll_seconds=min_poll_seconds)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def upload_id(data_dir: Path) -> str:
    """An upload of component `support`, logged on openai/gpt-4o."""
    return Store(data_dir).save_upload("records.jsonl", 1000, records(60)).id


@pytest.fixture
def aws() -> Aws:
    return Aws()


@pytest.fixture
def client(data_dir: Path, aws: Aws, upload_id: str) -> TestClient:
    return TestClient(create_app(data_dir, CATALOG, batch=aws.runs))


def request(upload_id: str, **overrides: Any) -> dict[str, Any]:
    body = {
        "upload_id": upload_id, "component": "support", "tier": "smoke", "spending_cap_usd": 1.0,
        "allow": ["anthropic"], "replay_speed": 1000.0, "confirmed": True,
    }  # fmt: skip
    return {**body, **overrides}


def start(client: TestClient, upload_id: str, **overrides: Any) -> dict[str, Any]:
    r = client.post("/api/runs", json=request(upload_id, **overrides))
    assert r.status_code == 201, r.text
    return r.json()


# --- settings ---------------------------------------------------------------------------------


def test_settings_come_from_the_latest_job_definition(aws: Aws):
    s = aws.runs.settings
    assert s.job_definition == "mme-worker-test:3"
    assert (s.bucket, s.runs_prefix, s.uploads_prefix) == ("mme-runs-1", "partners/test/runs", "partners/test/uploads")
    assert (s.kms_key_arn, s.keys_secret_id, s.log_group) == (KMS, SECRET, "/mme/run-workers")


def test_a_job_definition_without_the_worker_settings_is_refused(aws: Aws):
    with pytest.raises(BatchUnavailable, match="RESULT_BUCKET"):
        BatchRuns.read_settings(aws.batch, "test", "mme-runs", "mme-worker-test:2")
    with pytest.raises(BatchUnavailable, match="No active job definition"):
        BatchRuns.read_settings(aws.batch, "test", "mme-runs", "mme-worker-test:9")


# --- starting ---------------------------------------------------------------------------------


def test_start_puts_the_records_in_s3_under_the_partner_key_and_submits_a_job(client, aws, upload_id, data_dir):
    run = start(client, upload_id)
    assert (run["state"], run["stage"], run["batch_job_id"]) == ("queued", WAITING_STAGE, "job-1")

    key = f"partners/test/uploads/{upload_id}.jsonl"
    body, extra = aws.s3.objects[key]
    assert body == Store(data_dir).read_records_bytes(upload_id)
    assert extra == {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": KMS, "ContentType": "application/x-ndjson"}

    submitted = aws.batch.submitted[0]
    env = {e["name"]: e["value"] for e in submitted["containerOverrides"]["environment"]}
    assert submitted["jobQueue"] == "mme-runs" and submitted["jobDefinition"] == "mme-worker-test:3"
    assert submitted["timeout"] == {"attemptDurationSeconds": 1800} and submitted["retryStrategy"] == {"attempts": 1}
    assert env == {
        "RUN_ID": run["id"], "RECORDS_URI": f"s3://mme-runs-1/{key}", "COMPONENT": "support", "ALLOW": "anthropic",
        "TIER": "smoke", "CAP_USD": "1.0", "SPEED": "1000.0",
    }  # fmt: skip
    assert not any("KEY" in name for name in env), "provider keys never travel as overrides"


def test_the_records_are_uploaded_once_per_upload(client, aws, upload_id):
    first = start(client, upload_id)
    aws.batch.jobs[first["batch_job_id"]] = {"status": "FAILED", "statusReason": "boom"}
    start(client, upload_id)  # the first run is refreshed, found ended, and no longer blocks
    assert aws.s3.puts.count(f"partners/test/uploads/{upload_id}.jsonl") == 1


def test_an_aws_failure_at_submit_is_a_502_and_leaves_a_failed_run(client, aws, upload_id):
    aws.batch.fail_submit = ClientError("AccessDeniedException", "not authorized to perform: batch:SubmitJob")
    r = client.post("/api/runs", json=request(upload_id))
    assert r.status_code == 502
    assert "AccessDeniedException" in r.json()["detail"] and "batch:SubmitJob" in r.json()["detail"]
    (summary,) = client.get("/api/runs").json()
    assert summary["state"] == "failed"


def test_only_one_run_at_a_time(client, aws, upload_id):
    first = start(client, upload_id)
    aws.batch.jobs[first["batch_job_id"]] = {"status": "RUNNING"}
    assert client.post("/api/runs", json=request(upload_id)).status_code == 409


# --- keys -------------------------------------------------------------------------------------


def test_key_status_comes_from_the_secret_tag(data_dir, upload_id):
    aws = Aws(key_variables="ANTHROPIC_API_KEY")
    client = TestClient(create_app(data_dir, CATALOG, batch=aws.runs))
    providers = {p["name"]: p for p in client.get("/api/providers").json()}
    assert providers["anthropic"]["key_configured"] and not providers["openai"]["key_configured"]
    assert providers["openai"]["key_source"] == "secret" and providers["openai"]["key_secret"] == "mme/keys/test"

    # the current model is on openai, so the baseline needs its key too
    r = client.post("/api/runs", json=request(upload_id))
    assert r.status_code == 400
    assert r.json()["detail"] == "Add OPENAI_API_KEY to the mme/keys/test secret (the value is never shown)."
    assert aws.batch.submitted == []


def test_without_a_tag_every_key_is_assumed_set_and_the_worker_checks(client):
    providers = client.get("/api/providers").json()
    assert all(p["key_configured"] and p["key_source"] == "secret" for p in providers)


# --- following a run ----------------------------------------------------------------------------


def test_status_follows_the_job_through_to_the_report(client, aws, upload_id):
    run = start(client, upload_id)
    job = run["batch_job_id"]

    aws.batch.jobs[job] = {"status": "RUNNABLE"}
    got = client.get(f"/api/runs/{run['id']}").json()
    assert (got["state"], got["stage"], got["stages_done"]) == ("queued", WAITING_STAGE, [])

    aws.batch.jobs[job] = {"status": "STARTING"}
    got = client.get(f"/api/runs/{run['id']}").json()
    assert (got["state"], got["stage"], got["stages_done"]) == ("running", STARTING_STAGE, [WAITING_STAGE])

    aws.batch.jobs[job] = {"status": "RUNNING"}
    aws.s3.worker_writes(
        run["id"], "progress.json",
        {"stage": "baseline", "stages_done": ["prepare", "candidates"], "spent_usd": 0.12, "calls_done": 87, "calls_total": 150},
    )  # fmt: skip
    got = client.get(f"/api/runs/{run['id']}").json()
    assert (got["state"], got["stage"], got["spent_usd"]) == ("running", "baseline", 0.12)
    assert (got["calls_done"], got["calls_total"]) == (87, 150)
    assert got["stages_done"] == [WAITING_STAGE, STARTING_STAGE, "prepare", "candidates"]
    assert client.get(f"/api/runs/{run['id']}/report").status_code == 404

    aws.batch.jobs[job] = {"status": "SUCCEEDED"}
    stages = ["prepare", "candidates", "baseline", "report"]
    aws.s3.worker_writes(run["id"], "status.json", {"status": "completed", "has_report": True, "spent_usd": 0.31, "stages_done": stages})
    aws.s3.worker_writes(run["id"], "report.json", REPORT_JSON)
    got = client.get(f"/api/runs/{run['id']}").json()
    assert (got["state"], got["stage"], got["has_report"], got["spent_usd"]) == ("done", None, True, 0.31)
    assert (got["calls_done"], got["calls_total"]) == (None, None)
    assert got["stages_done"] == [WAITING_STAGE, STARTING_STAGE, *stages] and got["finished_at"]

    report = client.get(f"/api/runs/{run['id']}/report")
    assert report.status_code == 200 and report.json()["component"] == json.loads(REPORT_JSON)["component"]
    summary = client.get("/api/runs").json()[0]
    assert summary["state"] == "done" and summary["verdicts"]


def test_a_worker_that_never_started_shows_batchs_reason(client, aws, upload_id):
    run = start(client, upload_id)
    aws.batch.jobs[run["batch_job_id"]] = {
        "status": "FAILED", "statusReason": "Task failed to start",
        "container": {"reason": "CannotPullImageManifestError: manifest unknown"},
    }  # fmt: skip
    got = client.get(f"/api/runs/{run['id']}").json()
    assert got["state"] == "failed"
    assert got["error"] == "The worker did not finish: CannotPullImageManifestError: manifest unknown."


def test_a_setup_error_from_the_worker_is_shown_as_is(client, aws, upload_id):
    run = start(client, upload_id)
    aws.batch.jobs[run["batch_job_id"]] = {"status": "FAILED"}
    message = "missing provider keys: add ANTHROPIC_API_KEY to the provider-key secret arn:..."
    aws.s3.worker_writes(run["id"], "status.json", {"status": "failed", "error": message, "error_kind": "setup", "has_report": False})
    got = client.get(f"/api/runs/{run['id']}").json()
    assert (got["state"], got["error"]) == ("failed", f"The run failed: {message}.")


def test_an_internal_worker_error_points_to_the_log_not_the_details(client, aws, upload_id):
    run = start(client, upload_id)
    aws.batch.jobs[run["batch_job_id"]] = {"status": "FAILED", "container": {"logStreamName": "test/default/abc"}}
    aws.s3.worker_writes(run["id"], "status.json", {"status": "failed", "error": "KeyError: 'x' at /app/...", "error_kind": "internal"})
    got = client.get(f"/api/runs/{run['id']}").json()
    assert got["error"] == "Something went wrong in the worker. The worker's log is in CloudWatch: /mme/run-workers / test/default/abc."


def test_stop_terminates_the_job_and_keeps_the_partial_report(client, aws, upload_id):
    run = start(client, upload_id)
    aws.batch.jobs[run["batch_job_id"]] = {"status": "RUNNING"}
    stopping = client.post(f"/api/runs/{run['id']}/cancel")
    assert stopping.status_code == 200 and stopping.json()["state"] == "cancelling"
    assert aws.batch.terminated == [run["batch_job_id"]]
    assert client.get(f"/api/runs/{run['id']}").json()["state"] == "cancelling"  # still running in AWS

    # Batch marks a terminated job FAILED even though the worker exited cleanly
    aws.batch.jobs[run["batch_job_id"]] = {"status": "FAILED", "statusReason": "Stopped from the UI"}
    aws.s3.worker_writes(run["id"], "status.json", {"status": "stopped", "has_report": True, "spent_usd": 0.05})
    aws.s3.worker_writes(run["id"], "report.json", REPORT_JSON)
    got = client.get(f"/api/runs/{run['id']}").json()
    assert (got["state"], got["has_report"], got["error"]) == ("cancelled", True, None)
    assert client.post(f"/api/runs/{run['id']}/cancel").status_code == 409


def test_polling_aws_is_throttled(data_dir, upload_id):
    aws = Aws(min_poll_seconds=60)
    client = TestClient(create_app(data_dir, CATALOG, batch=aws.runs))
    run = start(client, upload_id)
    for _ in range(5):
        client.get(f"/api/runs/{run['id']}")
    assert aws.batch.describe_calls == 1


def test_a_restart_picks_batch_runs_up_again(data_dir, aws, upload_id):
    run = start(TestClient(create_app(data_dir, CATALOG, batch=aws.runs)), upload_id)
    aws.batch.jobs[run["batch_job_id"]] = {"status": "RUNNING"}

    # a new server process: new app and BatchRuns, same data directory and the same jobs in AWS
    restarted = TestClient(create_app(data_dir, CATALOG, batch=BatchRuns(aws.runs.settings, aws.batch, aws.s3, aws.secrets, 0.0)))
    assert restarted.get(f"/api/runs/{run['id']}").json()["state"] == "running"


def test_a_local_server_leaves_batch_runs_alone(data_dir, aws, upload_id):
    run = start(TestClient(create_app(data_dir, CATALOG, batch=aws.runs)), upload_id)
    local = TestClient(create_app(data_dir, CATALOG))
    got = local.get(f"/api/runs/{run['id']}").json()
    assert got["state"] == "queued"  # not marked failed: the job is still in AWS
    r = local.post(f"/api/runs/{run['id']}/cancel")
    assert r.status_code == 409 and "--executor batch" in r.json()["detail"]


# --- deleting ---------------------------------------------------------------------------------


def test_deleting_a_run_and_its_upload_removes_their_s3_files(client, aws, upload_id):
    run = start(client, upload_id)
    aws.batch.jobs[run["batch_job_id"]] = {"status": "RUNNING"}
    assert client.delete(f"/api/uploads/{upload_id}").status_code == 409  # in use

    aws.batch.jobs[run["batch_job_id"]] = {"status": "SUCCEEDED"}
    aws.s3.worker_writes(run["id"], "status.json", {"status": "completed", "has_report": True})
    aws.s3.worker_writes(run["id"], "report.json", REPORT_JSON)
    assert client.get(f"/api/runs/{run['id']}").json()["state"] == "done"

    assert client.delete(f"/api/runs/{run['id']}").status_code == 204
    assert not [k for k in aws.s3.objects if k.startswith(f"partners/test/runs/{run['id']}/")]
    assert client.delete(f"/api/uploads/{upload_id}").status_code == 204
    assert f"partners/test/uploads/{upload_id}.jsonl" not in aws.s3.objects


# --- the local executor still works ------------------------------------------------------------


def test_local_runs_still_execute_in_process(data_dir, upload_id, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # `with`: keep one event loop for the whole test, so the background task isn't cancelled between requests
    with TestClient(create_app(data_dir, CATALOG, factory_builder=lambda http, keys: factory())) as client:
        run = start(client, upload_id)
        assert run["batch_job_id"] is None
        deadline = time.monotonic() + 30
        while (got := client.get(f"/api/runs/{run['id']}").json())["state"] in ("queued", "running"):
            assert time.monotonic() < deadline, got
            time.sleep(0.05)
        assert (got["state"], got["has_report"]) == ("done", True), got
        providers = client.get("/api/providers").json()
    assert all(p["key_source"] == "environment" and p["key_secret"] is None for p in providers)
