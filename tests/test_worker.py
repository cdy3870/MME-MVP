"""The run worker image's entrypoint (docker/worker.py), with S3, Secrets Manager and the models faked."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from migration_eval.capture.sink import record_to_json

# tests/ has no __init__.py files, so pytest puts this directory on sys.path (prepend mode).
from test_run import CATALOG_YAML, factory, records

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("worker", ROOT / "docker" / "worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

KMS = "arn:aws:kms:us-east-1:111122223333:key/test"
SSE = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": KMS}


class FakeS3:
    def __init__(self, records_jsonl: bytes) -> None:
        self.records_jsonl = records_jsonl
        self.written: dict[str, tuple[bytes, dict[str, Any]]] = {}
        self.progress: list[dict[str, Any]] = []

    def download_file(self, bucket: str, key: str, path: str) -> None:
        assert (bucket, key) == ("mme-runs-1", "partners/test/uploads/u1.jsonl")
        Path(path).write_bytes(self.records_jsonl)

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str, **extra: Any) -> None:
        self.written[Key] = (Body, extra)
        if Key.endswith("progress.json"):
            self.progress.append(json.loads(Body))

    def upload_file(self, path: str, bucket: str, key: str, ExtraArgs: dict[str, Any] | None = None) -> None:
        self.written[key] = (Path(path).read_bytes(), ExtraArgs or {})

    def json(self, name: str) -> dict[str, Any]:
        return json.loads(self.written[f"partners/test/runs/run1/{name}"][0])


class FakeSecrets:
    def __init__(self, value: str) -> None:
        self.value = value

    def get_secret_value(self, SecretId: str) -> dict[str, str]:
        return {"SecretString": self.value}


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(CATALOG_YAML)
    return {
        "RUN_ID": "run1", "RECORDS_URI": "s3://mme-runs-1/partners/test/uploads/u1.jsonl", "COMPONENT": "support",
        "CAP_USD": "100", "ALLOW": "openai,anthropic", "TIER": "smoke", "SPEED": "1000",
        "RESULT_BUCKET": "mme-runs-1", "RESULT_PREFIX": "partners/test/runs", "RESULT_KMS_KEY_ID": KMS,
        "KEYS_SECRET_ID": "mme/keys/test", "CATALOG": str(catalog), "WORK_DIR": str(tmp_path / "work"),
    }  # fmt: skip


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3("".join(record_to_json(r) + "\n" for r in records(60)).encode())


BOTH_KEYS = json.dumps({"OPENAI_API_KEY": "sk-openai", "ANTHROPIC_API_KEY": "sk-anthropic", "UNRELATED": "x"})


def run(env, s3, secret=BOTH_KEYS, **kwargs) -> tuple[int, list[dict[str, str]]]:
    seen_keys: list[dict[str, str]] = []

    def builder(client, keys):
        seen_keys.append(keys)
        return factory()

    code = worker.run_worker(env, s3, FakeSecrets(secret), factory_builder=builder, **kwargs)
    return code, seen_keys


def test_a_run_writes_its_report_status_and_progress_under_the_partner_key(env, s3, tmp_path):
    code, seen_keys = run(env, s3)
    assert code == 0
    assert seen_keys == [{"openai": "sk-openai", "anthropic": "sk-anthropic"}]

    status = s3.json("status.json")
    assert (status["status"], status["has_report"], status["work_dir_deleted"]) == ("completed", True, True)
    assert status["stages_done"][:3] == ["prepare", "candidates", "baseline"] and status["stages_done"][-1] == "report"
    assert json.loads(s3.written["partners/test/runs/run1/report.json"][0])["component"] == "support"
    assert all(extra == SSE for _, extra in s3.written.values()), "everything is written under the partner's key"

    stages = [p["stage"] for p in s3.progress]
    assert stages[:3] == ["prepare", "candidates", "baseline"] and stages[-1] == "report"
    assert s3.progress[-1]["stages_done"] == status["stages_done"][:-1]
    assert not (tmp_path / "work" / "run1").exists()
    assert "sk-openai" not in json.dumps(status)


def test_progress_counts_the_requests_of_each_replaying_stage(env, s3):
    assert run(env, s3)[0] == 0
    baseline = [p for p in s3.progress if p["stage"] == "baseline" and p["calls_total"] is not None]
    assert baseline[0]["calls_done"] == 0 and baseline[-1]["calls_done"] == baseline[-1]["calls_total"] == 150  # 3 × 50
    candidate = [p for p in s3.progress if p["stage"].startswith("candidate ") and p["calls_total"] is not None]
    assert candidate and all(p["calls_total"] == 50 for p in candidate)
    assert all(p["calls_done"] is None for p in s3.progress if p["stage"] in ("prepare", "candidates", "report"))


def test_progress_writes_request_counts_at_most_every_two_seconds():
    written: list[dict[str, Any]] = []

    class Out:
        def put_json(self, name: str, data: dict[str, Any]) -> None:
            written.append(data)

    t = {"now": 0.0}
    progress = worker.Progress(Out(), worker.SpendMeter(1.0), clock=lambda: t["now"])
    progress.on_stage("baseline")
    for done in range(0, 11):  # 11 requests 0.5 s apart, plus the stage start
        progress.on_progress(done, 10)
        t["now"] += 0.5
    counts = [w["calls_done"] for w in written]
    assert counts == [None, 0, 4, 8, 10]  # stage start, (0, 10), then every 2 s, and the last one


def test_a_missing_key_is_a_setup_error_naming_the_variable_and_the_secret(env, s3):
    code, seen_keys = run(env, s3, secret=json.dumps({"OPENAI_API_KEY": "sk-openai"}))
    assert code == 1 and seen_keys == []
    status = s3.json("status.json")
    assert (status["status"], status["error_kind"], status["has_report"]) == ("failed", "setup", False)
    assert status["error"] == "missing provider keys: add ANTHROPIC_API_KEY to the provider-key secret mme/keys/test"


def test_a_secret_that_is_not_json_is_a_setup_error_without_its_value(env, s3):
    code, _ = run(env, s3, secret="sk-ant-plaintext")
    assert code == 1
    status = s3.json("status.json")
    assert status["error_kind"] == "setup" and "not a JSON object" in status["error"]
    assert "sk-ant-plaintext" not in json.dumps(status)


def test_a_missing_component_is_a_setup_error(env, s3):
    code, _ = run({**env, "COMPONENT": "billing"}, s3)
    assert code == 1 and s3.json("status.json")["error"] == "the records have no component 'billing'"


def test_a_run_error_is_reported_as_one(env, s3):
    code, _ = run({**env, "CAP_USD": "0.000001"}, s3)  # the baseline can't fit under the cap
    assert code == 1
    status = s3.json("status.json")
    assert status["error_kind"] == "run" and "spending cap" in status["error"]


def test_a_stop_still_writes_the_report(env, s3):
    code, _ = run(env, s3, should_stop=lambda: True)
    assert code == 0
    status = s3.json("status.json")
    assert (status["status"], status["has_report"]) == ("stopped", True)
    report = json.loads(s3.written["partners/test/runs/run1/report.json"][0])
    assert {c["verdict"] for c in report["candidates"]} == {"insufficient_data"}


def test_missing_settings_stop_before_anything_happens(env, s3):
    assert worker.run_worker({k: v for k, v in env.items() if k != "RUN_ID"}, s3, FakeSecrets(BOTH_KEYS)) == 2
    assert s3.written == {}
