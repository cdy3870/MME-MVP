"""Entrypoint for the run worker image: one evaluation run per container (MOD-135, RUN-3).

    1. download the records (v0 JSONL) from RECORDS_URI into WORK_DIR
    2. run `migration-eval run ... --yes`, streaming its output to the container log
    3. upload report.json (when the run wrote one) and status.json to
       s3://RESULT_BUCKET/RESULT_PREFIX/RUN_ID/
    4. delete WORK_DIR, so no partner data is left in the container (RUN-7)

Environment:
    RUN_ID, RECORDS_URI (s3://bucket/key), COMPONENT, CAP_USD, ALLOW (comma-separated providers),
    RESULT_BUCKET, RESULT_PREFIX; optional TIER (default smoke), SPEED (default 1), WORK_DIR
    (default /work).

    Provider keys (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY) come from one of:
      * KEYS_SECRET_ID: a Secrets Manager secret holding a JSON object of those variables, read
        with the job role (the Terraform setup, ACC-2); only the CLI's environment gets them
      * already in the environment, injected by the job definition's `secrets`
    Never as plain SubmitJob overrides, which show up in DescribeJobs.

    RESULT_KMS_KEY_ID, when set, encrypts report.json and status.json with that KMS key (the
    partner's; the Terraform bucket policy refuses partner-prefix writes without it).

SIGTERM (TerminateJob, a timeout) is passed on to the CLI. The CLI does not yet save partial
results when stopped (RUN-6), so a stopped run uploads status.json only.

Exit code: the CLI's, or 2 when the worker itself fails (missing settings, S3 errors).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

REQUIRED = ("RUN_ID", "RECORDS_URI", "COMPONENT", "CAP_USD", "ALLOW", "RESULT_BUCKET", "RESULT_PREFIX")
KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


def load_provider_keys(secret_id: str) -> dict[str, str]:
    """The provider key variables in the secret. Errors never include the secret's value."""
    value = boto3.client("secretsmanager").get_secret_value(SecretId=secret_id)["SecretString"]
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        raise ValueError(f"secret {secret_id} is not a JSON object like {{\"ANTHROPIC_API_KEY\": \"...\"}}")
    return {name: data[name] for name in KEY_VARIABLES if isinstance(data.get(name), str) and data[name].strip()}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri}")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not bucket or not key:
        raise ValueError(f"expected s3://bucket/key, got {uri}")
    return bucket, key


def main() -> int:
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        print(f"worker: missing environment variables: {', '.join(missing)}", flush=True)
        return 2
    env = os.environ
    run_id = env["RUN_ID"]
    out_prefix = f"{env['RESULT_PREFIX'].strip('/')}/{run_id}"
    work = Path(env.get("WORK_DIR", "/work")) / run_id
    records = work / "records.jsonl"
    report = work / "report.json"
    s3 = boto3.client("s3")
    kms_key = env.get("RESULT_KMS_KEY_ID")
    sse = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key} if kms_key else {}
    child_env = dict(env)
    status: dict[str, object] = {
        "run_id": run_id,
        "batch_job_id": env.get("AWS_BATCH_JOB_ID"),
        "attempt": int(env.get("AWS_BATCH_JOB_ATTEMPT", "1")),
        "started_at": now(),
    }

    stopped = False
    child: subprocess.Popen[bytes] | None = None

    def on_term(signum: int, frame: object) -> None:
        nonlocal stopped
        stopped = True
        print("worker: SIGTERM received, stopping the run", flush=True)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)

    try:
        if env.get("KEYS_SECRET_ID"):
            keys = load_provider_keys(env["KEYS_SECRET_ID"])
            child_env.update(keys)
            status["provider_keys"] = sorted(keys)  # names only
            print(f"worker: provider keys loaded: {', '.join(sorted(keys)) or 'none'}", flush=True)

        work.mkdir(parents=True, exist_ok=True)
        bucket, key = split_s3(env["RECORDS_URI"])
        s3.download_file(bucket, key, str(records))
        print(f"worker: downloaded {env['RECORDS_URI']} ({records.stat().st_size} bytes)", flush=True)

        command = [
            "migration-eval", "run", str(records),
            "--component", env["COMPONENT"],
            "--tier", env.get("TIER", "smoke"),
            "--cap", env["CAP_USD"],
            "--speed", env.get("SPEED", "1"),
            "--out", str(report),
            "--yes",  # the confirmation happened before submit (RUN-2), not inside the worker
        ]
        for provider in filter(None, (p.strip() for p in env["ALLOW"].split(","))):
            command += ["--allow", provider]
        print(f"worker: {' '.join(command)}", flush=True)

        if not stopped:
            child = subprocess.Popen(command, env=child_env)
            exit_code = child.wait()
        else:
            exit_code = 143
        status["cli_exit_code"] = exit_code

        if report.exists():
            s3.upload_file(str(report), env["RESULT_BUCKET"], f"{out_prefix}/report.json", ExtraArgs=sse or None)
            status["report"] = f"s3://{env['RESULT_BUCKET']}/{out_prefix}/report.json"
            print(f"worker: uploaded {status['report']}", flush=True)
    except Exception as exc:  # report the failure in status.json, then fail the job
        print(f"worker: failed: {type(exc).__name__}: {exc}", flush=True)
        status["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 2
    finally:
        shutil.rmtree(work, ignore_errors=True)
        status["work_dir_deleted"] = not work.exists()

    status.update(status="stopped" if stopped else ("completed" if exit_code == 0 else "failed"), finished_at=now())
    try:
        s3.put_object(
            Bucket=env["RESULT_BUCKET"],
            Key=f"{out_prefix}/status.json",
            Body=json.dumps(status, indent=2).encode(),
            ContentType="application/json",
            **sse,
        )
    except Exception as exc:
        print(f"worker: could not write status.json: {type(exc).__name__}: {exc}", flush=True)
        return exit_code or 2
    print(f"worker: {status['status']} (exit {exit_code})", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
