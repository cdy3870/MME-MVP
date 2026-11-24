"""Smoke-test AWS Batch job submission for run workers (MOD-135), with a dummy worker.

Registers a throwaway job definition, submits one job per run, polls it to the end, reads the
result the worker wrote to S3, prints the CloudWatch log tail, then deregisters the definition.
The worker is a tiny Python program (no partner data, no provider calls) that:

    * optionally reads the first KiB of an S3 object, to check the job role can decrypt it
      with the partner's KMS key (RUN-3)
    * "works" for --work-seconds, one step per second
    * on SIGTERM stops, and still writes what it finished (RUN-6)
    * writes result.json to s3://<bucket>/<prefix>/<run id>/

What it measures, against the gaps noted on MOD-135:
    * cold start: time from submit to RUNNING, with and without a warm instance
    * instance reuse: --repeat 2 shows whether the second run landed on the same EC2 instance
      (RUN-3 wants a fresh worker per run; Batch reuses instances)
    * per-run role: --job-role-arn is set on a new job definition revision for each run, since
      SubmitJob can't override the job role
    * stop: --stop-after N calls TerminateJob N seconds into RUNNING and times the halt (RUN-6: 30 s)
    * retries: --fail-first-attempt makes attempt 1 exit 1, to check --retries works

Prerequisites (until the Terraform from MOD-135 exists, create these by hand):
    * a managed EC2 compute environment (minvCpus 0) and a job queue attached to it
    * subnets with outbound internet (NAT or public IP): the worker pip-installs boto3
    * a job role trusted by ecs-tasks.amazonaws.com with s3:PutObject on the results prefix
      (plus s3:GetObject and kms:Decrypt for --upload-uri)
    * locally: `pip install boto3` and AWS credentials allowed batch:RegisterJobDefinition,
      SubmitJob, DescribeJobs, TerminateJob, DeregisterJobDefinition, iam:PassRole on the job
      role, ecs:DescribeContainerInstances, logs:GetLogEvents and s3:GetObject

--mode replay runs the real worker image (Dockerfile, docker/worker.py) instead of the dummy:
it downloads --records-uri, runs `migration-eval run`, and writes report.json and status.json.
That needs the image in ECR (--image), and for provider keys a Secrets Manager secret holding
a JSON object such as {"ANTHROPIC_API_KEY": "..."} (--keys-secret-arn) plus an execution role
that can read it (--execution-role-arn). The job role also needs s3:GetObject on the records.

Examples:
    python scripts/batch_smoke.py --queue mme-replay --bucket my-results \\
        --job-role-arn arn:aws:iam::123456789012:role/mme-run-worker --dry-run
    python scripts/batch_smoke.py --mode replay --queue mme-replay --bucket my-results \\
        --job-role-arn ... --image 123456789012.dkr.ecr.us-east-1.amazonaws.com/mme-run-worker:latest \\
        --records-uri s3://my-results/batch-smoke/inputs/records.jsonl --component support \\
        --allow anthropic --run-cap 1 --keys-secret-arn ... --execution-role-arn ...
    python scripts/batch_smoke.py --queue mme-replay --bucket my-results \\
        --job-role-arn arn:aws:iam::123456789012:role/mme-run-worker --repeat 2
    python scripts/batch_smoke.py ... --work-seconds 300 --stop-after 20

With the Terraform in infra/terraform, pass --job-definition instead of registering one: the
job definition already holds the image, the partner's job role, its bucket prefix, KMS key and
provider-key secret, so only the per-run settings are sent (plus timeout and retries):
    python scripts/batch_smoke.py --mode replay --queue mme-runs --job-definition mme-worker-test \\
        --records-uri s3://mme-runs-123456789012/partners/test/uploads/records.jsonl \\
        --component support-triage --allow anthropic --run-cap 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

DEFAULT_IMAGE = "public.ecr.aws/docker/library/python:3.12-slim"
LOG_GROUP = "/aws/batch/job"  # Batch's default when the job definition sets no logConfiguration
TERMINAL = ("SUCCEEDED", "FAILED")

# Runs inside the container as `python -c "$WORKER_SRC"`. Python is PID 1 (the shell `exec`s it),
# so the SIGTERM from TerminateJob reaches the handler below.
WORKER_SRC = r'''
import json, os, signal, socket, sys, time
from datetime import datetime, timezone
import boto3

stop = False
def on_term(signum, frame):
    global stop
    stop = True
    print("SIGTERM received, finishing up", flush=True)
signal.signal(signal.SIGTERM, on_term)

env = os.environ
now = lambda: datetime.now(timezone.utc).isoformat()
attempt = int(env.get("AWS_BATCH_JOB_ATTEMPT", "1"))
result = {
    "run_id": env["RUN_ID"],
    "batch_job_id": env.get("AWS_BATCH_JOB_ID"),
    "attempt": attempt,
    "compute_environment": env.get("AWS_BATCH_CE_NAME"),
    "hostname": socket.gethostname(),
    "started_at": now(),
}
print(f"worker started: run {env['RUN_ID']}, attempt {attempt}", flush=True)

if env.get("FAIL_FIRST_ATTEMPT") == "1" and attempt == 1:
    print("failing attempt 1 on purpose (--fail-first-attempt)", flush=True)
    sys.exit(1)

s3 = boto3.client("s3")
if env.get("UPLOAD_URI"):
    bucket, _, key = env["UPLOAD_URI"].removeprefix("s3://").partition("/")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-1023")  # needs kms:Decrypt
        result["upload"] = {
            "ok": True,
            "bytes_read": len(obj["Body"].read()),
            "encryption": obj.get("ServerSideEncryption"),
            "kms_key": obj.get("SSEKMSKeyId"),
        }
    except Exception as exc:
        result["upload"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(f"upload check: {result['upload']}", flush=True)

steps = 0
for _ in range(int(env.get("WORK_SECONDS", "30"))):
    if stop:
        break
    time.sleep(1)
    steps += 1
    if steps % 10 == 0:
        print(f"step {steps}", flush=True)

result.update(status="stopped" if stop else "completed", steps_done=steps, finished_at=now())
s3.put_object(
    Bucket=env["RESULT_BUCKET"],
    Key=env["RESULT_KEY"],
    Body=json.dumps(result, indent=2).encode(),
    ContentType="application/json",
)
print(f"wrote s3://{env['RESULT_BUCKET']}/{env['RESULT_KEY']}", flush=True)
'''

WORKER_COMMAND = [
    "sh",
    "-c",
    "pip install --quiet --disable-pip-version-check --root-user-action=ignore boto3"
    ' && exec python -c "$WORKER_SRC"',
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--queue", required=True, help="job queue name or ARN")
    p.add_argument(
        "--job-definition",
        help="use this existing job definition (name or name:revision, e.g. from Terraform) instead of "
        "registering one; replay mode only. Bucket and prefix are read from it.",
    )
    p.add_argument("--bucket", help="S3 bucket the worker writes results to (not with --job-definition)")
    p.add_argument("--prefix", default="batch-smoke", help="key prefix for results (default: %(default)s)")
    p.add_argument("--job-role-arn", help="IAM role the container runs as (RUN-3); not with --job-definition")
    p.add_argument("--region", help="AWS region (default: from your AWS config)")
    p.add_argument("--image", default=DEFAULT_IMAGE, help="worker image; needs sh, python and pip")
    p.add_argument("--vcpus", type=int, default=1)
    p.add_argument("--memory", type=int, default=2048, help="MiB")
    p.add_argument("--timeout", type=int, default=900, help="attemptDurationSeconds, min 60")
    p.add_argument("--retries", type=int, default=1, help="retryStrategy.attempts, 1-10")
    p.add_argument("--work-seconds", type=int, default=30, help="how long the dummy worker runs")
    p.add_argument("--upload-uri", help="s3://bucket/key to read with the job role (checks KMS access)")
    p.add_argument("--stop-after", type=int, help="TerminateJob this many seconds after RUNNING")
    p.add_argument("--fail-first-attempt", action="store_true", help="attempt 1 exits 1 (tests retries)")
    p.add_argument("--repeat", type=int, default=1, help="runs to do one after another (shows instance reuse)")
    p.add_argument("--poll", type=float, default=5.0, help="seconds between DescribeJobs calls")
    p.add_argument("--keep-definition", action="store_true", help="don't deregister the job definition")
    p.add_argument("--dry-run", action="store_true", help="print the API requests; call nothing")

    r = p.add_argument_group("replay mode (--mode replay: the real worker image)")
    r.add_argument("--mode", choices=("dummy", "replay"), default="dummy")
    r.add_argument("--records-uri", help="s3://bucket/key of a v0 JSONL records file")
    r.add_argument("--component", help="component to evaluate")
    r.add_argument("--allow", action="append", default=[], help="provider candidates may use; repeat it")
    r.add_argument("--tier", choices=("smoke", "full"), default="smoke")
    r.add_argument("--run-cap", type=float, default=1.0, help="spending cap for provider calls, USD")
    r.add_argument("--speed", type=float, default=1000.0, help="replay speed; high so logged gaps don't stretch the test")
    r.add_argument("--keys-secret-arn", help="Secrets Manager secret: JSON object of provider key variables")
    r.add_argument(
        "--key-vars",
        help="comma-separated key variables to read from the secret (default: the ones for --allow; "
        "add the current model's provider if it isn't allowed, since its baseline is replayed too)",
    )
    r.add_argument("--execution-role-arn", help="role ECS uses to read the secret (needed with --keys-secret-arn)")
    args = p.parse_args(argv)
    if args.timeout < 60:
        p.error("--timeout must be at least 60 (Batch minimum)")
    if not 1 <= args.retries <= 10:
        p.error("--retries must be 1-10")
    if args.fail_first_attempt and args.retries < 2:
        p.error("--fail-first-attempt needs --retries 2 or more, or the job just fails")
    if args.job_definition:
        if args.mode != "replay":
            p.error("--job-definition needs --mode replay (the dummy worker registers its own definition)")
        clashing = [flag for flag, value in (
            ("--bucket", args.bucket), ("--job-role-arn", args.job_role_arn),
            ("--keys-secret-arn", args.keys_secret_arn), ("--execution-role-arn", args.execution_role_arn),
        ) if value]
        if args.image != DEFAULT_IMAGE:
            clashing.append("--image")
        if clashing:
            p.error(f"{', '.join(clashing)} come from the job definition; drop them with --job-definition")
    else:
        missing = [flag for flag, value in (("--bucket", args.bucket), ("--job-role-arn", args.job_role_arn)) if not value]
        if missing:
            p.error(f"{' and '.join(missing)} required (or pass --job-definition)")
    if args.mode == "replay":
        missing = [f"--{name.replace('_', '-')}" for name in ("records_uri", "component") if not getattr(args, name)]
        if not args.allow:
            missing.append("--allow")
        if args.image == DEFAULT_IMAGE and not args.job_definition:
            missing.append("--image (the worker image in ECR)")
        if missing:
            p.error(f"--mode replay needs {', '.join(missing)}")
        if args.keys_secret_arn and not args.execution_role_arn:
            p.error("--keys-secret-arn needs --execution-role-arn (ECS reads the secret with it)")
        if args.fail_first_attempt:
            p.error("--fail-first-attempt only works with the dummy worker")
    return args


KEY_VARIABLES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


def key_vars(args: argparse.Namespace) -> list[str]:
    if args.key_vars:
        return [v.strip() for v in args.key_vars.split(",") if v.strip()]
    return [KEY_VARIABLES[p] for p in args.allow if p in KEY_VARIABLES]


def job_definition_request(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    """One definition revision per run: the job role can't be overridden at SubmitJob."""
    container: dict[str, Any] = {
        "image": args.image,
        "jobRoleArn": args.job_role_arn,
        "resourceRequirements": [
            {"type": "VCPU", "value": str(args.vcpus)},
            {"type": "MEMORY", "value": str(args.memory)},
        ],
    }
    if args.mode == "dummy":
        container["command"] = WORKER_COMMAND
        container["environment"] = [{"name": "WORKER_SRC", "value": WORKER_SRC}]
    elif args.keys_secret_arn:  # replay: the image's own CMD runs docker/worker.py
        # ECS reads each JSON key of the secret into its own variable: <arn>:<json key>::
        container["secrets"] = [{"name": v, "valueFrom": f"{args.keys_secret_arn}:{v}::"} for v in key_vars(args)]
        container["executionRoleArn"] = args.execution_role_arn
    return {
        "jobDefinitionName": "mme-batch-smoke",
        "type": "container",
        "platformCapabilities": ["EC2"],
        "containerProperties": container,
        "timeout": {"attemptDurationSeconds": args.timeout},
        "retryStrategy": {"attempts": args.retries},
        "tags": {"purpose": "batch-smoke", "run_id": run_id},
    }


def submit_request(args: argparse.Namespace, run_id: str, definition_arn: str) -> dict[str, Any]:
    if args.mode == "replay":
        env = {
            "RUN_ID": run_id,
            "RECORDS_URI": args.records_uri,
            "COMPONENT": args.component,
            "ALLOW": ",".join(args.allow),
            "TIER": args.tier,
            "CAP_USD": str(args.run_cap),
            "SPEED": str(args.speed),
        }
        if not args.job_definition:  # a Terraform job definition fixes these per partner
            env |= {"RESULT_BUCKET": args.bucket, "RESULT_PREFIX": args.prefix.strip("/")}
    else:
        env = {
            "RUN_ID": run_id,
            "RESULT_BUCKET": args.bucket,
            "RESULT_KEY": result_key(args, run_id),
            "WORK_SECONDS": str(args.work_seconds),
            "FAIL_FIRST_ATTEMPT": "1" if args.fail_first_attempt else "0",
        }
        if args.upload_uri:
            env["UPLOAD_URI"] = args.upload_uri
    return {
        "jobName": f"mme-smoke-{run_id}",
        "jobQueue": args.queue,
        "jobDefinition": definition_arn,
        "containerOverrides": {"environment": [{"name": k, "value": v} for k, v in env.items()]},
        # Set per run, overriding the job definition's defaults (MOD-135)
        "timeout": {"attemptDurationSeconds": args.timeout},
        "retryStrategy": {"attempts": args.retries},
        "tags": {"purpose": "batch-smoke", "run_id": run_id},
    }


def resolve_job_definition(batch: Any, args: argparse.Namespace) -> None:
    """Fill args.bucket and args.prefix from an existing job definition's environment."""
    name, _, revision = args.job_definition.partition(":")
    found = batch.describe_job_definitions(jobDefinitionName=name, status="ACTIVE")["jobDefinitions"]
    if revision:
        found = [d for d in found if str(d["revision"]) == revision]
    if not found:
        raise SystemExit(f"no active job definition {args.job_definition!r}")
    latest = max(found, key=lambda d: d["revision"])
    env = {e["name"]: e["value"] for e in latest["containerProperties"].get("environment", [])}
    if "RESULT_BUCKET" not in env or "RESULT_PREFIX" not in env:
        raise SystemExit(f"{name}:{latest['revision']} sets no RESULT_BUCKET/RESULT_PREFIX; is it the Terraform one?")
    args.bucket, args.prefix = env["RESULT_BUCKET"], env["RESULT_PREFIX"]
    print(f"using {name}:{latest['revision']} (results in s3://{args.bucket}/{args.prefix}/)")


def result_key(args: argparse.Namespace, run_id: str) -> str:
    """The dummy writes result.json; the real worker writes status.json (and report.json beside it)."""
    name = "status.json" if args.mode == "replay" else "result.json"
    return f"{args.prefix.strip('/')}/{run_id}/{name}"


def log(msg: str, t0: float) -> None:
    print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)


def instance_id(ecs: Any, container_instance_arn: str | None) -> str | None:
    """EC2 instance behind an ECS container instance ARN (…:container-instance/<cluster>/<id>)."""
    if not container_instance_arn:
        return None
    try:
        cluster = container_instance_arn.split("/")[-2]
        found = ecs.describe_container_instances(cluster=cluster, containerInstances=[container_instance_arn])
        return found["containerInstances"][0]["ec2InstanceId"]
    except Exception as exc:  # informational only
        return f"unknown ({type(exc).__name__})"


def wait_for_job(batch: Any, ecs: Any, args: argparse.Namespace, job_id: str, t0: float) -> dict[str, Any]:
    """Poll until the job ends, logging each state change; TerminateJob if --stop-after says so."""
    last_status = None
    running_since = None
    terminated_at = None
    seen_instances: set[str] = set()
    while True:
        job = batch.describe_jobs(jobs=[job_id])["jobs"][0]
        status = job["status"]
        if status != last_status:
            reason = f" ({job['statusReason']})" if job.get("statusReason") else ""
            log(f"{status}{reason}", t0)
            last_status = status
        if status == "RUNNING" and running_since is None:
            running_since = time.monotonic()
        container = job.get("container", {})
        ec2_id = instance_id(ecs, container.get("containerInstanceArn"))
        if ec2_id and ec2_id not in seen_instances:
            seen_instances.add(ec2_id)
            log(f"attempt on instance {ec2_id}", t0)
        if status in TERMINAL:
            if terminated_at is not None:
                log(f"halted {time.monotonic() - terminated_at:.1f}s after TerminateJob (RUN-6 wants ≤ 30s)", t0)
            job["_instances"] = sorted(seen_instances)
            return job
        if (
            args.stop_after is not None
            and running_since is not None
            and terminated_at is None
            and time.monotonic() - running_since >= args.stop_after
        ):
            batch.terminate_job(jobId=job_id, reason="batch_smoke --stop-after")
            terminated_at = time.monotonic()
            log("TerminateJob sent", t0)
        time.sleep(args.poll)


def print_logs(logs: Any, job: dict[str, Any], limit: int = 40) -> None:
    stream = job.get("container", {}).get("logStreamName")
    if not stream:
        print("  (no log stream)")
        return
    options = (job.get("container", {}).get("logConfiguration") or {}).get("options") or {}
    group = options.get("awslogs-group", LOG_GROUP)
    try:
        events = logs.get_log_events(logGroupName=group, logStreamName=stream, startFromHead=True, limit=limit)
    except Exception as exc:
        print(f"  (couldn't read logs: {type(exc).__name__}: {exc})")
        return
    for event in events["events"]:
        print(f"  {event['message']}")


def read_result(s3: Any, args: argparse.Namespace, run_id: str) -> dict[str, Any] | None:
    try:
        body = s3.get_object(Bucket=args.bucket, Key=result_key(args, run_id))["Body"].read()
        return json.loads(body)
    except Exception as exc:
        print(f"  no result at s3://{args.bucket}/{result_key(args, run_id)}: {type(exc).__name__}")
        return None


def timing(job: dict[str, Any]) -> str:
    """Batch's own timestamps (ms since epoch): queue wait and run time."""
    created, started, stopped = job.get("createdAt"), job.get("startedAt"), job.get("stoppedAt")
    parts = []
    if created and started:
        parts.append(f"submit→running {(started - created) / 1000:.0f}s")
    if started and stopped:
        parts.append(f"running {(stopped - started) / 1000:.0f}s")
    return ", ".join(parts) or "no timestamps"


def run_once(clients: dict[str, Any], args: argparse.Namespace, index: int) -> dict[str, Any]:
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    print(f"\n=== run {index + 1}/{args.repeat}: {run_id} ===")
    t0 = time.monotonic()
    batch = clients["batch"]

    if args.job_definition:
        definition_arn = args.job_definition
    else:
        definition = batch.register_job_definition(**job_definition_request(args, run_id))
        definition_arn = definition["jobDefinitionArn"]
        log(f"registered {definition_arn}", t0)
    try:
        job_id = batch.submit_job(**submit_request(args, run_id, definition_arn))["jobId"]
        log(f"submitted job {job_id}", t0)
        job = wait_for_job(batch, clients["ecs"], args, job_id, t0)
    finally:
        if not args.job_definition and not args.keep_definition:
            batch.deregister_job_definition(jobDefinition=definition_arn)
            log("deregistered job definition", t0)

    container = job.get("container", {})
    print(f"\nstatus: {job['status']}  exit code: {container.get('exitCode')}  attempts: {len(job.get('attempts', []))}")
    if container.get("reason"):
        print(f"container reason: {container['reason']}")
    print(f"timing: {timing(job)}")
    print(f"instances: {', '.join(job['_instances']) or 'unknown'}")
    print("\nlog tail:")
    print_logs(clients["logs"], job)
    print(f"\n{result_key(args, run_id).rsplit('/', 1)[-1]}:")
    result = read_result(clients["s3"], args, run_id)
    if result is not None:
        print(json.dumps(result, indent=2))
    return {"run_id": run_id, "status": job["status"], "instances": job["_instances"], "timing": timing(job)}


def dry_run(args: argparse.Namespace) -> None:
    run_id = "DRYRUN"
    if args.job_definition:
        print("SubmitJob:")
        print(json.dumps(submit_request(args, run_id, args.job_definition), indent=2))
        print(f"\nWorker would write status.json under the job definition's RESULT_BUCKET/RESULT_PREFIX/{run_id}/")
        return
    definition = job_definition_request(args, run_id)
    if args.mode == "dummy":
        definition["containerProperties"]["environment"] = [
            {"name": "WORKER_SRC", "value": f"<{len(WORKER_SRC)} chars of worker source>"}
        ]
    print("RegisterJobDefinition:")
    print(json.dumps(definition, indent=2))
    print("\nSubmitJob:")
    print(json.dumps(submit_request(args, run_id, "<jobDefinitionArn from RegisterJobDefinition>"), indent=2))
    print(f"\nWorker would write s3://{args.bucket}/{result_key(args, run_id)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        dry_run(args)
        return 0
    try:
        import boto3
    except ImportError:
        print("boto3 is required: pip install boto3", file=sys.stderr)
        return 2
    session = boto3.Session(region_name=args.region)
    clients = {name: session.client(name) for name in ("batch", "ecs", "logs", "s3")}
    if args.job_definition:
        resolve_job_definition(clients["batch"], args)

    summaries = [run_once(clients, args, i) for i in range(args.repeat)]

    if args.repeat > 1:
        print("\n=== summary ===")
        for s in summaries:
            print(f"{s['run_id']}  {s['status']:9}  {s['timing']:35}  {', '.join(s['instances']) or '?'}")
        firsts = [s["instances"][0] for s in summaries if s["instances"]]
        if len(firsts) != len(set(firsts)):
            print("instance REUSED across runs: a fresh worker per run (RUN-3) is not guaranteed")
        else:
            print("each run got its own instance")
    return 0 if all(s["status"] == "SUCCEEDED" for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
