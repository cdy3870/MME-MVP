# Run workers on AWS Batch (MOD-135)

Terraform for running each evaluation run as an AWS Batch job on EC2. Batch has no fee of its own
and the compute environment scales to zero, so nothing runs (or costs) between runs.

| File | What it creates |
|---|---|
| `batch.tf` | Launch template (encrypted root volume, IMDSv2), compute environment (EC2, `min_vcpus = 0`), job queue, one job definition per partner |
| `iam.tf` | EC2 instance role, one job role per partner, the `orchestrator` policy for whatever submits runs |
| `kms.tf` | One KMS key per partner (EXP-3), or bring your own via `kms_key_arn` |
| `storage.tf` | Bucket for uploads and run outputs (partner prefixes locked to the partner's key), ECR repository, worker log group, one provider-key secret per partner |
| `network.tf` | Outbound-only security group; the default VPC's public subnets unless `subnet_ids` is set |

## How a run uses it

1. The orchestrator uploads the partner's records to `s3://<bucket>/partners/<p>/uploads/`,
   encrypted with the partner's key (the bucket refuses anything else there).
2. It calls `SubmitJob` on the queue with the job definition `mme-worker-<p>`, the run's settings
   as environment overrides (`RUN_ID`, `RECORDS_URI`, `COMPONENT`, `ALLOW`, `TIER`, `CAP_USD`,
   `SPEED`) and the run's `timeout` and `retryStrategy`.
3. Batch starts an instance if none is free. The worker container (`docker/worker.py`) runs as
   the partner's job role: it reads the provider keys from the partner's secret, the records from
   S3, runs `migration-eval run`, and writes `report.json` and `status.json` to
   `partners/<p>/runs/<run id>/` under the partner's key.
4. The orchestrator polls `DescribeJobs` and reads the results from S3. Batch scales back to zero.

## Set up

Requires Terraform 1.6+ and the AWS CLI with admin-level credentials.

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars      # edit partners, instance types
terraform init
```

If you already created the `mme-run-worker` ECR repository by hand, adopt it instead of
recreating it (its images are kept):

```bash
terraform import aws_ecr_repository.worker mme-run-worker
```

Then:

```bash
terraform plan
terraform apply
```

Attach the orchestrator policy to yourself (or the server's role), so you can submit runs and
read partner data:

```bash
aws iam attach-user-policy --user-name <your IAM user> --policy-arn "$(terraform output -raw orchestrator_policy_arn)"
```

Push the worker image (from the repo root; `--platform` matters on Apple Silicon):

```bash
REPO=$(terraform -chdir=infra/terraform output -raw worker_image_repository)
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%%/*}"
docker build --platform linux/amd64 -t "${REPO}:latest" . && docker push "${REPO}:latest"
```

Store a partner's provider keys. Terraform creates the secret empty so keys never enter its state;
this reads the key without echoing it:

```bash
read -rs "KEY?Anthropic API key: "; echo
KEY="$KEY" python3 -c 'import json, os; print(json.dumps({"ANTHROPIC_API_KEY": os.environ["KEY"]}))' \
  | aws secretsmanager put-secret-value --secret-id mme/keys/test --secret-string file:///dev/stdin
unset KEY
```

## Smoke test

```bash
BUCKET=$(terraform -chdir=infra/terraform output -raw bucket)
KEY_ARN=$(terraform -chdir=infra/terraform output -json partner_kms_key_arns | python3 -c 'import json,sys; print(json.load(sys.stdin)["test"])')
aws s3 cp records.dummy.jsonl "s3://${BUCKET}/partners/test/uploads/records.jsonl" \
  --sse aws:kms --sse-kms-key-id "${KEY_ARN}"

python scripts/batch_smoke.py --mode replay --queue mme-runs --job-definition mme-worker-test \
  --records-uri "s3://${BUCKET}/partners/test/uploads/records.jsonl" \
  --component support-triage --allow anthropic --run-cap 1
```

`--sse-kms-key-id` must be the key's full ARN: the bucket policy compares the ARN exactly.

## Running from the UI

Start the server with the Batch executor, using AWS credentials that have the orchestrator policy:

```bash
pip install -e ".[aws]"
migration-eval serve --executor batch --partner test        # queue mme-runs, job definition mme-worker-test
cd frontend && npm run dev                                   # http://localhost:5173/?live=1
```

The UI works as before. Behind it, starting a run puts the upload's records in
`partners/test/uploads/` under the partner's key and submits a job; the run screen shows "Waiting for
a machine" and "Starting the worker" before the run's own stages, with the spend read from the
worker's `progress.json`. Stop calls TerminateJob; the worker writes a report of what it finished.
Restarting the server doesn't lose runs: they keep going in AWS and are picked up again.

The server can store keys in the secret but not read them back, so it can't tell which keys are set.
Tag the secret with the variable names it holds and the configure screen and the start checks use it:

```bash
aws secretsmanager tag-resource --secret-id mme/keys/test \
  --tags Key=mme:key-variables,Value=ANTHROPIC_API_KEY
```

Without the tag every provider is shown as having a key, and a missing one fails the run in the worker
with a message naming the variable.

## What this does not cover yet

* **Per-run roles (RUN-3).** Batch fixes the job role in the job definition and `SubmitJob` can't
  override it, so this sets up one role per partner. A role for each run would mean the
  orchestrator creating a role and a job definition revision per run (with `iam:CreateRole`
  under a permissions boundary) and deleting both afterwards.
* **A fresh instance per run (RUN-3).** Batch reuses running instances for the next queued job,
  possibly another partner's. The container's working data is deleted by the worker and each
  job gets its own container, but the host is shared. Separate compute environments per partner,
  or a job sized to fill a whole instance, would isolate hosts.
* **Instance deletion timing (RUN-7).** Batch terminates idle instances a few minutes after the
  queue empties, not when the report is written.
* **Retries.** Kept at 1: a retry restarts the run from the beginning with a fresh spend meter,
  so it could spend the cap again (RUN-4, RUN-5).
* **Remote state.** State is local; move it to an S3 backend before anyone else applies.

## Free Tier accounts

Accounts on the AWS Free Tier plan can only launch Free Tier eligible instance types; anything
else fails with "The specified instance type is not eligible for Free Tier" and jobs wait in
RUNNABLE. List what's allowed with:

```bash
aws ec2 describe-instance-types --filters Name=free-tier-eligible,Values=true \
  --query 'InstanceTypes[].[InstanceType,VCpuInfo.DefaultVCpus,MemoryInfo.SizeInMiB]' --output table
```

## Removing the hand-built resources

Once a smoke test passes on the Terraform setup, delete what was created by hand for the first
test (queue `mme-replay`, compute environment `mme-replay-cpu`, roles `mmeRunWorker`,
`mmeBatchInstanceRole`, `mmeBatchExecutionRole`, secret `mme-provider-keys`, bucket
`mme-batch-smoke-<account>`). Disable the queue and compute environment before deleting them.
