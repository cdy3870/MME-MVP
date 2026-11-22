# Roles:
#   instance role      the EC2 hosts: join the ECS cluster, pull from ECR, write container logs
#   <name>-worker-<p>  one job role per partner (RUN-3): read that partner's uploads and provider
#                      keys, write its run outputs, use only its KMS key
#   orchestrator       a managed policy for whatever submits runs (the server, or you with
#                      scripts/batch_smoke.py); attach it to that user or role
#
# RUN-3 asks for a role "for that run only". Batch fixes the job role in the job definition, so a
# true per-run role means creating a role and a job definition revision for every run. This file
# stops at one role per partner; see the README.

# ---- EC2 instance role -------------------------------------------------------------------------

data "aws_iam_policy_document" "ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "${var.name}-batch-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_role_policy_attachment" "instance_ecs" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.name}-batch-instance"
  role = aws_iam_role.instance.name
}

# ---- Per-partner job roles ---------------------------------------------------------------------

data "aws_iam_policy_document" "ecs_tasks_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "worker" {
  for_each = var.partners

  name               = "${var.name}-worker-${each.key}"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
  tags               = { partner = each.key }
}

data "aws_iam_policy_document" "worker" {
  for_each = var.partners

  statement {
    sid       = "ReadUploads"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.runs.arn}/partners/${each.key}/uploads/*"]
  }

  statement {
    sid       = "WriteRunOutputs"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.runs.arn}/partners/${each.key}/runs/*"]
  }

  statement {
    sid       = "UsePartnerKeyOnly"
    actions   = ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey"]
    resources = [local.partner_kms_arn[each.key]]
  }

  statement {
    sid       = "ReadProviderKeys"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.provider_keys[each.key].arn]
  }
}

resource "aws_iam_role_policy" "worker" {
  for_each = var.partners

  name   = "partner-data"
  role   = aws_iam_role.worker[each.key].id
  policy = data.aws_iam_policy_document.worker[each.key].json
}

# ---- Orchestrator ------------------------------------------------------------------------------

data "aws_iam_policy_document" "orchestrator" {
  statement {
    sid     = "SubmitRuns"
    actions = ["batch:SubmitJob"]
    resources = concat(
      [aws_batch_job_queue.runs.arn],
      [for jd in aws_batch_job_definition.worker : "${jd.arn_prefix}:*"],
    )
  }

  statement {
    sid       = "WatchAndStopRuns"
    actions   = ["batch:DescribeJobs", "batch:ListJobs", "batch:DescribeJobDefinitions", "batch:TerminateJob"]
    resources = ["*"] # DescribeJobs, ListJobs and DescribeJobDefinitions don't support resource scoping
  }

  statement {
    sid       = "PartnerData"
    actions   = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.runs.arn}/partners/*"]
  }

  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.runs.arn]
  }

  statement {
    sid       = "PartnerKeys"
    actions   = ["kms:Encrypt", "kms:Decrypt", "kms:GenerateDataKey"]
    resources = values(local.partner_kms_arn)
  }

  statement {
    sid       = "StoreProviderKeys" # `migration-eval keys add` (ACC-2): write, never read
    actions   = ["secretsmanager:PutSecretValue", "secretsmanager:DescribeSecret"]
    resources = [for s in aws_secretsmanager_secret.provider_keys : s.arn]
  }

  statement {
    sid       = "ReadWorkerLogs"
    actions   = ["logs:GetLogEvents"]
    resources = ["${aws_cloudwatch_log_group.worker.arn}:*"]
  }

  statement {
    sid       = "FindInstances" # batch_smoke.py reports which EC2 instance ran each job
    actions   = ["ecs:DescribeContainerInstances"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "orchestrator" {
  name        = "${var.name}-orchestrator"
  description = "Submit, watch and stop MME runs; read and write partner data"
  policy      = data.aws_iam_policy_document.orchestrator.json
}
