# AWS Batch on EC2 (MOD-135): one compute environment that scales to zero, one job queue, and one
# job definition per partner. The orchestrator submits a job per run with the run's settings as
# environment overrides, and sets the run's time limit and retry count at SubmitJob.

# Encrypted root volume for the hosts: the containers' working data (/work) lives on it, and it is
# deleted with the instance (RUN-3, RUN-7). IMDSv2 with a hop limit of 1 keeps containers from
# reaching the instance metadata service, so they get only their job role, never the host's.
resource "aws_launch_template" "worker" {
  name_prefix = "${var.name}-worker-"

  block_device_mappings {
    device_name = "/dev/xvda" # root device of the ECS-optimized Amazon Linux images Batch uses
    ebs {
      volume_size           = 30
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  tag_specifications {
    resource_type = "instance"
    tags          = { Name = "${var.name}-run-worker" }
  }
  tag_specifications {
    resource_type = "volume"
    tags          = { Name = "${var.name}-run-worker" }
  }
}

resource "aws_batch_compute_environment" "cpu" {
  name_prefix = "${var.name}-cpu-"
  type        = "MANAGED"
  # No service_role: Batch uses its service-linked role, which also allows in-place updates.

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    min_vcpus           = 0 # scale to zero: no instances, no cost, when no run is going
    max_vcpus           = var.max_vcpus
    instance_type       = var.instance_types
    subnets             = local.subnet_ids
    security_group_ids  = [aws_security_group.worker.id]
    instance_role       = aws_iam_instance_profile.instance.arn

    launch_template {
      launch_template_id = aws_launch_template.worker.id
      version            = aws_launch_template.worker.latest_version
    }

    tags = { Name = "${var.name}-run-worker" }
  }

  update_policy {
    job_execution_timeout_minutes = 30
    terminate_jobs_on_update      = false
  }

  depends_on = [aws_iam_role_policy_attachment.instance_ecs]

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_batch_job_queue" "runs" {
  name     = "${var.name}-runs"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.cpu.arn
  }
}

resource "aws_batch_job_definition" "worker" {
  for_each = var.partners

  name                  = "${var.name}-worker-${each.key}"
  type                  = "container"
  platform_capabilities = ["EC2"]
  propagate_tags        = true

  # Defaults only: the orchestrator overrides both per run at SubmitJob.
  retry_strategy {
    attempts = 1 # keep at 1 until stages resume without repeating paid calls (RUN-4, RUN-5)
  }
  timeout {
    attempt_duration_seconds = var.worker_timeout_seconds
  }

  container_properties = jsonencode({
    image      = "${aws_ecr_repository.worker.repository_url}:${var.worker_image_tag}"
    jobRoleArn = aws_iam_role.worker[each.key].arn
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.worker_vcpus) },
      { type = "MEMORY", value = tostring(var.worker_memory_mib) },
    ]
    # Fixed per partner. Per-run settings (RUN_ID, RECORDS_URI, COMPONENT, ...) come at SubmitJob.
    environment = [
      { name = "PARTNER", value = each.key },
      { name = "RESULT_BUCKET", value = aws_s3_bucket.runs.bucket },
      { name = "RESULT_PREFIX", value = "partners/${each.key}/runs" },
      { name = "RESULT_KMS_KEY_ID", value = local.partner_kms_arn[each.key] },
      { name = "KEYS_SECRET_ID", value = aws_secretsmanager_secret.provider_keys[each.key].arn },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.worker.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = each.key
      }
    }
  })

  tags = { partner = each.key }
}
