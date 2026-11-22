# S3 bucket for uploads and run outputs, ECR repository for the worker image, worker log group,
# and one Secrets Manager secret per partner for its provider keys.
#
# Bucket layout:
#   partners/<partner>/uploads/<file>          records (v0 JSONL), written by the orchestrator
#   partners/<partner>/runs/<run id>/          report.json and status.json, written by the worker
# Everything under partners/<partner>/ must be encrypted with that partner's KMS key; the bucket
# policy refuses any other write there.

resource "aws_s3_bucket" "runs" {
  bucket = "${var.name}-runs-${local.account_id}"
}

resource "aws_s3_bucket_public_access_block" "runs" {
  bucket                  = aws_s3_bucket.runs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "runs" {
  bucket = aws_s3_bucket.runs.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Default for anything outside partners/; partner prefixes are held to their own key below.
resource "aws_s3_bucket_server_side_encryption_configuration" "runs" {
  bucket = aws_s3_bucket.runs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "runs" {
  bucket = aws_s3_bucket.runs.id

  rule {
    id     = "expire-partner-data"
    status = "Enabled"
    filter {
      prefix = "partners/"
    }
    expiration {
      days = var.partner_data_retention_days
    }
  }

  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

data "aws_iam_policy_document" "runs_bucket" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.runs.arn, "${aws_s3_bucket.runs.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }

  # A write under partners/<p>/ must name that partner's key ARN exactly. A missing header also
  # matches StringNotEquals, so unencrypted or default-encrypted writes are refused too.
  dynamic "statement" {
    for_each = var.partners
    content {
      sid       = "RequirePartnerKey${replace(title(replace(statement.key, "-", " ")), " ", "")}"
      effect    = "Deny"
      actions   = ["s3:PutObject"]
      resources = ["${aws_s3_bucket.runs.arn}/partners/${statement.key}/*"]
      principals {
        type        = "*"
        identifiers = ["*"]
      }
      condition {
        test     = "StringNotEquals"
        variable = "s3:x-amz-server-side-encryption-aws-kms-key-id"
        values   = [local.partner_kms_arn[statement.key]]
      }
    }
  }
}

resource "aws_s3_bucket_policy" "runs" {
  bucket     = aws_s3_bucket.runs.id
  policy     = data.aws_iam_policy_document.runs_bucket.json
  depends_on = [aws_s3_bucket_public_access_block.runs]
}

resource "aws_ecr_repository" "worker" {
  name                 = "${var.name}-run-worker"
  image_tag_mutability = "MUTABLE" # `latest` is re-pushed; switch to IMMUTABLE once CI tags by commit

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "worker" {
  repository = aws_ecr_repository.worker.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 10 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/${var.name}/run-workers"
  retention_in_days = var.log_retention_days
}

# Provider API keys (ACC-2): a JSON object such as {"ANTHROPIC_API_KEY": "..."}. Terraform creates
# the empty secret; the value is set outside Terraform so keys never land in state.
resource "aws_secretsmanager_secret" "provider_keys" {
  for_each = var.partners

  name                    = "${var.name}/keys/${each.key}"
  description             = "Provider API keys for partner ${each.key}, read only by its run workers"
  kms_key_id              = local.partner_kms_arn[each.key]
  recovery_window_in_days = 7

  tags = { partner = each.key }
}
