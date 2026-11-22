output "job_queue" {
  value = aws_batch_job_queue.runs.name
}

output "job_definitions" {
  description = "Job definition name per partner (SubmitJob uses its latest active revision)."
  value       = { for name, jd in aws_batch_job_definition.worker : name => jd.name }
}

output "bucket" {
  value = aws_s3_bucket.runs.bucket
}

output "worker_image_repository" {
  value = aws_ecr_repository.worker.repository_url
}

output "partner_kms_key_arns" {
  value = local.partner_kms_arn
}

output "provider_key_secrets" {
  description = "Set each value to a JSON object such as {\"ANTHROPIC_API_KEY\": \"...\"}."
  value       = { for name, s in aws_secretsmanager_secret.provider_keys : name => s.name }
}

output "orchestrator_policy_arn" {
  description = "Attach to the user or role that submits runs."
  value       = aws_iam_policy.orchestrator.arn
}

output "log_group" {
  value = aws_cloudwatch_log_group.worker.name
}
