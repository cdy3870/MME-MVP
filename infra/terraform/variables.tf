variable "region" {
  description = "AWS region for everything."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Prefix for resource names."
  type        = string
  default     = "mme"
}

variable "subnet_ids" {
  description = "Subnets for the run workers. They need outbound internet (public IP or NAT) to reach ECR, Batch and the model providers. Empty means the default VPC's public subnets."
  type        = list(string)
  default     = []
}

variable "instance_types" {
  description = "EC2 instance types the compute environment may launch. On an AWS Free Tier plan only Free Tier eligible types launch (m7i-flex.large, c7i-flex.large, ...)."
  type        = list(string)
  default     = ["m7i-flex.large"]
}

variable "max_vcpus" {
  description = "Most vCPUs the compute environment runs at once; caps how many runs go in parallel and the EC2 bill."
  type        = number
  default     = 8
}

variable "worker_image_tag" {
  description = "Tag of the worker image in ECR. Prefer a git commit over latest once CI pushes images."
  type        = string
  default     = "latest"
}

variable "worker_vcpus" {
  type    = number
  default = 1
}

variable "worker_memory_mib" {
  type    = number
  default = 2048
}

variable "worker_timeout_seconds" {
  description = "Default time limit per attempt; the orchestrator overrides it per run at SubmitJob."
  type        = number
  default     = 3600
}

variable "log_retention_days" {
  description = "How long worker logs are kept in CloudWatch."
  type        = number
  default     = 14
}

variable "partner_data_retention_days" {
  description = "Uploads and run outputs under partners/ are deleted after this many days (NFR-3)."
  type        = number
  default     = 30
}

variable "partners" {
  description = <<-EOT
    One entry per partner. Each gets its own KMS key (unless kms_key_arn names an existing one),
    a job role and job definition that can use only that key, a secret for its provider keys, and
    the S3 prefix partners/<name>/. Names: lowercase letters, digits and hyphens.
  EOT
  type = map(object({
    kms_key_arn = optional(string)
  }))
  default = {
    test = {}
  }

  validation {
    condition     = alltrue([for name in keys(var.partners) : can(regex("^[a-z0-9][a-z0-9-]{0,30}$", name))])
    error_message = "Partner names must be lowercase letters, digits and hyphens, at most 31 characters."
  }
}
