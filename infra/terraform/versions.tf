terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # Local state for now. Before a second person applies this, move state to an S3 bucket with
  # locking (backend "s3" with use_lockfile = true) so two applies can't race.
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      project    = "mme"
      managed_by = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition
}
