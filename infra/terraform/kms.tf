# One customer-managed key per partner (EXP-3), unless the partner entry names an existing key.
# The default key policy delegates to IAM, so the job role and orchestrator policies in iam.tf
# decide who may use each key. CloudTrail records every use.

locals {
  # Partners that need a key created. The keys of this map come only from variables, so for_each
  # on resources that depend on the keys is known at plan and import time.
  partners_needing_key = { for name, p in var.partners : name => p if p.kms_key_arn == null }
}

resource "aws_kms_key" "partner" {
  for_each = local.partners_needing_key

  description             = "MME uploads, reports and provider keys for partner ${each.key}"
  enable_key_rotation     = true
  deletion_window_in_days = 30

  tags = { partner = each.key }
}

resource "aws_kms_alias" "partner" {
  for_each = local.partners_needing_key

  name          = "alias/${var.name}-${each.key}"
  target_key_id = aws_kms_key.partner[each.key].key_id
}

locals {
  partner_kms_arn = {
    for name, p in var.partners : name => p.kms_key_arn != null ? p.kms_key_arn : aws_kms_key.partner[name].arn
  }
}
