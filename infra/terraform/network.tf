# Subnets: the ones given, or the default VPC's public subnets.

data "aws_vpc" "default" {
  count   = length(var.subnet_ids) == 0 ? 1 : 0
  default = true
}

data "aws_subnets" "default_public" {
  count = length(var.subnet_ids) == 0 ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default[0].id]
  }
  filter {
    name   = "map-public-ip-on-launch"
    values = ["true"]
  }
}

locals {
  subnet_ids = length(var.subnet_ids) > 0 ? var.subnet_ids : data.aws_subnets.default_public[0].ids
}

data "aws_subnet" "first" {
  id = local.subnet_ids[0]
}

# Workers take no inbound connections; they only call out (ECR, Batch, S3, providers).
resource "aws_security_group" "worker" {
  name_prefix = "${var.name}-worker-"
  description = "MME run workers: outbound only"
  vpc_id      = data.aws_subnet.first.vpc_id

  egress {
    description      = "All outbound"
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}
