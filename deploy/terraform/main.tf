# BankLens Platform — Terraform stub for the host that runs it today.
#
# Honest scope: this describes the single EC2 host, its security group and
# the ECR repository that the GitHub Actions pipeline pushes to. It has been
# checked with `terraform validate` only; the live host was created by hand
# and is not imported into state. Postgres is not provisioned here: the
# platform runs an embedded cluster locally and expects a managed instance
# (RDS) in production, whose URL arrives through the secrets below.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "ap-south-1"
}

variable "instance_type" {
  type    = string
  default = "t3.small" # the platform needs more than the original t2.micro
}

variable "allowed_admin_cidr" {
  type        = string
  description = "CIDR allowed to reach SSH"
}

resource "aws_ecr_repository" "banklens" {
  name                 = "banklens"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "keep_recent" {
  repository = aws_ecr_repository.banklens.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last 10 images; the deploy script also prunes on the host"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_security_group" "banklens" {
  name        = "banklens-host"
  description = "BankLens host: HTTPS in, SSH from the admin CIDR"

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_admin_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

resource "aws_iam_role" "host" {
  name = "banklens-host"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_pull" {
  role       = aws_iam_role.host.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "host" {
  name = "banklens-host"
  role = aws_iam_role.host.name
}

resource "aws_instance" "banklens" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  vpc_security_group_ids = [aws_security_group.banklens.id]
  iam_instance_profile   = aws_iam_instance_profile.host.name

  root_block_device {
    volume_size = 30 # the silent no-op deploy of 2026-08-29 was a full 8 GB disk
  }

  user_data = <<-EOF
    #!/bin/bash
    dnf install -y docker nginx certbot
    systemctl enable --now docker nginx
  EOF

  tags = { Name = "banklens" }
}

output "host_public_ip" {
  value = aws_instance.banklens.public_ip
}

output "ecr_repository_url" {
  value = aws_ecr_repository.banklens.repository_url
}
