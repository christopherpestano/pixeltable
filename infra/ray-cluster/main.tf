terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# --- SSH key pair ---

resource "tls_private_key" "ssh" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "ray" {
  key_name   = "pixeltable-ray"
  public_key = tls_private_key.ssh.public_key_openssh
}

resource "local_file" "ssh_private_key" {
  content         = tls_private_key.ssh.private_key_pem
  filename        = pathexpand("~/.ssh/pixeltable-ray.pem")
  file_permission = "0600"
}

# --- Data sources ---

# Latest Amazon Deep Learning AMI (Amazon Linux 2) — has CUDA, cuDNN, PyTorch pre-installed
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI (Amazon Linux 2) Version *"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

data "aws_vpc" "default" {
  default = true
}

# --- Security group: SSH only ---

resource "aws_security_group" "ray_ssh" {
  name_prefix = "ray-cluster-"
  description = "SSH access only - Ray ports accessed via SSH tunnel"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ray-cluster-ssh"
  }
}

# --- Spot instance ---

resource "aws_spot_instance_request" "ray_head" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.ray.key_name
  vpc_security_group_ids = [aws_security_group.ray_ssh.id]
  spot_price             = var.spot_max_price
  spot_type              = "one-time"
  wait_for_fulfillment   = true
  availability_zone      = var.availability_zone

  # 150 GB root volume (DLAMI snapshot is ~105GB; extra space for model weights)
  root_block_device {
    volume_size = 150
    volume_type = "gp3"
  }

  user_data = file("${path.module}/user-data.sh")

  # Ensure the CloudWatch ec2:stop action works on this instance
  instance_initiated_shutdown_behavior = "terminate"

  tags = {
    Name = "ray-cluster-head"
  }

  timeouts {
    create = "10m"
  }
}
