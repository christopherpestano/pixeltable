output "region" {
  description = "AWS region"
  value       = var.region
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_spot_instance_request.ray_head.spot_instance_id
}

output "public_ip" {
  description = "Public IP of the Ray head node"
  value       = aws_spot_instance_request.ray_head.public_ip
}

output "ssh_tunnel_command" {
  description = "Run this to connect your local machine to the remote Ray cluster"
  value       = "ssh -L 10001:localhost:10001 -L 8265:localhost:8265 ec2-user@${aws_spot_instance_request.ray_head.public_ip} -i ~/.ssh/pixeltable-ray.pem"
}

output "pixeltable_config" {
  description = "Add this to ~/.pixeltable/config.toml (start.sh does this automatically)"
  value       = <<-EOT
    [ray]
    address = "ray://localhost:10001"
    runtime_env = '{"py_modules": ["/path/to/pixeltable"]}'
  EOT
}

output "destroy_reminder" {
  description = "Run this when done to avoid charges"
  value       = "terraform destroy -auto-approve"
}
