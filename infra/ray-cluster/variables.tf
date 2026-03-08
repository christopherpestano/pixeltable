variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (must be a GPU instance for CUDA workloads)"
  type        = string
  default     = "g4dn.xlarge"
}

variable "spot_max_price" {
  description = "Maximum hourly price for the spot instance (USD)"
  type        = string
  default     = "0.50"
}

variable "idle_timeout_minutes" {
  description = "Minutes of low CPU before the instance is automatically stopped"
  type        = number
  default     = 30
}
