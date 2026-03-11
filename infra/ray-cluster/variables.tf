variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (must be a GPU instance for CUDA workloads)"
  type        = string
  default     = "g5.xlarge"
}

variable "use_spot" {
  description = "Use spot pricing (false = on-demand)"
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Maximum hourly price for the spot instance (USD)"
  type        = string
  default     = "1.25"
}

variable "availability_zone" {
  description = "Availability zone for the instance (leave empty to use default)"
  type        = string
  default     = null
}

variable "ami_id" {
  description = "Custom AMI ID. Leave empty to use the base DLAMI."
  type        = string
  default     = ""
}

variable "idle_timeout_minutes" {
  description = "Minutes of low CPU before the instance is automatically stopped"
  type        = number
  default     = 720
}
