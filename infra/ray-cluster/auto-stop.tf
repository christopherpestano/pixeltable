resource "aws_cloudwatch_metric_alarm" "idle_stop" {
  alarm_name          = "ray-cluster-idle-stop"
  comparison_operator = "LessThanThreshold"
  # CloudWatch basic monitoring uses 5-min periods; idle_timeout_minutes should be a multiple of 5
  evaluation_periods  = var.idle_timeout_minutes / 5
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300 # 5 minutes
  statistic           = "Average"
  threshold           = 5
  alarm_description   = "Stop Ray GPU instance after ${var.idle_timeout_minutes} min of idle CPU"

  dimensions = {
    InstanceId = aws_instance.ray_head.id
  }

  alarm_actions = [
    "arn:aws:automate:${var.region}:ec2:stop",
  ]
}
