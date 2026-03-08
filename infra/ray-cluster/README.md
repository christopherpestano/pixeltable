# Ray GPU Cluster on AWS

Spin up a cheap GPU instance running Ray, connectable from your Mac via SSH tunnel. Auto-stops after 30 minutes of idle to avoid surprise bills.

**Cost**: ~$0.16/hr (g4dn.xlarge spot with T4 16GB GPU). Stopped instances cost only ~$0.80/month for 100GB EBS storage.

## Prerequisites

- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) configured (`aws configure`)
- [Terraform](https://developer.hashicorp.com/terraform/install) installed
- An SSH key pair is created automatically if you don't have one

## Quick Start

```bash
cd infra/ray-cluster
./start.sh
```

This handles everything: creates an SSH key pair, provisions the instance, waits for Ray, opens the SSH tunnel, and configures Pixeltable. Takes ~5 minutes.

When done:

```bash
./stop.sh
```

### Manual approach

If you prefer to do things step by step:

```bash
terraform init
terraform apply -var="key_name=YOUR_KEY_PAIR_NAME"

# Wait for Ray, then connect
ssh -L 10001:localhost:10001 -L 8265:localhost:8265 \
  ec2-user@<PUBLIC_IP> -i ~/.ssh/YOUR_KEY.pem

# Add to ~/.pixeltable/config.toml:
# [ray]
# address = "ray://localhost:10001"
# runtime_env = '{"py_modules": ["/path/to/pixeltable"]}'
```

## Local Code Sync

The `start.sh` script configures `runtime_env` so your local pixeltable source is uploaded to the cluster when Python connects. This means the remote workers run **your local code**, not the pip-installed version.

**Important**: Code is synced once at process startup. If you make local changes after starting your Python process, you must restart the process to pick them up. Re-running `start.sh` is not needed — just restart Python.

## Auto-Stop

A CloudWatch alarm monitors CPU utilization. If CPU stays below 5% for 30 minutes (configurable via `idle_timeout_minutes`), the instance is **stopped** (not terminated).

To resume a stopped instance:

```bash
aws ec2 start-instances --instance-ids <INSTANCE_ID>
```

## Tear Down

```bash
terraform destroy
```

This terminates the instance and deletes all associated resources.

## Alternative: Ray Cluster Launcher

If you prefer Ray's built-in tooling over Terraform:

```bash
ray up ray-cluster.yaml
# ... use the cluster ...
ray down ray-cluster.yaml
```

Note: The Ray launcher does not support auto-stop. You must `ray down` manually.

## Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `key_name` | (required) | Your AWS EC2 key pair name |
| `region` | `us-east-1` | AWS region |
| `instance_type` | `g4dn.xlarge` | EC2 instance type |
| `spot_max_price` | `0.25` | Max spot price (USD/hr) |
| `idle_timeout_minutes` | `30` | Minutes idle before auto-stop |
