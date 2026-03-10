#!/usr/bin/env bash
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
KEY_FILE="$HOME/.ssh/pixeltable-ray.pem"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PXT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PXT_CONFIG="$HOME/.pixeltable/config.toml"
SSH_PID_FILE="/tmp/pixeltable-ray-ssh.pid"
TERRAFORM_APPLIED=false

# All config lives in variables.tf (or terraform.tfvars).
# No env-var overrides — Terraform is the single source of truth.

# ── Helpers ─────────────────────────────────────────────────────────────────
info()  { echo "==> $*"; }
error() { echo "ERROR: $*" >&2; exit 1; }

cleanup_on_failure() {
  if [[ "$TERRAFORM_APPLIED" == "true" ]] && [[ -f "$SCRIPT_DIR/terraform.tfstate" ]]; then
    echo ""
    info "Cleaning up resources after failure..."
    cd "$SCRIPT_DIR"
    terraform destroy -auto-approve -input=false 2>/dev/null || true
    info "Cleanup complete. No lingering resources."
  fi

  if [[ -f "$SSH_PID_FILE" ]]; then
    kill "$(cat "$SSH_PID_FILE")" 2>/dev/null || true
    rm -f "$SSH_PID_FILE"
  fi
}
trap cleanup_on_failure EXIT

# ── Preflight checks ───────────────────────────────────────────────────────
command -v terraform >/dev/null || error "terraform not found. Install: https://developer.hashicorp.com/terraform/install"
command -v aws       >/dev/null || error "aws CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"

aws sts get-caller-identity >/dev/null 2>&1 \
  || error "AWS credentials not configured. Run 'aws configure' first."

# ── Terraform ───────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

info "Initializing Terraform..."
terraform init -input=false

TERRAFORM_APPLIED=true

info "Requesting spot instance..."
if ! terraform apply -auto-approve -input=false; then
  error "Terraform apply failed. Spot capacity may be unavailable. Try a different region in variables.tf or terraform.tfvars."
fi

PUBLIC_IP=$(terraform output -raw public_ip)
INSTANCE_ID=$(terraform output -raw instance_id)
info "Instance $INSTANCE_ID running at $PUBLIC_IP"

# ── Wait for Ray to be ready ───────────────────────────────────────────────
info "Waiting for Ray head node to be ready (this takes 3-5 minutes)..."
for i in $(seq 1 60); do
  if [[ $i -eq 60 ]]; then
    # Last attempt: show SSH errors for debugging
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
      -i "$KEY_FILE" "ec2-user@$PUBLIC_IP" \
      'test -f /home/ec2-user/ray-ready' && break
  else
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
      -i "$KEY_FILE" "ec2-user@$PUBLIC_IP" \
      'test -f /home/ec2-user/ray-ready' 2>/dev/null && break
  fi
  if [[ $i -eq 60 ]]; then
    trap - EXIT  # Don't destroy — keep instance alive for debugging
    echo ""
    echo "WARNING: Timed out waiting for Ray, but the instance is still running."
    echo "Debug:   ssh -i $KEY_FILE ec2-user@$PUBLIC_IP 'tail -100 /var/log/user-data.log'"
    echo "Destroy: cd $SCRIPT_DIR && terraform destroy -auto-approve -input=false"
    exit 1
  fi
  sleep 10
done
info "Ray is ready."

# ── SSH tunnel ──────────────────────────────────────────────────────────────
if [[ -f "$SSH_PID_FILE" ]] && kill -0 "$(cat "$SSH_PID_FILE")" 2>/dev/null; then
  kill "$(cat "$SSH_PID_FILE")" 2>/dev/null || true
fi

info "Starting SSH tunnel (ports 10001, 8265)..."
ssh -o StrictHostKeyChecking=no -N \
  -L 10001:localhost:10001 \
  -L 8265:localhost:8265 \
  -i "$KEY_FILE" "ec2-user@$PUBLIC_IP" &
SSH_PID=$!
echo "$SSH_PID" > "$SSH_PID_FILE"

sleep 2
if ! kill -0 "$SSH_PID" 2>/dev/null; then
  error "SSH tunnel failed to start."
fi

# ── Configure Pixeltable ───────────────────────────────────────────────────
mkdir -p "$(dirname "$PXT_CONFIG")"

info "Detecting GPUs on remote node..."
NUM_GPUS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_FILE" "ec2-user@$PUBLIC_IP" 'nvidia-smi -L 2>/dev/null | wc -l' || echo 0)
NUM_GPUS=$((NUM_GPUS + 0))  # ensure it's a number
NUM_CPUS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_FILE" "ec2-user@$PUBLIC_IP" 'nproc' || echo 1)
NUM_CPUS=$((NUM_CPUS + 0))
if [[ "$NUM_GPUS" -gt 0 ]]; then
  info "Detected $NUM_GPUS GPU(s) and $NUM_CPUS CPUs on remote node"
else
  info "No GPUs detected; $NUM_CPUS CPUs on remote node"
fi

RAY_SECTION=$(cat <<EOF

[ray]
address = "ray://localhost:10001"
num_cpus = $NUM_CPUS
num_gpus = $NUM_GPUS
runtime_env = '{"py_modules": ["$PXT_ROOT"], "excludes": [".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "infra", "docs", "notebooks", "**/*.mp4", "**/*.avi"]}'
EOF
)

if [[ -f "$PXT_CONFIG" ]] && grep -q '^\[ray\]' "$PXT_CONFIG"; then
  awk '/^\[ray\]$/{skip=1; next} /^\[/{skip=0} !skip' "$PXT_CONFIG" > "${PXT_CONFIG}.tmp"
  mv "${PXT_CONFIG}.tmp" "$PXT_CONFIG"
fi
echo "$RAY_SECTION" >> "$PXT_CONFIG"
info "Configured pixeltable to use remote Ray cluster with local source sync"

# ── Done ────────────────────────────────────────────────────────────────────
trap - EXIT
echo ""
echo "============================================"
echo "  Ray GPU cluster is ready!"
echo "============================================"
echo ""
echo "  Instance:    $INSTANCE_ID"
echo "  Public IP:   $PUBLIC_IP"
echo ""
echo "  Ray dashboard: http://localhost:8265"
echo "  Pixeltable is configured -- just run your code."
echo ""
echo "  To tear down:  $SCRIPT_DIR/stop.sh"
echo "============================================"
