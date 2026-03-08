#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_PID_FILE="/tmp/pixeltable-ray-ssh.pid"
PXT_CONFIG="$HOME/.pixeltable/config.toml"

# All config lives in variables.tf (or terraform.tfvars).
# No env-var overrides — Terraform is the single source of truth.

info() { echo "==> $*"; }

# ── Kill SSH tunnel ─────────────────────────────────────────────────────────
if [[ -f "$SSH_PID_FILE" ]]; then
  info "Stopping SSH tunnel..."
  kill "$(cat "$SSH_PID_FILE")" 2>/dev/null || true
  rm -f "$SSH_PID_FILE"
else
  info "No SSH tunnel running."
fi

# ── Remove [ray] section from pixeltable config ────────────────────────────
if [[ -f "$PXT_CONFIG" ]] && grep -q '^\[ray\]' "$PXT_CONFIG"; then
  info "Removing [ray] section from $PXT_CONFIG"
  awk '/^\[ray\]$/{skip=1; next} /^\[/{skip=0} !skip' "$PXT_CONFIG" > "${PXT_CONFIG}.tmp"
  mv "${PXT_CONFIG}.tmp" "$PXT_CONFIG"
fi

# ── Terraform destroy ───────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
if [[ -f "terraform.tfstate" ]]; then
  info "Destroying Terraform resources..."
  terraform destroy -auto-approve -input=false
else
  info "No Terraform state found, nothing to destroy."
fi

echo ""
echo "Cluster torn down. No more charges."
