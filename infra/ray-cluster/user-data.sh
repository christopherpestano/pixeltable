#!/bin/bash
set -exo pipefail

# Log everything for debugging: tail -f /var/log/user-data.log on the instance
exec > >(tee /var/log/user-data.log) 2>&1

# Run everything as ec2-user with their full login environment
sudo -u ec2-user bash -l -c '
  set -exo pipefail

  # Activate the pytorch conda env
  PYTORCH_ENV=$(conda env list | grep -m1 pytorch | awk "{print \$1}")
  conda activate "$PYTORCH_ENV"

  # Install Ray + Pixeltable deps
  pip install --no-cache-dir --only-binary=:all: "ray[default]" pixeltable diffusers transformers accelerate

  # Start Ray head node
  ray start --head \
    --port=6379 \
    --ray-client-server-port=10001 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

  touch ~/ray-ready
'

echo "===== Ray head node is ready ====="
