#!/bin/bash
set -exo pipefail

# Log everything for debugging: tail -f /var/log/user-data.log on the instance
exec > >(tee /var/log/user-data.log) 2>&1

# Activate ec2-user's conda pytorch environment
source /home/ec2-user/anaconda3/bin/activate
conda activate "$(conda env list | grep -m1 pytorch | awk '{print $1}')"

# Install Ray and Pixeltable dependencies.
# --only-binary=:all: ensures pre-built wheels (torch wheels bundle their own CUDA).
pip install --no-cache-dir --only-binary=:all: \
  'ray[default]' \
  pixeltable \
  diffusers \
  transformers \
  accelerate

# Start Ray head node as ec2-user (not root) so workers can access GPU properly
sudo -u ec2-user "$(which ray)" start --head \
  --port=6379 \
  --ray-client-server-port=10001 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

# Marker file so the user can check if setup is complete:
#   ssh ec2-user@<ip> 'test -f /home/ec2-user/ray-ready && echo READY'
touch /home/ec2-user/ray-ready
chown ec2-user:ec2-user /home/ec2-user/ray-ready

echo "===== Ray head node is ready ====="
